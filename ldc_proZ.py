#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ldc_pro.py — Pipeline RAG/GPT « B + C » :
- motifs dynamiques générés par GPT (gpt-4.1-mini)
- reranking GPT sur 50 dictées candidates
- synthèse ultra-courte
- BM25 + embeddings OpenAI (text-embedding-3-large)
- cache complet (embeddings + motifs + segments)
"""

# ============================================================
#  PARTIE 1 — Imports + utilitaires + client OpenAI + cache motifs
# ============================================================

import os
import re
import json
import argparse
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import fitz  # PyMuPDF
from unidecode import unidecode
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Client OpenAI
client = OpenAI()


# ------------------------------------------------------------
#   NORMALISATION Texte
# ------------------------------------------------------------

def normalize(text: str) -> str:
    """Minuscule + suppression des accents."""
    return unidecode(text.lower())


def tokenize(text: str) -> List[str]:
    """Normalisation + séparation en tokens alphanumériques."""
    t = normalize(text)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [tok for tok in t.split() if tok]


def clean_text(s: str) -> str:
    """
    Nettoyage :
    - supprime \n
    - supprime numéros de page (3 ou 4 chiffres)
    - supprime espaces multiples
    """
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"\b\d{3,4}\b", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def parse_json_object(raw: str) -> Dict:
    """
    Parse une réponse GPT censée être du JSON strict, avec un filet de sécurité
    si le modèle ajoute accidentellement du texte autour de l'objet.
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalisation dans [0,1], ou vecteur de zéros si constant."""
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ------------------------------------------------------------
#  CACHE MOTIFS GPT (./ldc_index/motifs/)
# ------------------------------------------------------------

def load_motifs_cache(cache_dir: str, pericope_hash: str) -> Optional[Dict]:
    """
    Charge le cache GPT pour une péricope donnée, si disponible.
    pericope_hash = hash simple dérivé du texte (sha1 ou autre).
    """
    motifs_dir = os.path.join(cache_dir, "motifs")
    os.makedirs(motifs_dir, exist_ok=True)
    path = os.path.join(motifs_dir, f"{pericope_hash}.json")

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_motifs_cache(cache_dir: str, pericope_hash: str, data: Dict):
    """Enregistre le dictionnaire {themes, keywords} dans le cache."""
    motifs_dir = os.path.join(cache_dir, "motifs")
    os.makedirs(motifs_dir, exist_ok=True)
    path = os.path.join(motifs_dir, f"{pericope_hash}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ------------------------------------------------------------
#  DATACLASSES
# ------------------------------------------------------------

@dataclass
class Dictee:
    tome: Optional[int]
    date: Optional[str]
    page_start: int
    text: str


@dataclass
class Segment:
    id: int
    dictee_index: int
    text: str
    tokens: List[str]

# ============================================================
#  PARTIE 2 — Motifs dynamiques GPT + Extraction PDF → Dictées → Segments
# ============================================================

# ------------------------------------------------------------
#  MOTIFS DYNAMIQUES GPT
# ------------------------------------------------------------

def hash_pericope(text: str) -> str:
    """Crée un hash court (hex) pour identifier la péricope dans le cache motifs."""
    import hashlib
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[:16]


def detect_dynamic_motifs_gpt(evangelium_text: str,
                              cache_dir: str = "ldc_index",
                              model_name: str = "gpt-4.1") -> Tuple[List[str], List[str]]:
    """
    Analyse la péricope avec GPT-4.1 pour extraire :
       - themes : motifs bibliques / théologiques profonds (2 à 5 mots)
       - keywords : 10 à 20 mots-clés concrets (tokens normalisés)

    Exigences garanties par le prompt :
       - Si un nom biblique apparaît dans la péricope (Noé, Abraham, Jonas…),
         GPT DOIT l'inclure dans keywords.
       - GPT DOIT ajouter au moins 4 mots associés (deluge, arche, arcenciel…).
       - Keywords en minuscules, sans accents, un seul token par mot.
       - Retour en JSON strict.

    Cache : ldc_index/motifs/<hash>.json
    """

    # --- 0. Cache : vérifier si motifs déjà générés ---
    pericope_hash = hash_pericope(evangelium_text)
    cached = load_motifs_cache(cache_dir, pericope_hash)
    if cached is not None:
        print(f"[INFO] Motifs dynamiques chargés depuis cache ({pericope_hash}).")
        return cached.get("themes", []), cached.get("keywords", [])

    # --- 1. Prompt optimisé GPT-4.1 ---
    prompt = f"""
Tu es un exégète catholique expert en typologie biblique, dans la symbolique chretienne, la mystique et expert de la divine volonté.

On te donne un passage d'Évangile. Tu dois renvoyer EXCLUSIVEMENT un JSON contenant :

1) "themes" :
   - 2 à 5 mots courts décrivant les motifs bibliques profonds du passage.
   - Ces thèmes DOIVENT inclure tout nom biblique explicitement présent dans le texte :
     (par ex: noe, abraham, jonas, moise, david, bartimee, lazare, marie…)
   - Ils peuvent inclure des images symboliques majeures (lumiere, foi, aveuglement, desert, eau, feu, pain…).

2) "keywords" :
   - 10 à 20 mots simples, en minuscules, sans accents, sans espace.
   - Mélange de termes littéraux du texte (ex: foi, voir, aveugle, maison, crier)
     ET de termes bibliques/typiques associés aux thèmes détectés.
   - SI un nom biblique est présent dans la péricope,
       tu DOIS l’inclure dans keywords,
       ET ajouter au moins quatre mots-clés directement associés :
         (ex pour "noe": deluge, arche, eaux, arcenciel, alliance, renouveau)
         (ex pour "abraham": sacrifice, foi, montagne, promesse)
         (ex pour "jonas": poisson, mer, conversion, ninive)
         (ex pour "bartimee": crier, vue, foi, lumiere)
   - Objectif : aider une recherche thématique (RAG).

Exigences de format :
- Retourne UNIQUEMENT un JSON strict :
  {{
    "themes": [...],
    "keywords": [...]
  }}
- Pas d’autres phrases, pas d’explications.

Passage d'Évangile :
\"\"\"{evangelium_text}\"\"\""""

    # --- 2. Appel GPT-4.1 ---
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "Tu es un théologien catholique rigoureux, précis et expert en typologie et de la divine volonté de louisa picaretta."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw = resp.choices[0].message.content.strip()

        # Parse JSON strict
        data = parse_json_object(raw)

        # --- 3. Normalisation des résultats ---
        themes_raw = data.get("themes", [])
        keywords_raw = data.get("keywords", [])

        # Normaliser les thèmes
        themes = [normalize(t).strip() for t in themes_raw if t.strip()]

        # Normaliser les keywords → tokens simples
        kw_set = set()
        for kw in keywords_raw:
            for tk in tokenize(kw):
                if len(tk) > 2:  # éviter "de", "et", "en"
                    kw_set.add(tk)

        keywords = sorted(kw_set)

        # --- 4. Enregistrement dans le cache ---
        save_motifs_cache(
            cache_dir,
            pericope_hash,
            {"themes": themes, "keywords": keywords}
        )

        print(f"[DEBUG] Thèmes dynamiques GPT-4.1 : {themes}")
        print(f"[DEBUG] Mots-clés dynamiques GPT-4.1 : {keywords}")

        return themes, keywords

    except Exception as e:
        print("[ERREUR] detect_dynamic_motifs_gpt (GPT-4.1) a échoué :", e)
        return [], []


# ------------------------------------------------------------
#  EXTRACTION PDF → PAGES
# ------------------------------------------------------------

def extract_pages(pdf_path: str) -> List[str]:
    """
    Ouvre le PDF via PyMuPDF et renvoie une liste de pages texte.
    """
    print(f"[INFO] Lecture du PDF : {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    print(f"[INFO] Nombre de pages extraites : {len(pages)}")
    return pages


# ------------------------------------------------------------
#  DÉCOUPE EN DICTÉES (via dates + tome)
# ------------------------------------------------------------

MONTHS_FR = (
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre",
    "décembre", "decembre"
)

DATE_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS_FR)})\s+((18|19|20)\d{{2}})\b",
    re.IGNORECASE,
)
TOME_RE = re.compile(r"Tome\s+(\d+)", re.IGNORECASE)


def parse_dictees(pages: List[str]) -> List[Dictee]:
    """
    Parcourt le texte page par page et découpe en dictées :
    - Une dictée commence dès qu’une date (JJ mois AAAA) est rencontrée.
    - Le tome courant est actualisé lorsqu'un "Tome X" est détecté.
    """
    dictees: List[Dictee] = []
    cur_tome: Optional[int] = None
    cur_date: Optional[str] = None
    cur_page = 0
    buf: List[str] = []
    started = False

    for p_i, text in enumerate(pages):
        for line in text.splitlines():
            l = line.strip()

            # Détection "Tome X"
            tm = TOME_RE.search(l)
            if tm:
                try:
                    cur_tome = int(tm.group(1))
                except ValueError:
                    pass

            # Détection date (souple : date présente quelque part dans la ligne)
            dm = DATE_RE.search(l)
            if dm:
                # Clôturer dictée précédente
                if started and buf:
                    dictees.append(
                        Dictee(
                            tome=cur_tome,
                            date=cur_date,
                            page_start=cur_page,
                            text="\n".join(buf).strip()
                        )
                    )
                    buf = []

                cur_date = dm.group(0)  # ex: "12 mars 1930"
                cur_page = p_i
                started = True
                continue

            if started:
                buf.append(line)

    # Dernière dictée
    if started and buf:
        dictees.append(
            Dictee(
                tome=cur_tome,
                date=cur_date,
                page_start=cur_page,
                text="\n".join(buf).strip()
            )
        )

    print(f"[INFO] Dictées extraites : {len(dictees)}")
    return dictees


# ------------------------------------------------------------
#  SEGMENTATION — dictées → segments de 200 mots
# ------------------------------------------------------------

def build_segments(dictees: List[Dictee],
                   seg_len: int = 250,
                   stride: int = 120) -> List[Segment]:
    """
    Découpe chaque dictée en segments glissants :
      - seg_len = ~200 mots
      - stride  = décalage (overlap ~80 mots)
    """
    segments: List[Segment] = []
    sid = 0

    for d_idx, d in enumerate(dictees):
        words = d.text.split()
        idx = 0
        while idx < len(words):
            chunk = words[idx: idx + seg_len]
            if not chunk:
                break
            seg_text = " ".join(chunk)
            tokens = tokenize(seg_text)
            segments.append(
                Segment(
                    id=sid,
                    dictee_index=d_idx,
                    text=seg_text,
                    tokens=tokens
                )
            )
            sid += 1
            if idx + seg_len >= len(words):
                break
            idx += stride

    print(f"[INFO] Segments construits : {len(segments)}")
    return segments
# ============================================================
#  PARTIE 3 — BM25 + Embeddings OpenAI + Scoring Hybride (keywords GPT)
# ============================================================

# ------------------------------------------------------------
#  BM25
# ------------------------------------------------------------

def build_bm25(segments: List[Segment]) -> BM25Okapi:
    """
    Construit un index BM25 sur la liste des segments.
    """
    corpus = [s.tokens for s in segments]
    bm25 = BM25Okapi(corpus)
    print("[INFO] Index BM25 construit.")
    return bm25


# ------------------------------------------------------------
#  EMBEDDINGS — text-embedding-3-large
# ------------------------------------------------------------

def embed_texts_openai(texts: List[str],
                       model_name: str = "text-embedding-3-large",
                       batch_size: int = 64) -> np.ndarray:
    """
    Encode une liste de textes via OpenAI Embeddings (puissant, économique).
    """
    all_vecs = []
    total = len(texts)
    print(f"[INFO] Encodage de {total} segments avec {model_name}…")

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model=model_name,
                input=batch
            )
        except Exception as e:
            raise RuntimeError(f"[ERREUR] Embeddings OpenAI batch {i}: {e}")

        for d in resp.data:
            all_vecs.append(d.embedding)

        print(f"[DEBUG] Batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} encodé.")

    embs = np.array(all_vecs, dtype=np.float32)
    print(f"[INFO] Embeddings shape = {embs.shape}")
    return embs


def build_embeddings_for_segments(segments: List[Segment],
                                  model_name: str = "text-embedding-3-large") -> np.ndarray:
    """
    Encode tous les segments et renvoie la matrice d'embeddings.
    """
    texts = [s.text for s in segments]
    return embed_texts_openai(texts, model_name=model_name)


# ------------------------------------------------------------
#  Scoring hybride BM25 + embeddings + bonus keywords GPT
# ------------------------------------------------------------

def score_segments_with_keywords(evangelium_text: str,
                                 keywords: List[str],
                                 segments: List[Segment],
                                 bm25: BM25Okapi,
                                 embs: np.ndarray,
                                 top_k_segments: int = 200
                                 ) -> List[Tuple[float, Segment]]:
    """
    Score chaque segment (BM25 + embeddings + bonus mots-clés GPT).
    Puis renvoie les top_k_segments meilleurs segments.

    Pipeline :
      - q_tokens = tokenize(péricope)
      - q_emb = embedding(enriched_text = péricope + keywords)
      - lexical_score = BM25
      - semantic_score = dot(emb, q_emb)
      - bonus = nombre de mots-clés dans le segment
      - final_score = 0.45*BM25 + 0.45*semantic + 0.10*bonus
    """

    # --- 1) Construire la requête enrichie pour embeddings ---
    if keywords:
        enriched = evangelium_text + "\nKEYWORDS: " + " ".join(keywords)
    else:
        enriched = evangelium_text

    try:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=[enriched]
        )
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"[ERREUR] Embedding de la requête enrichie : {e}")

    # --- 2) Scoring BM25 ---
    bm25_query = evangelium_text + " " + " ".join(keywords or [])
    q_tokens = tokenize(bm25_query)

    bm_scores = np.array(bm25.get_scores(q_tokens), dtype=float)

    # --- 3) Scoring sémantique ---
    sem_scores = embs @ q_emb  # produit scalaire = mesure de proximité

    # --- 4) Bonus des mots-clés GPT ---
    keyword_set = set(keywords)
    bonus = np.zeros(len(segments), dtype=float)
    if keyword_set:
        for seg in segments:
            overlap = len(set(seg.tokens) & keyword_set)
            if overlap > 0:
                bonus[seg.id] = overlap

    # --- 5) Normalisation ---
    lex_n = _norm(bm_scores)
    sem_n = _norm(sem_scores)
    bonus_n = _norm(bonus)

    # --- 6) Pondérations ---
    w_lex   = 0.45
    w_sem   = 0.45
    w_bonus = 0.10

    final_score = w_lex * lex_n + w_sem * sem_n + w_bonus * bonus_n

    # --- 7) Sélection des meilleurs segments ---
    idx_sorted = np.argsort(final_score)[::-1]
    top_idx = idx_sorted[:top_k_segments]

    print(f"[INFO] Top {top_k_segments} segments sélectionnés (avant regroupement dictées).")

    results = [(final_score[i], segments[i]) for i in top_idx]
    return results

# ============================================================
#  PARTIE 4 — Regroupement dictées & make_excerpt optimisée
# ============================================================

def group_segments_by_dictee(ranked_segments: List[Tuple[float, Segment]],
                             dictees: List[Dictee],
                             top_k_dicts_pre_rerank: int = 50
                             ) -> List[Tuple[float, Dictee, Segment]]:
    """
    Regroupe les segments scorés par dictée :
      - pour chaque dictée, conserve le segment au score maximal
      - retourne jusqu'à top_k_dicts_pre_rerank dictées candidates

    Cette liste est ensuite soumise à GPT pour le reranking (50 → 5).
    """
    by_dict: Dict[int, List[Tuple[float, Segment]]] = {}

    for score, seg in ranked_segments:
        by_dict.setdefault(seg.dictee_index, []).append((float(score), seg))

    rows: List[Tuple[float, Dictee, Segment]] = []
    for d_idx, hits in by_dict.items():
        hits.sort(key=lambda x: x[0], reverse=True)
        best_score, best_seg = hits[0]
        supporting_scores = [s for s, _ in hits[1:4]]
        support_bonus = 0.0
        if supporting_scores:
            support_bonus = 0.18 * sum(supporting_scores) / len(supporting_scores)
        density_bonus = min(len(hits), 5) * 0.01
        score = best_score + support_bonus + density_bonus
        dictee = dictees[d_idx]
        rows.append((score, dictee, best_seg))

    rows.sort(key=lambda x: x[0], reverse=True)
    candidates = rows[:top_k_dicts_pre_rerank]

    print(f"[INFO] Dictées candidates (pré-rerank GPT) : {len(candidates)}")
    return candidates


def make_excerpt(dictee: Dictee,
                 seg: Segment,
                 max_chars: int = 1500) -> str:
    """
    Produit un extrait lisible de la dictée autour du segment sélectionné.

    - Nettoie le texte complet de la dictée.
    - Cherche où se situe le segment dans la dictée (via un "probe" de 80 caractères).
    - Remonte au début de la phrase précédente (ponctuation forte).
    - Descend jusqu'à 5 fins de phrases après.
    - Tronque proprement à max_chars sans couper au milieu d'une phrase.

    Résultat : 3–6 phrases complètes, propres, adaptées à l'évaluation par GPT.
    """
    full = clean_text(dictee.text)
    snippet = clean_text(seg.text)

    if not snippet:
        # Si le segment est vide, on prend le début de la dictée
        excerpt = full[:max_chars]
        cut = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
        return excerpt[:cut+1].strip() if cut > 0 else excerpt.strip()

    probes = [snippet[:120], snippet[:80], snippet[:50]]
    words = snippet.split()
    if len(words) >= 10:
        probes.append(" ".join(words[:10]))
    if len(words) >= 7:
        probes.append(" ".join(words[:7]))

    pos = -1
    probe = ""
    for candidate in probes:
        candidate = clean_text(candidate)
        if len(candidate) < 20:
            continue
        found = full.find(candidate)
        if found != -1:
            pos = found
            probe = candidate
            break

    if pos == -1:
        # Impossible d'aligner le segment : mieux vaut garder le segment
        # sélectionné que revenir arbitrairement au début de la dictée.
        excerpt = snippet[:max_chars]
        cut = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
        return excerpt[:cut+1].strip() if cut > 0 else excerpt.strip()

    # --------------------------------------------------------
    # 1. Début de l'extrait : début de phrase précédente
    # --------------------------------------------------------
    last_dot = full.rfind(".", 0, pos)
    last_exc = full.rfind("!", 0, pos)
    last_q   = full.rfind("?", 0, pos)

    starts = [c for c in (last_dot, last_exc, last_q) if c != -1]
    if starts:
        start = max(starts) + 1
    else:
        start = 0

    # --------------------------------------------------------
    # 2. Fin de l'extrait : jusqu'à 5 fins de phrases après
    # --------------------------------------------------------
    end_candidates = []
    cursor = pos + len(probe)

    for _ in range(5):  # chercher jusqu'à 5 fins de phrase
        d = full.find(".", cursor)
        e = full.find("!", cursor)
        q = full.find("?", cursor)
        cands = [x for x in (d, e, q) if x != -1]
        if not cands:
            break
        nxt = min(cands)
        end_candidates.append(nxt)
        cursor = nxt + 1

    if end_candidates:
        end = min(max(end_candidates) + 1, len(full))
    else:
        end = min(len(full), start + max_chars)

    excerpt = full[start:end].strip()

    # --------------------------------------------------------
    # 3. Tronquer proprement à max_chars
    # --------------------------------------------------------
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
        cut = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
        if cut != -1:
            excerpt = excerpt[:cut+1]

    return clean_text(excerpt)

# ============================================================
#  PARTIE 5 — Reranking GPT (50 → 5) + Synthèse courte
# ============================================================

_RERANK_SYSTEM_PROMPT = """Tu es un théologien catholique, spécialiste de la Divine Volonté (Fiat) telle qu'expliquée dans le « Livre du Ciel » de Luisa Piccarreta. Ton rôle est de sélectionner, parmi une liste d'extraits du Livre du Ciel proposés par leurs identifiants (ID), ceux qui éclairent le mieux une péricope évangélique fournie par l'utilisateur.

CONTEXTE DE LA MISSION

Le « Livre du Ciel » est l'ensemble des dictées reçues par Luisa Piccarreta (1865–1947), mystique italienne. Ces dictées développent une théologie originale de la Divine Volonté : vivre dans le « Fiat », accomplir tous ses actes en union avec Jésus dans la Volonté divine, participer à l'œuvre de la Rédemption par des actes qui prolongent ceux du Verbe incarné, restaurer l'ordre originel voulu par Dieu sur la création, faire advenir le Règne du Fiat « sur la terre comme au Ciel ».

Chaque extrait du Livre du Ciel est rattaché à un tome et une date de dictée. Plusieurs extraits peuvent reprendre une même image (lumière, soleil, mer, ronde, Fiat, vie intérieure) sous des angles très différents : intérieur, contemplatif, eschatologique, christologique, mariologique, ecclésial, ascétique, réparateur. Ton discernement consiste précisément à choisir, parmi ces angles, ceux qui servent le mieux la péricope évangélique soumise.

CRITÈRES DE SÉLECTION (par ordre décroissant d'importance)

1. Adéquation thématique forte.
   L'extrait reprend, approfondit ou prolonge le motif central de la péricope (geste, parole, image, personnage, lieu). Le lien doit être net : explicite, ou très étroitement implicite. Les liens artificiels, forcés, ou tenant à un simple mot de surface sont à exclure.

2. Lumière propre de la Divine Volonté.
   L'extrait apporte un éclairage doctrinal qui appartient au registre du Livre du Ciel : Fiat intérieur, actes dans la Divine Volonté, fusion, réparation, prévenance d'amour, Règne du Fiat, conformité à la Volonté de Dieu, soleil de la Volonté divine. Il ne se borne pas à une moralisation générique qui pourrait s'appliquer à n'importe quel autre texte.

3. Concordance avec les thèmes et mots-clés signalés par l'utilisateur.
   Lorsque l'utilisateur fournit des thèmes et des mots-clés (issus d'une analyse préalable de la péricope), ces indications orientent fortement la sélection sans s'y substituer mécaniquement : un extrait à thème adjacent mais profondément aligné spirituellement reste éligible. À l'inverse, un extrait qui ne fait qu'effleurer plusieurs mots-clés en surface ne suffit pas.

4. Complémentarité des extraits retenus.
   Lorsque l'on doit en choisir plusieurs, ils doivent éclairer la péricope sous des angles distincts : intérieur, contemplatif, pratique, eschatologique, christologique, marial, ecclésial. Évite de retenir deux extraits qui développent essentiellement la même idée. La diversité d'angles est plus précieuse que la répétition d'un même point fort.

5. Ancrage direct sur le cœur de la péricope.
   Au moins un extrait doit éclairer le cœur de la péricope (geste, parole, événement central), et non un détail périphérique. Ne pas se contenter d'extraits qui ne touchent qu'un mot ou qu'une image secondaire.

À PROSCRIRE

- Choisir un extrait dont le lien avec la péricope est artificiel, forcé, ou ne tient qu'à un mot de surface.
- Privilégier mécaniquement les extraits comportant le plus grand nombre de mots-clés, au détriment de l'unité de sens.
- Retenir deux extraits qui disent essentiellement la même chose.
- Imposer une progression artificielle entre les extraits (introduction / développement / conclusion).
- S'éloigner du sens littéral et spirituel de la péricope au profit d'une lecture privée du Livre du Ciel.
- Surinterpréter un extrait au-delà de ce qu'il dit réellement.
- Projeter sur les extraits des idées qui n'y figurent pas.

FORMAT DE SORTIE OBLIGATOIRE

Tu retournes EXCLUSIVEMENT un objet JSON strict, sans aucun texte autour, sans markdown, sans commentaire, sans préambule, de la forme :

{"ids": [id1, id2, ...]}

où chaque id est un entier correspondant à l'un des IDs d'extraits fournis dans le message utilisateur. La liste doit contenir exactement le nombre d'extraits demandés par l'utilisateur (ou moins, uniquement si la liste candidate est plus courte). Le JSON doit être directement parseable par json.loads() en Python : guillemets droits, pas de virgule finale, pas de clé en plus.

POSTURE DE TRAVAIL

Tu travailles avec sobriété, fidélité au texte biblique et respect strict de la doctrine catholique. Tu fais confiance à la richesse propre du Livre du Ciel sans la surinterpréter. Tu privilégies toujours le sens fort et juste sur le sens spectaculaire ou surprenant. En cas d'hésitation entre deux extraits comparables, tu préfères celui dont le langage est le plus accessible et l'enracinement biblique le plus explicite. Tu te tiens à distance du pathos pieux comme de l'analyse froide : ton discernement est celui d'un exégète à la fois rigoureux et contemplatif."""


def rerank_with_gpt(evangelium_text: str,
                    candidate_dicts: List[Tuple[float, Dictee, Segment]],
                    themes: List[str],
                    keywords: List[str],
                    model_name: str = "gpt-4.1",
                    top_k_final: int = 5) -> List[int]:
    """
    Demande à GPT de choisir les top_k_final dictées les plus pertinentes
    parmi la liste candidate_dicts (jusqu'à 50 dictées).

    Retourne une liste d'indices (0-based) dans candidate_dicts.

    Optimisation : les consignes stables sont placées dans le system message
    (préfixe identique entre requêtes → bénéficie du prompt caching OpenAI
    automatique pour les préfixes ≥ 1024 tokens).
    """

    # 1) Préparer les extraits pour GPT
    sections = []
    for i, (score, dictee, seg) in enumerate(candidate_dicts):
        excerpt = make_excerpt(dictee, seg)
        sections.append(
            f"ID {i+1} — Tome {dictee.tome} — {dictee.date}\n{excerpt}"
        )
    joined = "\n\n---\n\n".join(sections)

    # 2) Contexte des thèmes / mots-clés dynamiques
    theme_info = ""
    if themes:
        theme_info = "Thèmes principaux du passage : " + ", ".join(themes) + ".\n"
    keyword_info = ""
    if keywords:
        keyword_info = "Mots-clés associés : " + ", ".join(keywords) + ".\n"

    # 3) Message utilisateur : uniquement la partie variable
    user_prompt = f"""PÉRICOPE ÉVANGÉLIQUE À ÉCLAIRER :

\"\"\"{evangelium_text}\"\"\"

{theme_info}{keyword_info}
EXTRAITS CANDIDATS DU LIVRE DU CIEL (chaque extrait porte un ID, un tome et une date) :

{joined}

CONSIGNE : Sélectionne exactement {top_k_final} extraits parmi les candidats ci-dessus, en appliquant les critères et contraintes définis dans tes instructions système. Retourne uniquement le JSON {{"ids": [...]}}."""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # pour un comportement stable
        )
        raw = resp.choices[0].message.content.strip()
        data = parse_json_object(raw)
        ids = data.get("ids", [])

        print("[DEBUG] Réponse brute GPT :", raw)
        print("[DEBUG] IDs renvoyés par GPT :", ids)

        indices: List[int] = []
        for i in ids:
            if isinstance(i, int) and 1 <= i <= len(candidate_dicts):
                indices.append(i - 1)

        if not indices:
            print("[WARN] GPT n'a pas renvoyé d'IDs valides, on garde les top locaux.")
            indices = list(range(min(top_k_final, len(candidate_dicts))))

        seen = set(indices)
        for fallback_idx in range(len(candidate_dicts)):
            if len(indices) >= min(top_k_final, len(candidate_dicts)):
                break
            if fallback_idx not in seen:
                indices.append(fallback_idx)
                seen.add(fallback_idx)

        return indices[:min(top_k_final, len(candidate_dicts))]

    except Exception as e:
        print("[ERREUR] rerank_with_gpt :", e)
        print("[INFO] On garde les top locaux par défaut.")
        return list(range(min(top_k_final, len(candidate_dicts))))


def summarize_with_gpt(evangelium_text: str,
                       passages: List[str],
                       model_name: str = "gpt-4.1") -> str:
    """
    Produit une synthèse UNIQUE, très concise (5–6 lignes),
    intégrant les cinq passages retenus.
    """
    if client is None:
        return "Synthèse non disponible (client OpenAI non configuré)."

    joined = "\n\n---\n\n".join(passages)

    prompt = f"""

On te donne :
1) Un passage d’Évangile ou autre texte théologique
2) Cinq extraits du « Livre du Ciel » qui l’éclairent.

Tâche :
Rédige une synthèse UNIQUE, claire et concise (6 à 8 lignes maximum),
qui montre comment les extraits du « Livre du Ciel » éclairent
le mouvement spirituel de la péricope évangélique.

Exigences :

1) Commence par identifier le mouvement principal de la péricope
   (ce qui est révélé, demandé ou mis en lumière),
   sans supposer nécessairement une action directe de Jésus.

2) Montre ensuite comment les extraits du Livre du Ciel
   approfondissent ce mouvement à la lumière de la Divine Volonté
   (Fiat, vie intérieure, actes dans la Volonté divine, réparation, union),
   en restant appuyé sur le contenu réel des extraits.

3) Conclus par une ouverture spirituelle sobre :
   ce que cette péricope, éclairée par ces extraits,
   appelle à vivre intérieurement aujourd’hui dans la Divine Volonté.

Contraintes de style :
- Un seul paragraphe continu.
- Pas de liste, pas de titres.
- Pas de formules générales abstraites non appuyées sur le texte.
- Lien explicite avec le contenu des extraits.
- Style théologique, clair, sobre et contemplatif.

Passage :
\"\"\"{evangelium_text}\"\"\"


Extraits du Livre du Ciel :
\"\"\"{joined}\"\"\"



"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "Tu es un exégète catholique, expet en mystique, spécialiste de la divine volonté."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Synthèse indisponible : {e}"
    

_EXPLAIN_SYSTEM_PROMPT = """Tu es un théologien catholique, exégète précis et synthétique, spécialiste de la Divine Volonté (Fiat) telle qu'enseignée à Luisa Piccarreta dans le « Livre du Ciel ». Ton rôle est de produire, pour CHAQUE extrait du Livre du Ciel qui te sera soumis par l'utilisateur, une courte explication (2 à 3 phrases) qui explicite en quoi ce passage éclaire la péricope évangélique fournie, à la lumière de la doctrine de la Divine Volonté.

CONTEXTE DE LA MISSION

Le « Livre du Ciel » est l'ensemble des dictées reçues par Luisa Piccarreta (1865–1947). Ces dictées développent une théologie originale du Fiat : vivre dans la Divine Volonté, accomplir tous ses actes en union avec Jésus, restaurer l'ordre originel voulu par Dieu, prolonger l'œuvre de la Rédemption par des actes intérieurs, faire advenir le Règne du Fiat. Ton commentaire doit toujours s'appuyer sur le contenu concret de l'extrait (un mot, une image, un geste, une phrase de Jésus à Luisa), jamais sur des généralités spirituelles plaquées de l'extérieur.

L'utilisateur t'enverra dans son message :
1) Une péricope évangélique (le passage de l'Évangile à éclairer).
2) Une liste d'extraits numérotés du Livre du Ciel (Passage 1, Passage 2, etc.).

Ta tâche est de produire UNE explication par extrait, dans l'ordre exact où ils sont fournis, en respectant strictement les contraintes ci-dessous.

CONTRAINTES DE STYLE ET DE CONTENU

1. Aucune formule générique d'introduction.
   Sont strictement INTERDITES les amorces du type :
   - "L'extrait du Livre du Ciel éclaire la péricope..."
   - "Ce passage du Livre du Ciel nous montre que..."
   - "Dans cet extrait, Luisa Piccarreta explique..."
   - "Ce texte est une belle illustration de..."
   Commence directement par le contenu théologique :
   - "Ce passage montre que..."
   - "Ici Jésus révèle à Luisa que..."
   - "Le geste évoqué dans la péricope trouve son écho dans..."
   - "L'image de [X] reprise ici éclaire le moment où..."

2. Aucune répétition du texte du passage.
   Tu ne paraphrases pas l'extrait, tu en dégages l'éclairage propre. Le lecteur a déjà lu l'extrait : il attend une mise en lumière, pas un résumé.

3. Appui concret obligatoire.
   Chaque explication doit se référer explicitement à un ou deux éléments concrets réellement présents dans l'extrait : un mot précis, une image, un geste, une parole de Jésus à Luisa, une comparaison. Si tu ne peux nommer aucun élément concret de l'extrait, c'est que ton explication est trop générale : reprends-la.

4. Une seule idée théologique centrale par explication.
   Évite de développer plusieurs angles dans la même réponse. Choisis l'angle qui éclaire le mieux la péricope et tiens-toi à lui. Mieux vaut une idée juste qu'une accumulation floue.

5. Lien explicite avec la péricope.
   Le rapport entre l'extrait et le passage évangélique doit être formulé clairement (par un mot, une formule, une analogie). L'explication ne reste jamais en l'air, suspendue à l'extrait seul. Si le lien n'est pas formulé, l'explication a manqué sa cible.

6. Longueur : 2 à 3 phrases, pas plus.
   Style sobre, précis, contemplatif. Pas de superlatifs gratuits, pas de pathos, pas d'exclamations.

7. Doctrine de la Divine Volonté.
   Lorsque c'est pertinent, mobilise les notions propres au Livre du Ciel (Fiat, actes dans la Volonté, fusion, réparation, Règne du Fiat, prévenance d'amour, soleil de la Volonté divine, vie intérieure). Mais seulement si l'extrait les évoque réellement. N'impose pas une grille doctrinale étrangère au texte.

À PROSCRIRE ABSOLUMENT

- Les formules abstraites non appuyées sur l'extrait ("ce passage révèle une grande profondeur spirituelle...").
- Les jugements généraux sans référence concrète ("c'est un texte particulièrement édifiant...").
- Les paraphrases déguisées en explication.
- Les digressions doctrinales sans lien avec la péricope.
- L'introduction d'idées absentes de l'extrait.
- Le pathos pieux et les exclamations dévotionnelles.
- Les répétitions d'une explication à l'autre (chaque éclairage doit être singulier).

FORMAT DE SORTIE OBLIGATOIRE

Tu retournes STRICTEMENT un objet JSON, sans aucun texte autour, sans markdown, sans commentaire, sans préambule, de la forme :

{
  "explanations": [
    "explication du passage 1",
    "explication du passage 2",
    ...
  ]
}

La liste "explanations" doit contenir exactement autant d'éléments que d'extraits fournis dans le message utilisateur, dans le même ordre. Le JSON doit être directement parseable par json.loads() en Python : guillemets droits, pas de virgule finale, pas de clé en plus.

POSTURE DE TRAVAIL

Tu travailles avec sobriété, précision et fidélité au texte. Tu fais confiance à la richesse propre du Livre du Ciel sans la surinterpréter. Tu préfères une explication courte mais juste à une explication ample mais flottante. Chaque mot doit porter. Ton style est celui d'un exégète à la fois rigoureux et contemplatif, qui sert le texte sans s'y substituer."""


def explain_passage_matches(evangelium_text: str,
                            passages: List[str],
                            model_name: str = "gpt-4.1") -> List[str]:
    """
    Pour chaque extrait retenu du Livre du Ciel, produit une courte explication
    (2–3 phrases) qui explicite en quoi ce passage éclaire la péricope évangélique
    selon la théologie de la Divine Volonté.

    Retourne une liste de textes (une explication par passage), dans le même ordre.

    IMPORTANT :
    - Cette fonction n'influence PAS la sélection des passages (elle est appelée
      après le rerank GPT).

    Optimisation : les consignes stables sont placées dans le system message
    (préfixe identique entre requêtes → bénéficie du prompt caching OpenAI
    automatique pour les préfixes ≥ 1024 tokens).
    """

    if client is None:
        return [""] * len(passages)

    # On construit un bloc de passages numérotés
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"Passage {i} :\n{p}\n")
    joined_passages = "\n\n".join(blocks)

    user_prompt = f"""PÉRICOPE ÉVANGÉLIQUE :

\"\"\"{evangelium_text}\"\"\"

EXTRAITS DU LIVRE DU CIEL :

{joined_passages}

CONSIGNE : Pour chaque Passage ci-dessus, produis une explication de 2 à 3 phrases conforme à tes instructions système. Retourne uniquement le JSON {{"explanations": [...]}} avec autant d'éléments que d'extraits, dans le même ordre."""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        data = parse_json_object(raw)
        explanations = data.get("explanations", [])

        # On s'assure que la longueur correspond au nombre de passages
        if len(explanations) < len(passages):
            explanations += [""] * (len(passages) - len(explanations))
        return explanations[:len(passages)]

    except Exception as e:
        print("[WARN] explain_passage_matches a échoué :", e)
        return ["" for _ in passages]

   

# ============================================================
#  PARTIE 6 — Construction/Chargement de l'index + main() / CLI
# ============================================================


def build_or_load_index(pdf_path: str,
                        cache_dir: str = "ldc_index",
                        embed_model_name: str = "text-embedding-3-large"
                        ) -> Tuple[List[Dictee], List[Segment], BM25Okapi, np.ndarray]:
    """
    Construit ou recharge :
      - la liste des dictées
      - la liste des segments
      - l'index BM25
      - les embeddings OpenAI

    Utilise un dossier de cache pour éviter de tout recalculer à chaque exécution.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dictees_path  = os.path.join(cache_dir, "dictees.pkl")
    segments_path = os.path.join(cache_dir, "segments.pkl")
    bm25_path     = os.path.join(cache_dir, "bm25.pkl")
    embs_path     = os.path.join(cache_dir, "embs.npy")

    # Si tout existe, on recharge
    if all(os.path.exists(p) for p in [dictees_path, segments_path, bm25_path, embs_path]):
        print("[INFO] Chargement index depuis cache…")
        with open(dictees_path, "rb") as f:
            dictees = pickle.load(f)
        with open(segments_path, "rb") as f:
            segments = pickle.load(f)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        embs = np.load(embs_path)
        return dictees, segments, bm25, embs

    # Sinon, on reconstruit tout
    pages   = extract_pages(pdf_path)
    dictees = parse_dictees(pages)
    segments = build_segments(dictees)
    bm25    = build_bm25(segments)
    embs    = build_embeddings_for_segments(segments, model_name=embed_model_name)

    # Enregistrer dans le cache
    with open(dictees_path, "wb") as f:
        pickle.dump(dictees, f)
    with open(segments_path, "wb") as f:
        pickle.dump(segments, f)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    np.save(embs_path, embs)

    return dictees, segments, bm25, embs


# ============================================================
#  MAIN / CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recherche des 5 dictées du Livre du Ciel les plus pertinentes pour une péricope évangélique."
    )
    parser.add_argument("--pdf", required=True, help="Chemin vers le fichier ldc.pdf")
    parser.add_argument("--query_file", help="Fichier texte contenant la péricope évangélique")
    parser.add_argument("--query", help="Péricope évangélique passée en argument")
    parser.add_argument("--top", type=int, default=5,
                        help="Nombre de dictées finales à afficher (après reranking GPT)")
    parser.add_argument("--cache_dir", default="ldc_index",
                        help="Dossier pour stocker l'index et les embeddings")
    parser.add_argument("--no_summary", action="store_true",
                        help="Ne pas générer de synthèse GPT")
    parser.add_argument(
    "--explain-matches",
    action="store_true",
    help="Produire une explication pour chaque passage retenu du Livre du Ciel."
)


    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Charger la péricope
    # --------------------------------------------------------
    if args.query_file:
        if not os.path.isfile(args.query_file):
            raise SystemExit(f"Erreur : fichier de péricope introuvable : {args.query_file}")
        with open(args.query_file, "r", encoding="utf-8") as f:
            evangelium_text = f.read()
    elif args.query:
        evangelium_text = args.query
    else:
        raise SystemExit("Erreur : fournir soit --query_file, soit --query.")

    # --------------------------------------------------------
    # 2. Construire ou charger l'index
    # --------------------------------------------------------
    dictees, segments, bm25, embs = build_or_load_index(
        args.pdf,
        cache_dir=args.cache_dir,
        embed_model_name="text-embedding-3-large"
    )

    # --------------------------------------------------------
    # 3. Détection de motifs / mots-clés dynamiques
    # --------------------------------------------------------
    motif_names, motif_keywords = detect_dynamic_motifs_gpt(evangelium_text, cache_dir=args.cache_dir)

    # motif_keywords servira à enrichir la requête et guider GPT

    # --------------------------------------------------------
    # 4. Scoring hybride -> TOP 50 segments
    # --------------------------------------------------------
    ranked_segments = score_segments_with_keywords(
        evangelium_text,
        motif_keywords,
        segments,
        bm25,
        embs,
        top_k_segments=50
    )

    # --------------------------------------------------------
    # 5. Regroupement par dictée -> ~50 dictées candidates
    # --------------------------------------------------------
    candidates = group_segments_by_dictee(
        ranked_segments,
        dictees,
        top_k_dicts_pre_rerank=50
    )

    print("[DEBUG] Nombre de dictées total :", len(dictees))
    print("[DEBUG] Nombre de segments scorés :", len(ranked_segments))
    print("[DEBUG] Nombre de dictées candidates avant GPT rerank :", len(candidates))

    if not candidates:
        print("Aucune dictée candidate trouvée.")
        return

    # --------------------------------------------------------
    # 6. Reranking GPT (optionnel)
    # --------------------------------------------------------
    final_indices = rerank_with_gpt(
        evangelium_text,
        candidates,
        motif_names,
        motif_keywords,
        top_k_final=args.top
    )

        # --------------------------------------------------------
        # --------------------------------------------------------
    # 7. Préparer les passages retenus
    # --------------------------------------------------------
    passages = []
    for idx in final_indices:
        score, d, seg = candidates[idx]
        excerpt = make_excerpt(d, seg)
        header = f"Tome {d.tome} — {d.date}"
        passages.append(f"{header}\n{excerpt}")

    # --------------------------------------------------------
    # 7bis. Explications par passage (si demandé)
    # --------------------------------------------------------
    if args.explain_matches:
        explanations = explain_passage_matches(evangelium_text, passages)
    else:
        explanations = [""] * len(passages)

    # --------------------------------------------------------
    # 7ter. Affichage passage + explication
    # --------------------------------------------------------
    print("\n===== PASSAGES RETENUS =====\n")
    for i, (block, expl) in enumerate(zip(passages, explanations), start=1):
        print(block)
        print()
        if args.explain_matches and expl.strip():
            print(expl.strip())
            print()

    # Afficher l’explication juste après, sans titre
    if args.explain_matches and i < len(explanations) and explanations[i].strip():
        print(explanations[i].strip(), "\n")


    # --------------------------------------------------------
    # 8. Synthèse optionnelle
    # --------------------------------------------------------
    if not args.no_summary and passages:
        print("Synthèse\n")
        summary = summarize_with_gpt(evangelium_text, passages)
        print(summary)





if __name__ == "__main__":
    main()
