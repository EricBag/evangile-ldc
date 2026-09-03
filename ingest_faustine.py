#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_faustine.py — Ajoute le « Petit Journal » de sainte Faustine à l'index.

Le principe directeur : **ne jamais recalculer les vecteurs du Livre du Ciel**.
Les embeddings de Luisa sont relus tels quels depuis `embs.npy` et les vecteurs
Faustine leur sont concaténés, en float16 comme eux.

Ce qui est écrit :

  embs.npy        vecteurs Luisa (inchangés) + vecteurs Faustine
  segments.pkl    segments Luisa (inchangés) + segments Faustine
  paragraphes.pkl unité de citation du corpus Faustine (nouveau fichier)
  bm25.pkl        reconstruit sur le corpus complet

`bm25.pkl` est le seul artefact intégralement recalculé : `rank_bm25` ne sait pas
ajouter un document à un index existant. C'est du calcul local, sans appel d'API,
et sans effet sur les embeddings.

Le script est **idempotent** : relancé, il repart des seuls segments Luisa et
remplace le lot Faustine au lieu de l'empiler.

Usage :
    python ingest_faustine.py --dry-run     # diagnostic + estimation, sans appel API
    python ingest_faustine.py               # ingestion réelle
"""

import argparse
import os
import pickle
import shutil
import sys
from typing import List

import numpy as np
from dotenv import load_dotenv

# `ldc_proZ` instancie son client OpenAI dès l'import : la clé doit être en
# place avant, comme dans main.py.
load_dotenv(override=False)

import faustine_parser as fp  # noqa: E402
from ldc_proZ import (  # noqa: E402
    SOURCE_FAUSTINE,
    SOURCE_LUISA,
    Segment,
    backfill_segment_sources,
    build_bm25,
    embed_texts_openai,
)

CACHE_DIR_DEFAUT = "ldc_index_word"
PDF_DEFAUT = os.path.join("data", "faustine", "petit_journal.pdf")
MODELE_EMBEDDINGS = "text-embedding-3-large"

#: Tarif public de text-embedding-3-large, pour la seule estimation affichée.
USD_PAR_MILLION_TOKENS = 0.13


def _chemins(cache_dir: str) -> dict:
    return {
        "segments": os.path.join(cache_dir, "segments.pkl"),
        "paragraphes": os.path.join(cache_dir, fp.FICHIER_PARAGRAPHES),
        "bm25": os.path.join(cache_dir, "bm25.pkl"),
        "embs": os.path.join(cache_dir, "embs.npy"),
    }


def refaire_metadonnees(cache_dir: str) -> None:
    """Recalcule la numérotation de référence des paragraphes déjà indexés.

    Ne touche qu'à `paragraphes.pkl`. Les segments, l'index BM25 et surtout les
    embeddings restent intacts : aucun réencodage, aucun appel d'API. C'est la
    voie à emprunter pour toute correction portant sur les seules métadonnées
    de citation.
    """
    chemins = _chemins(cache_dir)
    if not os.path.exists(chemins["paragraphes"]):
        raise SystemExit(
            f"[ERREUR] Corpus Faustine introuvable : {chemins['paragraphes']}"
        )

    with open(chemins["paragraphes"], "rb") as f:
        paragraphes = pickle.load(f)

    fp.renseigner_numeros_reference(paragraphes)

    _sauvegarder(chemins["paragraphes"])
    with open(chemins["paragraphes"], "wb") as f:
        pickle.dump(paragraphes, f)

    incertains = sum(1 for p in paragraphes if p.num_incertain)
    print(f"[INFO] {len(paragraphes)} paragraphe(s) renumérotés "
          f"(bascule au § {fp.BASCULE_NUM_REF} du PDF, "
          f"décalage +1 au-delà).")
    print(f"[INFO] {incertains} paragraphe(s) en zone incertaine "
          f"{fp.ZONE_NUM_INCERTAIN} : cités « § ~N ».")
    print("[INFO] Embeddings, segments et index BM25 inchangés.")


def _sauvegarder(chemin: str) -> None:
    """Copie de sûreté avant écrasement. `embs.npy` n'est pas reconstructible
    sans repayer l'encodage de tout le Livre du Ciel."""
    if os.path.exists(chemin):
        sauvegarde = chemin + ".bak"
        shutil.copy2(chemin, sauvegarde)
        print(f"[INFO] Sauvegarde : {os.path.basename(sauvegarde)}")


def _segments_luisa(segments: List[Segment]) -> List[Segment]:
    """Isole les segments du Livre du Ciel en vérifiant qu'ils ouvrent l'index.

    `Segment.id` sert d'indice de ligne dans `embs` : les vecteurs Luisa ne
    peuvent être conservés à l'identique que s'ils occupent bien le préfixe
    de la matrice.
    """
    backfill_segment_sources(segments)
    luisa = [s for s in segments if s.source == SOURCE_LUISA]
    attendus = list(range(len(luisa)))
    if [s.id for s in luisa] != attendus:
        raise SystemExit(
            "[ERREUR] Les segments du Livre du Ciel n'occupent pas les indices "
            "0..N-1 de l'index. Les embeddings ne peuvent pas être réutilisés "
            "en l'état : restaurez embs.npy/segments.pkl depuis leur .bak."
        )
    return luisa


def ingerer(pdf_path: str,
            cache_dir: str,
            min_chars: int,
            dry_run: bool) -> None:
    chemins = _chemins(cache_dir)

    for cle in ("segments", "embs"):
        if not os.path.exists(chemins[cle]):
            raise SystemExit(
                f"[ERREUR] Index introuvable : {chemins[cle]}. "
                "Construisez d'abord l'index du Livre du Ciel."
            )

    # --------------------------------------------------------
    # 1. Index existant
    # --------------------------------------------------------
    with open(chemins["segments"], "rb") as f:
        segments_existants = pickle.load(f)
    luisa = _segments_luisa(segments_existants)

    embs_existants = np.load(chemins["embs"])
    if embs_existants.shape[0] != len(segments_existants):
        raise SystemExit(
            f"[ERREUR] Incohérence index : {embs_existants.shape[0]} vecteurs "
            f"pour {len(segments_existants)} segments."
        )
    embs_luisa = embs_existants[:len(luisa)]

    deja_faustine = len(segments_existants) - len(luisa)
    print(f"[INFO] Segments Luisa conservés : {len(luisa)}")
    if deja_faustine:
        print(f"[INFO] Lot Faustine existant remplacé : {deja_faustine} segment(s)")

    # --------------------------------------------------------
    # 2. Corpus Faustine
    # --------------------------------------------------------
    print(f"[INFO] Lecture du PDF : {pdf_path}")
    paragraphes = fp.charger_paragraphes(pdf_path, min_chars=min_chars)
    if not paragraphes:
        raise SystemExit("[ERREUR] Aucun paragraphe extrait du Petit Journal.")
    fp.imprimer_rapport(fp.rapport_couverture(paragraphes))

    segments_faustine = fp.segmenter_paragraphes(paragraphes, id_offset=len(luisa))
    caracteres = sum(len(s.text) for s in segments_faustine)
    # ~4 caractères par token en français : l'ordre de grandeur suffit ici.
    tokens_estimes = caracteres / 4
    cout = tokens_estimes / 1_000_000 * USD_PAR_MILLION_TOKENS
    print(f"[INFO] Segments Faustine : {len(segments_faustine)} "
          f"({caracteres} caractères)")
    print(f"[INFO] Encodage estimé : ~{tokens_estimes:,.0f} tokens "
          f"≈ {cout:.2f} $ ({MODELE_EMBEDDINGS})")

    if dry_run:
        print("[INFO] --dry-run : aucun appel API, aucun fichier écrit.")
        return

    # --------------------------------------------------------
    # 3. Embeddings des seuls segments Faustine
    # --------------------------------------------------------
    embs_faustine = embed_texts_openai(
        [s.text for s in segments_faustine], model_name=MODELE_EMBEDDINGS,
    )
    if embs_faustine.shape[1] != embs_luisa.shape[1]:
        raise SystemExit(
            f"[ERREUR] Dimension incompatible : {embs_faustine.shape[1]} "
            f"vs {embs_luisa.shape[1]} pour le corpus existant."
        )

    # --------------------------------------------------------
    # 4. Écriture
    # --------------------------------------------------------
    for cle in ("embs", "segments", "bm25"):
        _sauvegarder(chemins[cle])

    segments_complets = luisa + segments_faustine
    embs_complets = np.concatenate(
        [embs_luisa, embs_faustine.astype(np.float16)], axis=0,
    )
    assert embs_complets.shape[0] == len(segments_complets)
    # Les vecteurs Luisa doivent sortir bit pour bit tels qu'ils sont entrés.
    assert np.array_equal(embs_complets[:len(luisa)], embs_luisa)

    bm25 = build_bm25(segments_complets)

    np.save(chemins["embs"], embs_complets)
    with open(chemins["segments"], "wb") as f:
        pickle.dump(segments_complets, f)
    with open(chemins["paragraphes"], "wb") as f:
        pickle.dump(paragraphes, f)
    with open(chemins["bm25"], "wb") as f:
        pickle.dump(bm25, f)

    print(f"[INFO] Index écrit : {len(segments_complets)} segments "
          f"({len(luisa)} {SOURCE_LUISA} + {len(segments_faustine)} {SOURCE_FAUSTINE}), "
          f"embs {embs_complets.shape} {embs_complets.dtype}")


def main() -> None:
    parseur = argparse.ArgumentParser(
        description="Ajoute le Petit Journal de sainte Faustine à l'index existant."
    )
    parseur.add_argument("--pdf", default=PDF_DEFAUT)
    parseur.add_argument("--cache-dir", default=CACHE_DIR_DEFAUT)
    parseur.add_argument("--min-chars", type=int, default=fp.MIN_CHARS_PARAGRAPHE,
                         help="Seuil de regroupement des paragraphes trop courts.")
    parseur.add_argument("--dry-run", action="store_true",
                         help="Diagnostic et estimation de coût, sans appel API.")
    parseur.add_argument("--refaire-metadonnees", action="store_true",
                         help="Recalculer la seule numérotation de référence "
                              "des paragraphes indexés (aucun réencodage).")
    options = parseur.parse_args()

    if options.refaire_metadonnees:
        refaire_metadonnees(options.cache_dir)
        return

    if not os.path.exists(options.pdf):
        raise SystemExit(f"[ERREUR] PDF introuvable : {options.pdf}")

    ingerer(options.pdf, options.cache_dir, options.min_chars, options.dry_run)


if __name__ == "__main__":
    sys.exit(main())
