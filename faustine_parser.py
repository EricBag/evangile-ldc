#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faustine_parser.py — Extraction et chunking du « Petit Journal » de sainte Faustine.

Pendant de `parse_dictees` / `build_segments` (ldc_proZ.py) pour le second corpus.
Différence structurante : l'unité de citation n'est pas une date de dictée mais un
**numéro de paragraphe** (« Petit Journal, § 742 »).

Le PDF fourni (`data/faustine/petit_journal.pdf`) est une capture de pages web
(JesusMarie.com / Église catholique au Gabon). Il porte trois artefacts qu'il faut
neutraliser avant tout découpage :

  1. du boilerplate répété à chaque page (« Retour », « Ste Faustine », mentions
     d'édition, marqueurs HTML résiduels « END CONTAINER ») ;
  2. des **compteurs de lignes** intercalés, tantôt seuls sur leur ligne, tantôt
     collés en préfixe du texte (« 281 444. Jeudi. L'adoration nocturne. ») ;
  3. des blocs dupliqués aux charnières de pages capturées.

S'y ajoute une contrainte propre à l'édition : certains numéros de § n'y figurent
tout simplement pas (le texte passe de « 68. » à « 71. »). Ces § sont absorbés
dans le paragraphe précédent, dont la plage `num_debut..num_fin` en garde la trace.
"""

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ldc_proZ import (
    BASCULE_NUM_REF,
    Paragraphe,
    Segment,
    SOURCE_FAUSTINE,
    ZONE_NUM_INCERTAIN,
    numero_incertain,
    numero_reference,
    tokenize,
)

# ------------------------------------------------------------
#  Paramètres
# ------------------------------------------------------------

#: Bornes du numérotage tel qu'il est imprimé dans le PDF.
NUM_PARA_MIN = 1
NUM_PARA_MAX = 1828

#: En deçà de ce nombre de caractères, un § est regroupé avec le suivant.
#: La plage `num_debut..num_fin` conserve alors les numéros fusionnés.
MIN_CHARS_PARAGRAPHE = 300

#: Découpe des paragraphes longs, alignée sur celle du Livre du Ciel.
SEG_LEN_MOTS = 250
SEG_STRIDE_MOTS = 120

#: Longueur minimale d'une suite de compteurs pour être reconnue comme telle.
#: Trois valeurs consécutives suffisent à écarter un « 12 » isolé dans le texte.
LONGUEUR_MIN_SUITE_COMPTEURS = 3


# ------------------------------------------------------------
#  Nettoyage
# ------------------------------------------------------------

#: Lignes de navigation / d'en-tête reproduites à l'identique sur chaque page.
BOILERPLATE_EXACT = frozenset({
    "Retour",
    "Sainte Faustine - Héléna Kowalska",
    "Le Petit Journal",
    "Petit Journal de Sœur Faustine",
    "Ste Faustine",
    "END CONTAINER",
    "END MAINBODY",
})

#: Fragments dont la présence suffit à écarter la ligne entière.
BOILERPLATE_FRAGMENTS = (
    "édition numérique par Anne",
    "www.JesusMarie.com",
    "l’apôtre de la Miséricorde Divine",
    "l'apôtre de la Miséricorde Divine",
    "Copyright ©",
    "Eglise Catholique au Gabon",
    "Journal Faustine",
)

CAHIER_RE = re.compile(r"^Cahier\s+([IVXLC]+|\d{1,2})\s*$", re.IGNORECASE)

#: Marqueur de paragraphe sûr : « 742. », « 296 . », ou « 999. » seul sur sa ligne
#: (le texte commence alors à la ligne suivante). Le point lève l'ambiguïté avec
#: un compteur de lignes.
MARQUEUR_STRICT_RE = re.compile(r"^(\d{1,4})\s*\.\s*")

#: Marqueur de paragraphe probable, dans les pages où la numérotation a perdu son
#: point à la capture : « 166 Je trouve… », « 1520Le Seigneur… », « 877, 12. I. 1937 ».
#: Beaucoup plus permissif, donc réservé au comblement des trous (cf.
#: `_completer_trous`) et jamais utilisé seul.
MARQUEUR_SOUPLE_RE = re.compile(
    r"^(\d{1,4})\s*[.,;]?\s*(?=[«\"A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŒÇ])"
)

#: Ligne ne portant qu'un nombre : soit un numéro de page, soit un marqueur de §
#: dont le texte commence à la ligne suivante. Seule la détection tranchera.
LIGNE_NUMERIQUE_RE = re.compile(r"^\d{1,5}$")

#: Compteur candidat : un nombre en tête de ligne **non** suivi d'un point.
COMPTEUR_RE = re.compile(r"^(\d{1,5})(?:\s+|$)")

#: Matière d'édition close le texte : imprimatur et mention finale. Tout ce qui
#: suit dans le dernier paragraphe relève de l'appareil éditorial, pas du Journal.
FIN_DU_TEXTE = (
    "Concordat cum originali",
    "fin du Petit Journal",
)

#: Pied de page recomposé : numéro de page collé à un reste d'en-tête courant
#: (« 0981 Faustine, », « 751 Faustine ch. », « 0020 Faustine cahier 2, »).
PIED_DE_PAGE_RE = re.compile(r"^\d{2,4}\s*(Faustine|cahier)\b", re.IGNORECASE)


def _normaliser(ligne: str) -> str:
    """Réduit toute suite d'espaces (y compris insécables) à une espace simple."""
    return " ".join(ligne.split())


def est_boilerplate(ligne: str) -> bool:
    """Vrai si la ligne relève de l'habillage du site capturé, pas du texte."""
    if ligne in BOILERPLATE_EXACT:
        return True
    return any(frag in ligne for frag in BOILERPLATE_FRAGMENTS)


def _candidat_compteur(ligne: str) -> Optional[Tuple[int, str]]:
    """Renvoie (valeur, reste de la ligne) si la ligne peut porter un compteur.

    Un marqueur de paragraphe est explicitement exclu : « 444. » et « 296 . »
    portent un point, un compteur jamais.
    """
    m = COMPTEUR_RE.match(ligne)
    if not m:
        return None
    reste = ligne[m.end():].strip()
    if reste.startswith("."):
        return None
    return int(m.group(1)), reste


def reperer_compteurs(lignes: Sequence[str]) -> set:
    """Indices des lignes portant un compteur de lignes du document capturé.

    On ne se fie pas à un seuil : on ne retient que les nombres qui s'inscrivent
    dans une **suite d'au moins `LONGUEUR_MIN_SUITE_COMPTEURS` valeurs
    consécutives**. Un nombre isolé en tête de phrase reste donc du texte.
    """
    reperes: set = set()
    suite: List[int] = []      # indices de lignes de la suite en cours
    precedent: Optional[int] = None

    def clore() -> None:
        if len(suite) >= LONGUEUR_MIN_SUITE_COMPTEURS:
            reperes.update(suite)
        suite.clear()

    for i, ligne in enumerate(lignes):
        candidat = _candidat_compteur(ligne)
        if candidat is None:
            continue
        valeur, _reste = candidat
        if precedent is not None and valeur == precedent + 1:
            suite.append(i)
        else:
            clore()
            suite.append(i)
        precedent = valeur

    clore()
    return reperes


def nettoyer_lignes(lignes_brutes: Sequence[str]) -> List[str]:
    """Boilerplate écarté, compteurs de lignes retirés, lignes vides supprimées."""
    lignes = [_normaliser(l) for l in lignes_brutes]
    lignes = [l for l in lignes if l]

    indices_compteurs = reperer_compteurs(lignes)

    nettoyees: List[str] = []
    for i, ligne in enumerate(lignes):
        if i in indices_compteurs:
            candidat = _candidat_compteur(ligne)
            if candidat is not None:
                _valeur, reste = candidat
                if not reste:
                    continue          # compteur seul sur sa ligne
                ligne = reste         # compteur collé en préfixe du texte
        if est_boilerplate(ligne) or PIED_DE_PAGE_RE.match(ligne):
            continue
        nettoyees.append(ligne)
    return nettoyees


# ------------------------------------------------------------
#  Détection des marqueurs de paragraphe
# ------------------------------------------------------------

def _sous_suite_croissante(candidats: Sequence[Tuple[int, int, int]]
                           ) -> List[Tuple[int, int, int]]:
    """Plus longue sous-suite strictement croissante des numéros candidats.

    Les blocs dupliqués aux charnières de pages capturées réintroduisent des
    numéros déjà rencontrés ; une simple lecture séquentielle s'y perdrait.
    Retenir la plus longue progression cohérente écarte ces rappels tout en
    conservant le maximum de marqueurs authentiques.

    Chaque candidat est un triplet (index de ligne, numéro, offset dans la ligne).
    """
    import bisect

    if not candidats:
        return []

    fins: List[int] = []          # plus petit numéro terminant une suite de longueur i+1
    indices_fins: List[int] = []  # index (dans `candidats`) de ces terminaisons
    parent: List[Optional[int]] = [None] * len(candidats)

    for j, (_ligne, numero, _offset) in enumerate(candidats):
        i = bisect.bisect_left(fins, numero)
        if i > 0:
            parent[j] = indices_fins[i - 1]
        if i == len(fins):
            fins.append(numero)
            indices_fins.append(j)
        elif fins[i] > numero:
            fins[i] = numero
            indices_fins[i] = j
        # `fins[i] == numero` : rappel d'un § deja rencontre. Le remplacer par
        # cette occurrence plus tardive n'allongerait aucune suite et deplacerait
        # le debut du paragraphe apres le bloc duplique -- on garde la premiere,
        # pour que le texte reste rattache au § qui l'ouvre.

    suite: List[Tuple[int, int, int]] = []
    courant: Optional[int] = indices_fins[-1]
    while courant is not None:
        suite.append(candidats[courant])
        courant = parent[courant]
    suite.reverse()
    return suite


def _candidats(lignes: Sequence[str], motif) -> List[Tuple[int, int, int]]:
    """Marqueurs potentiels reconnus par `motif`, dans l'ordre des lignes."""
    trouves: List[Tuple[int, int, int]] = []
    for i, ligne in enumerate(lignes):
        m = motif.match(ligne)
        if not m:
            continue
        numero = int(m.group(1))
        if NUM_PARA_MIN <= numero <= NUM_PARA_MAX:
            trouves.append((i, numero, m.end()))
    return trouves


def _completer_trous(retenus: List[Tuple[int, int, int]],
                     souples: Sequence[Tuple[int, int, int]]
                     ) -> List[Tuple[int, int, int]]:
    """Comble les trous du numérotage avec les marqueurs à ponctuation dégradée.

    Un candidat souple n'est admis que s'il s'insère **exactement** dans un trou :
    numéro et position tous deux strictement compris entre les deux marqueurs sûrs
    qui l'encadrent. Une date ou un nombre en tête de phrase ne satisfait
    pratiquement jamais ces deux conditions à la fois.
    """
    if not retenus or not souples:
        return retenus

    par_ligne = {ligne for ligne, _num, _off in retenus}
    complets = list(retenus)

    for k in range(len(retenus) - 1):
        ligne_g, num_g, _ = retenus[k]
        ligne_d, num_d, _ = retenus[k + 1]
        if num_d - num_g <= 1:
            continue
        dans_le_trou = [
            c for c in souples
            if ligne_g < c[0] < ligne_d
            and num_g < c[1] < num_d
            and c[0] not in par_ligne
        ]
        complets.extend(_sous_suite_croissante(dans_le_trou))

    complets.sort(key=lambda c: c[0])
    return complets


def detecter_marqueurs(lignes: Sequence[str]) -> List[Tuple[int, int, int]]:
    """Marqueurs de paragraphe retenus, sous forme (index ligne, numéro, offset).

    Deux passes : la plus longue progression cohérente parmi les marqueurs sûrs
    (« 742. »), puis comblement des trous par les marqueurs à ponctuation
    dégradée (« 166 Je trouve… »), qui ne sont jamais admis d'eux-mêmes.
    """
    stricts = _sous_suite_croissante(_candidats(lignes, MARQUEUR_STRICT_RE))
    souples = _candidats(lignes, MARQUEUR_SOUPLE_RE)
    return _completer_trous(stricts, souples)


# ------------------------------------------------------------
#  Construction des paragraphes
# ------------------------------------------------------------

def construire_paragraphes(lignes_nettoyees: Sequence[str],
                           min_chars: int = MIN_CHARS_PARAGRAPHE
                           ) -> List[Paragraphe]:
    """Assemble les paragraphes numérotés à partir des lignes déjà nettoyées.

    Les § plus courts que `min_chars` sont regroupés avec le suivant ; la plage
    `num_debut..num_fin` garde la trace des numéros réunis, de sorte que la
    citation reste exacte (« §§ 280–295 »).
    """
    marqueurs = detecter_marqueurs(lignes_nettoyees)
    if not marqueurs:
        return []

    # Cahier courant : dernière mention « Cahier N » rencontrée avant le §.
    cahier_par_ligne: List[Optional[str]] = []
    cahier: Optional[str] = None
    for ligne in lignes_nettoyees:
        m = CAHIER_RE.match(ligne)
        if m:
            cahier = m.group(1).upper()
        cahier_par_ligne.append(cahier)

    bruts: List[Paragraphe] = []
    for k, (i_ligne, numero, offset) in enumerate(marqueurs):
        fin = marqueurs[k + 1][0] if k + 1 < len(marqueurs) else len(lignes_nettoyees)
        morceaux = [lignes_nettoyees[i_ligne][offset:]]
        # Les lignes réduites à un nombre et non retenues comme marqueur sont des
        # numéros de page : elles n'appartiennent pas au texte du paragraphe.
        morceaux.extend(l for l in lignes_nettoyees[i_ligne + 1:fin]
                        if not LIGNE_NUMERIQUE_RE.match(l))
        texte = " ".join(m.strip() for m in morceaux if m.strip()).strip()
        texte = _couper_matiere_finale(texte)
        if not texte:
            continue
        # Le § suivant détecté peut être postérieur de plusieurs numéros : les
        # numéros absents de l'édition sont absorbés ici, la plage les enregistre.
        num_fin = (marqueurs[k + 1][1] - 1) if k + 1 < len(marqueurs) else numero
        bruts.append(Paragraphe(
            num_debut=numero,
            num_fin=max(numero, num_fin),
            cahier=cahier_par_ligne[i_ligne],
            text=texte,
        ))

    regroupes = _regrouper_courts(bruts, min_chars)
    renseigner_numeros_reference(regroupes)
    return regroupes


def renseigner_numeros_reference(paragraphes: Sequence[Paragraphe]) -> None:
    """Renseigne `num_ref_debut`, `num_ref_fin` et `num_incertain` sur place.

    Appliquée après le regroupement, pour que la plage de référence recouvre
    bien celle du PDF. N'agit que sur des métadonnées : appelable sur un index
    déjà constitué, sans toucher ni aux segments ni aux embeddings.
    """
    for para in paragraphes:
        para.num_ref_debut = numero_reference(para.num_debut)
        para.num_ref_fin = numero_reference(para.num_fin)
        para.num_incertain = (numero_incertain(para.num_debut)
                              or numero_incertain(para.num_fin))


def _couper_matiere_finale(texte: str) -> str:
    """Retranche l'appareil éditorial qui suit la dernière ligne du Journal."""
    coupe = len(texte)
    for marqueur in FIN_DU_TEXTE:
        pos = texte.find(marqueur)
        if pos != -1:
            coupe = min(coupe, pos)
    return texte[:coupe].strip()


def _regrouper_courts(paragraphes: Sequence[Paragraphe],
                      min_chars: int) -> List[Paragraphe]:
    """Fusionne chaque § trop court avec le suivant, en étendant sa plage.

    Le dernier § n'ayant pas de suivant, il est rattaché au précédent s'il est
    lui-même trop court.
    """
    if min_chars <= 0:
        return list(paragraphes)

    regroupes: List[Paragraphe] = []
    en_attente: Optional[Paragraphe] = None

    for para in paragraphes:
        if en_attente is not None:
            para = Paragraphe(
                num_debut=en_attente.num_debut,
                num_fin=max(en_attente.num_fin, para.num_fin),
                cahier=en_attente.cahier or para.cahier,
                text=f"{en_attente.text} {para.text}".strip(),
            )
            en_attente = None
        if len(para.text) < min_chars:
            en_attente = para
            continue
        regroupes.append(para)

    if en_attente is not None:
        if regroupes:
            dernier = regroupes[-1]
            regroupes[-1] = Paragraphe(
                num_debut=dernier.num_debut,
                num_fin=max(dernier.num_fin, en_attente.num_fin),
                cahier=dernier.cahier,
                text=f"{dernier.text} {en_attente.text}".strip(),
            )
        else:
            regroupes.append(en_attente)

    return regroupes


# ------------------------------------------------------------
#  Segmentation
# ------------------------------------------------------------

def segmenter_paragraphes(paragraphes: Sequence[Paragraphe],
                          id_offset: int = 0,
                          seg_len: int = SEG_LEN_MOTS,
                          stride: int = SEG_STRIDE_MOTS) -> List[Segment]:
    """Découpe les paragraphes en segments indexables.

    Un § court tient dans un seul segment ; un § long est découpé en fenêtres
    glissantes, comme les dictées du Livre du Ciel. Tous les segments d'un même
    § pointent vers lui : l'unité de citation reste le paragraphe.

    `id_offset` est le nombre de segments déjà présents dans l'index : les ids
    Faustine prolongent la numérotation Luisa sans la perturber, puisque
    `Segment.id` sert d'index dans la matrice d'embeddings.
    """
    segments: List[Segment] = []
    sid = id_offset

    for p_idx, para in enumerate(paragraphes):
        mots = para.text.split()
        debut = 0
        while debut < len(mots):
            morceau = mots[debut: debut + seg_len]
            if not morceau:
                break
            texte = " ".join(morceau)
            segments.append(Segment(
                id=sid,
                dictee_index=p_idx,
                text=texte,
                tokens=tokenize(texte),
                source=SOURCE_FAUSTINE,
            ))
            sid += 1
            if debut + seg_len >= len(mots):
                break
            debut += stride

    return segments


# ------------------------------------------------------------
#  Pipeline complet + diagnostic
# ------------------------------------------------------------

def extraire_lignes_pdf(pdf_path: str) -> List[str]:
    """Lignes brutes du PDF, dans l'ordre des pages."""
    import fitz  # PyMuPDF — importé ici pour garder le module testable sans PDF

    doc = fitz.open(pdf_path)
    lignes: List[str] = []
    for page in doc:
        lignes.extend(page.get_text("text").splitlines())
    return lignes


def charger_paragraphes(pdf_path: str,
                        min_chars: int = MIN_CHARS_PARAGRAPHE
                        ) -> List[Paragraphe]:
    """PDF → paragraphes prêts à indexer."""
    lignes = nettoyer_lignes(extraire_lignes_pdf(pdf_path))
    return construire_paragraphes(lignes, min_chars=min_chars)


#: Nom du fichier d'index propre au corpus Faustine, dans le dossier de cache.
FICHIER_PARAGRAPHES = "paragraphes.pkl"


def charger_paragraphes_indexes(cache_dir: str) -> List[Paragraphe]:
    """Paragraphes déjà indexés, ou liste vide si le corpus n'a pas été ingéré.

    Appelé au démarrage de l'application : l'absence du fichier signifie
    simplement que `ingest_faustine.py` n'a pas encore été lancé, auquel cas
    seul le Livre du Ciel est disponible.
    """
    import pickle

    chemin = os.path.join(cache_dir, FICHIER_PARAGRAPHES)
    if not os.path.exists(chemin):
        return []
    with open(chemin, "rb") as f:
        paragraphes = pickle.load(f)

    # `pickle` restaure les dataclasses sans passer par `__init__` : les
    # paragraphes sérialisés avant l'ajout de la numérotation de référence
    # n'en portent aucune trace. On la recalcule plutôt que de citer un
    # numéro faux — c'est un calcul de métadonnées, sans effet sur l'index.
    if any(getattr(p, "num_ref_debut", None) is None for p in paragraphes):
        print("[INFO] Numérotation de référence recalculée au chargement "
              "(index antérieur à son introduction).")
        renseigner_numeros_reference(paragraphes)
    return paragraphes


def rapport_couverture(paragraphes: Sequence[Paragraphe]) -> Dict:
    """Diagnostic de l'extraction : couverture du numérotage et gabarits.

    Un § « couvert » est un § dont le numéro tombe dans la plage d'un paragraphe
    extrait — y compris lorsqu'il a été absorbé faute de marqueur dans l'édition.
    """
    couverts: set = set()
    for para in paragraphes:
        couverts.update(range(para.num_debut, para.num_fin + 1))

    attendus = set(range(NUM_PARA_MIN, NUM_PARA_MAX + 1))
    longueurs = sorted(len(p.text) for p in paragraphes)
    mediane = longueurs[len(longueurs) // 2] if longueurs else 0

    return {
        "paragraphes": len(paragraphes),
        "num_min": min((p.num_debut for p in paragraphes), default=None),
        "num_max": max((p.num_fin for p in paragraphes), default=None),
        "marqueurs_distincts": len({p.num_debut for p in paragraphes}),
        "numeros_couverts": len(couverts & attendus),
        "numeros_absents": sorted(attendus - couverts),
        "plages_fusionnees": sum(1 for p in paragraphes if p.num_fin > p.num_debut),
        "longueur_mediane": mediane,
        "longueur_max": longueurs[-1] if longueurs else 0,
        "total_caracteres": sum(longueurs),
    }


def imprimer_rapport(rapport: Dict) -> None:
    """Affiche le rapport de couverture sous une forme lisible en console."""
    absents = rapport["numeros_absents"]
    print(f"[INFO] Paragraphes extraits : {rapport['paragraphes']} "
          f"(§ {rapport['num_min']} → {rapport['num_max']})")
    print(f"[INFO] Marqueurs distincts trouvés : {rapport['marqueurs_distincts']}")
    print(f"[INFO] Numéros couverts : {rapport['numeros_couverts']}/"
          f"{NUM_PARA_MAX - NUM_PARA_MIN + 1}")
    print(f"[INFO] Paragraphes à plage multiple : {rapport['plages_fusionnees']}")
    print(f"[INFO] Longueur médiane : {rapport['longueur_mediane']} caractères "
          f"(max {rapport['longueur_max']}, total {rapport['total_caracteres']})")
    if absents:
        apercu = ", ".join(str(n) for n in absents[:20])
        suite = f" … et {len(absents) - 20} autre(s)" if len(absents) > 20 else ""
        print(f"[WARN] {len(absents)} numéro(s) non couvert(s) : {apercu}{suite}")


if __name__ == "__main__":
    import argparse

    parseur = argparse.ArgumentParser(
        description="Diagnostic de l'extraction du Petit Journal de sainte Faustine."
    )
    parseur.add_argument("--pdf", default="data/faustine/petit_journal.pdf")
    parseur.add_argument("--min-chars", type=int, default=MIN_CHARS_PARAGRAPHE)
    parseur.add_argument("--montrer", type=int, metavar="N",
                         help="Afficher le paragraphe couvrant le § N.")
    options = parseur.parse_args()

    paras = charger_paragraphes(options.pdf, min_chars=options.min_chars)
    imprimer_rapport(rapport_couverture(paras))

    if options.montrer is not None:
        for para in paras:
            if para.num_debut <= options.montrer <= para.num_fin:
                plage = (f"§ {para.num_debut}" if para.num_debut == para.num_fin
                         else f"§§ {para.num_debut}–{para.num_fin}")
                print(f"\n--- {plage} (cahier {para.cahier}) ---")
                print(para.text[:1200])
                break
        else:
            print(f"\n[WARN] Aucun paragraphe ne couvre le § {options.montrer}.")
