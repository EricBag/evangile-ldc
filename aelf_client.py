"""
aelf_client.py — Client API AELF
================================
Récupération des lectures du jour depuis l'API publique AELF
(https://api.aelf.org/v1/messes/{date}/{zone}).

4 fonctions publiques :
    - fetch_messe(date, zone)              -> dict       (cache 24h)
    - extract_evangile(messe)              -> (ref, titre, texte)
    - extract_premiere_lecture(messe)      -> (ref, titre, texte)
    - extract_contexte_liturgique(messe)   -> str

Le texte renvoyé par les extractors est nettoyé : HTML → texte brut,
acclamation finale (« – Parole du Seigneur. » / « – Acclamons la
Parole de Dieu. ») retirée.
"""

from __future__ import annotations

import html
import logging
import re
import time
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

AELF_BASE_URL = "https://api.aelf.org/v1/messes"
HTTP_TIMEOUT = 10        # secondes
MAX_RETRIES = 1          # une tentative supplémentaire après le premier échec
RETRY_BACKOFF = 1.5      # secondes entre deux tentatives


class AelfError(RuntimeError):
    """Erreur lors de l'accès ou du parsing de l'API AELF."""


# ============================================================
# 1) Récupération distante
# ============================================================

@lru_cache(maxsize=128)
def fetch_messe(date: str, zone: str = "romain") -> dict:
    """
    GET https://api.aelf.org/v1/messes/{date}/{zone}

    Args:
        date: date au format ISO "YYYY-MM-DD".
        zone: "romain", "france", "belgique", "canada", "luxembourg",
              "suisse", "afrique" (cf. AELF). Défaut "romain".

    Returns:
        dict : JSON décodé de l'API (clés `informations` et `messes`).

    Raises:
        AelfError : si timeout, statut HTTP ≠ 200, ou JSON invalide,
                    après MAX_RETRIES tentatives supplémentaires.
    """
    url = f"{AELF_BASE_URL}/{date}/{zone}"
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning(
                "AELF fetch failed (attempt %d/%d) for %s: %s",
                attempt + 1, MAX_RETRIES + 1, url, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF)

    raise AelfError(
        f"Échec de récupération AELF pour {date}/{zone}: {last_exc}"
    ) from last_exc


# ============================================================
# 2) Nettoyage HTML → texte brut
# ============================================================

# NB : la regex BR absorbe l'éventuel `\n` qui suit le tag dans le HTML
# source d'AELF — sinon chaque ligne de poésie se retrouve séparée par
# deux sauts (le `\n` issu de `<br />` + le `\n` littéral du source).
# L'indentation typographique (`&nbsp;`) en tête de ligne est préservée.
_RE_BR = re.compile(r"<br\s*/?>(?:[ \t]*\n)?", re.IGNORECASE)
_RE_P_OPEN = re.compile(r"<p[^>]*>\s*", re.IGNORECASE)
_RE_P_CLOSE = re.compile(r"\s*</p\s*>", re.IGNORECASE)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

_ACCLAMATIONS = (
    "– Acclamons la Parole de Dieu.",
    "– Parole du Seigneur.",
    "— Acclamons la Parole de Dieu.",
    "— Parole du Seigneur.",
    "- Acclamons la Parole de Dieu.",
    "- Parole du Seigneur.",
    "Acclamons la Parole de Dieu.",
    "Parole du Seigneur.",
)


def _html_to_text(content: str) -> str:
    """Convertit le HTML AELF en texte brut lisible.

    `<br />` → `\\n`, `</p>` → `\\n\\n`, balises résiduelles supprimées,
    entités HTML décodées, espaces de fin de ligne nettoyés, blocs
    vides multiples compactés.
    """
    if not content:
        return ""
    s = content
    s = _RE_BR.sub("\n", s)
    s = _RE_P_CLOSE.sub("\n\n", s)
    s = _RE_P_OPEN.sub("", s)
    s = _RE_TAG.sub("", s)
    s = html.unescape(s)
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = _RE_MULTI_NEWLINE.sub("\n\n", s)
    return s.strip()


def _strip_acclamation(text: str) -> str:
    """Retire l'acclamation finale si la dernière ligne non vide y correspond."""
    if not text:
        return text
    lines = text.rstrip().split("\n")
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return text
    last = lines[i].strip()
    for acc in _ACCLAMATIONS:
        if last == acc or last.endswith(acc):
            del lines[i:]
            return "\n".join(lines).rstrip()
    return text


def _clean_titre(titre: Optional[str]) -> str:
    """Décode entities + strip. Retourne '' si vide/None."""
    if not titre:
        return ""
    return html.unescape(titre).strip()


# ============================================================
# 3) Extraction sémantique
# ============================================================

def _find_lecture(messe_data: dict, type_lecture: str) -> dict:
    """Retourne la 1ère lecture du `type` donné dans `messes[0]`.

    Raises:
        AelfError : si la structure est inattendue ou la lecture absente.
    """
    if not isinstance(messe_data, dict):
        raise AelfError("Réponse AELF inattendue : dict attendu.")
    messes = messe_data.get("messes") or []
    if not messes:
        raise AelfError("Réponse AELF sans messe (clé 'messes' vide).")
    messe = messes[0]
    for lec in messe.get("lectures") or []:
        if lec.get("type") == type_lecture:
            return lec
    raise AelfError(
        f"Lecture de type '{type_lecture}' introuvable dans la messe."
    )


def extract_evangile(messe_data: dict) -> tuple[str, str, str]:
    """Extrait l'évangile du jour.

    Returns:
        (ref, titre, texte) : `titre` vaut "" si absent. `texte` est
        nettoyé (HTML→texte, acclamation finale retirée).

    Raises:
        AelfError : si la structure est invalide ou l'évangile absent.
    """
    lec = _find_lecture(messe_data, "evangile")
    ref = (lec.get("ref") or "").strip()
    titre = _clean_titre(lec.get("titre"))
    texte = _strip_acclamation(_html_to_text(lec.get("contenu") or ""))
    return ref, titre, texte


def extract_premiere_lecture(messe_data: dict) -> tuple[str, str, str]:
    """Extrait la 1ère lecture du jour (type == 'lecture_1').

    Returns:
        (ref, titre, texte). Mêmes garanties que `extract_evangile`.

    Raises:
        AelfError : si la 1ère lecture est absente.
    """
    lec = _find_lecture(messe_data, "lecture_1")
    ref = (lec.get("ref") or "").strip()
    titre = _clean_titre(lec.get("titre"))
    texte = _strip_acclamation(_html_to_text(lec.get("contenu") or ""))
    return ref, titre, texte


def extract_contexte_liturgique(messe_data: dict) -> str:
    """Construit un libellé lisible du contexte liturgique du jour.

    Priorité :
        1. `informations.jour_liturgique_nom` (souvent le plus complet,
           ex. "7ème Dimanche de Pâques (semaine III du Psautier)")
        2. `informations.fete` (ex. "Nativité du Seigneur")
        3. Fallback "{jour} de la {semaine}"

    Returns:
        str : "" si aucune information exploitable.
    """
    if not isinstance(messe_data, dict):
        return ""
    info = messe_data.get("informations") or {}
    nom = (info.get("jour_liturgique_nom") or "").strip()
    if nom:
        return nom
    fete = (info.get("fete") or "").strip()
    if fete:
        return fete
    jour = (info.get("jour") or "").strip()
    semaine = (info.get("semaine") or "").strip()
    if jour and semaine:
        return f"{jour.capitalize()} de la {semaine}"
    return jour.capitalize() or semaine
