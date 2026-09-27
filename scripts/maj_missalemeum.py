"""
maj_missalemeum.py — Vendoring du moteur de calendrier de Missale Meum
=====================================================================
Recopie dans vendor/missalemeum/ la seule partie « calendrier » du projet
Missale Meum (https://github.com/mmolenda/missalemeum, licence MIT), qui
détermine l'office célébré chaque jour selon les rubriques de 1962.

Le correctif appliqué est volontairement minimal :
  * imports `api.*` réécrits en `vendor.missalemeum.*` ;
  * imports du parseur de propres (`api.propers.*`) neutralisés : les textes
    sont lus par notre propre module `tradi.py`, pas par Missale Meum ;
  * `utils.py` réduit aux trois fonctions dont le calendrier a besoin
    (le module d'origine dépend de PyYAML et des suppléments).

Usage :
    python scripts/maj_missalemeum.py            # commit épinglé ci-dessous
    python scripts/maj_missalemeum.py <sha>      # autre commit (puis mettre
                                                 # à jour COMMIT_EPINGLE)
"""

import re
import sys
import urllib.request
from pathlib import Path

DEPOT = "mmolenda/missalemeum"
COMMIT_EPINGLE = "7be440e994f4fa8a3aa5e19f5e9da0ecb8dc0145"

BASE_DIR = Path(__file__).resolve().parent.parent
CIBLE = BASE_DIR / "vendor" / "missalemeum"

FICHIERS = [
    "LICENSE",
    "backend/api/kalendar/__init__.py",
    "backend/api/kalendar/factory.py",
    "backend/api/kalendar/models.py",
    "backend/api/kalendar/rules.py",
    "backend/api/constants/__init__.py",
    "backend/api/constants/common.py",
    "backend/api/constants/la/__init__.py",
    "backend/api/constants/la/blocks.py",
    "backend/api/constants/la/translation.py",
]

UTILS = '''"""Extrait de `api/utils.py` (Missale Meum) : fonctions utiles au calendrier."""
import re
from typing import List, Pattern, Union

from vendor.missalemeum.constants.common import CUSTOM_PREFACES


'''


def telecharger(sha: str, chemin: str) -> str:
    url = f"https://raw.githubusercontent.com/{DEPOT}/{sha}/{chemin}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def corriger(texte: str) -> str:
    texte = re.sub(r"\bapi\.(constants|kalendar|utils)\b",
                   r"vendor.missalemeum.\1", texte)
    texte = texte.replace("f'api.constants.", "f'vendor.missalemeum.constants.")
    # Parseur de propres non embarqué : les textes viennent de tradi.py.
    texte = re.sub(r"^from api\.propers\.\w+ import .*$",
                   "# (vendoring) parseur de propres non embarqué", texte,
                   flags=re.M)
    return texte


def extraire_utils(source: str) -> str:
    """Garde match_all, match_first et get_custom_preface de api/utils.py."""
    blocs = []
    for nom in ("match_all", "match_first", "get_custom_preface"):
        m = re.search(rf"^def {nom}\(.*?(?=^def |^class |\Z)", source,
                      flags=re.M | re.S)
        if not m:
            raise RuntimeError(f"{nom} introuvable dans api/utils.py")
        blocs.append(m.group(0).rstrip() + "\n")
    return UTILS + "\n\n".join(blocs)


def main() -> None:
    sha = sys.argv[1] if len(sys.argv) > 1 else COMMIT_EPINGLE
    for chemin in FICHIERS:
        rel = chemin.removeprefix("backend/api/")
        dst = CIBLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        texte = telecharger(sha, chemin)
        if chemin.endswith(".py"):
            texte = corriger(texte)
        dst.write_text(texte, encoding="utf-8", newline="\n")
    (CIBLE / "utils.py").write_text(
        extraire_utils(telecharger(sha, "backend/api/utils.py")),
        encoding="utf-8", newline="\n")
    (CIBLE / "__init__.py").write_text(
        f'"""Calendrier 1962 de Missale Meum (MIT), commit {sha}."""\n',
        encoding="utf-8", newline="\n")
    (BASE_DIR / "vendor" / "__init__.py").touch()
    print(f"Missale Meum @ {sha} copié dans {CIBLE}")


if __name__ == "__main__":
    main()
