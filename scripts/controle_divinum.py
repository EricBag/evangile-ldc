"""
controle_divinum.py — Contrôle indépendant sur le site Divinum Officium
======================================================================
Script ponctuel, hors application. Pour chaque jour d'une période, récupère
la messe du jour sur www.divinumofficium.com (rubriques 1960) et note le nom
de la messe et la référence de l'Évangile affichés par le site.

Les résultats sont enregistrés au fur et à mesure dans
data/tradi/controle_divinum_<années>.json ; une relance reprend là où le
script s'était arrêté. La comparaison avec tradi.py est faite par
scripts/rapport_tradi.py, qui liste les écarts dans le rapport annuel.

Une requête par seconde au plus, pour ne pas surcharger le site.

Usage :
    python scripts/controle_divinum.py [debut] [fin]
Par défaut : 29/11/2026 → 27/11/2027.
"""

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
URL = "https://www.divinumofficium.com/cgi-bin/missa/missa.pl"
AGENT = "evangile-ldc/controle-ponctuel (comparaison de calendrier, 1 requete/s)"
PAUSE = 1.0

_TITRE = re.compile(r'<FORM[^>]*>\s*<P ALIGN="CENTER"><FONT[^>]*>(.*?)</FONT>',
                    re.S | re.I)
_EVANGELIUM = re.compile(r"<I>Evangelium</I>", re.I)
_REF_ROUGE = re.compile(r'<FONT COLOR="red"><I>([^<]+)</I></FONT>', re.I)
_FORME_REF = re.compile(r"^[1-4]?\s?[A-Za-zÀ-ÿ]+\.?,?\s+\d")


def texte(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def analyser(page: str) -> dict:
    """Nom de la messe et référence de l'Évangile d'une page du site."""
    titre = _TITRE.search(page)
    resultat = {"titre": texte(titre.group(1)) if titre else None, "reference": None}
    debut = _EVANGELIUM.search(page)
    if debut:
        for m in _REF_ROUGE.finditer(page, debut.end()):
            candidat = texte(m.group(1))
            if _FORME_REF.match(candidat):
                resultat["reference"] = candidat
                break
    return resultat


def recuperer(jour: date) -> dict:
    params = urllib.parse.urlencode({
        "version": "Rubrics 1960 - 1960",
        "lang2": "Francais",
        "date": jour.strftime("%m-%d-%Y"),
        "command": "praySanctaMissa",
    })
    req = urllib.request.Request(f"{URL}?{params}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return analyser(r.read().decode("utf-8", errors="replace"))


def main() -> None:
    debut = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 11, 29)
    fin = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2027, 11, 27)
    sortie = BASE_DIR / "data" / "tradi" / f"controle_divinum_{debut.year}-{fin.year}.json"
    resultats = json.loads(sortie.read_text(encoding="utf-8")) if sortie.exists() else {}

    j = debut
    while j <= fin:
        cle = j.isoformat()
        if cle not in resultats or resultats[cle].get("erreur"):
            try:
                resultats[cle] = recuperer(j)
            except Exception as exc:  # réseau : noté, repris à la prochaine relance
                resultats[cle] = {"titre": None, "reference": None, "erreur": str(exc)}
            print(cle, resultats[cle], flush=True)
            sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=1,
                                         sort_keys=True), encoding="utf-8")
            time.sleep(PAUSE)
        j += timedelta(days=1)
    manquants = sum(1 for v in resultats.values() if not v.get("reference"))
    print(f"{len(resultats)} jours enregistrés, {manquants} sans référence → {sortie}")


if __name__ == "__main__":
    main()
