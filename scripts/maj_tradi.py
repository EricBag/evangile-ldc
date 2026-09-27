"""
maj_tradi.py — Mise à jour des textes de la messe traditionnelle (1962)
======================================================================
Retélécharge depuis le dépôt Divinum Officium les seuls fichiers texte utiles
à l'Évangile du jour et les recopie dans data/tradi/, en conservant
l'arborescence d'origine :

    data/tradi/missa/{Latin,Francais}/{Tempora,Sancti,Commune}
    data/tradi/horas/{Latin,Francais}/Commune

Le Commun des messes n'est pas dans missa/*/Commune (deux fichiers seulement) :
Divinum Officium résout les renvois « @Commune/Cx » vers horas/<langue>/Commune,
dont les fichiers portent aussi les sections de la messe ([Evangelium]…).

Usage :
    python scripts/maj_tradi.py            # dernière version de master
    python scripts/maj_tradi.py <sha>      # un commit précis

Nécessite git. Clone partiel (sparse, sans historique) dans un dossier
temporaire : seuls les dossiers ci-dessus sont téléchargés.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

DEPOT = "https://github.com/DivinumOfficium/divinum-officium.git"
BASE_DIR = Path(__file__).resolve().parent.parent
CIBLE = BASE_DIR / "data" / "tradi"

DOSSIERS = [
    f"web/www/{partie}/{langue}/{dossier}"
    for langue in ("Latin", "Francais")
    for partie, dossier in (("missa", "Tempora"), ("missa", "Sancti"),
                            ("missa", "Commune"), ("horas", "Commune"))
]


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else "master"
    with tempfile.TemporaryDirectory() as tmp:
        git("clone", "--quiet", "--filter=blob:none", "--no-checkout",
            "--depth", "1" if ref == "master" else "1000", DEPOT, tmp)
        git("sparse-checkout", "set", "--no-cone", "/LICENSE", *DOSSIERS, cwd=tmp)
        git("checkout", "--quiet", ref, cwd=tmp)
        sha = git("rev-parse", "HEAD", cwd=tmp)

        for sous in ("missa", "horas"):
            shutil.rmtree(CIBLE / sous, ignore_errors=True)
        shutil.copyfile(Path(tmp) / "LICENSE", CIBLE / "LICENSE")
        n = 0
        for dossier in DOSSIERS:
            src = Path(tmp) / dossier
            dst = CIBLE / dossier.removeprefix("web/www/")
            for f in sorted(src.glob("*.txt")):
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst / f.name)
                n += 1

    # Trace de la version embarquée dans SOURCE.md
    source = CIBLE / "SOURCE.md"
    if source.exists():
        texte = source.read_text(encoding="utf-8")
        texte = re.sub(r"(Commit embarqué : `)[0-9a-f]+(` \()[^)]*\)",
                       rf"\g<1>{sha}\g<2>{date.today().isoformat()})", texte)
        source.write_text(texte, encoding="utf-8")
    print(f"{n} fichiers copiés depuis Divinum Officium @ {sha}")


if __name__ == "__main__":
    main()
