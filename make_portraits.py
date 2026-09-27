# -*- coding: utf-8 -*-
"""
make_portraits.py — Génère le diptyque de l'en-tête depuis les images sources.

    static/luisa.jpg      portrait de Luisa Piccarreta,  depuis logo.jpg
    static/faustine.jpg   portrait de Sainte Faustine,   depuis data/faustine/sfi.jpg

Les deux sorties font 200 × 300 (2:3). `logo.jpg` est en 300 × 197, donc en
paysage : un portrait vertical « au même ratio » en est la transposition, à
budget de pixels constant. `logo.jpg` sert de source et n'est jamais réécrit.

Les deux sources ne se ressemblent pas : Luisa est une photographie ancienne en
noir et blanc incrustée sur un ciel très clair, Sainte Faustine une peinture
colorée sur fond sombre. Pour qu'elles fassent la paire côte à côte, on les
ramène à la même luminosité moyenne et à la même saturation. L'ajustement se
fait par correction gamma, qui déplace les demi-teintes sans écraser les noirs,
plutôt que par un réglage de luminosité, qui délaverait l'habit
de Sainte Faustine.

À exécuter au besoin (après changement d'une source ou d'un cadrage) :

    python make_portraits.py            # variante retenue : légère désaturation
    python make_portraits.py --gris     # variante niveaux de gris
"""
import argparse
import os

from PIL import Image, ImageEnhance, ImageOps, ImageStat

RACINE = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(RACINE, "static")

#: Dimensions des portraits produits, en pixels (ratio 2:3).
LARGEUR, HAUTEUR = 200, 300

#: Recadrages en buste (x0, y0, x1, y1), au ratio 2:3, visage et livre visibles.
#: Luisa          — source 300 × 197, visage en (200, 57), chapelet dans les mains.
#: Sainte Faustine — source 722 × 423, visage en (315, 92), Petit Journal
#:                   en (285-350, 275-345).
PORTRAITS = {
    "luisa": {"source": ("logo.jpg",), "boite": (137, 8, 263, 197)},
    "faustine": {"source": ("data", "faustine", "sfi.jpg"), "boite": (197, 0, 434, 355)},
}

#: Cibles communes d'harmonisation. La luminance visée penche vers l'image la
#: plus sombre : viser la moyenne exacte des deux délaverait l'habit
#: de Sainte Faustine, qui occupe l'essentiel de son cadre.
LUM_CIBLE = 106.0
SAT_CIBLE = 46.0

#: Qualité JPEG, calée pour rester dans l'ordre de grandeur de `logo.jpg`.
QUALITE = 88


def luminance(image):
    """Luminosité moyenne de l'image, sur 0-255."""
    return ImageStat.Stat(ImageOps.grayscale(image)).mean[0]


def saturation(image):
    """Saturation moyenne de l'image, sur 0-255."""
    return ImageStat.Stat(image.convert("HSV").split()[1]).mean[0]


def gamma(image, exposant):
    """Applique une correction gamma : > 1 assombrit, < 1 éclaircit."""
    table = [min(255, int(round(255.0 * (i / 255.0) ** exposant))) for i in range(256)]
    return image.point(table * len(image.getbands()))


def viser_luminance(image, cible, tolerance=0.5):
    """Cherche par dichotomie le gamma amenant la luminance moyenne sur `cible`."""
    bas, haut, exposant = 0.2, 5.0, 1.0
    for _ in range(40):
        exposant = (bas + haut) / 2
        ecart = luminance(gamma(image, exposant)) - cible
        if abs(ecart) < tolerance:
            break
        # gamma > 1 assombrit : si l'image reste trop claire, on monte.
        bas, haut = (exposant, haut) if ecart > 0 else (bas, exposant)
    return gamma(image, exposant)


def viser_saturation(image, cible, tolerance=0.5):
    """Cale la saturation moyenne sur `cible`.

    Le facteur de `ImageEnhance.Color` n'agit pas linéairement sur la saturation
    moyenne mesurée en HSV : on approche donc par corrections successives.
    """
    for _ in range(20):
        actuelle = saturation(image)
        if actuelle < 1 or abs(actuelle - cible) < tolerance:
            break
        image = ImageEnhance.Color(image).enhance(cible / actuelle)
    return image


def harmoniser(image, en_gris):
    """Ramène l'image aux tons communs du diptyque.

    La correction gamma passe en premier : elle déplace la saturation perçue, on
    cale donc celle-ci après, sur l'image déjà mise à la bonne luminosité.
    """
    if en_gris:
        return viser_luminance(ImageOps.grayscale(image).convert("RGB"), LUM_CIBLE)
    return viser_saturation(viser_luminance(image, LUM_CIBLE), SAT_CIBLE)


def produire(nom, reglages, en_gris):
    source = os.path.join(RACINE, *reglages["source"])
    if not os.path.exists(source):
        raise SystemExit("[ERREUR] Source introuvable : %s" % source)

    image = (Image.open(source).convert("RGB")
             .crop(reglages["boite"])
             .resize((LARGEUR, HAUTEUR), Image.LANCZOS))
    image = harmoniser(image, en_gris)

    chemin = os.path.join(SORTIE, "%s.jpg" % nom)
    image.save(chemin, "JPEG", quality=QUALITE, optimize=True, progressive=True)
    print("[OK] %-20s %d x %d  luminance %.1f  saturation %.1f  %d octets"
          % (chemin, image.width, image.height, luminance(image),
             saturation(image), os.path.getsize(chemin)))


def main():
    parseur = argparse.ArgumentParser(
        description="Génère les portraits de l'en-tête depuis les images sources.")
    parseur.add_argument("--gris", action="store_true",
                         help="produit la variante en niveaux de gris "
                              "au lieu de la légère désaturation")
    arguments = parseur.parse_args()

    for nom, reglages in sorted(PORTRAITS.items()):
        produire(nom, reglages, arguments.gris)


if __name__ == "__main__":
    main()
