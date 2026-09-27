# -*- coding: utf-8 -*-
"""
make_diptyque.py — Génère l'image d'accueil : deux médaillons côte à côte.

    static/diptyque.png      Sainte Faustine à gauche, Luisa Piccarreta à droite
    static/diptyque@2x.png   même image pour les écrans haute densité

Les deux portraits reçoivent le même traitement : recadrage carré en buste,
à la même échelle de visage ; harmonisation de luminosité et de saturation
(fonctions de `make_portraits.py`) ; découpe ronde lissée ; bordure fine
dorée (#c9b074, celle des portraits de l'application). L'image ne contient
aucun texte : les noms figurent dans la page (texte d'accueil, attribut alt).

Sources :
    Sainte Faustine  data/faustine/sfi.jpg    (722 × 423)
    Luisa Piccarreta static/luisa_icon.jpg    (400 × 388), le portrait des
                     icônes PWA, plus grand et mieux cadré que logo.jpg.
Les icônes PWA ne sont pas touchées (voir make_icons.py).

    python make_diptyque.py
"""
import os

from PIL import Image, ImageDraw

from make_portraits import harmoniser, luminance, saturation

RACINE = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(RACINE, "static")

#: Mise en page à l'échelle 1 (pixels CSS) ; la version 2x double tout.
DIAMETRE = 116          # médaillon
BORDURE = 1.5           # épaisseur de la bordure
ECART = 44              # espace entre les deux médaillons
MARGE = 4               # marge autour de l'ensemble

OR = (201, 176, 116, 255)           # #c9b074

#: Recadrages carrés (x0, y0, x1, y1), en buste, réglés à l'œil pour que les
#: deux visages aient la même taille dans le médaillon (le voile de Sainte
#: Faustine monte plus haut que la tête de Luisa, d'où un cadre plus serré).
PORTRAITS = [
    ("Sainte Faustine", ("data", "faustine", "sfi.jpg"), (215, 0, 415, 200)),
    ("Luisa Piccarreta", ("static", "luisa_icon.jpg"), (0, 0, 388, 388)),
]

SURECHANTILLONNAGE = 4  # découpe ronde et bordure dessinées en 4x puis réduites


def medaillon(source, boite, diametre, bordure):
    """Portrait harmonisé, découpé en disque, cerclé d'or (RGBA)."""
    image = Image.open(os.path.join(RACINE, *source)).convert("RGB").crop(boite)
    grand = diametre * SURECHANTILLONNAGE
    image = harmoniser(image.resize((grand, grand), Image.LANCZOS), en_gris=False)

    masque = Image.new("L", (grand, grand), 0)
    ImageDraw.Draw(masque).ellipse((0, 0, grand - 1, grand - 1), fill=255)
    disque = Image.new("RGBA", (grand, grand), (0, 0, 0, 0))
    disque.paste(image, (0, 0), masque)

    trait = round(bordure * SURECHANTILLONNAGE)
    ImageDraw.Draw(disque).ellipse((trait // 2, trait // 2, grand - 1 - trait // 2,
                                    grand - 1 - trait // 2),
                                   outline=OR, width=trait)
    return disque.resize((diametre, diametre), Image.LANCZOS), image


def composer(echelle):
    d = DIAMETRE * echelle
    marge, ecart = MARGE * echelle, ECART * echelle
    toile = Image.new("RGBA", (2 * d + ecart + 2 * marge, d + 2 * marge), (0, 0, 0, 0))
    for i, (nom, source, boite) in enumerate(PORTRAITS):
        disque, portrait = medaillon(source, boite, d, BORDURE * echelle)
        toile.alpha_composite(disque, (marge + i * (d + ecart), marge))
        if echelle == 1:
            print("[OK] %-17s luminance %.1f  saturation %.1f"
                  % (nom, luminance(portrait), saturation(portrait)))
    return toile


def main():
    for echelle, suffixe in ((1, ""), (2, "@2x")):
        chemin = os.path.join(SORTIE, "diptyque%s.png" % suffixe)
        composer(echelle).save(chemin, "PNG", optimize=True)
        with Image.open(chemin) as im:
            print("[OK] %-28s %d x %d  %d octets"
                  % (chemin, im.width, im.height, os.path.getsize(chemin)))


if __name__ == "__main__":
    main()
