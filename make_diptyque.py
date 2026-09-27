# -*- coding: utf-8 -*-
"""
make_diptyque.py — Génère l'image d'accueil : deux médaillons côte à côte.

    static/diptyque.png      Sainte Faustine à gauche, Luisa Piccarreta à droite
    static/diptyque@2x.png   même image pour les écrans haute densité

Les deux portraits reçoivent le même traitement : recadrage carré en buste,
à la même échelle de visage ; harmonisation de luminosité et de saturation
(fonctions de `make_portraits.py`) ; découpe ronde lissée ; bordure fine
gris-bleu (#5a7a99, l'accent bleu de l'application). Le nom de chaque sainte
est écrit dessous en Cormorant Garamond, la police des titres de l'app.

Sources :
    Sainte Faustine  data/faustine/sfi.jpg    (722 × 423)
    Luisa Piccarreta static/luisa_icon.jpg    (400 × 388), le portrait des
                     icônes PWA, plus grand et mieux cadré que logo.jpg.
Les icônes PWA ne sont pas touchées (voir make_icons.py).

La police (licence SIL OFL) est téléchargée au premier lancement depuis le
dépôt Google Fonts dans .cache/fonts/ (ignoré par Git).

    python make_diptyque.py
"""
import os
import urllib.request

from PIL import Image, ImageDraw, ImageFont

from make_portraits import harmoniser, luminance, saturation

RACINE = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(RACINE, "static")

#: Mise en page à l'échelle 1 (pixels CSS) ; la version 2x double tout.
DIAMETRE = 116          # médaillon
BORDURE = 1.5           # épaisseur de la bordure
ECART = 44              # espace entre les deux médaillons
MARGE = 8               # marge autour de l'ensemble
TAILLE_NOM = 17         # corps du nom
ESPACE_NOM = 7          # entre le médaillon et le nom

BLEU_GRIS = (90, 122, 153, 255)     # #5a7a99
ENCRE = (74, 56, 32, 255)           # #4a3820, couleur des sous-titres

#: Recadrages carrés (x0, y0, x1, y1), en buste, réglés à l'œil pour que les
#: deux visages aient la même taille dans le médaillon (le voile de Sainte
#: Faustine monte plus haut que la tête de Luisa, d'où un cadre plus serré).
PORTRAITS = [
    ("Sainte Faustine", ("data", "faustine", "sfi.jpg"), (215, 0, 415, 200)),
    ("Luisa Piccarreta", ("static", "luisa_icon.jpg"), (0, 0, 388, 388)),
]

POLICE_URL = ("https://raw.githubusercontent.com/google/fonts/main/ofl/"
              "cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf")
POLICE = os.path.join(RACINE, ".cache", "fonts", "CormorantGaramond[wght].ttf")
GRAISSE = 500           # comme les titres de l'app (font-weight: 500)
SURECHANTILLONNAGE = 4  # découpe ronde et bordure dessinées en 4x puis réduites


def police(taille):
    if not os.path.exists(POLICE):
        os.makedirs(os.path.dirname(POLICE), exist_ok=True)
        print("[..] Téléchargement de Cormorant Garamond (OFL)")
        urllib.request.urlretrieve(POLICE_URL, POLICE)
    f = ImageFont.truetype(POLICE, taille)
    f.set_variation_by_axes([GRAISSE])
    return f


def medaillon(source, boite, diametre, bordure):
    """Portrait harmonisé, découpé en disque, cerclé de gris-bleu (RGBA)."""
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
                                   outline=BLEU_GRIS, width=trait)
    return disque.resize((diametre, diametre), Image.LANCZOS), image


def composer(echelle):
    d = DIAMETRE * echelle
    marge, ecart = MARGE * echelle, ECART * echelle
    fonte = police(TAILLE_NOM * echelle)
    hauteur_nom = fonte.getbbox("Ag")[3]
    largeur = 2 * d + ecart + 2 * marge
    hauteur = marge + d + ESPACE_NOM * echelle + hauteur_nom + marge
    toile = Image.new("RGBA", (largeur, hauteur), (0, 0, 0, 0))
    dessin = ImageDraw.Draw(toile)

    for i, (nom, source, boite) in enumerate(PORTRAITS):
        disque, portrait = medaillon(source, boite, d, BORDURE * echelle)
        x = marge + i * (d + ecart)
        toile.alpha_composite(disque, (x, marge))
        centre = x + d / 2
        dessin.text((centre, marge + d + ESPACE_NOM * echelle), nom,
                    font=fonte, fill=ENCRE, anchor="mt")
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
