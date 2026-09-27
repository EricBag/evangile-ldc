# -*- coding: utf-8 -*-
"""
make_diptyque.py — Génère l'image d'accueil : deux médaillons côte à côte.

    static/diptyque.png      Sainte Faustine à gauche, Luisa Piccarreta à droite
    static/diptyque@2x.png   même image pour les écrans haute densité

Traitement retenu (case « c1 » de la planche comparative) : noir et blanc
doux, bordure dorée fine #c9b074. Chaîne appliquée à chaque portrait :

  1. recadrage carré en buste, à la même échelle de visage ;
  2. passage en niveaux de gris, à 232 px (médaillon en 2x) ;
  3. égalisation locale du contraste (CLAHE, écrite en numpy), force réglée
     par portrait (`limite`) ;
  4. alignement sur une cible commune de luminance moyenne et d'écart-type,
     mesurés dans le médaillon : les deux photos, très différentes au départ
     (peinture sombre, photographie claire), s'accordent ;
  5. noir et blanc doux : noirs relevés, blancs adoucis ;
  6. découpe ronde lissée, anneau doré. Hors du médaillon, transparent.

La version 1x est la réduction de la version 2x, pour un rendu identique.
L'image ne contient aucun texte : les noms figurent dans la page.

Sources :
    Sainte Faustine  data/faustine/sfi.jpg    (722 × 423)
    Luisa Piccarreta static/luisa_icon.jpg    (400 × 388), le portrait des
                     icônes PWA, plus grand et mieux cadré que logo.jpg.
Les icônes PWA ne sont pas touchées (voir make_icons.py).

    python make_diptyque.py
"""
import os

import numpy as np
from PIL import Image

RACINE = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(RACINE, "static")

#: Mise en page en 2x (la version 1x en est la réduction de moitié).
D = 232                 # diamètre du médaillon (116 px CSS)
ECART = 88              # espace entre les médaillons (44 px CSS)
MARGE = 8               # marge autour de l'ensemble (4 px CSS)

OR = (201, 176, 116)    # #c9b074

#: Cible commune après égalisation (0-255), mesurée dans le médaillon.
CIBLE_MOYENNE = 138.0
CIBLE_ECART = 52.0
TUILES = (4, 4)         # grille de l'égalisation locale

#: Portraits : (nom, source, recadrage carré (x0, y0, x1, y1), limite CLAHE).
#: Recadrages réglés à l'œil pour que les deux visages aient la même taille
#: (le voile de Sainte Faustine monte plus haut que la tête de Luisa, d'où un
#: cadre plus serré). La limite fixe la force de l'égalisation locale.
PORTRAITS = [
    ("Sainte Faustine", ("data", "faustine", "sfi.jpg"), (215, 0, 415, 200), 2.0),
    ("Luisa Piccarreta", ("static", "luisa_icon.jpg"), (0, 0, 388, 388), 2.0),
]


def clahe(gris, tuiles=TUILES, limite=2.0):
    """Égalisation d'histogramme adaptative à contraste limité (uint8 → float 0-255)."""
    h, w = gris.shape
    ty, tx = tuiles
    th, tw = h // ty, w // tx
    luts = np.zeros((ty, tx, 256), dtype=np.float64)
    for i in range(ty):
        for j in range(tx):
            tuile = gris[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            hist = np.bincount(tuile.ravel(), minlength=256).astype(np.float64)
            plafond = limite * tuile.size / 256.0
            exces = np.maximum(hist - plafond, 0).sum()
            hist = np.minimum(hist, plafond) + exces / 256.0
            cdf = hist.cumsum()
            luts[i, j] = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1) * 255.0
    # Interpolation bilinéaire entre les tables des centres de tuiles.
    yc = (np.arange(h) + 0.5) / th - 0.5
    xc = (np.arange(w) + 0.5) / tw - 0.5
    y0 = np.clip(np.floor(yc).astype(int), 0, ty - 1)
    x0 = np.clip(np.floor(xc).astype(int), 0, tx - 1)
    y1 = np.clip(y0 + 1, 0, ty - 1)
    x1 = np.clip(x0 + 1, 0, tx - 1)
    wy = np.clip(yc - y0, 0, 1)[:, None]
    wx = np.clip(xc - x0, 0, 1)[None, :]
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    v = gris.astype(int)
    haut = luts[Y0, X0, v] * (1 - wx) + luts[Y0, X1, v] * wx
    bas = luts[Y1, X0, v] * (1 - wx) + luts[Y1, X1, v] * wx
    return (haut * (1 - wy) + bas * wy).clip(0, 255)


def disque(taille, adoucir=1.0):
    """(masque du médaillon 0-1 au bord lissé, rayon normalisé)."""
    y, x = np.mgrid[0:taille, 0:taille]
    r = np.hypot(x - (taille - 1) / 2, y - (taille - 1) / 2)
    return np.clip((taille / 2 - r) / adoucir, 0, 1), r / (taille / 2)


def gris_egalise(source, boite, limite):
    """Gris (0-1) égalisé puis aligné sur la cible commune."""
    image = (Image.open(os.path.join(RACINE, *source)).convert("L")
             .crop(boite).resize((D, D), Image.LANCZOS))
    g = clahe(np.asarray(image), limite=limite)
    dedans = disque(D)[0] > 0.5
    g = (g - g[dedans].mean()) / max(g[dedans].std(), 1e-6) * CIBLE_ECART + CIBLE_MOYENNE
    return g.clip(0, 255) / 255.0


#: Courbe « noir et blanc doux ». Base : noirs relevés (0,07), blancs adoucis.
#: Au-dessus de SEUIL_BLANCS, levée progressive des hautes lumières (blancs du
#: voile et du col vers 230/255 au lieu de 214), puis épaule douce au-dessus de
#: EPAULE : la courbe tend vers le blanc sans l'atteindre (max ≈ 252), donc
#: sans écrêtage. En dessous de SEUIL_BLANCS, la courbe de base est inchangée.
SEUIL_BLANCS = 0.6
LEVEE_BLANCS = 0.196
EPAULE = 0.88


def noir_blanc_doux(g):
    """Courbe douce, blancs remontés sans écrêtage (0-1 → 0-255)."""
    t = np.clip((g - SEUIL_BLANCS) / (1 - SEUIL_BLANCS), 0, 1)
    y = 0.07 + 0.88 * g ** 0.92 + LEVEE_BLANCS * t * t
    y = np.where(y <= EPAULE, y, 1 - (1 - EPAULE) * np.exp(-(y - EPAULE) / (1 - EPAULE)))
    return y * 255.0


def medaillon(g):
    """Médaillon RGBA : portrait en noir et blanc doux, anneau doré (~3 px en 2x)."""
    masque, r = disque(D, adoucir=1.2)
    anneau = np.clip(1 - np.abs(r * D / 2 - (D / 2 - 2.2)) / 1.6, 0, 1)
    gris = noir_blanc_doux(g)[..., None]
    rgb = gris * (1 - anneau[..., None]) + np.array(OR) * anneau[..., None]
    rgba = np.dstack([rgb, masque * 255.0]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def composer():
    """Diptyque en 2x."""
    toile = Image.new("RGBA", (2 * D + ECART + 2 * MARGE, D + 2 * MARGE), (0, 0, 0, 0))
    for i, (nom, source, boite, limite) in enumerate(PORTRAITS):
        toile.alpha_composite(medaillon(gris_egalise(source, boite, limite)),
                              (MARGE + i * (D + ECART), MARGE))
    return toile


def main():
    image_2x = composer()
    image_1x = image_2x.resize((image_2x.width // 2, image_2x.height // 2), Image.LANCZOS)
    for image, suffixe in ((image_1x, ""), (image_2x, "@2x")):
        chemin = os.path.join(SORTIE, "diptyque%s.png" % suffixe)
        image.save(chemin, "PNG", optimize=True)
        print("[OK] %-28s %d x %d  %d octets"
              % (chemin, image.width, image.height, os.path.getsize(chemin)))


if __name__ == "__main__":
    main()
