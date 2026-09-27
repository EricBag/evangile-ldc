"""
make_icons.py — Génère les icônes de l'application (PWA, favicon)
=================================================================
Icône (variante B) : les visages de Sainte Faustine et de Luisa Piccarreta,
têtes presque jointives, en gros plan.

Montage propre à l'icône : l'image du montage (montage_luisa_faustine.py)
est reconstruite avec Faustine placée tout contre Luisa, à la même hauteur
(l'image d'accueil ne change pas). Luisa reste identique pixel pour pixel ;
Faustine passe derrière l'épaule de Luisa.

Contraintes (icône « maskable ») :
  * les deux visages tiennent entiers dans le cercle central de 80 % de
    l'icône (zone sûre des icônes maskable) et le remplissent presque ;
    l'auréole et le rayonnement peuvent être coupés ;
  * fond de ciel jusqu'aux bords, sans transparence ; aucun texte.

    python make_icons.py
"""
import numpy as np
from PIL import Image

import montage_luisa_faustine as montage

ICONES = [
    ("static/icon-512.png", 512),
    ("static/icon-192.png", 192),
    ("static/icon-180.png", 180),
    ("static/favicon.png", 32),
]

#: Placement de Faustine pour l'icône (yeux), tout contre Luisa, et fondu de
#: protection réduit : Faustine passe derrière l'épaule de Luisa. Au-delà de
#: x = 540, le menton de Faustine serait coupé par la silhouette protégée.
POSITION_ICONE = (540, montage.LUISA_YEUX[1])
FONDU_ICONE = 10

#: Visages, assimilés à des cercles (centre, rayon) dans l'image du montage.
#: Faustine : relatif à ses yeux ; Luisa : fixe.
VISAGE_FAUSTINE = ((3, 27), 75)
VISAGE_LUISA = ((730, 286), 75)

#: Part du cercle sûr (80 % de l'icône) occupée par le cercle qui englobe les
#: deux visages : ils le remplissent presque, avec 3 % de marge au bord.
REMPLISSAGE = 0.97


def cercle_englobant(c1, r1, c2, r2):
    """Plus petit cercle contenant deux cercles : (centre, rayon)."""
    (x1, y1), (x2, y2) = c1, c2
    d = float(np.hypot(x2 - x1, y2 - y1))
    if d + r2 <= r1:
        return c1, r1
    if d + r1 <= r2:
        return c2, r2
    rayon = (d + r1 + r2) / 2
    t = (rayon - r1) / d
    return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t), rayon


def image_icone():
    """Icône carrée pleine résolution (PIL RGB) et son cadrage."""
    im, _ = montage.construire(verbeux=False, position=POSITION_ICONE,
                               fondu=FONDU_ICONE, degagement_min=None, affiner=True)
    (dx, dy), rf = VISAGE_FAUSTINE
    faustine = ((POSITION_ICONE[0] + dx, POSITION_ICONE[1] + dy), rf)
    centre, rayon = cercle_englobant(*faustine, *VISAGE_LUISA)
    cote = int(np.ceil(2 * rayon / (0.8 * REMPLISSAGE)))
    x0 = round(centre[0] - cote / 2)
    y0 = round(centre[1] - cote / 2)
    assert x0 >= 0 and y0 >= 0 and x0 + cote <= im.width and y0 + cote <= im.height,         "cadrage hors de l'image"
    icone = im.crop((x0, y0, x0 + cote, y0 + cote))
    return icone, {"cote": cote, "boite": (x0, y0, x0 + cote, y0 + cote),
                   "rayon_visages": rayon, "remplissage_zone_sure": rayon / (0.4 * cote)}


def main() -> None:
    icone, infos = image_icone()
    print("Cadrage : carré de %d px %s, visages dans %.0f %% du cercle sûr"
          % (infos["cote"], infos["boite"], 100 * infos["remplissage_zone_sure"]))
    for chemin, taille in ICONES:
        icone.resize((taille, taille), Image.LANCZOS).save(chemin, "PNG", optimize=True)
        print(f"  {chemin}  ({taille}x{taille})")


if __name__ == "__main__":
    main()
