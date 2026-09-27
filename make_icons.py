"""
make_icons.py — Génère les icônes de l'application (PWA, favicon)
=================================================================
Icône : Sainte Faustine et Luisa Piccarreta dans la lumière, tirée de
l'image complète du montage (montage_luisa_faustine.py), cadrée pour un
masque rond.

Contraintes (icône « maskable ») :
  * les deux visages, avec l'auréole de Faustine et le rayonnement autour de
    la tête de Luisa, tiennent entièrement dans le cercle central de 80 % de
    l'icône (zone sûre des icônes maskable), centrés ;
  * fond de ciel bleu jusqu'aux bords, sans transparence, pour que le
    lanceur puisse découper n'importe quelle forme ;
  * aucun texte.

Le montage ne fait que 750 px de haut et les visages sont dans sa moitié
haute : le carré de l'icône déborde au-dessus de l'image. Ce débord est
comblé par un prolongement du ciel (reflet du haut de l'image, de plus en
plus flou, fondu vers le bleu), sans toucher aux figures.

    python make_icons.py
"""
import numpy as np
from PIL import Image
from scipy import ndimage

import montage_luisa_faustine as montage

ICONES = [
    ("static/icon-512.png", 512),
    ("static/icon-192.png", 192),
    ("static/icon-180.png", 180),
    ("static/favicon.png", 32),
]

#: Part du cercle sûr (80 % de l'icône) réellement occupée par le cercle qui
#: englobe les deux têtes : 8 % de marge avant le bord du masque (au-delà,
#: le carré sortirait de l'image en largeur).
REMPLISSAGE = 0.92

#: Hauteur (px) de la bande de ciel pur du haut du montage, sans figure,
#: reflétée pour prolonger le ciel.
BANDE_CIEL = 100
#: Lignes du haut de l'image rendues progressivement floues vers le raccord,
#: et force de ce flou.
TRANSITION = 70
FLOU = 18


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


def prolonger_ciel(image, haut):
    """Ajoute `haut` px de ciel au-dessus de l'image (tableau float H × W × 3).

    Pour qu'aucun raccord ne se voie, le haut de l'image (ciel seul) devient
    progressivement flou en montant, et le ciel ajouté prolonge ce flou (bande
    de ciel reflétée, très floutée) en se fondant vers le bleu du ciel.
    """
    if haut <= 0:
        return image, 0
    h, w, _ = image.shape
    zone = image[:h // 3].reshape(-1, 3)
    saturation = zone.max(axis=1) - zone.min(axis=1)
    bleu = np.median(zone[saturation > np.percentile(saturation, 75)], axis=0)

    # Haut de l'image : du net (ligne TRANSITION) au flou (ligne 0).
    image = image.copy()
    flou = ndimage.gaussian_filter(image[:BANDE_CIEL + 40], (FLOU, FLOU, 0))
    t = np.linspace(0, 1, TRANSITION)[:, None, None]       # 0 en haut, 1 en bas
    t = t * t * (3 - 2 * t)
    image[:TRANSITION] = flou[:TRANSITION] * (1 - t) + image[:TRANSITION] * t

    # Ciel ajouté : reflet flou de la bande de ciel, étiré, fondu vers le bleu.
    # Reflet : la ligne à la distance d au-dessus du raccord reprend la ligne d
    # de l'image (ligne 0 du ciel ajouté = ligne 0 de l'image).
    bande = np.ascontiguousarray(flou[:BANDE_CIEL])
    reflet = np.stack([np.asarray(Image.fromarray(bande[..., c].astype(np.float32), "F")
                                  .resize((w, haut), Image.BICUBIC)) for c in range(3)], axis=-1)
    reflet = ndimage.gaussian_filter(reflet, (FLOU, FLOU, 0))
    k = np.linspace(0, 1, haut)[:, None, None]             # 0 au raccord, 1 en haut
    vers_bleu = 0.6 * k * k * (3 - 2 * k)
    ciel = reflet * (1 - vers_bleu) + bleu[None, None] * vers_bleu
    return np.concatenate([ciel[::-1], image], axis=0), haut


def image_icone():
    """Icône carrée pleine résolution (PIL RGB) et son cadrage."""
    im, reperes = montage.construire(verbeux=False)
    centre, rayon = cercle_englobant(*reperes["aureole"], *reperes["luisa"])
    cote = int(np.ceil(2 * rayon / (0.8 * REMPLISSAGE)))
    x0 = round(centre[0] - cote / 2)
    y0 = round(centre[1] - cote / 2)
    image = np.asarray(im).astype(float)
    image, ajout = prolonger_ciel(image, max(0, -y0))
    y0 += ajout
    assert 0 <= x0 and x0 + cote <= image.shape[1], "cadrage hors de l'image en largeur"
    assert y0 + cote <= image.shape[0], "cadrage hors de l'image en hauteur"
    carre = image[y0:y0 + cote, x0:x0 + cote]
    icone = Image.fromarray(np.rint(carre).clip(0, 255).astype(np.uint8), "RGB")
    return icone, {"cote": cote, "ciel_ajoute": ajout, "rayon_tetes": rayon,
                   "remplissage_zone_sure": rayon / (0.4 * cote)}


def main() -> None:
    icone, infos = image_icone()
    print("Cadrage : carré de %d px, ciel prolongé de %d px, têtes dans %.0f %% du "
          "cercle sûr" % (infos["cote"], infos["ciel_ajoute"],
                          100 * infos["remplissage_zone_sure"]))
    for chemin, taille in ICONES:
        icone.resize((taille, taille), Image.LANCZOS).save(chemin, "PNG", optimize=True)
        print(f"  {chemin}  ({taille}x{taille})")


if __name__ == "__main__":
    main()
