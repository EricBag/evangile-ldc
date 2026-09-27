# -*- coding: utf-8 -*-
"""
montage_luisa_faustine.py — Sainte Faustine ajoutée à l'image de Luisa Piccarreta.

Montage scripté (Pillow + numpy + scipy), sans IA générative.

Base : luisa-piccarreta.webp (1024 × 750, ciel bleu, rayons derrière Luisa).
La figure de Luisa n'est jamais modifiée : une zone protégée couvre toute sa
silhouette (tête, buste, bras, livre, couverture) ; toute retouche s'éteint
avant d'y entrer, et le script vérifie que ses pixels sont identiques à
l'original.

Sainte Faustine (data/faustine/sfi.jpg), dans le ciel à gauche, au même plan
que Luisa :
  1. détourage par ses habits sombres (voile, habit), visage et guimpe
     inclus ; liseré clair du bord droit du voile retiré ; tête et épaules
     seulement, bords fondus, bas du buste perdu dans les nuages ;
  2. même échelle de visage que Luisa (distance yeux-menton : 85 px contre
     51 px dans la source, ×1,66), sans miroir, rapprochée de Luisa avec un
     dégagement d'au moins 25 px entre sa tête et ses épaules et la silhouette ;
  3. lumière venant de la droite, comme pour Luisa ;
  4. étalonnage sur Luisa : niveaux de l'habit, luminance médiane du visage,
     blanc maximal et netteté identiques (ni plus blanc ni plus net) ; teinte
     bleutée relevée sur Luisa ;
  5. halo de lumière blanche douce derrière la tête, plus discret que le
     rayonnement de Luisa, et auréole : anneau blanc net et fin, légèrement
     plus large que la tête, avec une lueur douce (aucune autour de Luisa).

Sorties : static/accueil_luisa_faustine.{webp,jpg} et @2x (image d'accueil
et de la page de connexion) ; image pleine taille dans static/_essais/.
Source de Luisa : data/luisa/luisa-piccarreta.webp.

    python montage_luisa_faustine.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _racine():
    """Racine du dépôt (dossier contenant data/faustine)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "data", "faustine")):
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("[ERREUR] racine du dépôt introuvable")
        d = parent
    return d


RACINE = _racine()
BASE = next(p for p in (os.path.join(RACINE, "data", "luisa", "luisa-piccarreta.webp"),
                        os.path.join(RACINE, "static", "_essais", "luisa-piccarreta.webp"))
            if os.path.exists(p))
FAUSTINE = os.path.join(RACINE, "data", "faustine", "sfi.jpg")
STATIQUE = os.path.join(RACINE, "static")
ESSAIS = os.path.join(STATIQUE, "_essais")          # hors dépôt : image pleine taille

#: Image d'accueil, 1,8:1 : 360 × 200 (1x) et 720 × 400 (2x), en WebP et en
#: JPEG de repli. Qualités réglées pour rester sous 100 ko en 2x.
NOM_ACCUEIL = "accueil_luisa_faustine"
QUALITE_WEBP = 82
QUALITE_JPEG = 84
POIDS_MAX_2X = 100 * 1024

# ------------------------------------------------------------- Mesures
#: Luisa (image de base) : milieu des yeux, distance yeux-menton.
LUISA_YEUX = (746, 263)
LUISA_YEUX_MENTON = 85.0
#: Faustine (sfi.jpg) : milieu des yeux, distance yeux-menton.
FAUSTINE_YEUX = (319, 75)
FAUSTINE_YEUX_MENTON = 51.0
ECHELLE = LUISA_YEUX_MENTON / FAUSTINE_YEUX_MENTON

#: Zones de mesure sur Luisa (coordonnées de l'image de base).
LUISA_VISAGE = (705, 240, 790, 335)
LUISA_TETE = (650, 160, 830, 360)
LUISA_HABIT = (560, 390, 880, 500)
LUISA_PERSONNE = (640, 165, 910, 520)
#: Zones de mesure sur Faustine (coordonnées de sfi.jpg).
FAUSTINE_VISAGE = (290, 60, 350, 122)
FAUSTINE_TETE = (255, 40, 380, 160)       # visage, guimpe et col

#: Silhouette de Luisa (contour large, coordonnées de l'image de base) :
#: tête, épaules, bras, livre à gauche, couverture et mains.
SILHOUETTE_LUISA = [
    (330, 448), (492, 436), (560, 342), (648, 300), (650, 205), (700, 158),
    (792, 156), (832, 222), (836, 300), (905, 330), (955, 420), (975, 560),
    (905, 650), (800, 710), (600, 725), (440, 650), (428, 560), (330, 545),
]
#: Largeur (px) du fondu qui éteint toute retouche avant la silhouette :
#: assez large pour que le buste de Faustine s'efface sans arête.
FONDU_PROTECTION = 60
#: Dégagement minimal (px) entre la tête et les épaules de Faustine
#: (alpha > 0,5, au-dessus de LIMITE_BUSTE) et la silhouette de Luisa. Plus
#: bas, le buste peut se fondre : le fondu de protection l'efface avant la
#: silhouette (bras gauche et livre de Luisa, sous y = 390).
DEGAGEMENT = 25
LIMITE_BUSTE = 390
#: Position des yeux de Faustine : rapprochée de Luisa (≈ 85 px plus près que
#: la première version), sans la serrer ; même hauteur que les yeux de Luisa.
FAUSTINE_POSITION = (310, LUISA_YEUX[1])

#: Lumière venant de la droite : ± LUMIERE niveaux de luminance.
LUMIERE = 14.0
FORCE_HALO = 0.4          # halo diffus, atténué pour que l'auréole s'y fonde

#: Auréole de Faustine : anneau blanc net et fin derrière la tête, avec une
#: légère lueur. Rayon = demi-largeur de la tête × AUREOLE_MARGE.
AUREOLE = True
AUREOLE_MARGE = 1.12
AUREOLE_TRAIT = 3.0       # épaisseur du trait (px)
AUREOLE_OPACITE = 0.6     # opacité du trait
AUREOLE_LUEUR = 0.28      # intensité de la lueur autour du trait
AUREOLE_FLOU_LUEUR = 5.0  # étendue de la lueur (px)


def luminance(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def redimensionner(tableau, taille):
    return np.asarray(Image.fromarray(tableau.astype(np.float32), "F").resize(taille, Image.LANCZOS))


# ------------------------------------------------------------- Zone protégée

#: Secteur de l'épaule gauche de Luisa (x0, y0, x1, y1) où le contour large
#: englobe du ciel clair ; option `affiner` de zone_luisa().
SECTEUR_EPAULE = (440, 290, 650, 460)
SEUIL_CIEL_EPAULE = 150


def zone_luisa(forme, fondu=FONDU_PROTECTION, base=None, affiner=False):
    """(silhouette de Luisa, facteur 0-1 éteignant les retouches à son approche).

    `affiner` (avec `base`) : dans le secteur de l'épaule, le ciel clair
    compris dans le contour large est retiré de la silhouette ; l'habit sombre
    de Luisa y reste protégé. Utile quand Faustine passe derrière l'épaule
    (icône)."""
    h, w = forme
    masque = Image.new("L", (w, h), 0)
    ImageDraw.Draw(masque).polygon(SILHOUETTE_LUISA, fill=255)
    silhouette = np.asarray(masque) > 0
    if affiner:
        x0, y0, x1, y1 = SECTEUR_EPAULE
        secteur = np.zeros_like(silhouette)
        secteur[y0:y1, x0:x1] = True
        ciel = ndimage.gaussian_filter(luminance(base), 1.5) > SEUIL_CIEL_EPAULE
        silhouette = silhouette & ~(secteur & ciel)
        # Nettoyage, puis marge de 2 px autour de l'habit.
        silhouette = ndimage.binary_opening(silhouette, iterations=2)
        silhouette = ndimage.binary_dilation(silhouette, iterations=2) & (np.asarray(masque) > 0)
    distance = ndimage.distance_transform_edt(~silhouette)
    return silhouette, smoothstep(0, fondu, distance)


# ------------------------------------------------------------- Détourage

def masque_faustine(src):
    """Alpha (0-1) de Faustine dans les coordonnées de sfi.jpg."""
    g = ndimage.gaussian_filter(luminance(src), 1.2)
    sombre = g < 85
    # Fermeture large, sur une image agrandie d'une marge : la guimpe blanche
    # touche le fond à droite, il faut fermer le contour du voile pour que le
    # visage soit « dans » la silhouette, sans rogner le haut du voile.
    marge = 30
    ferme = ndimage.binary_closing(np.pad(sombre, marge), structure=np.ones((15, 15)),
                                   iterations=3)[marge:-marge, marge:-marge]
    silhouette = ndimage.binary_fill_holes(ferme | sombre)
    etiquettes, n = ndimage.label(silhouette)
    if n > 1:
        tailles = ndimage.sum(silhouette, etiquettes, range(1, n + 1))
        silhouette = etiquettes == (1 + int(np.argmax(tailles)))
    h, w = g.shape
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = FAUSTINE_YEUX
    # Érosion légère partout (pas de liseré du fond), plus forte sur le bord
    # droit du voile, où la peinture a un filet clair (x ≈ 372-376).
    erodee = ndimage.binary_erosion(silhouette, iterations=2)
    erodee_droite = ndimage.binary_erosion(silhouette, iterations=6)
    silhouette = np.where(x > cx + 36, erodee_droite, erodee)
    # Tête et épaules : ellipse étroite autour du buste (bras tendu exclu),
    # pour que Faustine puisse se rapprocher de Luisa.
    ellipse = ((x - cx) / 125.0) ** 2 + ((y - (cy + 70)) / 200.0) ** 2
    zone = 1 - smoothstep(0.75, 1.0, ellipse)
    # Bas du buste perdu dans les nuages.
    zone *= 1 - smoothstep(cy + 115, cy + 215, y)
    alpha = ndimage.gaussian_filter(silhouette.astype(float), 2.2) * zone
    return np.clip(alpha, 0, 1)


# ------------------------------------------------------------- Étalonnage

def table_teinte(base):
    """Table luminance → RGB relevée sur Luisa (pixels peu saturés)."""
    x0, y0, x1, y1 = LUISA_PERSONNE
    zone = base[y0:y1, x0:x1].reshape(-1, 3)
    lum = luminance(zone)
    garde = (zone.max(axis=1) - zone.min(axis=1)) < 70      # exclut le ciel
    zone, lum = zone[garde], lum[garde]
    points, couleurs = [], []
    for a in range(0, 256, 8):
        sel = (lum >= a) & (lum < a + 8)
        if sel.sum() >= 20:
            points.append(lum[sel].mean())
            couleurs.append(np.median(zone[sel], axis=0))
    return np.array(points), np.array(couleurs)


def appliquer_teinte(lum, table):
    points, couleurs = table
    out = np.stack([np.interp(lum, points, couleurs[:, k]) for k in range(3)], axis=-1)
    bas, haut = lum < points[0], lum > points[-1]
    out[bas] = lum[bas, None] * (couleurs[0] / max(points[0], 1))
    out[haut] = 255 - (255 - lum[haut, None]) * ((255 - couleurs[-1]) / max(255 - points[-1], 1))
    return out


def detail_fin(lum, box):
    x0, y0, x1, y1 = box
    zone = lum[y0:y1, x0:x1]
    return float(np.std(zone - ndimage.gaussian_filter(zone, 1.5)))


def references_luisa(base):
    lb = luminance(base)
    x0, y0, x1, y1 = LUISA_VISAGE
    visage = float(np.median(lb[y0:y1, x0:x1]))
    x0, y0, x1, y1 = LUISA_HABIT
    zone = lb[y0:y1, x0:x1]
    habit = float(np.median(zone[zone < 120]))
    x0, y0, x1, y1 = LUISA_TETE
    tete = base[y0:y1, x0:x1]
    peu_sature = (tete.max(axis=-1) - tete.min(axis=-1)) < 45      # sans le ciel
    blanc = float(np.percentile(luminance(tete)[peu_sature], 99.5))
    return {"visage": visage, "habit": habit, "blanc": blanc,
            "detail": detail_fin(lb, LUISA_VISAGE)}


def boite_echelle(boite, echelle):
    return tuple(round(v * echelle) for v in boite)


def preparer_faustine(base, ref):
    """Faustine mise à l'échelle et étalonnée sur Luisa : (rgb, alpha, mesures)."""
    src = np.asarray(Image.open(FAUSTINE).convert("RGB")).astype(float)
    alpha = masque_faustine(src)
    lum = luminance(src)

    # Niveaux : habit et visage calés sur ceux de Luisa.
    x0, y0, x1, y1 = FAUSTINE_VISAGE
    visage_f = np.median(lum[y0:y1, x0:x1])
    habit_f = np.median(lum[(alpha > 0.9) & (lum < 85)])
    pente = (ref["visage"] - ref["habit"]) / (visage_f - habit_f)
    lum = (lum - habit_f) * pente + ref["habit"]

    taille = (round(src.shape[1] * ECHELLE), round(src.shape[0] * ECHELLE))
    lum, alpha = redimensionner(lum, taille), redimensionner(alpha, taille)
    visage = boite_echelle(FAUSTINE_VISAGE, ECHELLE)
    tete = boite_echelle(FAUSTINE_TETE, ECHELLE)

    # Lumière venant de la droite : le côté droit s'éclaircit, le gauche
    # s'assombrit, en douceur, sur la largeur de la tête.
    cx = FAUSTINE_YEUX[0] * ECHELLE
    xs = np.arange(lum.shape[1])[None, :]
    lum = lum + LUMIERE * np.tanh((xs - cx) / (60 * ECHELLE))

    # Visage : même luminance médiane que Luisa (recalée après l'éclairage).
    vx0, vy0, vx1, vy1 = visage
    lum = lum + (ref["visage"] - np.median(lum[vy0:vy1, vx0:vx1]))

    # Netteté : flou jusqu'au détail fin du visage de Luisa, pas au-delà.
    sigma = 0.0
    for s in np.arange(0.0, 3.01, 0.1):
        essai = ndimage.gaussian_filter(lum, s) if s else lum
        if detail_fin(essai, visage) <= ref["detail"]:
            sigma, lum = s, essai
            break

    # Blanc maximal : hautes lumières (guimpe, col) ramenées au blanc de Luisa,
    # par compression linéaire au-dessus du visage.
    tx0, ty0, tx1, ty1 = tete
    blanc_f = np.percentile(lum[ty0:ty1, tx0:tx1], 99.5)
    seuil = ref["visage"]
    if blanc_f > ref["blanc"]:
        haut = lum > seuil
        lum[haut] = seuil + (lum[haut] - seuil) * (ref["blanc"] - seuil) / (blanc_f - seuil)
    lum = np.clip(lum, 0, 255)

    mesures = {
        "visage": float(np.median(lum[vy0:vy1, vx0:vx1])),
        "blanc": float(np.percentile(lum[ty0:ty1, tx0:tx1], 99.5)),
        "detail": detail_fin(lum, visage),
        "flou": sigma,
    }
    return appliquer_teinte(lum, table_teinte(base)), np.clip(alpha, 0, 1), mesures


# ------------------------------------------------------------- Montage

def halo(forme, centre, rayon, force):
    """Halo blanc doux, à peine rayonnant."""
    h, w = forme
    y, x = np.mgrid[0:h, 0:w]
    dx, dy = x - centre[0], y - centre[1]
    r = np.hypot(dx, dy) / rayon
    rayons = 1 + 0.18 * np.cos(14 * np.arctan2(dy, dx)) * np.clip(r, 0, 1.5)
    return np.clip(force * np.exp(-(r ** 2) * 1.6) * rayons, 0, 1)


def placement(alpha, yeux, forme):
    """Coin supérieur gauche et boîte visible de Faustine dans l'image."""
    ox = round(yeux[0] - FAUSTINE_YEUX[0] * ECHELLE)
    oy = round(yeux[1] - FAUSTINE_YEUX[1] * ECHELLE)
    h, w = forme
    fh, fw = alpha.shape
    return ox, oy, (max(ox, 0), max(oy, 0), min(ox + fw, w), min(oy + fh, h))


def degagement(alpha, yeux, silhouette):
    """Distance minimale (px) entre la tête et les épaules de Faustine et la silhouette."""
    ox, oy, (x0, y0, x1, y1) = placement(alpha, yeux, silhouette.shape)
    corps = np.zeros(silhouette.shape, bool)
    corps[y0:y1, x0:x1] = alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox] > 0.5
    corps[LIMITE_BUSTE:] = False
    distance = ndimage.distance_transform_edt(~silhouette)
    return float(distance[corps].min()) if corps.any() else np.inf


def geometrie_tete(alpha, yeux, forme):
    """(centre, demi-largeur) de la tête de Faustine dans l'image.

    Demi-largeur : boîte de l'alpha > 0,5 entre le haut du voile et le menton.
    Centre horizontal : milieu du haut de la tête (du sommet du voile aux
    yeux), pour ne pas être tiré à gauche par le pan de voile qui tombe.
    """
    ox, oy, (x0, y0, x1, y1) = placement(alpha, yeux, forme)
    menton = round(yeux[1] + FAUSTINE_YEUX_MENTON * ECHELLE)
    a = np.zeros(forme)
    a[y0:y1, x0:x1] = alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    lignes, colonnes = np.nonzero(a[:menton] > 0.5)
    haut, gauche, droite = lignes.min(), colonnes.min(), colonnes.max()
    _, colonnes_haut = np.nonzero(a[haut:round(yeux[1])] > 0.5)
    centre_x = (colonnes_haut.min() + colonnes_haut.max()) / 2
    return (centre_x, (haut + menton) / 2), (droite - gauche) / 2


def aureole(forme, centre, rayon):
    """Opacité (0-1) de l'auréole : anneau net antialiasé + lueur douce."""
    h, w = forme
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - centre[0], y - centre[1])
    trait = np.clip(1 - np.abs(r - rayon) / (AUREOLE_TRAIT / 2) + 0.5, 0, 1)
    lueur = ndimage.gaussian_filter(trait, AUREOLE_FLOU_LUEUR)
    lueur /= max(lueur.max(), 1e-6)
    return np.clip(trait * AUREOLE_OPACITE + lueur * AUREOLE_LUEUR, 0, 1)


def composer(base, yeux, rgb, alpha, protection):
    sortie = base.copy()
    h, w = base.shape[:2]
    centre = (yeux[0], yeux[1] - 0.45 * LUISA_YEUX_MENTON)
    i = (halo((h, w), centre, 150, FORCE_HALO) * protection)[..., None]
    sortie = 255 - (255 - sortie) * (1 - i)                      # mélange « écran »
    if AUREOLE:
        # Blanc pur, peint derrière Faustine : sa tête en masque le bas.
        centre_tete, demi_largeur = geometrie_tete(alpha, yeux, (h, w))
        o = (aureole((h, w), centre_tete, demi_largeur * AUREOLE_MARGE) * protection)[..., None]
        sortie = sortie * (1 - o) + 255.0 * o
    ox, oy, (x0, y0, x1, y1) = placement(alpha, yeux, (h, w))
    a = alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox] * protection[y0:y1, x0:x1]
    f = rgb[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    sortie[y0:y1, x0:x1] = f * a[..., None] + sortie[y0:y1, x0:x1] * (1 - a[..., None])
    return sortie


#: Tête de Luisa avec le rayonnement proche (cercle englobant), pour les
#: recadrages qui doivent la garder entière (icônes, voir make_icons.py).
LUISA_TETE_CERCLE = ((740, 258), 118)


def construire(verbeux=True, position=FAUSTINE_POSITION, fondu=FONDU_PROTECTION,
               degagement_min=DEGAGEMENT, affiner=False):
    """Image complète (PIL, 1024 × 750) et repères utiles aux recadrages :
    {"yeux_faustine", "aureole": (centre, rayon), "luisa": (centre, rayon)}.

    Par défaut, le montage de l'image d'accueil. `position` (yeux de
    Faustine), `fondu` (fondu de protection autour de Luisa) et
    `degagement_min` (None : pas de contrôle) et `affiner` (voir zone_luisa)
    permettent d'autres placements,
    comme celui de l'icône (make_icons.py). Luisa reste identique dans tous
    les cas (contrôlé)."""
    base = np.asarray(Image.open(BASE).convert("RGB")).astype(float)
    silhouette, protection = zone_luisa(base.shape[:2], fondu, base, affiner)
    ref = references_luisa(base)
    rgb, alpha, mesures = preparer_faustine(base, ref)

    x_yeux, y_yeux = position
    if degagement_min is not None:
        assert degagement(alpha, (x_yeux, y_yeux), silhouette) >= degagement_min,             "Faustine trop proche de la silhouette de Luisa"
    image = composer(base, (x_yeux, y_yeux), rgb, alpha, protection)

    ecart = np.abs(np.rint(image) - base)[silhouette].max()
    if verbeux:
        print("[OK] yeux de Faustine en (%d, %d) ; dégagement %.0f px ; écart max dans la "
              "silhouette de Luisa = %.0f" % (x_yeux, y_yeux,
                                               degagement(alpha, (x_yeux, y_yeux), silhouette), ecart))
        print("     visage  Faustine %.1f / Luisa %.1f" % (mesures["visage"], ref["visage"]))
        print("     blanc   Faustine %.1f / Luisa %.1f" % (mesures["blanc"], ref["blanc"]))
        print("     détail  Faustine %.2f / Luisa %.2f (flou %.1f)"
              % (mesures["detail"], ref["detail"], mesures["flou"]))
    assert ecart == 0, "la figure de Luisa a été modifiée"

    centre_tete, demi_largeur = geometrie_tete(alpha, (x_yeux, y_yeux), base.shape[:2])
    reperes = {
        "yeux_faustine": (x_yeux, y_yeux),
        # Rayon de l'auréole, lueur comprise.
        "aureole": (centre_tete, demi_largeur * AUREOLE_MARGE + AUREOLE_FLOU_LUEUR * 2),
        "luisa": LUISA_TETE_CERCLE,
    }
    return Image.fromarray(np.rint(image).clip(0, 255).astype(np.uint8), "RGB"), reperes


def main():
    im, reperes = construire()
    x_yeux, y_yeux = reperes["yeux_faustine"]
    os.makedirs(ESSAIS, exist_ok=True)
    im.save(os.path.join(ESSAIS, "luisa_faustine_v1.png"), optimize=True)

    # Recadrage pour l'accueil sur mobile (1,8:1), centré sur les deux figures.
    cx = (x_yeux + LUISA_YEUX[0]) / 2
    largeur = 880
    boite = (round(cx - largeur / 2), 40, round(cx + largeur / 2), 40 + round(largeur / 1.8))
    for l, suffixe in ((720, "@2x"), (360, "")):
        rec = im.crop(boite).resize((l, round(l / 1.8)), Image.LANCZOS)
        for ext, options in (("webp", {"quality": QUALITE_WEBP, "method": 6}),
                             ("jpg", {"quality": QUALITE_JPEG, "optimize": True,
                                      "progressive": True})):
            chemin = os.path.join(STATIQUE, f"{NOM_ACCUEIL}{suffixe}.{ext}")
            rec.save(chemin, **options)
            poids = os.path.getsize(chemin)
            print("[OK] %-34s %d x %d  %6d octets" % (os.path.relpath(chemin, RACINE),
                                                      rec.width, rec.height, poids))
            if suffixe == "@2x":
                assert poids < POIDS_MAX_2X, f"{chemin} dépasse 100 ko"
    print("[OK] recadrage mobile", boite)
    return im, boite


if __name__ == "__main__":
    main()
