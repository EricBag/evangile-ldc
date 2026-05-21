"""
make_icons.py — Génère les icônes PWA depuis static/luisa.jpg
=============================================================
Crée 3 PNG carrés : portrait de Luisa recadré en cercle sur fond
ivoire (#fdfaf2), photo occupant ~80 % (marge ivoire autour).

À exécuter au besoin (après changement de luisa.jpg) :
    .venv/Scripts/python.exe make_icons.py
"""
from PIL import Image, ImageDraw

SRC = "static/luisa.jpg"
IVORY = (253, 250, 242)        # #fdfaf2
PHOTO_RATIO = 0.80             # la photo occupe 80 % de l'icône
SUPERSAMPLE = 4                # anti-crénelage du cercle

ICONS = [
    ("static/icon-180.png", 180),
    ("static/icon-192.png", 192),
    ("static/favicon.png", 32),
]


def make_icon(out_path: str, size: int) -> None:
    big = size * SUPERSAMPLE

    # Fond ivoire (carré plein, jusqu'aux bords -> compatible "maskable")
    icon = Image.new("RGBA", (big, big), IVORY + (255,))

    # Crop carré centré de la source
    src = Image.open(SRC).convert("RGB")
    w, h = src.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    src_sq = src.crop((left, top, left + side, top + side))

    # Redimensionnement de la photo à 80 % de l'icône
    photo_size = int(big * PHOTO_RATIO)
    photo = src_sq.resize((photo_size, photo_size), Image.LANCZOS).convert("RGBA")

    # Masque circulaire
    mask = Image.new("L", (photo_size, photo_size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, photo_size - 1, photo_size - 1), fill=255)

    # Collage centré
    offset = (big - photo_size) // 2
    icon.paste(photo, (offset, offset), mask)

    # Réduction finale (anti-crénelage)
    icon = icon.resize((size, size), Image.LANCZOS)
    icon.save(out_path, "PNG", optimize=True)
    print(f"  {out_path}  ({size}x{size})")


def main() -> None:
    print("Génération des icônes PWA :")
    for path, size in ICONS:
        make_icon(path, size)
    print("Terminé.")


if __name__ == "__main__":
    main()
