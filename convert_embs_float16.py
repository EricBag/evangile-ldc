"""
convert_embs_float16.py — Script one-shot
=========================================
Convertit ldc_index_word/embs.npy de float32 -> float16 pour passer
sous la limite GitHub de 100 Mio par fichier.

À exécuter UNE SEULE FOIS :
    .venv/Scripts/python.exe convert_embs_float16.py

Peut être supprimé après usage.
"""
import os
import numpy as np

EMBS_PATH = os.path.join("ldc_index_word", "embs.npy")


def _size_mo(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    if not os.path.exists(EMBS_PATH):
        raise SystemExit(f"Introuvable : {EMBS_PATH}")

    before_mo = _size_mo(EMBS_PATH)
    embs = np.load(EMBS_PATH)
    print(f"AVANT  : {before_mo:.2f} Mo  | shape={embs.shape} dtype={embs.dtype}")

    if embs.dtype == np.float16:
        print("Déjà en float16 — rien à faire.")
        return

    embs_float16 = embs.astype(np.float16)
    np.save(EMBS_PATH, embs_float16)

    after_mo = _size_mo(EMBS_PATH)
    reloaded = np.load(EMBS_PATH)
    print(f"APRÈS  : {after_mo:.2f} Mo  | shape={reloaded.shape} dtype={reloaded.dtype}")
    print(f"Gain   : -{before_mo - after_mo:.2f} Mo ({100 * (1 - after_mo / before_mo):.0f} %)")
    print("OK — sous 100 Mio." if after_mo < 100 else "ATTENTION — toujours au-dessus de 100 Mo.")


if __name__ == "__main__":
    main()
