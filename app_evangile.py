"""
app_evangile.py — Évangile du jour × Livre du Ciel
====================================================
Version : logo agrandi, sans "Thèmes détectés", sans synthèse théologique.
Sortie = 2 extraits du Livre du Ciel + leurs éclairages.
"""

from pathlib import Path
import os
import base64
import streamlit as st

# ============================================================
# 1) CLÉ OPENAI — AVANT TOUT IMPORT DE ldc_proZ
# ============================================================

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# ============================================================
# 2) Imports lourds
# ============================================================

from ldc_proZ import (
    build_or_load_index,
    detect_dynamic_motifs_gpt,
    score_segments_with_keywords,
    group_segments_by_dictee,
    rerank_with_gpt,
    make_excerpt,
    explain_passage_matches,
)


# ------------------------------------------------------------
# Configuration page
# ------------------------------------------------------------

st.set_page_config(
    page_title="Évangile du jour — Livre du Ciel",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "ldc.pdf"
DEFAULT_CACHE = BASE_DIR / "ldc_index_word"
LOGO_PATH = BASE_DIR / "logo.jpg"


# ============================================================
# 3) LOGO : image si présente, sinon SVG de secours
# ============================================================

_LOGO_SVG_FALLBACK = """
<div style="text-align:center; margin:0;">
  <svg width="60" height="60" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;">
    <path d="M6 10 C 6 9, 7 8, 8 8 L 19 10 L 19 32 L 8 30 C 7 30, 6 29, 6 28 Z"
          fill="#f9fafb" stroke="#111827" stroke-width="1.2" stroke-linejoin="round"/>
    <path d="M34 10 C 34 9, 33 8, 32 8 L 21 10 L 21 32 L 32 30 C 33 30, 34 29, 34 28 Z"
          fill="#f9fafb" stroke="#111827" stroke-width="1.2" stroke-linejoin="round"/>
    <line x1="20" y1="10" x2="20" y2="32" stroke="#111827" stroke-width="1.2"/>
  </svg>
</div>
"""


def _build_logo_html(width: int = 220, radius: int = 16) -> str:
    if not LOGO_PATH.exists():
        return _LOGO_SVG_FALLBACK

    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return _LOGO_SVG_FALLBACK

    ext = LOGO_PATH.suffix.lower()
    mime = "imag
