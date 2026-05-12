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
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    return f"""
    <div style="text-align:center; margin: 0 0 1.4rem 0;">
      <img src="data:{mime};base64,{b64}"
           alt="Logo"
           style="width:{width}px; height:auto;
                  border-radius:{radius}px;
                  box-shadow: 0 6px 22px rgba(17, 24, 39, 0.12);
                  display:inline-block;" />
    </div>
    """


LOGO_HTML = _build_logo_html(width=220, radius=16)


# ============================================================
# 4) CSS personnalisé
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

/* Forcer le th\u00e8me clair (pour les navigateurs en mode sombre) */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
.main, section.main {
    background-color: #faf9f4 !important;
    color: #111827 !important;
}
.stApp * {
    color: inherit;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {height: 0; visibility: hidden;}

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 760px !important;
}

html, body, [class*="st-"], button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
}

h1 {
    font-size: 2.6rem !important;
    line-height: 1.15 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.3rem !important;
    text-align: center !important;
}

h2 {
    font-size: 1.7rem !important;
    margin-top: 2rem !important;
    margin-bottom: 0.8rem !important;
}

.stTextArea textarea {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
    background: #ffffff !important;
    color: #111827 !important;
    caret-color: #111827 !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
}
.stTextArea textarea:focus {
    border-color: #111827 !important;
    box-shadow: 0 0 0 3px rgba(17, 24, 39, 0.08) !important;
}

input[type="password"], input[type="text"] {
    color: #111827 !important;
    caret-color: #111827 !important;
}

.stButton > button {
    background-color: #111827 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #111827 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.12) !important;
}

.stDownloadButton > button {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background-color: #f9fafb !important;
    border-color: #9ca3af !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    padding: 1.5rem 1.75rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

blockquote {
    border-left: 3px solid #9ca3af !important;
    padding: 0.6rem 0 0.6rem 1.25rem !important;
    margin: 1rem 0 !important;
    color: #111827 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 1.05rem !important;
    font-style: normal !important;
    font-weight: 400 !important;
    line-height: 1.75 !important;
    background: #fafafa !important;
    border-radius: 0 8px 8px 0 !important;
    padding-right: 1rem !important;
}

hr {
    border: none !important;
    border-top: 1px solid #e5e7eb !important;
    margin: 2rem 0 !important;
}

.stAlert {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def check_password() -> bool:
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("app_password"):
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    try:
        configured = st.secrets.get("app_password")
    except Exception:
        configured = None

    if not configured:
        return True

    if st.session_state.get("password_correct"):
        return True

    st.markdown(LOGO_HTML, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>Accès protégé</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#6b7280; margin-bottom:2rem;'>"
        "Veuillez saisir le mot de passe pour accéder à l'application.</p>",
        unsafe_allow_html=True,
    )
    st.text_input(
        "Mot de passe",
        type="password",
        on_change=password_entered,
        key="password",
        label_visibility="collapsed",
        placeholder="Mot de passe",
    )
    if st.session_state.get("password_correct") is False:
        st.error("Mot de passe incorrect")
    return False


if not check_password():
    st.stop()


def _build_export(evangile_text: str, passages: list) -> str:
    lines = []
    lines.append("ÉVANGILE")
    lines.append("=" * 60)
    lines.append(evangile_text.strip())
    lines.append("")
    lines.append("EXTRAITS DU LIVRE DU CIEL")
    lines.append("=" * 60)
    for i, p in enumerate(passages, 1):
        lines.append(f"\nExtrait {i} — Tome {p['tome']} — {p['date']}")
        lines.append("-" * 60)
        lines.append(p["extrait"])
        if p.get("explication"):
            lines.append("")
            lines.append("Éclairage :")
            lines.append(p["explication"])
    return "\n".join(lines)


def analyser_evangile(evangelium_text: str,
                      dictees, segments, bm25, embs,
                      cache_dir: Path) -> list:
    motif_names, motif_keywords = detect_dynamic_motifs_gpt(
        evangelium_text, cache_dir=str(cache_dir),
    )
    ranked_segments = score_segments_with_keywords(
        evangelium_text, motif_keywords, segments, bm25, embs,
        top_k_segments=200,
    )
    candidates = group_segments_by_dictee(
        ranked_segments, dictees, top_k_dicts_pre_rerank=50,
    )
    if not candidates:
        return []

    final_indices = rerank_with_gpt(
        evangelium_text, candidates, motif_names, motif_keywords,
        top_k_final=2,
    )

    passages = []
    for idx in final_indices:
        _score, d, seg = candidates[idx]
        passages.append({
            "tome": d.tome,
            "date": d.date,
            "extrait": make_excerpt(d, seg),
        })

    textes = [p["extrait"] for p in passages]
    explications = explain_passage_matches(evangelium_text, textes)
    for p, expl in zip(passages, explications):
        p["explication"] = expl

    return passages


@st.cache_resource(show_spinner="Chargement de l'index du Livre du Ciel…")
def load_index(pdf_path: str, cache_dir: str):
    return build_or_load_index(
        pdf_path,
        cache_dir=cache_dir,
        embed_model_name="text-embedding-3-large",
    )


st.markdown(LOGO_HTML, unsafe_allow_html=True)
st.markdown("<h1>Évangile du jour</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:1rem; "
    "margin-top:-0.3rem; margin-bottom:2.5rem;'>"
    "À la lumière du <em>Livre du Ciel</em> de Luisa Piccarreta"
    "</p>",
    unsafe_allow_html=True,
)

pdf_p = DEFAULT_PDF
cache_p = DEFAULT_CACHE

if not pdf_p.exists():
    st.error(f"PDF introuvable : {pdf_p}")
    st.stop()

try:
    dictees, segments, bm25, embs = load_index(str(pdf_p), str(cache_p))
except Exception as e:
    st.error(f"Erreur de chargement de l'index : {e}")
    st.stop()

evangile_text = st.text_area(
    "Texte de l'évangile",
    height=240,
    placeholder="Collez ici le texte de l'évangile du jour…",
    label_visibility="collapsed",
)

_, c, _ = st.columns([1, 1, 1])
with c:
    launch = st.button("Analyser", type="primary", use_container_width=True)

if launch:
    if not evangile_text.strip():
        st.warning("Veuillez d'abord saisir le texte de l'évangile.")
        st.stop()

    with st.spinner("Recherche dans le Livre du Ciel…"):
        try:
            passages = analyser_evangile(
                evangile_text, dictees, segments, bm25, embs, cache_p,
            )
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.stop()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<h2>Extraits du Livre du Ciel</h2>", unsafe_allow_html=True)

    if not passages:
        st.info("Aucun extrait pertinent trouvé pour ce texte.")
    else:
        for i, p in enumerate(passages, start=1):
            with st.container(border=True):
                st.markdown(
                    f"<p style='color:#6b7280; font-size:0.8rem; "
                    f"text-transform:uppercase; letter-spacing:0.08em; "
                    f"margin-bottom:0.3rem;'>"
                    f"Extrait {i} · Tome {p['tome']} · {p['date']}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> {p['extrait']}")
                if p.get("explication"):
                    st.markdown(
                        f"<p style='color:#6b7280; font-size:0.8rem; "
                        f"text-transform:uppercase; letter-spacing:0.08em; "
                        f"margin-top:1rem; margin-bottom:0.3rem;'>Éclairage</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color:#111827; line-height:1.7;'>{p['explication']}</p>",
                        unsafe_allow_html=True,
                    )

        export_txt = _build_export(evangile_text, passages)
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        _, cdl, _ = st.columns([1, 1, 1])
        with cdl:
            st.download_button(
                "Télécharger",
                data=export_txt.encode("utf-8"),
                file_name="evangile_ldc.txt",
                mime="text/plain",
                use_container_width=True,
            )
