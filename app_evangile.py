"""
app_evangile.py — Mini appli Streamlit (version Cloud)
======================================================

Colle un évangile du jour, obtiens 2 extraits du Livre du Ciel
+ une synthèse théologique à la lumière de la Divine Volonté.

Version protégée par mot de passe pour déploiement sur
Streamlit Community Cloud (lien privé).

Lancement local :
    streamlit run app_evangile.py

Déploiement :
    voir GUIDE_DEPLOIEMENT.md
"""

from pathlib import Path
import os
import streamlit as st

# On réutilise tout le pipeline existant
from ldc_proZ import (
    build_or_load_index,
    detect_dynamic_motifs_gpt,
    score_segments_with_keywords,
    group_segments_by_dictee,
    rerank_with_gpt,
    make_excerpt,
    explain_passage_matches,
    client,
)


# ------------------------------------------------------------
# Configuration page
# ------------------------------------------------------------

st.set_page_config(
    page_title="Évangile du jour — Livre du Ciel",
    page_icon="📖",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "ldc.pdf"
DEFAULT_CACHE = BASE_DIR / "ldc_index_word"


# ------------------------------------------------------------
# Récupération de la clé OpenAI depuis les secrets Streamlit
#
# Sur Streamlit Cloud, on configure OPENAI_API_KEY dans Settings → Secrets.
# En local, la variable d'environnement OPENAI_API_KEY suffit.
# ------------------------------------------------------------

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ------------------------------------------------------------
# Mot de passe simple (lien privé)
# ------------------------------------------------------------

def check_password() -> bool:
    """Retourne True si l'utilisateur a entré le bon mot de passe."""

    def password_entered():
        if st.session_state.get("password") == st.secrets.get("app_password"):
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Si pas de mot de passe configuré dans les secrets → pas de protection
    if not st.secrets.get("app_password"):
        return True

    if st.session_state.get("password_correct"):
        return True

    st.title("🔒 Accès protégé")
    st.text_input(
        "Mot de passe",
        type="password",
        on_change=password_entered,
        key="password",
    )
    if st.session_state.get("password_correct") is False:
        st.error("😕 Mot de passe incorrect")
    return False


if not check_password():
    st.stop()


# ------------------------------------------------------------
# Helpers d'export
# ------------------------------------------------------------

def _build_export(evangile_text: str, result: dict) -> str:
    lines = []
    lines.append("ÉVANGILE")
    lines.append("=" * 60)
    lines.append(evangile_text.strip())
    lines.append("")
    lines.append("EXTRAITS DU LIVRE DU CIEL")
    lines.append("=" * 60)
    for i, p in enumerate(result["passages"], 1):
        lines.append(f"\nExtrait {i} — Tome {p['tome']} — {p['date']}")
        lines.append("-" * 60)
        lines.append(p["extrait"])
        if p.get("explication"):
            lines.append("")
            lines.append("Éclairage :")
            lines.append(p["explication"])
    if result.get("synthese"):
        lines.append("")
        lines.append("SYNTHÈSE THÉOLOGIQUE")
        lines.append("=" * 60)
        lines.append(result["synthese"])
    return "\n".join(lines)


# ------------------------------------------------------------
# Synthèse théologique adaptée (2 extraits)
# ------------------------------------------------------------

def synthese_theologique(evangelium_text: str,
                         passages: list,
                         model_name: str = "gpt-4.1") -> str:
    """Synthèse théologique unique (8-12 lignes) calibrée pour 2 extraits."""
    if not passages:
        return ""

    joined = "\n\n---\n\n".join(passages)

    prompt = f"""
Tu es un théologien catholique, expert en mystique et spécialiste de la
Divine Volonté (Fiat) telle qu'enseignée par Luisa Piccarreta dans le
« Livre du Ciel ».

On te donne :
1) Un passage d'Évangile.
2) Deux extraits du « Livre du Ciel » sélectionnés pour l'éclairer.

Tâche :
Rédige une synthèse théologique UNIQUE, claire et substantielle
(8 à 12 lignes), qui montre comment ces deux extraits éclairent en
profondeur la péricope évangélique.

Structure attendue (sans titres apparents) :

1) Identifie d'abord le mouvement principal de la péricope
   (ce qui est révélé, demandé ou mis en lumière).

2) Montre ensuite, en t'appuyant explicitement sur les deux extraits,
   comment la doctrine de la Divine Volonté approfondit ce mouvement :
   Fiat, vie intérieure, actes accomplis dans la Volonté divine,
   réparation, union. Articule clairement la COMPLÉMENTARITÉ
   des deux extraits.

3) Conclus par une ouverture spirituelle sobre :
   ce que la péricope, ainsi éclairée, appelle à vivre intérieurement
   aujourd'hui dans la Divine Volonté.

Contraintes de style :
- Un seul paragraphe continu.
- Pas de liste, pas de titres, pas de numérotation.
- Style théologique sobre, précis, contemplatif.
- Lien explicite et concret avec le contenu des deux extraits.

Passage d'Évangile :
\"\"\"{evangelium_text}\"\"\"

Extraits du Livre du Ciel :
\"\"\"{joined}\"\"\"
"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "Tu es un théologien catholique expert en mystique et en Divine Volonté."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Synthèse indisponible : {e}"


# ------------------------------------------------------------
# Pipeline complet : évangile → 2 extraits + synthèse
# ------------------------------------------------------------

def analyser_evangile(evangelium_text: str,
                      dictees, segments, bm25, embs,
                      cache_dir: Path) -> dict:
    """Reprend la logique de process_evangelium() en forçant top=2."""

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
        return {"passages": [], "synthese": "", "motifs": motif_names}

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

    synthese = synthese_theologique(evangelium_text, textes)

    return {
        "passages": passages,
        "synthese": synthese,
        "motifs": motif_names,
    }


# ------------------------------------------------------------
# Cache : l'index est chargé une seule fois pour toute la session
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement de l'index du Livre du Ciel (peut prendre quelques minutes au premier lancement)…")
def load_index(pdf_path: str, cache_dir: str):
    return build_or_load_index(
        pdf_path,
        cache_dir=cache_dir,
        embed_model_name="text-embedding-3-large",
    )


# ------------------------------------------------------------
# Interface utilisateur
# ------------------------------------------------------------

st.title("📖 Évangile du jour — Livre du Ciel")
st.caption(
    "Collez le texte d'un évangile : 2 extraits du Livre du Ciel "
    "(Luisa Piccarreta) + une synthèse théologique à la lumière "
    "de la Divine Volonté."
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
    placeholder=(
        "Collez ici le texte de l'évangile du jour.\n\n"
        "Exemple :\n"
        "En ce temps-là, Jésus disait à ses disciples : "
        "« À quoi vais-je comparer cette génération ? »…"
    ),
)

col1, _ = st.columns([1, 5])
with col1:
    launch = st.button("🔍 Analyser", type="primary", use_container_width=True)

if launch:
    if not evangile_text.strip():
        st.warning("Veuillez d'abord saisir le texte de l'évangile.")
        st.stop()

    with st.spinner("Analyse en cours (motifs → scoring → reranking → synthèse)…"):
        try:
            result = analyser_evangile(
                evangile_text, dictees, segments, bm25, embs, cache_p,
            )
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.stop()

    if result.get("motifs"):
        st.markdown("**Thèmes détectés :** " + ", ".join(result["motifs"]))

    st.divider()
    st.subheader("📜 Extraits du Livre du Ciel")

    if not result["passages"]:
        st.info("Aucun extrait pertinent trouvé pour ce texte.")
    else:
        for i, p in enumerate(result["passages"], start=1):
            with st.container(border=True):
                st.markdown(f"**Extrait {i} — Tome {p['tome']} — {p['date']}**")
                st.markdown(f"> {p['extrait']}")
                if p.get("explication"):
                    st.markdown("**🔎 Éclairage**")
                    st.markdown(f"*{p['explication']}*")

    if result.get("synthese"):
        st.divider()
        st.subheader("✨ Synthèse théologique")
        st.markdown(result["synthese"])

        export_txt = _build_export(evangile_text, result)
        st.download_button(
            "💾 Télécharger en .txt",
            data=export_txt.encode("utf-8"),
            file_name="evangile_ldc.txt",
            mime="text/plain",
        )
