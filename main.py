"""
main.py — Évangile du jour × Livre du Ciel
==========================================
Backend FastAPI servant l'application redesignée en HTML libre.
La logique métier (AELF, ldc_proZ, cache) est strictement identique à
app_evangile.py, seule la couche présentation change.
"""

from pathlib import Path
import os
import json
import secrets
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException, Cookie, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Charger .env (mais ne pas écraser les vars système existantes)
load_dotenv(override=False)

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "ldc.pdf"
DEFAULT_CACHE = BASE_DIR / "ldc_index_word"
CACHE_EVANGILES_DIR = BASE_DIR / "cache_evangiles"

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
SESSION_SECRET = secrets.token_hex(32)  # régénéré à chaque démarrage

# ============================================================
# Imports lourds (après chargement .env pour OPENAI_API_KEY)
# ============================================================

import aelf_client
from ldc_proZ import (
    build_or_load_index,
    detect_dynamic_motifs_gpt,
    score_segments_with_keywords,
    group_segments_by_dictee,
    rerank_with_gpt,
    make_excerpt,
    explain_passage_matches,
    evangile_hash,
)

# ============================================================
# Cache des évangiles déjà analysés
# ============================================================

def _cache_path(text: str) -> Path:
    return CACHE_EVANGILES_DIR / f"{evangile_hash(text)}.json"

def _load_cached_passages(text: str):
    path = _cache_path(text)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        passages = data.get("passages")
        if isinstance(passages, list) and passages:
            return passages
        return None
    except Exception:
        return None

def _save_cached_passages(text: str, passages: list) -> None:
    if not passages:
        return
    try:
        CACHE_EVANGILES_DIR.mkdir(exist_ok=True)
        path = _cache_path(text)
        data = {
            "cached_at": datetime.now().isoformat(timespec="seconds"),
            "evangile_preview": text.strip()[:200],
            "passages": passages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _clear_cache_evangiles() -> int:
    if not CACHE_EVANGILES_DIR.exists():
        return 0
    count = 0
    for f in CACHE_EVANGILES_DIR.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count

# ============================================================
# Logique métier (identique à app_evangile.py)
# ============================================================

def build_evangile_state(date_iso: str, include_premiere_lecture: bool) -> dict:
    """Construit le texte pré-rempli et le contexte liturgique pour une date."""
    try:
        messe = aelf_client.fetch_messe(date_iso)
    except aelf_client.AelfError as exc:
        return {"context": "", "text": "", "error": str(exc)}

    context = aelf_client.extract_contexte_liturgique(messe)
    parts: list[str] = []

    try:
        ref, titre, texte = aelf_client.extract_evangile(messe)
        header = f"ÉVANGILE ({ref})"
        if titre:
            header += f"\n{titre}"
        parts.append(f"{header}\n\n{texte}")
    except aelf_client.AelfError:
        pass

    if include_premiere_lecture:
        try:
            ref, titre, texte = aelf_client.extract_premiere_lecture(messe)
            header = f"─── 1ÈRE LECTURE ({ref}) ───"
            if titre:
                header += f"\n{titre}"
            parts.append(f"{header}\n\n{texte}")
        except aelf_client.AelfError:
            pass

    return {"context": context, "text": "\n\n".join(parts), "error": None}


def analyser_evangile(evangelium_text: str, dictees, segments, bm25, embs, cache_dir: Path) -> list:
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

# ============================================================
# Chargement de l'index (une fois au démarrage)
# ============================================================

print("Chargement de l'index du Livre du Ciel...")
if not DEFAULT_PDF.exists():
    raise RuntimeError(f"PDF introuvable : {DEFAULT_PDF}")

DICTEES, SEGMENTS, BM25, EMBS = build_or_load_index(
    str(DEFAULT_PDF),
    cache_dir=str(DEFAULT_CACHE),
    embed_model_name="text-embedding-3-large",
)
print("Index chargé.")

# ============================================================
# Application FastAPI
# ============================================================

app = FastAPI(title="Évangile du jour")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------------------------------------------------
# Auth helpers (cookie de session simple)
# ------------------------------------------------------------

def is_authenticated(session: Optional[str]) -> bool:
    if not APP_PASSWORD:
        return True  # pas de mot de passe configuré → accès libre
    return session == SESSION_SECRET

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, session: Optional[str] = Cookie(default=None)):
    """Page d'accueil. Redirige vers login si non authentifié."""
    if not is_authenticated(session):
        return templates.TemplateResponse(request, "login.html", {"request": request, "error": None})

    # Pré-charger l'évangile du jour
    today = date.today().isoformat()
    state = build_evangile_state(today, include_premiere_lecture=False)

    JOURS_FR = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
    MOIS_FR = ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"]
    d = date.fromisoformat(today)
    today_human = f"{JOURS_FR[d.weekday()]} {d.day} {MOIS_FR[d.month-1]} {d.year}"

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "today": today,
        "today_human": today_human,
        "context": state["context"],
        "evangile_text": state["text"],
        "error": state["error"],
    })

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, password: str = Form(...)):
    """Vérifie le mot de passe et pose un cookie de session."""
    if password == APP_PASSWORD:
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="session", value=SESSION_SECRET, httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse(
        request,
        "login.html",
        {"request": request, "error": "Mot de passe incorrect"},
        status_code=401,
    )

@app.post("/api/evangile")
async def api_evangile(
    date_iso: str = Form(...),
    include_premiere_lecture: bool = Form(False),
    session: Optional[str] = Cookie(default=None),
):
    """Récupère l'évangile AELF pour une date (utilisé pour rafraîchir sans reload)."""
    if not is_authenticated(session):
        raise HTTPException(status_code=401, detail="Non authentifié")

    state = build_evangile_state(date_iso, include_premiere_lecture)
    return JSONResponse(state)

@app.post("/api/eclairage")
async def api_eclairage(
    evangile_text: str = Form(...),
    session: Optional[str] = Cookie(default=None),
):
    """Analyse l'évangile et renvoie les 2 extraits + éclairages."""
    if not is_authenticated(session):
        raise HTTPException(status_code=401, detail="Non authentifié")

    if not evangile_text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")

    # Cache check
    cached = _load_cached_passages(evangile_text)
    if cached is not None:
        return JSONResponse({"passages": cached, "from_cache": True})

    try:
        passages = analyser_evangile(
            evangile_text, DICTEES, SEGMENTS, BM25, EMBS, DEFAULT_CACHE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse : {e}")

    _save_cached_passages(evangile_text, passages)
    return JSONResponse({"passages": passages, "from_cache": False})

@app.post("/api/admin/clear-cache")
async def api_admin_clear_cache(
    admin_password: str = Form(...),
    session: Optional[str] = Cookie(default=None),
):
    """Vide le cache des évangiles analysés."""
    if not is_authenticated(session):
        raise HTTPException(status_code=401, detail="Non authentifié")
    if not ADMIN_PASSWORD or admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Mot de passe admin incorrect")

    n = _clear_cache_evangiles()
    return JSONResponse({"deleted": n})

@app.get("/api/admin/cache-count")
async def api_admin_cache_count(session: Optional[str] = Cookie(default=None)):
    """Renvoie le nombre d'évangiles en cache (lecture seule, accès libre tant que session OK)."""
    if not is_authenticated(session):
        raise HTTPException(status_code=401, detail="Non authentifié")
    if not CACHE_EVANGILES_DIR.exists():
        return JSONResponse({"count": 0})
    count = sum(1 for _ in CACHE_EVANGILES_DIR.glob("*.json"))
    return JSONResponse({"count": count})

# ============================================================
# Lancement
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
