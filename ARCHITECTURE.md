# Architecture du projet — Évangile du jour × Livre du Ciel

Ce document décrit **toutes les briques** du projet (langages, frameworks, services)
et **comment elles sont reliées**, du code jusqu'au smartphone de l'utilisateur.

> 💡 Sur GitHub, le schéma ci-dessous s'affiche directement en image.
> Tu peux aussi ouvrir `architecture.html` dans un navigateur pour le voir en grand.

## Vue d'ensemble

```mermaid
flowchart TB
    subgraph DEV["🧑‍💻 Ton poste (développement)"]
        CODE["Code source<br/>Python + HTML/CSS/JS"]
        GIT["Git"]
    end

    subgraph GH["☁️ GitHub"]
        REPO["Dépôt EricBag/evangile-ldc<br/>branche main"]
    end

    subgraph RW["🚂 Railway — hébergement (HTTPS)"]
        BUILD["Builder Railpack + mise<br/>installe Python 3.11"]
        UVICORN["uvicorn<br/>serveur web ASGI"]
        subgraph APP["Application FastAPI · main.py"]
            FASTAPI["FastAPI<br/>routes + API + sécurité"]
            JINJA["Jinja2<br/>pages HTML (index, login)"]
            STATIC["Fichiers statiques<br/>CSS · JS · icônes · manifest · sw.js"]
            RAG["ldc_proZ.py<br/>moteur de recherche (RAG)"]
            AELFC["aelf_client.py"]
            CACHE[("cache_evangiles/<br/>résultats en JSON")]
            PDF["ldc.pdf<br/>Livre du Ciel"]
        end
    end

    subgraph LIBS["📚 Bibliothèques Python (utilisées par le RAG)"]
        PYMUPDF["PyMuPDF<br/>lit le PDF"]
        BM25["rank-bm25<br/>recherche par mots-clés"]
        NUMPY["numpy<br/>calcul sur embeddings"]
    end

    subgraph EXT["🌐 Services externes"]
        OPENAI["API OpenAI<br/>GPT-4.1 + embeddings"]
        AELF["API AELF<br/>lectures liturgiques du jour"]
    end

    subgraph USER["📱 Utilisateur"]
        PHONE["Smartphone / navigateur<br/>PWA installable"]
    end

    CODE --> GIT -->|git push| REPO
    REPO -->|auto-deploy| BUILD --> UVICORN --> FASTAPI

    FASTAPI --> JINJA
    FASTAPI --> STATIC
    FASTAPI --> AELFC
    FASTAPI --> RAG
    RAG --> CACHE
    RAG --> PDF
    RAG --> PYMUPDF
    RAG --> BM25
    RAG --> NUMPY
    RAG -->|requêtes IA payantes| OPENAI
    AELFC -->|HTTP| AELF

    PHONE <-->|HTTPS| FASTAPI
    STATIC -.->|sw.js + manifest<br/>= installation| PHONE
```

## Comment lire ce schéma

### 1. Le parcours du code (déploiement)
1. Tu écris le **code** sur ton poste et tu l'envoies avec **Git** (`git push`).
2. **GitHub** reçoit le code (dépôt `EricBag/evangile-ldc`, branche `main`).
3. **Railway** détecte le push et lance un **déploiement automatique** :
   - le *builder* (Railpack + `mise`) installe **Python 3.11** ;
   - puis démarre **uvicorn**, le serveur web qui fait tourner l'application.

### 2. Le parcours d'une visite (utilisation)
1. Depuis son **smartphone**, l'utilisateur ouvre le site en **HTTPS** (PWA installable
   grâce à `sw.js` + `manifest.json`).
2. **FastAPI** (`main.py`) reçoit la demande et :
   - sert les **pages HTML** via **Jinja2** et les **fichiers statiques** (CSS, JS, icônes) ;
   - récupère l'**évangile du jour** via `aelf_client.py` → **API AELF** ;
   - lance l'analyse via `ldc_proZ.py` (le **moteur RAG**), qui :
     - lit le **`ldc.pdf`** avec **PyMuPDF**,
     - recherche les passages avec **rank-bm25** (mots-clés) et **numpy** (embeddings),
     - interroge l'**API OpenAI** (GPT-4.1) pour affiner et expliquer ;
   - met les résultats en **cache** (`cache_evangiles/`) pour ne pas repayer OpenAI
     deux fois le même évangile.

## Qui fait quoi — récapitulatif

| Brique | Type | Rôle dans le projet |
|---|---|---|
| **Python 3.11** | Langage | Tout le backend |
| **FastAPI** | Framework web | Cœur de l'app : routes, API, sécurité (mots de passe, rate limiting) |
| **uvicorn** | Serveur ASGI | Fait tourner FastAPI en production |
| **Jinja2** | Moteur de templates | Génère les pages HTML (`index.html`, `login.html`) |
| **HTML/CSS/JS + PWA** | Frontend | Interface ; `manifest.json` + `sw.js` = installation sur smartphone |
| **ldc_proZ.py** | Module maison | Moteur de recherche (RAG) dans le Livre du Ciel |
| **PyMuPDF** | Bibliothèque | Extrait le texte du `ldc.pdf` |
| **rank-bm25** | Bibliothèque | Recherche par mots-clés |
| **numpy** | Bibliothèque | Calculs sur les embeddings |
| **aelf_client.py** | Module maison | Récupère les lectures du jour |
| **API OpenAI** | Service externe | GPT-4.1 + embeddings (analyse, **payant**) |
| **API AELF** | Service externe | Évangile et lectures liturgiques du jour |
| **cache_evangiles/** | Données | Évangiles déjà analysés (évite de repayer OpenAI) |
| **Git** | Outil | Versionne et envoie le code |
| **GitHub** | Service | Héberge le code source |
| **Railway** | Service | Héberge et exécute l'application en ligne (auto-deploy) |
