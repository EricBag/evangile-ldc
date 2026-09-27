# Missale Meum — moteur de calendrier (copie embarquée)

- **Projet** : https://github.com/mmolenda/missalemeum
- **Commit** : `7be440e994f4fa8a3aa5e19f5e9da0ecb8dc0145` (2026-09-21)
- **Licence** : MIT, « Copyright (c) 2022 Marcin Molenda » — voir [`LICENSE`](LICENSE).
- **Rôle** : déterminer l'office célébré chaque jour selon les rubriques de
  1960 (temporal, sanctoral, préséance, vigiles, Quatre-Temps).
- **Dépendance** : `python-dateutil` (calcul de Pâques). Testé sous
  Python 3.11.10 (version de Railway), bien que le projet d'origine déclare
  Python ≥ 3.13 : la partie calendrier n'utilise rien de plus récent.

## Fichiers copiés

`backend/api/kalendar/{factory,models,rules}.py`,
`backend/api/constants/{__init__,common}.py`,
`backend/api/constants/la/{blocks,translation}.py`, et un extrait de
`backend/api/utils.py` (`match_all`, `match_first`, `get_custom_preface`).

## Correctif appliqué (automatique)

Fait par `scripts/maj_missalemeum.py`, qui recopie tout depuis le commit
épinglé :

1. imports `api.*` réécrits en `vendor.missalemeum.*` ;
2. imports du parseur de propres (`api.propers.*`) neutralisés : les textes
   sont lus par `tradi.py`. Les méthodes `get_proper()` / `has_proper()` des
   modèles, qui en dépendent, ne sont jamais appelées ;
3. `utils.py` réduit aux trois fonctions utilisées par le calendrier
   (l'original dépend de PyYAML et des suppléments).

Ne pas modifier ces fichiers à la main : changer `COMMIT_EPINGLE` dans le
script et le relancer, puis `python scripts/rapport_tradi.py` et les tests.
