# Textes de la messe traditionnelle (missel romain de 1962)

## Provenance

- **Projet** : Divinum Officium — https://github.com/DivinumOfficium/divinum-officium
  (site : https://www.divinumofficium.com)
- **Commit embarqué : `5cf0e7f0a3f2c8e1567e0bb0fef2e662250125a0` (2026-09-27)**
- **Licence** : MIT, « Copyright (c) 2026 Divinum Officium ». Le texte complet
  est dans [`LICENSE`](LICENSE), à conserver avec les fichiers.
  Licence vérifiée via l'API GitHub (`license.spdx_id = MIT`).

Seuls les fichiers texte utiles à l'Évangile du jour sont embarqués, dans
l'arborescence d'origine (pas de Git LFS) :

| Dossier local | Origine dans le dépôt |
|---|---|
| `missa/Latin/{Tempora,Sancti,Commune}` | `web/www/missa/Latin/…` |
| `missa/Francais/{Tempora,Sancti,Commune}` | `web/www/missa/Francais/…` |
| `horas/Latin/Commune`, `horas/Francais/Commune` | `web/www/horas/…/Commune` |

Le Commun des messes (renvois `@Commune/Cx`, règles `vide Cx`) n'est plus dans
`missa/*/Commune`, qui ne contient que deux fichiers : Divinum Officium le
résout dans `horas/<langue>/Commune`, dont les fichiers portent aussi les
sections de la messe (`[Evangelium]`…).

Mise à jour : `python scripts/maj_tradi.py` (dernière version de `master`) ou
`python scripts/maj_tradi.py <sha>` ; le script réécrit la ligne « Commit
embarqué » ci-dessus. Relancer ensuite `python scripts/rapport_tradi.py` et
les tests (`python -m unittest discover -s tests`).

## Traduction française

**Traduction non identifiée.**

Ce qui a pu être établi à partir de l'historique du dépôt (sans permettre
d'identifier la traduction utilisée pour les Évangiles) :

- Le temporal français (`missa/Francais/Tempora`, anciennement
  `missa/French/Tempora`) a été ajouté en 2018-2019 par fr. Romain Marie
  (compte BrRoman ; commits `0684c054` « French/Tempora: files added. » et
  `880e4d38` « French: Pent19* », PR #1020 et #1134). Ni les commits ni les
  PR n'indiquent de source.
- La PR #536 (fusionnée le 16/06/2016) remplaçait la traduction française de
  l'**Ordinaire** et du **Commun** de la messe (tirée d'introibo.fr) par celle
  du missel de l'abbaye du Barroux. Elle ne concerne ni le temporal ni le
  sanctoral, et les fichiers qu'elle modifiait (`missa/French/Commune`) ne
  sont plus ceux qu'utilise Divinum Officium aujourd'hui (le Commun est
  désormais lu dans `horas/Francais/Commune`, créé en 2025).
- `web/www/missa/source.txt` ne documente que les sources latines et
  anglaises.

La mention affichée dans l'application est donc : « Missel romain de 1962 —
texte : Divinum Officium (divinumofficium.com), traduction non identifiée ».

## Corrections locales

`corrections.json` corrige des coquilles de référence de Divinum Officium
(fichier, langue, référence erronée, référence corrigée, justification) ;
chaque correction appliquée est listée dans le rapport annuel. Les
signalements correspondants à poster en amont sont dans `SIGNALEMENTS.md`.
Après une mise à jour des textes, supprimer les entrées devenues inutiles
(le rapport ne les listera plus comme appliquées).

## Calendrier

Le choix de la messe du jour (rubriques de 1960) est fait par le moteur de
calendrier de Missale Meum (MIT), copié dans `vendor/missalemeum/` — voir
`vendor/missalemeum/VENDORED.md`.
