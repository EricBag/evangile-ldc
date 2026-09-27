# Signalements à poster sur le dépôt Divinum Officium

Dépôt : https://github.com/DivinumOfficium/divinum-officium/issues/new
Constatés sur le commit `5cf0e7f0a3f2c8e1567e0bb0fef2e662250125a0`, et
visibles sur le site le 27/09/2026 (contrôle `scripts/controle_divinum.py`).
En attendant une correction en amont, les deux coquilles sont corrigées
localement par `corrections.json`.

Textes en anglais, langue des tickets du dépôt. Titre et corps prêts à coller.

---

## 1. Référence française erronée : jeudi de la IIe semaine de Carême (25/02/2027)

**Titre**

```
French missa: wrong Gospel reference in Tempora/Quad2-4 (Luc 6:19-31 instead of Luc 16:19-31)
```

**Corps**

```
File: web/www/missa/Francais/Tempora/Quad2-4.txt, line 28 ([Evangelium] section)

Current:  !Luc 6:19-31
Expected: !Luc 16:19-31

The French text that follows is the parable of the rich man and Lazarus
("Il y avait un homme riche, qui était vêtu de pourpre et de lin…"),
i.e. Luke 16:19-31, which is also the reference given by the Latin file
(web/www/missa/Latin/Tempora/Quad2-4.txt: "!Luc 16:19-31").
The chapter number lost its leading "1".

Seen on commit 5cf0e7f0a3f2c8e1567e0bb0fef2e662250125a0 (Feria V of the
2nd week of Lent, e.g. 2027-02-25, Rubrics 1960, French).
```

---

## 2. Référence latine erronée : vigile des saints Pierre et Paul (28/06/2027)

**Titre**

```
Latin missa: wrong Gospel reference in Sancti/06-28r (Joannes 21:15-10 instead of 21:15-19)
```

**Corps**

```
File: web/www/missa/Latin/Sancti/06-28r.txt, line 38 ([Evangelium] section)

Current:  !Joannes 21:15-10
Expected: !Joannes 21:15-19

The verse range is descending (15-10). The Latin text ends with
"Hoc autem dixit, signíficans, qua morte clarificatúrus esset Deum",
which is John 21:19; the French file (web/www/missa/Francais/Sancti/06-28r.txt,
line 28) already reads "!Joannes 21:15-19". The website shows
"Joann 21:15-10" for the Vigil of Ss. Peter and Paul (e.g. 2027-06-28,
Rubrics 1960).

Seen on commit 5cf0e7f0a3f2c8e1567e0bb0fef2e662250125a0.
```

---

## 3. Commun inexistant « C4D » dans trois fichiers français du sanctoral

Constaté en traitant saint Martin pape (12/11/2027) : la règle française
renvoie à un Commun qui n'existe nulle part dans le dépôt. `tradi.py` suit
alors la règle latine (voir le rapport, « Autres incidents »).

**Titre**

```
French missa: "vide C4D" points to a non-existent Commune (Sancti/07-13, 11-12, 11-23t)
```

**Corps**

```
Three French Sancti files refer to a Commune "C4D" that does not exist
anywhere in the repository (no Commune/C4D.txt in missa/ or horas/, any
language):

- web/www/missa/Francais/Sancti/07-13.txt, line 6: "vide C4D;"
  (Latin file: [Rank] "vide C2b-1", [Rule] "vide C4b-1")
- web/www/missa/Francais/Sancti/11-12.txt, line 3: "vide C4D"
  (Latin file: "vide C2b-1", St Martin I, Pope and Martyr)
- web/www/missa/Francais/Sancti/11-23t.txt, line 3: "vide C4D;"
  (no Latin counterpart file)

Presumably the Latin targets were intended. Seen on commit
5cf0e7f0a3f2c8e1567e0bb0fef2e662250125a0.
