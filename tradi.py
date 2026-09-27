"""
tradi.py — Évangile du jour selon le missel romain de 1962
==========================================================
Forme traditionnelle (rubriques de 1960, saint Pie V / saint Jean XXIII).

Deux briques :
  * le **calendrier** : quel office est célébré tel jour. Fourni par le moteur
    de Missale Meum (MIT), copié dans vendor/missalemeum/
    (voir scripts/maj_missalemeum.py) ;
  * les **textes** : fichiers de Divinum Officium (MIT) embarqués dans
    data/tradi/ (voir scripts/maj_tradi.py et data/tradi/SOURCE.md).

Tous les fichiers sont lus et parsés une seule fois, à la construction de
`MesseTraditionnelle` ; les calendriers annuels sont mis en cache.

Choix de la messe principale
----------------------------
Quelques jours ont plusieurs messes (le moteur renvoie alors plusieurs
célébrations, suffixées m1, m2, m3) :
  * Noël (25/12) : on retient la **messe du jour** (3e messe, « Puer natus »),
    la plus représentative de la fête, dont l'Évangile est le prologue de Jean ;
  * Commémoraison des fidèles défunts (02/11) : on retient la **1re messe**,
    la seule que tout prêtre célèbre si l'on n'en dit qu'une ;
  * tout autre cas de messes multiples : la 1re, signalée dans les logs.

Féries sans Évangile propre
---------------------------
Quand le fichier d'une férie n'a ni [Evangelium], ni renvoi « @ », ni règle
d'héritage (« vide »/« ex »), l'Évangile du dimanche précédent n'est repris
que pour les féries **après l'Épiphanie, après la Pentecôte, de l'Avent et du
temps pascal** (après l'octave de Pâques). Exceptions à messe propre, jamais
reprises : Quatre-Temps de l'Avent, lundi des Rogations, vigile et fête de
l'Ascension, vigile de la Pentecôte ; vendredi et samedi après l'Ascension :
messe de l'Ascension. Du 2 au 5 janvier : messe de l'octave de Noël.
Partout ailleurs (Carême, Quatre-Temps, octaves…), l'absence est une erreur
explicite (`TradiError`), jamais une reprise tacite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

log = logging.getLogger("tradi")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "tradi"

FRANCAIS = "Francais"
LATIN = "Latin"

MENTION_SOURCE = ("Missel romain de 1962 — texte : Divinum Officium "
                  "(divinumofficium.com), traduction non identifiée")
MENTION_LATIN = "Traduction française indisponible : texte latin"


def temps_pascal(jour: date) -> bool:
    """De Pâques au samedi après la Pentecôte inclus."""
    from dateutil.easter import easter
    paques = easter(jour.year)
    return 0 <= (jour - paques).days <= 55


class TradiError(RuntimeError):
    """Évangile introuvable ou données incohérentes pour une date."""


# ============================================================
# Conditions de rubriques (évaluées pour le missel de 1962)
# ============================================================

def condition_vraie(cond: str) -> bool:
    """Évalue une condition Divinum Officium pour les rubriques de 1960.

    Grammaire utile : [sed|vero] [nisi|non] atome (et atome)* (aut …)*.
    Atomes reconnus vrais : « rubrica 196 », « rubrica 1960 » (préfixe de
    1960), « communi Summorum Pontificum » (commun des papes, en vigueur
    depuis 1942) et « ad missam ». Tout autre atome (autres rubriques,
    « tempore … », « feria … ») est faux.
    """
    s = cond.strip().lower()
    for mot in ("sed ", "vero "):
        if s.startswith(mot):
            s = s[len(mot):]
    negation = False
    for mot in ("nisi ", "non "):
        if s.startswith(mot):
            negation, s = True, s[len(mot):]
    if not s:
        return not negation

    def atome(a: str) -> bool:
        a = a.strip()
        if a.startswith("non "):
            return not atome(a[4:])
        if a.startswith("rubrica "):
            x = a[len("rubrica "):].strip()
            return x.isdigit() and len(x) >= 3 and "1960".startswith(x)
        return "summorum pontificum" in a or a == "ad missam"

    vrai = any(all(atome(a) for a in alt.split(" et "))
               for alt in s.split(" aut "))
    return vrai != negation


_COND_LIGNE = re.compile(
    r"\(((?:sed |vero |deinde |atque )?[^()]*?\b(?:rubrica|communi|deinde)\b[^()]*)\)")
_VERBES_INCLURE = ("dicitur", "dicuntur")
_VERBES_OMETTRE = ("omittitur", "omittuntur")


@dataclass
class _Condition:
    vraie: bool
    arriere: bool        # « sed/vero » : agit sur ce qui précède
    omettre: bool        # verbe omittitur/omittuntur (sinon dicitur)
    bloc: bool           # « hæc versus » : tout le bloc, pas une seule ligne


def _analyser_condition(texte: str) -> _Condition:
    mots = texte.strip().split()
    arriere = bool(mots) and mots[0].lower() in ("sed", "vero")
    bas = texte.lower()
    omettre = any(v in bas for v in _VERBES_OMETTRE)
    bloc = "hæc versus" in bas or "haec versus" in bas
    reste = bas
    for m in ("hæc versus", "haec versus", "hic versus", "semper",
              *_VERBES_INCLURE, *_VERBES_OMETTRE):
        reste = reste.replace(m, " ")
    reste = re.sub(r"^\s*(sed|vero|deinde|atque)\b", " ", reste)
    reste = " ".join(reste.split())
    return _Condition(vraie=condition_vraie(reste) if reste else True,
                      arriere=arriere, omettre=omettre, bloc=bloc)


def appliquer_conditions(lignes: List[str]) -> List[str]:
    """Applique les conditions en ligne « (…) » d'un corps de section.

    Sémantique retenue (sous-ensemble de Divinum Officium, suffisant pour les
    Évangiles, rangs et règles du missel) :
      * condition vraie à portée arrière (sed/vero) : retire la ligne
        précédente, ou le bloc précédent si « hæc versus » ;
      * « … dicitur » faux : la ligne (ou le texte) qui suit est ignorée ;
      * « … omittitur » vrai, portée avant : la ligne qui suit est ignorée.
    Une condition peut ouvrir ou fermer une ligne de texte (forme en ligne).
    """
    sortie: List[str] = []
    debut_bloc = 0
    sauter = False

    def texte(t: str) -> None:
        nonlocal sauter, debut_bloc
        if sauter:
            sauter = False
            return
        if not t.strip():
            sortie.append("")
            debut_bloc = len(sortie)
            return
        sortie.append(t.strip())

    def condition(c: _Condition) -> None:
        nonlocal sauter, debut_bloc
        if c.arriere and c.vraie:
            if c.bloc:
                del sortie[debut_bloc:]
            elif sortie and sortie[-1]:
                sortie.pop()
        if c.omettre:
            sauter = c.vraie and not c.arriere
        else:
            sauter = not c.vraie
        debut_bloc = len(sortie)

    for ligne in lignes:
        pos = 0
        morceaux = list(_COND_LIGNE.finditer(ligne))
        if not morceaux:
            texte(ligne)
            continue
        for m in morceaux:
            avant = ligne[pos:m.start()]
            if avant.strip():
                texte(avant)
            condition(_analyser_condition(m.group(1)))
            pos = m.end()
        if ligne[pos:].strip():
            texte(ligne[pos:])
    return sortie


# ============================================================
# Fichiers Divinum Officium
# ============================================================

_ENTETE = re.compile(r"^\[([^\]]+)\]\s*(\((.*)\))?\s*$")
_RENVOI = re.compile(r"^@([^:\s]*)(?::([^:]*))?(?::(.*))?\s*$")
_SUBST = re.compile(r"s/((?:\\.|[^/])*)/((?:\\.|[^/])*)/(\w*)")

#: Sections de prières renvoyées par les Évangiles de la Passion, sans objet
#: pour l'affichage du texte évangélique.
SECTIONS_IGNOREES = {"Munda Cor Passionis"}


@dataclass
class Fichier:
    sections: Dict[str, List[str]]
    renvoi_global: Optional[str] = None     # fichier commençant par « @… »

    def regle_vide(self) -> Optional[str]:
        """Cible d'une règle « vide X » / « ex X » ([Rule] puis [Rank]).

        Exception : quand le [Rank] retenu pour 1960 fait du jour une simple
        férie sans source, un « ex X » de [Rule] est un reste de l'ancienne
        octave (ex. Pent02-3 : « ex Tempora/Pent01-4 », octave de la
        Fête-Dieu abolie en 1955) et n'est pas suivi — comme sur le site
        Divinum Officium. Les « vide X » restent suivis (Quadp2-3).
        """
        rang = self.sections.get("Rank", [])
        ferie_sans_source = bool(rang) and all(
            len(l.split(";;")) < 4 and "Feria" in l for l in rang if l.strip())
        for nom in ("Rule", "Rank"):
            for ligne in self.sections.get(nom, []):
                for item in re.split(r";;|;", ligne):
                    item = item.strip()
                    if item.startswith("ex ") and ferie_sans_source:
                        continue
                    if item.startswith(("vide ", "ex ")):
                        return item.split(None, 1)[1].strip()
        return None


def lire_fichier(chemin: Path) -> Fichier:
    """Découpe un fichier en sections, variante 1960 retenue."""
    variantes: Dict[str, List[Tuple[Optional[str], List[str]]]] = {}
    renvoi_global = None
    courant: Optional[List[str]] = None
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        m = _ENTETE.match(ligne)
        if m:
            courant = []
            variantes.setdefault(m.group(1).strip(), []).append((m.group(3), courant))
            continue
        if courant is None:
            if ligne.startswith("@") and renvoi_global is None:
                renvoi_global = ligne[1:].strip()
            continue
        courant.append(ligne)

    sections: Dict[str, List[str]] = {}
    for nom, liste in variantes.items():
        retenue = None
        for cond, corps in liste:
            if cond is None:
                if retenue is None:
                    retenue = corps
            elif condition_vraie(cond):
                retenue = corps          # la variante conditionnelle l'emporte
        if retenue is not None:
            lignes = appliquer_conditions(retenue)
            while lignes and not lignes[-1]:
                lignes.pop()
            sections[nom] = lignes
    return Fichier(sections, renvoi_global)


def normaliser_cible(cible: str, chemin_courant: str) -> str:
    """Chemin complet d'une cible de renvoi ou de règle.

    « C4a » → « Commune/C4a » ; un nom nu autre qu'un commun désigne un
    fichier du même dossier (« vide Quadp2-0 » depuis Tempora/Quadp2-3) ;
    le zéro manquant d'un sanctoral est rétabli (« Sancti/9-12 »).
    """
    cible = cible.strip().removesuffix(".txt")
    if "/" not in cible:
        dossier = "Commune" if re.match(r"^C\d", cible) else chemin_courant.split("/")[0]
        cible = f"{dossier}/{cible}"
    return re.sub(r"^Sancti/(\d)-", r"Sancti/0\1-", cible)


# ============================================================
# Références bibliques
# ============================================================

LIVRES = {
    "matt": "Matthieu", "matth": "Matthieu", "mt": "Matthieu",
    "marc": "Marc", "mc": "Marc", "mk": "Marc", "mar": "Marc",
    "luc": "Luc", "lc": "Luc", "lk": "Luc", "luke": "Luc",
    "joann": "Jean", "joannes": "Jean", "joan": "Jean", "joh": "Jean",
    "ioann": "Jean", "ioannes": "Jean", "ioan": "Jean",
    "jo": "Jean", "jn": "Jean", "jean": "Jean", "john": "Jean",
}


def formater_reference(brute: str) -> str:
    """« !Matt 22:1-14 » → « Matthieu 22, 1-14 »."""
    s = brute.lstrip("!").strip().rstrip(".").strip()
    s = re.sub(r"^([A-Za-zÀ-ÿ]+),\s*(\d+)\.\s*", r"\1 \2:", s)  # « Matt, 28. 1-7 »
    m = re.match(r"^([A-Za-zÀ-ÿ]+)\.?\s+(.*)$", s)
    if not m:
        return s
    livre = LIVRES.get(m.group(1).lower(), m.group(1))
    passages = re.sub(r"(\d+):(\d)", r"\1, \2", m.group(2))
    return f"{livre} {passages}"


def cle_reference(brute: str) -> str:
    """Forme comparable d'une référence (livre canonique + chiffres)."""
    ref = formater_reference(brute)
    livre, _, reste = ref.partition(" ")
    return livre + " " + re.sub(r"[^0-9]+", ".", reste).strip(".")


# ============================================================
# Noms français des jours du temporal
# ============================================================

JOURS = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]


def _romain(n: int) -> str:
    valeurs = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in valeurs:
        while n >= v:
            s, n = s + r, n - v
    return s


def _ordinal(n: int) -> str:
    return "Ier" if n == 1 else f"{_romain(n)}e"


NOMS_TEMPORA = {
    "Adv3-3": "Mercredi des Quatre-Temps de l'Avent",
    "Adv3-5": "Vendredi des Quatre-Temps de l'Avent",
    "Adv3-6": "Samedi des Quatre-Temps de l'Avent",
    "Nat1-0": "Dimanche dans l'octave de Noël",
    "Nat2-0": "Fête du Très Saint Nom de Jésus",
    "Nat29": "Ve jour dans l'octave de Noël",
    "Nat30": "VIe jour dans l'octave de Noël",
    "Nat31": "VIIe jour dans l'octave de Noël",
    "Epi1-0": "Fête de la Sainte Famille",
    "Quadp1-0": "Dimanche de la Septuagésime",
    "Quadp2-0": "Dimanche de la Sexagésime",
    "Quadp3-0": "Dimanche de la Quinquagésime",
    "Quadp3-3": "Mercredi des Cendres",
    "Quadp3-4": "Jeudi après les Cendres",
    "Quadp3-5": "Vendredi après les Cendres",
    "Quadp3-6": "Samedi après les Cendres",
    "Quad1-3": "Mercredi des Quatre-Temps de Carême",
    "Quad1-5": "Vendredi des Quatre-Temps de Carême",
    "Quad1-6": "Samedi des Quatre-Temps de Carême",
    "Quad5-0": "Ier dimanche de la Passion",
    "Quad6-0": "IIe dimanche de la Passion ou des Rameaux",
    "Quad6-1": "Lundi saint",
    "Quad6-2": "Mardi saint",
    "Quad6-3": "Mercredi saint",
    "Quad6-4": "Jeudi saint",
    "Quad6-5": "Vendredi saint",
    "Quad6-6": "Samedi saint — Vigile pascale",
    "Pasc0-0": "Dimanche de Pâques",
    "Pasc1-0": "Dimanche in albis, octave de Pâques",
    "Pasc5-1": "Lundi des Rogations",
    "Pasc5-2": "Mardi des Rogations",
    "Pasc5-3": "Vigile de l'Ascension",
    "Pasc5-4": "Ascension de Notre-Seigneur",
    "Pasc6-0": "Dimanche après l'Ascension",
    "Pasc6-6": "Vigile de la Pentecôte",
    "Pasc7-0": "Dimanche de la Pentecôte",
    "Pasc7-3": "Mercredi des Quatre-Temps de Pentecôte",
    "Pasc7-5": "Vendredi des Quatre-Temps de Pentecôte",
    "Pasc7-6": "Samedi des Quatre-Temps de Pentecôte",
    "Pent01-0": "Fête de la Très Sainte Trinité",
    "Pent01-4": "Fête du Très Saint Sacrement",
    "Pent02-5": "Fête du Sacré-Cœur de Jésus",
    "093-3": "Mercredi des Quatre-Temps de septembre",
    "093-5": "Vendredi des Quatre-Temps de septembre",
    "093-6": "Samedi des Quatre-Temps de septembre",
}

#: Noms usuels des principales fêtes du sanctoral. Les fichiers français de
#: Divinum Officium donnent souvent le nom latin ([Officium]) ; pour les
#: autres jours, ce nom d'origine est affiché tel quel.
NOMS_SANCTI = {
    "12-08": "Immaculée Conception de la Bienheureuse Vierge Marie",
    "12-24": "Vigile de Noël",
    "12-25m1": "Nativité de Notre-Seigneur — messe de minuit",
    "12-25m2": "Nativité de Notre-Seigneur — messe de l'aurore",
    "12-25m3": "Nativité de Notre-Seigneur — messe du jour",
    "12-26": "Saint Étienne, premier martyr",
    "12-27": "Saint Jean, apôtre et évangéliste",
    "12-28": "Saints Innocents",
    "01-01": "Octave de la Nativité de Notre-Seigneur",
    "01-06": "Épiphanie de Notre-Seigneur",
    "01-13": "Commémoraison du Baptême de Notre-Seigneur",
    "02-02": "Purification de la Bienheureuse Vierge Marie",
    "03-19": "Saint Joseph, époux de la Bienheureuse Vierge Marie",
    "03-25": "Annonciation de la Bienheureuse Vierge Marie",
    "05-01": "Saint Joseph artisan",
    "05-31": "Bienheureuse Vierge Marie Reine",
    "06-24": "Nativité de saint Jean-Baptiste",
    "06-29": "Saints Pierre et Paul, apôtres",
    "07-01": "Très Précieux Sang de Notre-Seigneur Jésus-Christ",
    "07-02": "Visitation de la Bienheureuse Vierge Marie",
    "08-06": "Transfiguration de Notre-Seigneur",
    "08-15": "Assomption de la Bienheureuse Vierge Marie",
    "08-22": "Cœur Immaculé de la Bienheureuse Vierge Marie",
    "09-08": "Nativité de la Bienheureuse Vierge Marie",
    "09-14": "Exaltation de la Sainte Croix",
    "09-15": "Sept Douleurs de la Bienheureuse Vierge Marie",
    "09-29": "Dédicace de saint Michel archange",
    "10-07": "Notre-Dame du Rosaire",
    "10-11": "Maternité de la Bienheureuse Vierge Marie",
    "10-DU": "Fête du Christ-Roi",
    "11-01": "Toussaint",
    "11-02m1": "Commémoraison de tous les fidèles défunts — 1re messe",
    "11-02m2": "Commémoraison de tous les fidèles défunts — 2e messe",
    "11-02m3": "Commémoraison de tous les fidèles défunts — 3e messe",
}

_TEMPS = {
    "Adv": "de l'Avent",
    "Epi": "après l'Épiphanie",
    "Quad": "de Carême",
    "Pasc": "après Pâques",
    "Pent": "après la Pentecôte",
}
_QUADP = {1: "la Septuagésime", 2: "la Sexagésime", 3: "la Quinquagésime"}


def forme_de_base(nom: str) -> str:
    """Epi1-0a, Pent01-0r, Pent03-1Feria… → Epi1-0, Pent01-0, Pent03-1."""
    return re.sub(r"(Feria|[a-z]+)$", "", nom)


def nom_tempora(nom: str) -> Optional[str]:
    """Nom français d'un jour du temporal (identifiant Divinum Officium)."""
    base = forme_de_base(nom)
    if base in NOMS_TEMPORA:
        return NOMS_TEMPORA[base]
    m = re.match(r"^(Adv|Epi|Quadp|Quad|Pasc|Pent)(\d+)-(\d)$", base)
    if not m:
        return None
    temps, n, j = m.group(1), int(m.group(2)), int(m.group(3))
    if temps == "Quadp":
        dimanche = f"le dimanche de {_QUADP.get(n, '')}"
        return f"{JOURS[j]} après {dimanche}"
    if temps == "Pasc" and n == 0:
        return f"{JOURS[j]} de Pâques"
    if temps == "Pasc" and n == 7:
        return f"{JOURS[j]} de Pentecôte"
    if temps == "Quad" and n == 5 and j:
        return f"{JOURS[j]} de la semaine de la Passion"
    # Pasc1-0 (in albis) est le Ier dimanche après Pâques, Pasc2-0 le IIe…
    dimanche = f"{_ordinal(n)} dimanche {_TEMPS[temps]}"
    if j == 0:
        return dimanche[0].upper() + dimanche[1:]
    return f"{JOURS[j]} après le {dimanche}"


# ============================================================
# Résultat
# ============================================================

@dataclass
class Evangile:
    date: str
    office: str                 # identifiant calendrier, ex. tempora:Pent19-0:2:g
    fichier: str                # fichier Divinum Officium, ex. Tempora/Pent19-0
    nom_messe: str
    reference: str              # « Matthieu 22, 1-14 »
    reference_brute: str        # « !Matt 22:1-14 »
    texte: str
    latin: bool = False         # repli sur le texte latin
    reprise_dimanche: Optional[str] = None   # fichier de la messe reprise
    messes: List[str] = field(default_factory=list)  # si plusieurs messes
    reference_latine: Optional[str] = None   # si elle diffère de la française

    @property
    def mention(self) -> str:
        if self.latin:
            return f"{MENTION_SOURCE}. {MENTION_LATIN}."
        return MENTION_SOURCE + "."


class ResultatEvangile(NamedTuple):
    reference: str                    # référence brute, ex. « !Matt 22:1-14 »
    paragraphes: List[str]
    latin: bool                       # texte (en partie) latin
    reference_latine: Optional[str] = None   # si elle diffère de la française


# ============================================================
# Moteur
# ============================================================

#: Féries où la reprise est interdite parce que la messe est propre : un
#: fichier sans Évangile y est une erreur de données, pas une férie ordinaire.
#: Quatre-Temps de l'Avent ; lundi des Rogations, vigile et fête de
#: l'Ascension, vigile de la Pentecôte (vérifiés : tous ont leur [Evangelium]).
_SANS_REPRISE = {"Adv3-3", "Adv3-5", "Adv3-6",
                 "Pasc5-1", "Pasc5-3", "Pasc5-4", "Pasc6-6"}
#: Vendredi et samedi après l'Ascension : messe de l'Ascension (Divinum
#: Officium le dit déjà par « vide Tempora/Pasc5-4 » ; filet de sécurité).
_REPRISE_ASCENSION = {"Pasc5-5", "Pasc5-6"}
#: Féries qui reprennent la messe du dimanche précédent. Temps pascal : à
#: partir de la semaine in albis (Pasc1) ; l'octave de Pâques (Pasc0) et
#: celle de la Pentecôte (Pasc7) ont leurs messes propres.
_FERIE_REPRISE = re.compile(r"^(Adv|Epi|Pent|Pasc)(\d+)-([1-6])$")

#: Textes français plus longs que la péricope de 1962 : on coupe au verset
#: de début (marqueur) ; faute de marqueur, on affiche la référence réelle du
#: texte français plutôt que celle du missel.
DECOUPES_FRANCAIS = {
    # Mardi saint : 1962 commence à Marc 14, 32 (Gethsémani) ; le français
    # donne toute la Passion depuis 14, 1.
    "Tempora/Quad6-2": ("Ils arrivent en un domaine appelé Gethsémani",
                        "!Marc 14:1-72; 15:1-46"),
}

#: Marqueur de Missale Meum pour une férie sans messe propre : l'office réel
#: est alors celui du temporal (`Day.tempora`).
FERIE_MISSALEMEUM = ":feria:4:w"

#: Identifiants du calendrier sans fichier homonyme dans Divinum Officium.
#: C10t (samedi de la Sainte Vierge après la Pentecôte) n'existe plus que dans
#: les surcharges locales de Missale Meum, dont l'Évangile est
#: « @Commune/C11:Evangelium » (Luc 11, 27-28, messe Salve sancta Parens).
ALIAS_FICHIERS = {
    "Commune/C10t": "Commune/C11",
}

#: Liturgies restaurées de la Semaine sainte (1955) : Divinum Officium donne
#: tout le déroulé dans [Prelude], sans section [Evangelium]. On y repère
#: l'Évangile par sa ligne de référence.
EVANGILES_DANS_PRELUDE = {
    "Tempora/Quad6-5r": r"^!Joannes 18",     # Vendredi saint : Passion selon saint Jean
    "Tempora/Quad6-6r": r"^!Matt, 28",       # Vigile pascale : Matthieu 28, 1-7
}

#: Correctif du calendrier Missale Meum : sa règle « Sept Douleurs le vendredi
#: de la Passion » s'applique avant les règles des fêtes de 1re classe et
#: renvoie la férie en supprimant toute autre observance. Une fête de
#: 1re classe tombant ce jour-là (saint Joseph le 19/03/2027, Annonciation)
#: disparaissait. Elle est rétablie ici, la férie n'étant que de 3e classe.
VENDREDI_PASSION_MISSALEMEUM = "tempora:Quad5-5Feria:3:v"

NOMS_COMMUNE = {
    "Commune/C11": "Messe de la Sainte Vierge le samedi",
    "Commune/C10": "Messe de la Sainte Vierge le samedi",
    "Commune/C10a": "Messe de la Sainte Vierge le samedi (Avent)",
    "Commune/C10b": "Messe de la Sainte Vierge le samedi (temps de Noël)",
    "Commune/C10c": "Messe de la Sainte Vierge le samedi",
    "Commune/C10Pasc": "Messe de la Sainte Vierge le samedi (temps pascal)",
}


class _Boucle(TradiError):
    pass


class MesseTraditionnelle:
    """Charge les textes une fois et répond « quel Évangile tel jour ? »."""

    def __init__(self, racine: Path = DATA_DIR):
        self.racine = Path(racine)
        self._fichiers: Dict[Tuple[str, str], Fichier] = {}
        for sous in ("missa", "horas"):
            for chemin in sorted((self.racine / sous).glob("*/*/*.txt")):
                langue, dossier = chemin.parts[-3], chemin.parts[-2]
                cle = (langue, f"{dossier}/{chemin.stem}")
                # missa/ prioritaire sur horas/ (Commun des messes)
                if sous == "horas" and cle in self._fichiers:
                    continue
                self._fichiers[cle] = lire_fichier(chemin)
        if not self._fichiers:
            raise TradiError(f"Aucun texte trouvé dans {self.racine}")
        #: Incidents de résolution rencontrés (pour le rapport annuel).
        self.incidents: List[str] = []
        log.info("Messe 1962 : %d fichiers chargés", len(self._fichiers))

    # ---------------- Textes ----------------

    def fichier(self, langue: str, chemin: str) -> Optional[Fichier]:
        return self._fichiers.get((langue, chemin))

    def section(self, chemin: str, nom: str, langue: str,
                _pile: Tuple = (), pascal: bool = False) -> Optional[Tuple[List[str], bool]]:
        """Lignes de la section `nom` du fichier `chemin`, renvois résolus.

        Renvoie (lignes, latin) où `latin` indique qu'un morceau a dû être
        pris dans le texte latin. None si la section est introuvable.

        Comme Divinum Officium, on superpose le fichier vernaculaire au
        fichier latin : une section absente du français est prise dans le
        latin (ses renvois restant résolus en français), et la structure
        d'héritage (renvoi de fichier, règle « vide »/« ex ») est lue dans le
        latin quand le fichier français ne la donne pas.

        `pascal` : au temps pascal, une règle « vide Cx » renvoie au commun
        pascal « Cxp » quand il existe (C2a → C2ap), comme le fait le site.
        """
        cle = (chemin, nom, langue)
        if cle in _pile:
            raise _Boucle("Renvoi circulaire : " + " → ".join(
                f"{c}:{n}" for c, n, _ in (*_pile, cle)))
        pile = (*_pile, cle)
        f = self.fichier(langue, chemin)
        fl = self.fichier(LATIN, chemin) if langue != LATIN else None
        if f is None and fl is None:
            return None
        if f is not None and nom in f.sections:
            return self._developper(f.sections[nom], chemin, nom, langue, pile,
                                    pascal=pascal)
        if fl is not None and nom in fl.sections:
            return self._developper(fl.sections[nom], chemin, nom, langue, pile,
                                    corps_latin=True, pascal=pascal)
        for attribut in ("renvoi_global", "regle_vide"):
            cible = None
            for source in (f, fl):
                if source is not None and cible is None:
                    v = getattr(source, attribut)
                    cible = v() if callable(v) else v
            if cible:
                cible = normaliser_cible(cible.split(":")[0], chemin)
                if pascal and attribut == "regle_vide" and cible.startswith("Commune/") \
                        and self.fichier(LATIN, cible + "p"):
                    cible += "p"
                r = self.section(cible, nom, langue, pile, pascal)
                if r is not None:
                    return r
        return None

    def _developper(self, lignes, chemin, nom, langue, pile, corps_latin=False,
                    pascal=False):
        sortie: List[str] = []
        latin = False
        for ligne in lignes:
            m = _RENVOI.match(ligne)
            if not m:
                sortie.append(ligne)
                if corps_latin and ligne.strip():
                    latin = True
                continue
            cible = normaliser_cible(m.group(1), chemin) if m.group(1) else chemin
            sous = (m.group(2) or "").strip() or nom
            if sous in SECTIONS_IGNOREES:
                continue
            r = self.section(cible, sous, langue, pile, pascal)
            if r is None:
                msg = f"Renvoi non résolu : {langue}/{chemin}:{nom} → {ligne}"
                self.incidents.append(msg)
                log.warning(msg)
                continue
            morceau, lat = r
            latin = latin or lat
            if m.group(3):
                morceau = self._substituer(morceau, m.group(3), ligne)
            sortie.extend(morceau)
        return sortie, latin

    def _substituer(self, lignes: List[str], subs: str, ligne: str) -> List[str]:
        texte = "\n".join(lignes)
        for motif, rempl, drapeaux in _SUBST.findall(subs):
            try:
                texte = re.sub(motif, rempl.replace("\\", "\\\\"), texte,
                               count=0 if "g" in drapeaux else 1,
                               flags=re.M if "m" in drapeaux else 0)
            except re.error as exc:
                self.incidents.append(f"Substitution ignorée ({exc}) : {ligne}")
        return texte.split("\n")

    # ---------------- Évangile d'un fichier ----------------

    def _evangile_brut(self, chemin: str, langue: str, pascal: bool = False):
        """(annonce, référence brute, paragraphes, latin) ou None."""
        r = self.section(chemin, "Evangelium", langue, pascal=pascal)
        if r is None and chemin in EVANGILES_DANS_PRELUDE:
            return self._evangile_prelude(chemin, langue, pascal)
        if r is None:
            return None
        lignes, latin = r
        lignes = [l for l in lignes if l.strip() and l.strip() != "_"]
        i_ref = next((i for i, l in enumerate(lignes) if l.startswith("!")), None)
        if i_ref is None:
            raise TradiError(f"{langue}/{chemin} : Évangile sans référence")
        paragraphes = []
        for l in lignes[i_ref + 1:]:
            if l.startswith("!"):
                continue
            l = re.sub(r"^v\.\s*", "", l).replace("((", "(").replace("))", ")")
            paragraphes.append(" ".join(l.split()))
        if not paragraphes:
            raise TradiError(f"{langue}/{chemin} : Évangile sans texte")
        return lignes[i_ref], paragraphes, latin

    def _evangile_prelude(self, chemin: str, langue: str, pascal: bool = False):
        """Évangile inclus dans le déroulé complet d'une liturgie ([Prelude]).

        Le texte suit la ligne de référence, entrecoupé de rubriques (« !… »)
        ignorées, jusqu'à la réponse « R. », un séparateur « _ » ou une
        nouvelle partie numérotée.
        """
        r = self.section(chemin, "Prelude", langue, pascal=pascal)
        if r is None:
            return None
        lignes, latin = r
        motif = re.compile(EVANGILES_DANS_PRELUDE[chemin])
        i_ref = next((i for i, l in enumerate(lignes) if motif.match(l)), None)
        if i_ref is None:
            return None
        paragraphes = []
        for l in lignes[i_ref + 1:]:
            l = l.strip()
            if l == "_" or l.startswith(("R.", "S.", "!!")) or re.match(r"^!\s*\d", l):
                break
            if l and not l.startswith("!"):
                paragraphes.append(" ".join(l.split()))
        if not paragraphes:
            return None
        return lignes[i_ref].strip(), paragraphes, latin

    def evangile_fichier(self, chemin: str, pascal: bool = False) -> Optional["ResultatEvangile"]:
        """Évangile d'un fichier, en français de préférence.

        Si les références française et latine diffèrent (coquille de l'une ou
        de l'autre dans Divinum Officium), le texte français est gardé : l'écart
        est journalisé, noté dans `incidents` et renvoyé dans
        `reference_latine` pour le rapport.
        Renvoie None si aucun Évangile n'est trouvé.
        """
        la = self._evangile_brut(chemin, LATIN, pascal)
        fr = self._evangile_brut(chemin, FRANCAIS, pascal)
        if fr is None and la is None:
            return None
        if fr is None:
            log.warning("%s : traduction française indisponible, texte latin", chemin)
            return ResultatEvangile(la[0], la[1], True)
        ref, paragraphes, latin = fr
        if latin:
            log.warning("%s : traduction française indisponible, texte latin", chemin)
        elif chemin in DECOUPES_FRANCAIS:
            ref, paragraphes = self._decouper(chemin, ref, paragraphes)
        ref_latine = None
        if la is not None and cle_reference(ref) != cle_reference(la[0]):
            ref_latine = la[0]
            msg = (f"{chemin} : écart de référence, française {ref!r} / latine "
                   f"{la[0]!r} (texte français conservé)")
            self.incidents.append(msg)
            log.warning(msg)
        return ResultatEvangile(ref, paragraphes, latin, ref_latine)

    def _decouper(self, chemin, ref, paragraphes):
        """Coupe un texte français au verset de début de la péricope de 1962.

        Sans marqueur repérable, le texte reste entier et la référence
        affichée devient celle du texte français réellement donné.
        """
        marqueur, ref_texte_complet = DECOUPES_FRANCAIS[chemin]
        for i, p in enumerate(paragraphes):
            pos = p.find(marqueur)
            if pos >= 0:
                return ref, [p[pos:], *paragraphes[i + 1:]]
        msg = f"{chemin} : début de péricope introuvable, référence du texte complet"
        self.incidents.append(msg)
        log.warning(msg)
        return ref_texte_complet, paragraphes

    # ---------------- Calendrier ----------------

    @staticmethod
    @lru_cache(maxsize=8)
    def _calendrier(annee: int):
        from vendor.missalemeum.kalendar.factory import MissalFactory
        return MissalFactory().create(annee, "la")

    def office_du_jour(self, jour: date):
        """(observance retenue ou None, identifiants des messes du jour).

        None signifie une férie sans office du temporal (début janvier) :
        `evangile_du_jour` décide alors s'il y a reprise.
        """
        journee = self._calendrier(jour.year).get_day(jour)
        celebrations = [o for o in journee.celebration if o.id != FERIE_MISSALEMEUM]
        if celebrations and celebrations[0].id == VENDREDI_PASSION_MISSALEMEUM:
            fete = self._fete_de_premiere_classe(jour)
            if fete is not None:
                log.info("%s : %s l'emporte sur le vendredi de la Passion "
                         "(correctif du calendrier)", jour, fete.id)
                return fete, []
        if not celebrations:
            return (journee.tempora[0] if journee.tempora else None), []
        if len(celebrations) == 1:
            return celebrations[0], []
        ids = [o.id for o in celebrations]
        principale = celebrations[0]
        for o in celebrations:
            if o.name == "12-25m3":         # Noël : messe du jour
                principale = o
        if not any(o.name.startswith(("12-25m", "11-02m")) for o in celebrations):
            log.warning("%s : plusieurs messes %s, la 1re est retenue", jour, ids)
        return principale, ids

    @staticmethod
    def _fete_de_premiere_classe(jour: date):
        """Fête de 1re classe du sanctoral fixée à cette date, s'il y en a une."""
        from vendor.missalemeum.constants import BLOCKS
        from vendor.missalemeum.kalendar.models import Observance
        prefixe = f"sancti:{jour:%m-%d}:"
        fetes = [Observance(i, jour, "la") for i in BLOCKS["la"].SANCTI
                 if i.startswith(prefixe)]
        return next((o for o in fetes if o.rank == 1), None)

    @staticmethod
    def chemin_fichier(observance, jour: date) -> str:
        """Fichier Divinum Officium d'une observance du calendrier."""
        cle = f"{observance.flexibility.capitalize()}/{observance.name}"
        if cle == "Tempora/Nat1-1":
            # Jour dans l'octave de Noël : fichiers Nat29/Nat30/Nat31.
            return f"Tempora/Nat{jour.day:02d}"
        return ALIAS_FICHIERS.get(cle, cle)

    def nom_messe(self, observance, chemin: str) -> str:
        if observance.flexibility == "tempora":
            nom = nom_tempora(chemin.split("/")[-1]) or nom_tempora(observance.name)
            if nom:
                return nom
        elif observance.name in NOMS_SANCTI:
            return NOMS_SANCTI[observance.name]
        elif chemin in NOMS_COMMUNE:
            return NOMS_COMMUNE[chemin]
        for langue in (FRANCAIS, LATIN):
            f = self.fichier(langue, chemin)
            if f and f.sections.get("Officium"):
                return f.sections["Officium"][0].strip()
        return observance.title or observance.name

    def _ferie_sans_temporal(self, jour: date) -> Tuple[str, str]:
        """(messe reprise, nom) pour une férie sans office du temporal.

        * 2 au 5 janvier : messe de l'octave de Noël (Sancti/01-01) ;
        * 7 au 12 janvier : messe précédente, Épiphanie ou dimanche après
          l'Épiphanie (Tempora/Epi1-0a).
        """
        jour_fr = JOURS[(jour.weekday() + 1) % 7]
        if jour.month == 1 and 2 <= jour.day <= 5:
            return "Sancti/01-01", f"{jour_fr} du temps de Noël (messe de l'octave de Noël)"
        j = jour
        while j.weekday() != 6 and (j.month, j.day) != (1, 6) \
                and (j.month, j.day) != (1, 1):
            j = date.fromordinal(j.toordinal() - 1)
        nom = f"{jour_fr} après l'Épiphanie (férie)"
        if (j.month, j.day) == (1, 6):
            return "Sancti/01-06", nom
        precedente, _ = self.office_du_jour(j) if j.weekday() == 6 else (None, [])
        if precedente is not None and precedente.name == "Epi1-0":
            return "Tempora/Epi1-0a", nom     # dimanche après l'Épiphanie
        raise TradiError(
            f"{jour} : férie hors temporal sans Évangile propre "
            "(aucune reprise autorisée)")

    def _messe_reprise(self, observance, chemin: str) -> str:
        """Messe reprise par une férie du temporal sans Évangile propre."""
        base = forme_de_base(observance.name)
        m = _FERIE_REPRISE.match(base)
        if observance.flexibility == "tempora" and base in _REPRISE_ASCENSION:
            return "Tempora/Pasc5-4"
        if observance.flexibility != "tempora" or not m or base in _SANS_REPRISE \
                or (m.group(1) == "Pasc" and not 1 <= int(m.group(2)) <= 6):
            raise TradiError(
                f"pas d'Évangile pour {chemin} (ni section, ni renvoi, "
                "ni règle de reprise applicable)")
        dimanche = f"Tempora/{m.group(1)}{m.group(2)}-0"
        return next((c for c in (dimanche + "a", dimanche)
                     if self.fichier(LATIN, c)), dimanche)

    def evangile_du_jour(self, jour: date) -> Evangile:
        observance, messes = self.office_du_jour(jour)
        pascal = temps_pascal(jour)
        reprise = None
        if observance is None:
            chemin = f"Feria/{jour.isoformat()}"
            reprise, nom = self._ferie_sans_temporal(jour)
            office = FERIE_MISSALEMEUM
            r = self.evangile_fichier(reprise, pascal)
        else:
            chemin = self.chemin_fichier(observance, jour)
            office, nom = observance.id, self.nom_messe(observance, chemin)
            try:
                r = self.evangile_fichier(chemin, pascal)
            except _Boucle as exc:
                self.incidents.append(f"{jour} : {exc}")
                raise
            if r is None:
                reprise = self._messe_reprise(observance, chemin)
                r = self.evangile_fichier(reprise, pascal)
        if r is None:
            raise TradiError(f"messe reprise {reprise} sans Évangile")
        ref, paragraphes, latin, ref_latine = r
        return Evangile(
            date=jour.isoformat(),
            office=office,
            fichier=chemin,
            nom_messe=nom,
            reference=formater_reference(ref),
            reference_brute=ref,
            texte="\n\n".join(paragraphes),
            latin=latin,
            reprise_dimanche=reprise,
            messes=messes,
            reference_latine=formater_reference(ref_latine) if ref_latine else None,
        )

    # ---------------- Interface avec l'API ----------------

    def etat_pour_api(self, date_iso: str) -> dict:
        """État de l'écran « Évangile du jour » pour la forme de 1962.

        Même structure que `build_evangile_state` (main.py) — context, text,
        error — complétée des champs propres à cette forme. Le texte suit le
        format produit pour AELF (« ÉVANGILE (réf) », ligne vide, texte) :
        le moteur d'éclairage l'analyse donc sans distinction.
        """
        etat = {"context": "", "text": "", "error": None, "messe": "",
                "reference": "", "source": MENTION_SOURCE + ".", "latin": False}
        try:
            e = self.evangile_du_jour(date.fromisoformat(date_iso))
        except Exception as exc:  # une date sans Évangile ne doit pas faire échouer la requête
            log.warning("Messe 1962 indisponible le %s : %s", date_iso, exc)
            etat["error"] = "Évangile du missel de 1962 indisponible pour cette date."
            return etat
        etat.update(
            context=e.nom_messe,
            text=f"ÉVANGILE ({e.reference})\n\n{e.texte}",
            messe=e.nom_messe,
            reference=e.reference,
            source=e.mention,
            latin=e.latin,
        )
        return etat
