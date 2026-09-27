"""
rapport_tradi.py — Contrôle annuel du résolveur de la messe de 1962
===================================================================
Fait tourner `tradi.MesseTraditionnelle` sur chaque jour d'une période et
produit un rapport Markdown : jours sans Évangile, renvois « @ » non résolus,
jours en repli latin, féries résolues par reprise, écarts de référence entre
français et latin, et — si `scripts/controle_divinum.py` a été lancé — écarts
avec le site Divinum Officium.

Usage :
    python scripts/rapport_tradi.py [debut] [fin] [sortie.md]
Par défaut : année liturgique 2026-2027 (29/11/2026 → 27/11/2027), rapport
écrit dans data/tradi/RAPPORT_2026-2027.md. Aucun accès réseau.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tradi  # noqa: E402

JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]

#: Écarts avec le site Divinum Officium examinés et assumés, par fichier
#: retenu par tradi.py ou par date (MM-JJ), avec la raison en une ligne.
ECARTS_ASSUMES = {
    "Sancti/12-25m3": "Noël : messe du jour retenue parmi les trois messes "
                      "(le site affiche la messe de minuit).",
    "Tempora/Quad6-4r": "Jeudi saint : messe du soir (Cène du Seigneur), messe "
                        "principale (le site affiche la messe chrismale).",
    "Tempora/Pasc5-1": "Lundi des Rogations : messe propre des Rogations "
                       "(le site affiche la messe de férie).",
    "Sancti/12-28": "Saints Innocents (IIe classe) ; le site affiche un jour dans "
                    "l'octave avec Luc 2, 42-52, anomalie probable du site.",
    "08-22": "Un dimanche de IIe classe l'emporte sur une fête de IIe classe qui "
             "n'est pas du Seigneur (rubriques de 1960) : dimanche, pas le Cœur Immaculé.",
    "07-16": "Notre-Dame du Mont-Carmel : commémoraison dans le calendrier de 1960 "
             "(AAS 1960), messe de la férie ; le site en fait une IIIe classe.",
}


def ecart_assume(jour, e):
    if e is not None and e.fichier in ECARTS_ASSUMES:
        return ECARTS_ASSUMES[e.fichier]
    return ECARTS_ASSUMES.get(f"{jour:%m-%d}")


def main() -> None:
    debut = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 11, 29)
    fin = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2027, 11, 27)
    sortie = Path(sys.argv[3]) if len(sys.argv) > 3 else \
        tradi.DATA_DIR / f"RAPPORT_{debut.year}-{fin.year}.md"
    fichier_controle = tradi.DATA_DIR / f"controle_divinum_{debut.year}-{fin.year}.json"
    controle = json.loads(fichier_controle.read_text(encoding="utf-8")) \
        if fichier_controle.exists() else None

    moteur = tradi.MesseTraditionnelle()
    lignes, erreurs, latin, reprises, multiples, incidents = [], [], [], [], [], []
    ecarts_fr_la, noms_origine, ecarts_site, sans_site = [], [], [], []
    corrections, assumes = [], []
    concordants = 0
    j = debut
    while j <= fin:
        avant = len(moteur.incidents)
        e = None
        try:
            e = moteur.evangile_du_jour(j)
        except tradi.TradiError as exc:
            message = str(exc).removeprefix(f"{j} : ")
            erreurs.append((j, message))
            lignes.append(f"| {j} {JOURS[j.weekday()]} | — | — | **ERREUR** | {message} |")
        else:
            notes = []
            if e.latin:
                latin.append((j, e))
                notes.append("repli latin")
            if e.reprise_dimanche:
                reprises.append((j, e))
                notes.append(f"reprise de {e.reprise_dimanche}")
            if e.messes:
                multiples.append((j, e))
                notes.append("messe principale parmi " + ", ".join(e.messes))
            for c in e.corrections:
                corrections.append((j, c))
                notes.append(f"référence corrigée : {c}")
            if e.reference_latine:
                ecarts_fr_la.append((j, e))
                notes.append(f"réf. latine : {e.reference_latine}")
            observance, _ = moteur.office_du_jour(j)
            if observance is not None and observance.flexibility == "sancti" \
                    and observance.name not in tradi.NOMS_SANCTI:
                noms_origine.append((j, e))
            lignes.append(f"| {j} {JOURS[j.weekday()]} | {e.nom_messe} | "
                          f"{e.reference} | {'la' if e.latin else 'fr'} | "
                          f"`{e.fichier}` {'; '.join(notes)} |")
        for inc in moteur.incidents[avant:]:
            incidents.append((j, inc))

        if controle is not None:
            site = controle.get(j.isoformat()) or {}
            ref_site = site.get("reference")
            if not ref_site:
                sans_site.append((j, site.get("titre"), site.get("erreur")))
            else:
                cle_site = tradi.cle_reference(ref_site)
                refs_nous = [] if e is None else \
                    [e.reference_brute] + ([e.reference_latine] if e.reference_latine else [])
                if any(tradi.cle_reference(r) == cle_site for r in refs_nous):
                    concordants += 1
                elif e is not None and any(
                        tradi.cle_reference(c["reference_erronee"]) == cle_site
                        for liste in (moteur.corrections.get((e.fichier, langue), [])
                                      for langue in (tradi.FRANCAIS, tradi.LATIN))
                        for c in liste):
                    raison = ("Le site reproduit une coquille de référence de Divinum "
                              "Officium, corrigée par corrections.json.")
                    assumes.append((j, e, site, raison))
                elif ecart_assume(j, e):
                    assumes.append((j, e, site, ecart_assume(j, e)))
                else:
                    ecarts_site.append((j, e, site))
        j += timedelta(days=1)

    nb = (fin - debut).days + 1
    non_resolus = [(d, i) for d, i in incidents if i.startswith("Renvoi non résolu")]
    autres = [(d, i) for d, i in incidents
              if not i.startswith("Renvoi non résolu") and "écart de référence" not in i]
    md = [
        f"# Rapport messe 1962 — {debut.strftime('%d/%m/%Y')} → {fin.strftime('%d/%m/%Y')}",
        "",
        "Généré par `scripts/rapport_tradi.py`. Calendrier : Missale Meum "
        "(vendor/missalemeum) ; textes : Divinum Officium (data/tradi).",
        "",
        "## Synthèse",
        "",
        "| Contrôle | Jours |",
        "|---|---|",
        f"| Jours examinés | {nb} |",
        f"| Évangile trouvé | {nb - len(erreurs)} |",
        f"| **Jours sans Évangile (erreur)** | {len(erreurs)} |",
        f"| Renvois « @ » non résolus | {len(non_resolus)} |",
        f"| Jours en repli latin | {len(latin)} |",
        f"| Féries résolues par reprise (dimanche ou messe précédente) | {len(reprises)} |",
        f"| Jours à plusieurs messes (principale retenue) | {len(multiples)} |",
        f"| Corrections de référence appliquées (corrections.json) | {len(corrections)} |",
        f"| Écarts de référence français / latin (texte français conservé) | {len(ecarts_fr_la)} |",
        f"| Autres incidents | {len(autres)} |",
        f"| Sanctoral affiché avec le nom d'origine Divinum Officium | {len(noms_origine)} |",
    ]
    if controle is not None:
        md += [
            f"| Contrôle site Divinum Officium : références concordantes | {concordants} |",
            f"| Contrôle site Divinum Officium : écarts assumés | {len(assumes)} |",
            f"| **Contrôle site Divinum Officium : écarts non expliqués** | {len(ecarts_site)} |",
            f"| Contrôle site Divinum Officium : référence non lue sur le site | {len(sans_site)} |",
        ]
    else:
        md += ["| Contrôle site Divinum Officium | non effectué "
               "(lancer `scripts/controle_divinum.py`) |"]
    md.append("")

    def section(titre, elements, fmt, vide="Aucun.", intro=None):
        md.extend([f"## {titre}", ""])
        if intro:
            md.extend([intro, ""])
        md.extend(fmt(x) for x in elements) if elements else md.append(vide)
        md.append("")

    section("Jours sans Évangile", erreurs, lambda x: f"- {x[0]} : {x[1]}")
    section("Renvois « @ » non résolus", non_resolus, lambda x: f"- {x[0]} : {x[1]}")
    section("Jours en repli latin (traduction française indisponible)", latin,
            lambda x: f"- {x[0]} : {x[1].nom_messe} — {x[1].reference} (`{x[1].fichier}`)")
    section("Féries résolues par reprise (dimanche ou messe précédente)", reprises,
            lambda x: f"- {x[0]} : {x[1].nom_messe} → `{x[1].reprise_dimanche}` ({x[1].reference})")
    section("Jours à plusieurs messes", multiples,
            lambda x: f"- {x[0]} : retenue `{x[1].fichier}` parmi {', '.join(x[1].messes)}")
    section("Corrections de référence appliquées", corrections,
            lambda x: f"- {x[0]} : {x[1]}",
            intro="Coquilles de Divinum Officium corrigées par `data/tradi/corrections.json` "
                  "(justification de chaque entrée dans le fichier).")
    section("Écarts de référence français / latin", ecarts_fr_la,
            lambda x: f"- {x[0]} : {x[1].nom_messe} (`{x[1].fichier}`) — française "
                      f"**{x[1].reference}** (affichée), latine {x[1].reference_latine}",
            intro="Le texte français est conservé ; l'écart est journalisé.")
    section("Autres incidents", autres, lambda x: f"- {x[0]} : {x[1]}")

    if controle is not None:
        def fmt_site(x):
            j, e, site = x
            nous = "ERREUR (pas d'Évangile)" if e is None else \
                f"{e.reference} — {e.nom_messe} (`{e.fichier}`)"
            return (f"| {j} {JOURS[j.weekday()]} | {nous} | "
                    f"{tradi.formater_reference(site['reference'])} — {site.get('titre') or '?'} |")
        intro = ("Référence de l'Évangile affichée par www.divinumofficium.com "
                 "(rubriques 1960, `scripts/controle_divinum.py`) comparée à celle "
                 "de `tradi.py` (française ou latine).")
        md.extend(["## Contrôle indépendant : écarts non expliqués avec le site Divinum Officium",
                   "", intro, ""])
        if ecarts_site:
            md.extend(["| Date | tradi.py | Site Divinum Officium |", "|---|---|---|",
                       *map(fmt_site, ecarts_site)])
        else:
            md.append("Aucun écart.")
        md.extend(["", "## Contrôle indépendant : écarts assumés", ""])
        if assumes:
            md.extend(["| Date | tradi.py | Site Divinum Officium | Raison |",
                       "|---|---|---|---|",
                       *(fmt_site(x[:3]) + f" {x[3]} |" for x in assumes)])
        else:
            md.append("Aucun.")
        md.append("")
        section("Contrôle indépendant : référence non lue sur le site", sans_site,
                lambda x: f"- {x[0]} : {x[1] or 'titre non lu'}"
                          + (f" ({x[2]})" if x[2] else ""))

    section("Sanctoral : nom d'origine Divinum Officium (souvent latin)", noms_origine,
            lambda x: f"- {x[0]} : {x[1].nom_messe} (`{x[1].fichier}`)")
    md.extend(["## Détail jour par jour", "",
               "| Date | Messe | Évangile | Langue | Fichier / notes |",
               "|---|---|---|---|---|", *lignes, ""])

    sortie.write_text("\n".join(md), encoding="utf-8")
    bilan_site = "" if controle is None else \
        f", site DO : {concordants} concordants / {len(assumes)} écarts assumés / " \
        f"{len(ecarts_site)} écarts non expliqués / {len(sans_site)} non lus"
    print(f"{nb} jours : {len(erreurs)} erreurs, {len(non_resolus)} renvois non "
          f"résolus, {len(latin)} replis latins, {len(reprises)} reprises, "
          f"{len(corrections)} corrections, {len(ecarts_fr_la)} écarts FR/LA, "
          f"{len(autres)} autres incidents"
          f"{bilan_site} → {sortie}")


if __name__ == "__main__":
    main()
