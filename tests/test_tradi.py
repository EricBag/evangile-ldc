# -*- coding: utf-8 -*-
"""Tests de la messe traditionnelle (missel de 1962).

Exécution : `python -m unittest discover -s tests`

Les tests « dates » tournent sur les données réellement embarquées
(data/tradi, vendor/missalemeum) ; les tests du parseur utilisent de petits
fichiers écrits dans un dossier temporaire. Aucun accès réseau.
"""

import tempfile
import textwrap
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import tradi


class TestDatesDeReference(unittest.TestCase):
    """Dates → Évangile attendu, forme de 1962."""

    @classmethod
    def setUpClass(cls):
        cls.moteur = tradi.MesseTraditionnelle()

    def evangile(self, iso):
        return self.moteur.evangile_du_jour(date.fromisoformat(iso))

    def verifier(self, iso, reference, fichier=None):
        e = self.evangile(iso)
        self.assertEqual(e.reference, reference, f"{iso} : {e.nom_messe}")
        self.assertFalse(e.latin, f"{iso} : repli latin inattendu")
        self.assertTrue(e.texte.strip())
        if fichier:
            self.assertEqual(e.fichier, fichier)
        return e

    def test_premier_dimanche_de_l_avent(self):
        e = self.verifier("2026-11-29", "Luc 21, 25-33", "Tempora/Adv1-0")
        self.assertEqual(e.nom_messe, "Ier dimanche de l'Avent")

    def test_dimanche_l_emporte_sur_saint_francois(self):
        e = self.verifier("2026-10-04", "Matthieu 22, 1-14", "Tempora/Pent19-0")
        self.assertEqual(e.nom_messe, "XIXe dimanche après la Pentecôte")

    def test_saint_michel(self):
        self.verifier("2026-09-29", "Matthieu 18, 1-10", "Sancti/09-29")

    def test_mercredi_des_cendres(self):
        e = self.verifier("2027-02-10", "Matthieu 6, 16-21")
        self.assertEqual(e.nom_messe, "Mercredi des Cendres")

    def test_paques(self):
        self.verifier("2027-03-28", "Marc 16, 1-7", "Tempora/Pasc0-0")

    def test_pentecote(self):
        self.verifier("2027-05-16", "Jean 14, 23-31", "Tempora/Pasc7-0")

    def test_christ_roi(self):
        e = self.verifier("2026-10-25", "Jean 18, 33-37", "Sancti/10-DU")
        self.assertEqual(e.nom_messe, "Fête du Christ-Roi")

    def test_defunts_premiere_messe(self):
        # Trois messes ce jour-là : la 1re est retenue (choix documenté
        # dans tradi.py).
        e = self.verifier("2026-11-02", "Jean 5, 25-29", "Sancti/11-02m1")
        self.assertEqual(len(e.messes), 3)

    def test_immaculee_conception(self):
        self.verifier("2026-12-08", "Luc 1, 26-28", "Sancti/12-08")

    def test_noel_messe_du_jour(self):
        # Messe du jour (3e messe) parmi les trois messes de Noël.
        e = self.verifier("2026-12-25", "Jean 1, 1-14", "Sancti/12-25m3")
        self.assertEqual(len(e.messes), 3)
        self.assertIn("messe du jour", e.nom_messe)

    def test_ferie_apres_la_pentecote_reprend_le_dimanche(self):
        # Mardi 6 juillet 2027, férie sans fête : Évangile du VIIe dimanche.
        e = self.verifier("2027-07-06", "Matthieu 7, 15-21", "Tempora/Pent07-2")
        self.assertEqual(e.reprise_dimanche, "Tempora/Pent07-0")
        dimanche = self.evangile("2027-07-04")
        self.assertEqual(e.texte, dimanche.texte)

    def test_ferie_de_l_avent_reprend_le_dimanche(self):
        e = self.verifier("2026-12-01", "Luc 21, 25-33")
        self.assertEqual(e.reprise_dimanche, "Tempora/Adv1-0")

    def test_ferie_du_temps_pascal_reprend_le_dimanche(self):
        # Mardi après le dimanche in albis : Évangile de ce dimanche.
        e = self.verifier("2027-04-06", "Jean 20, 19-31", "Tempora/Pasc1-2")
        self.assertEqual(e.reprise_dimanche, "Tempora/Pasc1-0")

    def test_lundi_des_rogations_messe_propre(self):
        e = self.verifier("2027-05-03", "Luc 11, 5-13", "Tempora/Pasc5-1")
        self.assertIsNone(e.reprise_dimanche)

    def test_vigile_de_la_pentecote_messe_propre(self):
        e = self.verifier("2027-05-15", "Jean 14, 15-21")
        self.assertIsNone(e.reprise_dimanche)

    def test_ferie_du_temps_de_noel_messe_de_l_octave(self):
        # 2 au 5 janvier : messe de l'octave de Noël (Luc 2, 21).
        e = self.verifier("2027-01-04", "Luc 2, 21")
        self.assertEqual(e.reprise_dimanche, "Sancti/01-01")

    def test_mardi_saint_commence_a_gethsemani(self):
        e = self.verifier("2027-03-23", "Marc 14, 32-72; 15, 1-46", "Tempora/Quad6-2")
        self.assertTrue(e.texte.startswith("Ils arrivent en un domaine appelé Gethsémani"))

    def test_saint_joseph_un_vendredi_de_la_passion(self):
        # Correctif du calendrier : la fête de 1re classe l'emporte sur la
        # férie (3e classe) du vendredi de la Passion.
        self.verifier("2027-03-19", "Matthieu 1, 18-21", "Sancti/03-19")

    def test_martyr_au_temps_pascal_commun_pascal(self):
        # Saint Fidèle (vide C2a) au temps pascal : commun C2ap, Jean 15, 1-7.
        self.verifier("2027-04-24", "Jean 15, 1-7", "Sancti/04-24")

    def test_ancienne_octave_de_la_fete_dieu_ignoree(self):
        # Pent02-3 garde « ex Tempora/Pent01-4 » dans [Rule], mais son [Rank]
        # de 1960 est une simple férie : reprise du IIe dimanche après la
        # Pentecôte (Luc 14, 16-24), comme sur le site Divinum Officium.
        e = self.verifier("2027-06-02", "Luc 14, 16-24", "Tempora/Pent02-3")
        self.assertEqual(e.reprise_dimanche, "Tempora/Pent02-0")

    def test_coquille_francaise_corrigee(self):
        # « Luc 6, 19-31 » dans le fichier français : corrigé par
        # data/tradi/corrections.json, texte français conservé.
        e = self.verifier("2027-02-25", "Luc 16, 19-31", "Tempora/Quad2-4")
        self.assertIsNone(e.reference_latine)
        self.assertEqual(len(e.corrections), 1)
        self.assertIn("Lazare", e.texte)

    def test_coquille_latine_tracee(self):
        # « Jean 21, 15-10 » dans le fichier latin : l'affichage (français)
        # était déjà juste, la correction est tracée.
        e = self.verifier("2027-06-28", "Jean 21, 15-19", "Sancti/06-28r")
        self.assertIsNone(e.reference_latine)
        self.assertEqual(e.corrections,
                         ["Sancti/06-28r (Latin) : Jean 21, 15-10 → Jean 21, 15-19"])

    def test_rameaux_passion_en_francais(self):
        # Texte français extrait d'Evangelium2 de Quad6-0 (Matthieu 26, 1 –
        # 27, 60 d'un seul tenant), à partir du verset 26, 36.
        e = self.verifier("2027-03-21", "Matthieu 26, 36-75; 27, 1-60", "Tempora/Quad6-0r")
        self.assertTrue(e.texte.startswith("Alors Jésus arriva avec eux dans une "
                                           "propriété appelée Gethsémani"))
        self.assertTrue(e.texte.endswith("il s’en alla."))

    def test_saint_martin_pape_en_francais(self):
        # Règle française « vide C4D » (fichier inexistant) : la règle latine
        # « vide C2b-1 » est suivie, commun lu en français.
        self.verifier("2027-11-12", "Matthieu 16, 13-19", "Sancti/11-12")

    def test_saint_didace_troisieme_classe(self):
        # Surcharge de rang (calendrier de 1960) : IIIe classe, l'emporte sur
        # la messe de la Vierge le samedi (IVe classe) retenue par Missale Meum.
        e = self.verifier("2027-11-13", "Luc 12, 32-34", "Sancti/11-13")
        self.assertIn("Didace", e.nom_messe)

    def test_mont_carmel_reste_une_commemoraison(self):
        e = self.verifier("2027-07-16", "Luc 16, 1-9", "Tempora/Pent08-5")
        self.assertEqual(e.reprise_dimanche, "Tempora/Pent08-0")

    def test_octave_de_noel(self):
        # Jours dans l'octave : Évangile de la messe de l'aurore (Luc 2, 15-20).
        e = self.verifier("2026-12-29", "Luc 2, 15-20", "Tempora/Nat29")
        self.assertIsNone(e.reprise_dimanche)

    def test_vendredi_saint_passion_selon_saint_jean(self):
        e = self.verifier("2027-03-26", "Jean 18, 1-40; 19, 1-42")
        self.assertGreater(len(e.texte), 5000)

    def test_etat_pour_api(self):
        etat = self.moteur.etat_pour_api("2026-10-04")
        self.assertIsNone(etat["error"])
        self.assertTrue(etat["text"].startswith("ÉVANGILE (Matthieu 22, 1-14)\n\n"))
        self.assertEqual(etat["messe"], "XIXe dimanche après la Pentecôte")
        self.assertIn("traduction non identifiée", etat["source"])
        self.assertFalse(etat["latin"])

    def test_etat_pour_api_erreur(self):
        with mock.patch.object(self.moteur, "evangile_du_jour",
                               side_effect=tradi.TradiError("pas d'Évangile")):
            etat = self.moteur.etat_pour_api("2027-04-06")
        self.assertEqual(etat["text"], "")
        self.assertTrue(etat["error"])


def ecrire(racine, chemin, contenu):
    f = Path(racine) / chemin
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(textwrap.dedent(contenu).lstrip(), encoding="utf-8")


class TestParseur(unittest.TestCase):
    """Résolution des renvois, conditions, repli latin, sur données factices."""

    def moteur(self, fichiers):
        tmp = tempfile.mkdtemp()
        for chemin, contenu in fichiers.items():
            ecrire(tmp, chemin, contenu)
        return tradi.MesseTraditionnelle(Path(tmp))

    def test_renvois_en_chaine(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\n@Sancti/01-02\n",
            "missa/Latin/Sancti/01-02.txt": "[Evangelium]\n@Commune/C4:Evangelium\n",
            "horas/Latin/Commune/C4.txt": "[Evangelium]\nSequentia\n!Matt 25:14-23\nIn illo tempore.\n",
            "missa/Francais/Sancti/01-01.txt": "[Evangelium]\n@Sancti/01-02\n",
            "missa/Francais/Sancti/01-02.txt": "[Evangelium]\n@Commune/C4:Evangelium\n",
            "horas/Francais/Commune/C4.txt": "[Evangelium]\nSuite\n!Mt 25:14-23\nEn ce temps-là.\n",
        })
        ref, paragraphes, latin = m.evangile_fichier("Sancti/01-01")[:3]
        self.assertEqual(tradi.formater_reference(ref), "Matthieu 25, 14-23")
        self.assertEqual(paragraphes, ["En ce temps-là."])
        self.assertFalse(latin)

    def test_renvoi_circulaire_detecte(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\n@Sancti/01-02\n",
            "missa/Latin/Sancti/01-02.txt": "[Evangelium]\n@Sancti/01-01\n",
        })
        with self.assertRaises(tradi.TradiError):
            m.evangile_fichier("Sancti/01-01")

    def test_renvoi_non_resolu_signale(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\nSequentia\n!Luc 1:1-4\nTexte.\n@Sancti/99-99\n",
        })
        m.evangile_fichier("Sancti/01-01")
        self.assertTrue(any("non résolu" in i for i in m.incidents))

    def test_repli_latin_si_traduction_absente(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\nSequentia\n!Joann 1:1-5\nIn principio.\n",
            "missa/Francais/Sancti/01-01.txt": "[Officium]\nSaint X\n",
        })
        _, paragraphes, latin = m.evangile_fichier("Sancti/01-01")[:3]
        self.assertTrue(latin)
        self.assertEqual(paragraphes, ["In principio."])

    def test_heritage_lu_dans_le_latin(self):
        # Le fichier français n'a pas de [Rule] : la règle « vide » du latin
        # s'applique, et le commun est lu en français.
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Rule]\nvide C4;\n",
            "missa/Francais/Sancti/01-01.txt": "[Officium]\nSaint X\n",
            "horas/Latin/Commune/C4.txt": "[Evangelium]\nS\n!Matt 5:13-19\nVos estis.\n",
            "horas/Francais/Commune/C4.txt": "[Evangelium]\nS\n!Matt 5:13-19\nVous êtes.\n",
        })
        _, paragraphes, latin = m.evangile_fichier("Sancti/01-01")[:3]
        self.assertEqual(paragraphes, ["Vous êtes."])
        self.assertFalse(latin)

    def test_regle_vers_fichier_inexistant_suit_le_latin(self):
        m = self.moteur({
            "missa/Latin/Sancti/11-12.txt": "[Rule]\nvide C2b-1;\n",
            "missa/Francais/Sancti/11-12.txt": "[Rule]\nvide C4D\n",
            "horas/Latin/Commune/C2b-1.txt": "[Evangelium]\nS\n!Matt 16:13-19\nVenit Jesus.\n",
            "horas/Francais/Commune/C2b-1.txt": "[Evangelium]\nS\n!Matt 16:13-19\nJésus vint.\n",
        })
        r = m.evangile_fichier("Sancti/11-12")
        self.assertEqual(r.paragraphes, ["Jésus vint."])
        self.assertFalse(r.latin)
        self.assertTrue(any("C4D" in i for i in m.incidents))

    def test_vide_relatif_au_dossier_courant(self):
        m = self.moteur({
            "missa/Latin/Tempora/Quadp2-3.txt": "[Rule]\nvide Quadp2-0;\n",
            "missa/Latin/Tempora/Quadp2-0.txt": "[Evangelium]\nS\n!Luc 8:4-15\nSemen.\n",
        })
        self.assertEqual(m.evangile_fichier("Tempora/Quadp2-3")[0], "!Luc 8:4-15")

    def test_ecart_de_reference_garde_le_francais(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 16:19-31\nHomo quidam.\n",
            "missa/Francais/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 6:19-31\nUn homme.\n",
        })
        r = m.evangile_fichier("Sancti/01-01")
        self.assertFalse(r.latin)
        self.assertEqual(r.paragraphes, ["Un homme."])
        self.assertEqual(r.reference, "!Luc 6:19-31")
        self.assertEqual(r.reference_latine, "!Luc 16:19-31")
        self.assertTrue(any("écart de référence" in i for i in m.incidents))

    def test_table_de_corrections(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 16:19-31\nHomo quidam.\n",
            "missa/Francais/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 6:19-31\nUn homme.\n",
            "corrections.json": """[{"fichier": "Sancti/01-01", "langue": "Francais",
                "reference_erronee": "Luc 6, 19-31", "reference_corrigee": "Luc 16, 19-31",
                "justification": "test"}]""",
        })
        r = m.evangile_fichier("Sancti/01-01")
        self.assertEqual(tradi.formater_reference(r.reference), "Luc 16, 19-31")
        self.assertEqual(r.paragraphes, ["Un homme."])
        self.assertIsNone(r.reference_latine)
        self.assertEqual(len(r.corrections), 1)
        self.assertFalse(any("écart de référence" in i for i in m.incidents))

    def test_decoupe_du_texte_francais(self):
        latin = "[Evangelium]\nS\n!Marc 14:32-72; 15, 1-46\nEt veniunt.\n"
        m = self.moteur({
            "missa/Latin/Tempora/Quad6-2.txt": latin,
            "missa/Francais/Tempora/Quad6-2.txt":
                "[Evangelium]\nS\n!Marc 14:32-72; 15, 1-46\nDébut. "
                "Ils arrivent en un domaine appelé Gethsémani, et la suite.\nFin.\n",
        })
        r = m.evangile_fichier("Tempora/Quad6-2")
        self.assertEqual(r.paragraphes,
                         ["Ils arrivent en un domaine appelé Gethsémani, et la suite.", "Fin."])
        self.assertIsNone(r.reference_latine)

    def test_decoupe_impossible_affiche_la_reference_reelle(self):
        m = self.moteur({
            "missa/Latin/Tempora/Quad6-2.txt": "[Evangelium]\nS\n!Marc 14:32-72; 15, 1-46\nEt.\n",
            "missa/Francais/Tempora/Quad6-2.txt":
                "[Evangelium]\nS\n!Marc 14:32-72; 15, 1-46\nTexte sans le marqueur.\n",
        })
        r = m.evangile_fichier("Tempora/Quad6-2")
        self.assertEqual(tradi.formater_reference(r.reference), "Marc 14, 1-72; 15, 1-46")
        self.assertEqual(r.paragraphes, ["Texte sans le marqueur."])

    def test_variante_de_section_1960(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": textwrap.dedent("""\
                [Evangelium]
                S
                !Matt 1:1-2
                Ancien.

                [Evangelium] (rubrica 1960)
                S
                !Matt 3:1-2
                Nouveau.
                """),
        })
        self.assertEqual(m.evangile_fichier("Sancti/01-01")[1], ["Nouveau."])

    def test_regles_de_reprise_du_temps_pascal(self):
        m = self.moteur({
            "missa/Latin/Tempora/Pasc2-0.txt": "[Evangelium]\nS\n!Joann 10:11-16\nT.\n",
            "missa/Latin/Tempora/Pasc2-3.txt": "[Officium]\nFeria IV\n",
            "missa/Latin/Tempora/Pasc5-4.txt": "[Evangelium]\nS\n!Marc 16:14-20\nT.\n",
            "missa/Latin/Tempora/Pasc5-5.txt": "[Officium]\nFeria VI\n",
            "missa/Latin/Tempora/Pasc5-1.txt": "[Officium]\nRogationes\n",
            "missa/Latin/Tempora/Pasc6-6.txt": "[Officium]\nVigilia\n",
            "missa/Latin/Tempora/Pasc0-2.txt": "[Officium]\nFeria III in octava\n",
            "missa/Latin/Tempora/Quad2-3.txt": "[Officium]\nFeria IV\n",
        })

        def office(nom):
            obs = mock.Mock(flexibility="tempora", id=f"tempora:{nom}:4:w")
            obs.name = nom
            return obs

        def evangile(nom):
            with mock.patch.object(m, "office_du_jour", return_value=(office(nom), [])):
                return m.evangile_du_jour(date(2027, 5, 7))

        self.assertEqual(evangile("Pasc2-3").reprise_dimanche, "Tempora/Pasc2-0")
        self.assertEqual(evangile("Pasc2-3Feria").reprise_dimanche, "Tempora/Pasc2-0")
        self.assertEqual(evangile("Pasc5-5").reprise_dimanche, "Tempora/Pasc5-4")
        # Messes propres (Rogations, vigile de la Pentecôte), octave de Pâques,
        # Carême : jamais de reprise.
        for nom in ("Pasc5-1", "Pasc6-6", "Pasc0-2", "Quad2-3"):
            with self.assertRaises(tradi.TradiError, msg=nom):
                evangile(nom)

    def test_ferie_des_quatre_temps_sans_evangile_leve_une_erreur(self):
        m = self.moteur({
            "missa/Latin/Tempora/Adv3-3.txt": "[Officium]\nFeria IV\n",
            "missa/Latin/Tempora/Adv3-0.txt": "[Evangelium]\nS\n!Joann 1:19-28\nT.\n",
            "missa/Latin/Tempora/Pent05-3.txt": "[Officium]\nFeria IV\n",
            "missa/Latin/Tempora/Pent05-0.txt": "[Evangelium]\nS\n!Luc 5:1-11\nT.\n",
        })

        def office(nom):
            obs = mock.Mock(flexibility="tempora", id=f"tempora:{nom}:4:v")
            obs.name = nom
            return obs

        with mock.patch.object(m, "office_du_jour", return_value=(office("Adv3-3"), [])):
            with self.assertRaises(tradi.TradiError):
                m.evangile_du_jour(date(2026, 12, 16))
        with mock.patch.object(m, "office_du_jour", return_value=(office("Pent05-3"), [])):
            e = m.evangile_du_jour(date(2027, 6, 23))
            self.assertEqual(e.reprise_dimanche, "Tempora/Pent05-0")


class TestConditionsEtReferences(unittest.TestCase):

    def test_condition_vraie(self):
        self.assertTrue(tradi.condition_vraie("rubrica 196"))
        self.assertTrue(tradi.condition_vraie("rubrica 1955 aut rubrica 1960"))
        self.assertTrue(tradi.condition_vraie("communi Summorum Pontificum"))
        self.assertFalse(tradi.condition_vraie("rubrica 1570"))
        self.assertFalse(tradi.condition_vraie("nisi rubrica 196"))
        self.assertFalse(tradi.condition_vraie("sed non rubrica 1960"))

    def test_conditions_en_ligne_de_la_passion(self):
        lignes = [
            "(rubrica 1955 aut rubrica 1960 dicitur)",
            "!Marc 14:32-72",
            "(rubrica 1570 aut rubrica 1910 aut rubrica divino afflatu dicitur)",
            "!Marc 14:1-72",
            "(deinde dicuntur)",
            "Version longue.",
            "(sed rubrica 1955 aut rubrica 1960 hæc versus omittuntur)",
            "",
            "Gethsémani.  (deinde dicuntur)",
        ]
        self.assertEqual(tradi.appliquer_conditions(lignes),
                         ["!Marc 14:32-72", "", "Gethsémani."])

    def test_sed_rubrica_ancienne_ignoree(self):
        lignes = ["@Commune/C3a-1", "(sed rubrica 1570)", "@Commune/C3a"]
        self.assertEqual(tradi.appliquer_conditions(lignes), ["@Commune/C3a-1"])

    def test_formater_reference(self):
        self.assertEqual(tradi.formater_reference("!Matt 22:1-14"), "Matthieu 22, 1-14")
        self.assertEqual(tradi.formater_reference("!Joannes 14:23-31"), "Jean 14, 23-31")
        self.assertEqual(tradi.formater_reference("!Luc 21:25-33."), "Luc 21, 25-33")
        self.assertEqual(tradi.formater_reference("!Matt, 28. 1-7"), "Matthieu 28, 1-7")
        self.assertEqual(tradi.formater_reference("!Marc 14:32-72; 15, 1-46"),
                         "Marc 14, 32-72; 15, 1-46")

    def test_noms_du_temporal(self):
        self.assertEqual(tradi.nom_tempora("Pent19-0"), "XIXe dimanche après la Pentecôte")
        self.assertEqual(tradi.nom_tempora("Pent03-2Feria"),
                         "Mardi après le IIIe dimanche après la Pentecôte")
        self.assertEqual(tradi.nom_tempora("Quad1-0"), "Ier dimanche de Carême")
        self.assertEqual(tradi.nom_tempora("Pasc2-0"), "IIe dimanche après Pâques")


if __name__ == "__main__":
    unittest.main()
