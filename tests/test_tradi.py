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

    def test_ferie_du_temps_pascal_n_est_pas_reprise(self):
        # Hors Épiphanie, Pentecôte et Avent : erreur explicite, pas de reprise.
        with self.assertRaises(tradi.TradiError):
            self.evangile("2027-04-06")

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
        ref, paragraphes, latin = m.evangile_fichier("Sancti/01-01")
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
        ref, paragraphes, latin = m.evangile_fichier("Sancti/01-01")
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
        _, paragraphes, latin = m.evangile_fichier("Sancti/01-01")
        self.assertEqual(paragraphes, ["Vous êtes."])
        self.assertFalse(latin)

    def test_vide_relatif_au_dossier_courant(self):
        m = self.moteur({
            "missa/Latin/Tempora/Quadp2-3.txt": "[Rule]\nvide Quadp2-0;\n",
            "missa/Latin/Tempora/Quadp2-0.txt": "[Evangelium]\nS\n!Luc 8:4-15\nSemen.\n",
        })
        self.assertEqual(m.evangile_fichier("Tempora/Quadp2-3")[0], "!Luc 8:4-15")

    def test_incoherence_de_reference_prefere_le_latin(self):
        m = self.moteur({
            "missa/Latin/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 16:19-31\nHomo quidam.\n",
            "missa/Francais/Sancti/01-01.txt": "[Evangelium]\nS\n!Luc 6:19-31\nUn homme.\n",
        })
        _, paragraphes, latin = m.evangile_fichier("Sancti/01-01")
        self.assertTrue(latin)
        self.assertEqual(paragraphes, ["Homo quidam."])

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
