# -*- coding: utf-8 -*-
"""Tests du chunking du « Petit Journal » (faustine_parser).

Exécution : `python -m unittest discover -s tests`

Aucun accès réseau ni au PDF : les cas sont construits à la main, sur des lignes
reproduisant les artefacts réels du document capturé.

`ldc_proZ` instancie un client OpenAI à l'import ; une clé factice suffit à le
satisfaire, aucun appel n'étant émis ici.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "cle-factice-pour-les-tests")

import faustine_parser as fp  # noqa: E402
from ldc_proZ import SOURCE_FAUSTINE  # noqa: E402


def texte_long(marque: str, longueur: int = 400) -> str:
    """Un corps de paragraphe assez long pour ne pas déclencher le regroupement."""
    return (marque + " ") * (longueur // (len(marque) + 1) + 1)


class TestNettoyage(unittest.TestCase):

    def test_boilerplate_ecarte(self):
        lignes = fp.nettoyer_lignes([
            "Retour",
            "Ste Faustine",
            "édition numérique par Anne Speeckaert et www.JesusMarie.com",
            "END CONTAINER",
            "Texte véritable du Journal.",
        ])
        self.assertEqual(lignes, ["Texte véritable du Journal."])

    def test_pied_de_page_recompose_ecarte(self):
        lignes = fp.nettoyer_lignes([
            "0981 Faustine,",
            "751 Faustine ch.",
            "0020 Faustine cahier 2,",
            "Texte véritable.",
        ])
        self.assertEqual(lignes, ["Texte véritable."])

    def test_compteurs_seuls_supprimes(self):
        lignes = fp.nettoyer_lignes([
            "Première ligne de texte.",
            "12",
            "13",
            "14",
            "Seconde ligne de texte.",
        ])
        self.assertEqual(lignes,
                         ["Première ligne de texte.", "Seconde ligne de texte."])

    def test_compteurs_en_prefixe_retires(self):
        lignes = fp.nettoyer_lignes([
            "281 444. Jeudi. L’adoration nocturne.",
            "282 Quand je suis venue pour adorer,",
            "283 un recueillement intérieur me saisit.",
        ])
        self.assertEqual(lignes, [
            "444. Jeudi. L’adoration nocturne.",
            "Quand je suis venue pour adorer,",
            "un recueillement intérieur me saisit.",
        ])

    def test_nombre_isole_reste_du_texte(self):
        """Une suite trop courte n'est pas un compteur : le nombre est du texte."""
        lignes = fp.nettoyer_lignes([
            "Un jour de fête.",
            "12 août, je priais dans ma cellule.",
            "Puis je suis sortie.",
        ])
        self.assertIn("12 août, je priais dans ma cellule.", lignes)

    def test_marqueur_de_paragraphe_jamais_pris_pour_un_compteur(self):
        """« 296 . » porte un point : c'est un §, pas un compteur, même en suite."""
        lignes = fp.nettoyer_lignes([
            "294 . Texte du paragraphe 294.",
            "295 . Texte du paragraphe 295.",
            "296 . Mon Bien suprême, je désire Vous aimer.",
        ])
        self.assertEqual(len(lignes), 3)
        self.assertTrue(lignes[2].startswith("296 ."))


class TestDetectionDesMarqueurs(unittest.TestCase):

    def test_marqueurs_simples(self):
        lignes = ["1. Alpha.", "2. Beta.", "3. Gamma."]
        self.assertEqual([n for _, n, _ in fp.detecter_marqueurs(lignes)],
                         [1, 2, 3])

    def test_bloc_duplique_ignore(self):
        """Le rappel d'un § déjà vu ne doit pas rompre la progression."""
        lignes = ["320. Alpha.", "321. Beta.", "321. Beta (rappel).", "322. Gamma."]
        retenus = fp.detecter_marqueurs(lignes)
        self.assertEqual([n for _, n, _ in retenus], [320, 321, 322])
        self.assertEqual([i for i, _, _ in retenus], [0, 1, 3])

    def test_marqueur_seul_sur_sa_ligne(self):
        lignes = ["999.", "Le texte commence à la ligne suivante.", "1000. Suite."]
        self.assertEqual([n for _, n, _ in fp.detecter_marqueurs(lignes)],
                         [999, 1000])

    def test_comblement_dun_trou_par_marqueur_degrade(self):
        """« 166 Je trouve… », sans point, comble le trou entre 165 et 167."""
        lignes = [
            "165. Texte du paragraphe cent soixante-cinq.",
            "166 Je trouve toujours lumière et force dans la prière.",
            "167. Texte du paragraphe cent soixante-sept.",
        ]
        self.assertEqual([n for _, n, _ in fp.detecter_marqueurs(lignes)],
                         [165, 166, 167])

    def test_nombre_hors_trou_non_promu(self):
        """Un nombre en tête de phrase hors de tout trou reste du texte."""
        lignes = [
            "165. Texte du paragraphe cent soixante-cinq.",
            "166. Texte du paragraphe cent soixante-six.",
            "900 Jours plus tard, je repris courage.",
            "167. Texte du paragraphe cent soixante-sept.",
        ]
        self.assertEqual([n for _, n, _ in fp.detecter_marqueurs(lignes)],
                         [165, 166, 167])

    def test_numeros_hors_bornes_rejetes(self):
        """Une date en tête de ligne dépasse la borne du numérotage."""
        lignes = ["1. Alpha.", "1934. Le jour de l’Assomption.", "2. Beta."]
        self.assertEqual([n for _, n, _ in fp.detecter_marqueurs(lignes)], [1, 2])


class TestConstructionDesParagraphes(unittest.TestCase):

    def test_paragraphe_isole(self):
        lignes = ["742. " + texte_long("deux résolutions générales")]
        paras = fp.construire_paragraphes(lignes)
        self.assertEqual(len(paras), 1)
        self.assertEqual((paras[0].num_debut, paras[0].num_fin), (742, 742))
        self.assertTrue(paras[0].text.startswith("deux résolutions"))

    def test_numeros_absents_absorbes_dans_la_plage(self):
        """Le passage de 68. à 71. laisse 69 et 70 dans la plage du §68."""
        lignes = ["68. " + texte_long("souffrance"), "71. " + texte_long("plock")]
        paras = fp.construire_paragraphes(lignes)
        self.assertEqual((paras[0].num_debut, paras[0].num_fin), (68, 70))
        self.assertEqual((paras[1].num_debut, paras[1].num_fin), (71, 71))

    def test_paragraphe_court_regroupe_avec_le_suivant(self):
        lignes = ["10. Trop court.", "11. " + texte_long("suite")]
        paras = fp.construire_paragraphes(lignes)
        self.assertEqual(len(paras), 1)
        self.assertEqual((paras[0].num_debut, paras[0].num_fin), (10, 11))
        self.assertTrue(paras[0].text.startswith("Trop court. "))

    def test_dernier_paragraphe_court_rattache_au_precedent(self):
        lignes = ["10. " + texte_long("corps"), "11. Trop court."]
        paras = fp.construire_paragraphes(lignes)
        self.assertEqual(len(paras), 1)
        self.assertEqual((paras[0].num_debut, paras[0].num_fin), (10, 11))
        self.assertTrue(paras[0].text.endswith("Trop court."))

    def test_regroupement_desactivable(self):
        lignes = ["10. Trop court.", "11. " + texte_long("suite")]
        paras = fp.construire_paragraphes(lignes, min_chars=0)
        self.assertEqual([(p.num_debut, p.num_fin) for p in paras],
                         [(10, 10), (11, 11)])

    def test_cahier_courant_conserve(self):
        lignes = ["Cahier III", "742. " + texte_long("corps")]
        paras = fp.construire_paragraphes(lignes)
        self.assertEqual(paras[0].cahier, "III")

    def test_numeros_de_page_exclus_du_texte(self):
        lignes = ["742. " + texte_long("corps"), "0853", "suite du paragraphe."]
        paras = fp.construire_paragraphes(lignes)
        self.assertNotIn("0853", paras[0].text)
        self.assertTrue(paras[0].text.endswith("suite du paragraphe."))

    def test_matiere_editoriale_finale_retranchee(self):
        lignes = ["1827. " + texte_long("silence intérieur")
                  + " Concordat cum originali Cracovie, 18 septembris 1968"]
        paras = fp.construire_paragraphes(lignes)
        self.assertNotIn("Concordat", paras[0].text)
        self.assertNotIn("1968", paras[0].text)


class TestSegmentation(unittest.TestCase):

    def test_source_et_offset_des_identifiants(self):
        paras = fp.construire_paragraphes(["742. " + texte_long("corps")])
        segments = fp.segmenter_paragraphes(paras, id_offset=1000)
        self.assertTrue(segments)
        self.assertEqual(segments[0].id, 1000)
        self.assertTrue(all(s.source == SOURCE_FAUSTINE for s in segments))

    def test_identifiants_consecutifs(self):
        paras = fp.construire_paragraphes(
            ["10. " + texte_long("alpha"), "11. " + texte_long("beta")])
        segments = fp.segmenter_paragraphes(paras, id_offset=7)
        self.assertEqual([s.id for s in segments],
                         list(range(7, 7 + len(segments))))

    def test_paragraphe_long_decoupe_mais_cite_dune_seule_piece(self):
        """Plusieurs segments, mais tous rattachés au même paragraphe."""
        lignes = ["742. " + " ".join(f"mot{i}" for i in range(900))]
        paras = fp.construire_paragraphes(lignes)
        segments = fp.segmenter_paragraphes(paras)
        self.assertGreater(len(segments), 1)
        self.assertEqual({s.dictee_index for s in segments}, {0})

    def test_tokens_renseignes(self):
        paras = fp.construire_paragraphes(["742. " + texte_long("Miséricorde")])
        segment = fp.segmenter_paragraphes(paras)[0]
        self.assertIn("misericorde", segment.tokens)


class TestNumerotationDeReference(unittest.TestCase):
    """Le PDF décroche d'une unité au-delà d'un point de bascule."""

    #: (numéro imprimé dans le PDF, numéro de référence attendu).
    #: Repères sans décalage, puis repères décalés de +1.
    REPERES = ((300, 300), (309, 309), (310, 310),
               (318, 319), (419, 420), (474, 475), (698, 699),
               (1145, 1146), (1540, 1541), (1827, 1828))

    def test_reperes(self):
        for num_pdf, num_ref in self.REPERES:
            self.assertEqual(fp.numero_reference(num_pdf), num_ref,
                             f"§ {num_pdf} du PDF")

    def test_avant_la_bascule_aucun_decalage(self):
        self.assertEqual(fp.numero_reference(fp.BASCULE_NUM_REF - 1),
                         fp.BASCULE_NUM_REF - 1)

    def test_a_partir_de_la_bascule_decalage_de_un(self):
        self.assertEqual(fp.numero_reference(fp.BASCULE_NUM_REF),
                         fp.BASCULE_NUM_REF + 1)

    def test_bascule_dans_lintervalle_impose_par_les_reperes(self):
        """Les repères 310→310 et 318→319 encadrent étroitement la bascule."""
        self.assertGreater(fp.BASCULE_NUM_REF, 310)
        self.assertLessEqual(fp.BASCULE_NUM_REF, 318)

    def test_zone_incertaine_reduite_au_trou_de_numerotation(self):
        """Seuls les § 311 à 317 du PDF dépendent encore du point de bascule."""
        self.assertEqual(fp.ZONE_NUM_INCERTAIN, (311, 317))
        for certain in (300, 309, 310, 318, 319, 419):
            self.assertFalse(fp.numero_incertain(certain), f"§ {certain}")
        for incertain in (311, 314, 317):
            self.assertTrue(fp.numero_incertain(incertain), f"§ {incertain}")

    def test_aucun_repere_nest_en_zone_incertaine(self):
        """Un repère vérifié ne doit jamais être affiché comme approximatif."""
        for num_pdf, _num_ref in self.REPERES:
            self.assertFalse(fp.numero_incertain(num_pdf), f"§ {num_pdf} du PDF")

    def test_champs_renseignes_a_la_construction(self):
        lignes = ["742. " + texte_long("corps")]
        para = fp.construire_paragraphes(lignes)[0]
        self.assertEqual(para.num_ref_debut, 743)
        self.assertEqual(para.num_ref_fin, 743)
        self.assertFalse(para.num_incertain)

    def test_plage_reportee_sur_la_numerotation_de_reference(self):
        lignes = ["68. " + texte_long("alpha"), "71. " + texte_long("beta")]
        para = fp.construire_paragraphes(lignes)[0]
        self.assertEqual((para.num_debut, para.num_fin), (68, 70))
        self.assertEqual((para.num_ref_debut, para.num_ref_fin), (68, 70))

    def test_paragraphe_couvrant_le_trou_marque_incertain(self):
        """Le § 310 absorbe le trou 311–317 : sa plage dépend de la bascule."""
        lignes = ["310. " + texte_long("corps"), "318. " + texte_long("suite")]
        para = fp.construire_paragraphes(lignes)[0]
        self.assertEqual((para.num_debut, para.num_fin), (310, 317))
        self.assertTrue(para.num_incertain)
        self.assertEqual(para.num_ref_debut, 310)     # 310 < 311 : pas de décalage
        self.assertEqual(para.num_ref_fin, 318)       # 317 >= 311 : décalé

    def test_plage_enjambant_le_trou_reste_certaine(self):
        """Enjamber le trou ne rend pas la citation approximative.

        Une plage 305–319 englobe la zone floue, mais ses deux bornes valent
        305 et 320 quel que soit le point de bascule : ce sont elles que l'on
        cite, et elles sont déterminées. Seules les bornes comptent.
        """
        lignes = ["305. " + texte_long("corps"), "320. " + texte_long("suite")]
        para = fp.construire_paragraphes(lignes)[0]
        self.assertEqual((para.num_debut, para.num_fin), (305, 319))
        self.assertFalse(para.num_incertain)
        self.assertEqual((para.num_ref_debut, para.num_ref_fin), (305, 320))


class TestRapportDeCouverture(unittest.TestCase):

    def test_plages_comptees_comme_couvertes(self):
        lignes = ["68. " + texte_long("alpha"), "71. " + texte_long("beta")]
        rapport = fp.rapport_couverture(fp.construire_paragraphes(lignes))
        self.assertEqual(rapport["paragraphes"], 2)
        self.assertEqual(rapport["numeros_couverts"], 4)     # 68, 69, 70, 71
        self.assertEqual(rapport["plages_fusionnees"], 1)
        self.assertNotIn(69, rapport["numeros_absents"])
        self.assertIn(1, rapport["numeros_absents"])


CACHE_INDEX = "ldc_index_word"
INDEX_PRESENT = os.path.exists(
    os.path.join(CACHE_INDEX, fp.FICHIER_PARAGRAPHES))


@unittest.skipUnless(INDEX_PRESENT,
                     "corpus Faustine non ingéré (ingest_faustine.py)")
class TestCorpusIndexe(unittest.TestCase):
    """Vérifications sur le corpus réellement indexé, pas sur des cas construits."""

    @classmethod
    def setUpClass(cls):
        cls.paragraphes = fp.charger_paragraphes_indexes(CACHE_INDEX)
        cls.par_reference = {}
        for para in cls.paragraphes:
            for num in range(para.num_ref_debut, para.num_ref_fin + 1):
                cls.par_reference.setdefault(num, para)

    def test_les_sept_reperes_sont_couverts(self):
        for _num_pdf, num_ref in TestNumerotationDeReference.REPERES:
            self.assertIn(num_ref, self.par_reference, f"§ {num_ref} de référence")

    def test_699_est_le_message_au_monde_entier(self):
        para = self.par_reference[699]
        self.assertIn("parle au monde entier de mon inconcevable miséricorde",
                      para.text.lower())

    def test_420_est_le_dimanche_de_quasimodo(self):
        self.assertTrue(
            self.par_reference[420].text.startswith("Dimanche de Quasimodo"))

    def test_309_est_lacte_doffrande(self):
        self.assertIn("offrande", self.par_reference[309].text.lower())

    def test_310_est_la_permission_du_confesseur(self):
        self.assertTrue(self.par_reference[310].text.startswith(
            "Quand j’ai reçu de mon confesseur la permission"))

    def test_319_est_ladoration_nocturne_du_9_aout_1934(self):
        texte = self.par_reference[319].text
        self.assertTrue(texte.startswith("9.8.1934."))
        self.assertIn("adoration nocturne", texte.lower())

    def test_un_seul_paragraphe_reste_incertain(self):
        """Le trou 311–317 n'est absorbé que par le § 310 du PDF."""
        incertains = [p for p in self.paragraphes if p.num_incertain]
        self.assertEqual(len(incertains), 1)
        self.assertEqual((incertains[0].num_debut, incertains[0].num_fin),
                         (310, 317))

    def test_1828_est_le_dernier_paragraphe(self):
        self.assertIs(self.par_reference[1828], self.paragraphes[-1])

    def test_numerotation_de_reference_croissante_et_sans_trou(self):
        precedent = 0
        for para in self.paragraphes:
            self.assertEqual(para.num_ref_debut, precedent + 1,
                             f"discontinuité avant le § {para.num_ref_debut}")
            self.assertGreaterEqual(para.num_ref_fin, para.num_ref_debut)
            precedent = para.num_ref_fin

    def test_zone_incertaine_uniquement_marquee(self):
        for para in self.paragraphes:
            attendu = (fp.numero_incertain(para.num_debut)
                       or fp.numero_incertain(para.num_fin))
            self.assertEqual(para.num_incertain, attendu,
                             f"§ {para.num_debut} du PDF")


if __name__ == "__main__":
    unittest.main()
