# -*- coding: utf-8 -*-
"""Tests du filtrage par source, du regroupement et de la citation.

Exécution : `python -m unittest discover -s tests`

Aucun accès réseau : le client OpenAI est remplacé par un double pour la seule
requête d'embedding effectuée par `score_segments_with_keywords`.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "cle-factice-pour-les-tests")

import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

import ldc_proZ as L  # noqa: E402


DIM = 4


def segment(sid, unite, source, texte):
    return L.Segment(id=sid, dictee_index=unite, text=texte,
                     tokens=L.tokenize(texte), source=source)


class _EmbeddingFactice:
    """Double du client OpenAI : renvoie toujours le même vecteur de requête."""

    def __init__(self, vecteur):
        self.vecteur = vecteur
        self.embeddings = self

    def create(self, model, input):
        donnee = type("D", (), {"embedding": list(self.vecteur)})()
        return type("R", (), {"data": [donnee]})()


class BaseCorpus(unittest.TestCase):
    """Corpus minimal : 2 dictées et 2 paragraphes, 2 segments chacun."""

    def setUp(self):
        self.segments = [
            segment(0, 0, L.SOURCE_LUISA, "la divine volonté et le fiat"),
            segment(1, 0, L.SOURCE_LUISA, "le soleil de la volonté divine"),
            segment(2, 1, L.SOURCE_LUISA, "la ronde dans la volonté"),
            segment(3, 0, L.SOURCE_FAUSTINE, "la miséricorde envers les pécheurs"),
            segment(4, 0, L.SOURCE_FAUSTINE, "jésus j ai confiance en vous"),
            segment(5, 1, L.SOURCE_FAUSTINE, "l heure de la miséricorde"),
        ]
        self.dictees = [
            L.Dictee(tome=3, date="12 mars 1930", page_start=0,
                     text="la divine volonté et le fiat. le soleil de la volonté divine."),
            L.Dictee(tome=7, date="4 avril 1926", page_start=0,
                     text="la ronde dans la volonté."),
        ]
        self.paragraphes = [
            L.Paragraphe(num_debut=742, num_fin=742, cahier="II",
                         text="la miséricorde envers les pécheurs. "
                              "jésus j ai confiance en vous.",
                         num_ref_debut=743, num_ref_fin=743),
            L.Paragraphe(num_debut=280, num_fin=295, cahier="I",
                         text="l heure de la miséricorde.",
                         num_ref_debut=280, num_ref_fin=295),
        ]
        self.bm25 = BM25Okapi([s.tokens for s in self.segments])
        # Vecteurs orthogonaux : chaque segment est proche d'un axe distinct.
        self.embs = np.zeros((len(self.segments), DIM), dtype=np.float32)
        for i in range(len(self.segments)):
            self.embs[i, i % DIM] = 1.0

    def scorer(self, sources, mots_cles=None):
        original = L.client
        L.client = _EmbeddingFactice(np.ones(DIM) / np.sqrt(DIM))
        try:
            return L.score_segments_with_keywords(
                "miséricorde et volonté divine", mots_cles or [],
                self.segments, self.bm25, self.embs,
                top_k_segments=10, sources=sources,
            )
        finally:
            L.client = original


class TestNormalisationDesSources(unittest.TestCase):

    def test_defaut_tout_le_corpus(self):
        self.assertEqual(L.normalize_sources(None), L.SOURCES_CONNUES)
        self.assertEqual(L.normalize_sources([]), L.SOURCES_CONNUES)

    def test_ordre_canonique_impose(self):
        self.assertEqual(L.normalize_sources(["faustine", "luisa"]),
                         (L.SOURCE_LUISA, L.SOURCE_FAUSTINE))

    def test_valeur_inconnue_ignoree(self):
        self.assertEqual(L.normalize_sources(["faustine", "inconnue"]),
                         (L.SOURCE_FAUSTINE,))

    def test_demande_entierement_invalide_retombe_sur_tout(self):
        self.assertEqual(L.normalize_sources(["inconnue"]), L.SOURCES_CONNUES)


class TestSelectionParSource(BaseCorpus):

    def test_indices_par_source(self):
        self.assertEqual(
            list(L.select_indices_by_source(self.segments, [L.SOURCE_FAUSTINE])),
            [3, 4, 5])
        self.assertEqual(
            list(L.select_indices_by_source(self.segments, [L.SOURCE_LUISA])),
            [0, 1, 2])

    def test_toutes_sources_conserve_lordre_complet(self):
        self.assertEqual(
            list(L.select_indices_by_source(self.segments, None)),
            list(range(len(self.segments))))

    def test_segment_sans_champ_source_rattache_a_luisa(self):
        """Un segment issu d'un pickle antérieur au champ reste du Livre du Ciel."""
        ancien = self.segments[0]
        del ancien.source
        self.assertIn(0, list(L.select_indices_by_source(
            self.segments, [L.SOURCE_LUISA])))


class TestScoringFiltre(BaseCorpus):

    def test_faustine_seule_ne_remonte_que_faustine(self):
        resultats = self.scorer([L.SOURCE_FAUSTINE])
        self.assertTrue(resultats)
        self.assertTrue(all(seg.source == L.SOURCE_FAUSTINE
                            for _score, seg in resultats))
        self.assertEqual({seg.id for _s, seg in resultats}, {3, 4, 5})

    def test_luisa_seule_ne_remonte_que_luisa(self):
        resultats = self.scorer([L.SOURCE_LUISA])
        self.assertEqual({seg.id for _s, seg in resultats}, {0, 1, 2})

    def test_les_deux_remontent_tout(self):
        resultats = self.scorer(None)
        self.assertEqual({seg.id for _s, seg in resultats}, set(range(6)))

    def test_les_scores_restent_apparies_a_leur_segment(self):
        """Le remappage d'indices après filtrage ne doit pas décaler les scores."""
        complet = {seg.id: score for score, seg in self.scorer(None)}
        for score, seg in self.scorer(None):
            self.assertEqual(complet[seg.id], score)


class TestNormalisationLocale(unittest.TestCase):
    """La normalisation doit porter sur les seuls segments retenus.

    Corpus construit pour que le Livre du Ciel domine très largement la
    composante sémantique. Normalisées globalement, les valeurs Faustine
    resteraient écrasées près de zéro et la pondération 0.45/0.45 ne porterait
    plus sur des grandeurs comparables.
    """

    def setUp(self):
        self.segments = [
            segment(0, 0, L.SOURCE_LUISA, "volonté divine fiat"),
            segment(1, 1, L.SOURCE_LUISA, "volonté divine soleil"),
            segment(2, 0, L.SOURCE_FAUSTINE, "silence intérieur"),
            segment(3, 1, L.SOURCE_FAUSTINE, "miséricorde miséricorde"),
        ]
        self.bm25 = BM25Okapi([s.tokens for s in self.segments])
        requete = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        # Le segment 3 domine Faustine sur les deux composantes à la fois.
        self.embs = np.array(
            [requete, requete, requete * 0.10, requete * 0.11], dtype=np.float32)
        self.requete = requete

    def scorer(self, sources):
        original = L.client
        L.client = _EmbeddingFactice(self.requete)
        try:
            return L.score_segments_with_keywords(
                "miséricorde", [], self.segments, self.bm25, self.embs,
                top_k_segments=10, sources=sources,
            )
        finally:
            L.client = original

    def test_le_meilleur_de_la_source_filtree_atteint_le_maximum(self):
        resultats = dict((seg.id, score) for score, seg in
                         self.scorer([L.SOURCE_FAUSTINE]))
        # 0.45 (lexical) + 0.45 (sémantique), bonus nul faute de mots-clés.
        self.assertAlmostEqual(resultats[3], 0.9, places=6)

    def test_sans_filtre_le_meme_segment_reste_ecrase(self):
        resultats = dict((seg.id, score) for score, seg in self.scorer(None))
        self.assertLess(resultats[3], 0.6)


class TestRegroupementParUnite(BaseCorpus):

    def scores_factices(self):
        return [(1.0 - i * 0.1, seg) for i, seg in enumerate(self.segments)]

    def test_unites_des_deux_corpus_non_confondues(self):
        """Dictée 0 et paragraphe 0 partagent un indice : ils doivent rester distincts."""
        groupes = L.group_segments_by_unit(
            self.scores_factices(), self.dictees, self.paragraphes)
        unites = [unite for _score, unite, _seg in groupes]
        self.assertEqual(len(unites), 4)
        self.assertEqual(sum(1 for u in unites if isinstance(u, L.Dictee)), 2)
        self.assertEqual(sum(1 for u in unites if isinstance(u, L.Paragraphe)), 2)

    def test_paragraphes_absents_ecartes(self):
        """Sans corpus Faustine chargé, aucune unité Faustine n'est proposée."""
        groupes = L.group_segments_by_unit(
            self.scores_factices(), self.dictees, paragraphes=None)
        self.assertTrue(all(isinstance(u, L.Dictee) for _s, u, _seg in groupes))

    def test_compatibilite_du_regroupement_par_dictee(self):
        groupes = L.group_segments_by_dictee(self.scores_factices(), self.dictees)
        self.assertTrue(all(isinstance(u, L.Dictee) for _s, u, _seg in groupes))
        self.assertEqual(len(groupes), 2)

    def test_sources_presentes(self):
        groupes = L.group_segments_by_unit(
            self.scores_factices(), self.dictees, self.paragraphes)
        self.assertEqual(L.sources_presentes(groupes),
                         (L.SOURCE_LUISA, L.SOURCE_FAUSTINE))
        luisa_seul = [g for g in groupes if isinstance(g[1], L.Dictee)]
        self.assertEqual(L.sources_presentes(luisa_seul), (L.SOURCE_LUISA,))


class TestCitation(BaseCorpus):

    def test_reference_paragraphe_isole(self):
        """La citation porte le numéro de référence, pas celui du PDF."""
        self.assertEqual(L.reference_faustine(self.paragraphes[0]),
                         "Petit Journal, § 743")

    def test_reference_plage(self):
        self.assertEqual(L.reference_faustine(self.paragraphes[1]),
                         "Petit Journal, §§ 280–295")

    def test_libelle_de_source_centralise(self):
        self.assertEqual(L.SOURCE_FAUSTINE_LABEL, "Petit Journal")
        for para in self.paragraphes:
            self.assertTrue(
                L.reference_faustine(para).startswith(L.SOURCE_FAUSTINE_LABEL))

    def test_numerotation_de_reference_obligatoire(self):
        """Un paragraphe non renuméroté doit échouer plutôt que citer un faux numéro."""
        orphelin = L.Paragraphe(num_debut=742, num_fin=742, cahier=None,
                                text="texte")
        with self.assertRaises(ValueError):
            L.reference_faustine(orphelin)

    def test_numero_incertain_marque_par_un_tilde(self):
        incertain = L.Paragraphe(num_debut=317, num_fin=317, cahier=None,
                                 text="texte", num_ref_debut=318,
                                 num_ref_fin=318, num_incertain=True)
        self.assertEqual(L.reference_faustine(incertain),
                         "Petit Journal, § ~318")

    def test_tilde_pose_borne_par_borne(self):
        """Le § 310 est vérifié, le 317 non : seule la fin porte le tilde."""
        para = L.Paragraphe(num_debut=310, num_fin=317, cahier=None,
                            text="texte", num_ref_debut=310,
                            num_ref_fin=318, num_incertain=True)
        self.assertEqual(L.reference_faustine(para),
                         "Petit Journal, §§ 310–~318")

    def test_plage_entierement_certaine_sans_tilde(self):
        para = L.Paragraphe(num_debut=474, num_fin=475, cahier=None,
                            text="texte", num_ref_debut=475,
                            num_ref_fin=476, num_incertain=False)
        self.assertEqual(L.reference_faustine(para),
                         "Petit Journal, §§ 475–476")

    def test_description_faustine(self):
        description = L.decrire_passage(self.paragraphes[0], self.segments[3])
        self.assertEqual(description["source"], L.SOURCE_FAUSTINE)
        self.assertEqual(description["auteur"], "sainte Faustine")
        self.assertIn(L.SOURCE_FAUSTINE_LABEL, description["reference"])
        self.assertLessEqual(len(description["extrait"]),
                             L.MAX_CHARS_CITATION_FAUSTINE)

    def test_description_luisa_conserve_tome_et_date(self):
        description = L.decrire_passage(self.dictees[0], self.segments[0])
        self.assertEqual(description["source"], L.SOURCE_LUISA)
        self.assertEqual(description["tome"], 3)
        self.assertEqual(description["date"], "12 mars 1930")
        self.assertEqual(description["reference"], "Tome 3 — 12 mars 1930")

    def test_plafond_dur_de_la_citation(self):
        long = L.Paragraphe(num_debut=1, num_fin=1, cahier=None,
                            text="mot " * 500)
        extrait = L.extrait_cite_faustine(long, None, max_chars=120)
        self.assertLessEqual(len(extrait), 120)
        self.assertTrue(extrait.endswith("…"))

    def test_paragraphe_court_cite_en_entier(self):
        court = L.Paragraphe(num_debut=1, num_fin=1, cahier=None,
                             text="Une phrase brève.")
        self.assertEqual(L.extrait_cite_faustine(court), "Une phrase brève.")

    def test_citation_preferee_sur_fin_de_phrase(self):
        para = L.Paragraphe(
            num_debut=1, num_fin=1, cahier=None,
            text="Première phrase complète et suffisamment longue pour compter. "
                 "Seconde phrase qui déborde largement le plafond fixé ici.")
        extrait = L.extrait_cite_faustine(para, None, max_chars=80)
        self.assertTrue(extrait.endswith("."))
        self.assertFalse(extrait.endswith("…"))


class TestAttributionDansLesPrompts(BaseCorpus):

    def test_entete_mono_source_sans_attribution(self):
        description = L.decrire_passage(self.dictees[0], self.segments[0])
        self.assertEqual(L._entete_passage(description, False),
                         "Tome 3 — 12 mars 1930")

    def test_entete_bi_source_avec_auteur_et_oeuvre(self):
        description = L.decrire_passage(self.paragraphes[0], self.segments[3])
        entete = L._entete_passage(description, True)
        self.assertEqual(entete, "sainte Faustine — Petit Journal, § 743")

    def test_entete_bi_source_ne_repete_pas_loeuvre(self):
        """L'œuvre est nommée une fois : « Petit Journal » ne doit pas doubler."""
        for para, seg in ((self.paragraphes[0], self.segments[3]),
                          (self.paragraphes[1], self.segments[5])):
            entete = L._entete_passage(L.decrire_passage(para, seg), True)
            self.assertEqual(entete.count(L.SOURCE_FAUSTINE_LABEL), 1, entete)

    def test_entete_bi_source_luisa(self):
        description = L.decrire_passage(self.dictees[0], self.segments[0])
        self.assertEqual(L._entete_passage(description, True),
                         "Luisa Piccarreta — Livre du Ciel, Tome 3 — 12 mars 1930")

    def test_un_prompt_par_combinaison_de_sources(self):
        combinaisons = [(L.SOURCE_LUISA,), (L.SOURCE_FAUSTINE,),
                        (L.SOURCE_LUISA, L.SOURCE_FAUSTINE)]
        for cle in combinaisons:
            self.assertIn(cle, L._RERANK_SYSTEM_PROMPTS)
            self.assertIn(cle, L._EXPLAIN_SYSTEM_PROMPTS)
            self.assertIn(cle, L._INTITULE_EXTRAITS)
            self.assertIn(cle, L._INTITULE_EXPLICATION)

    def test_prompt_luisa_seul_inchange(self):
        """Le corpus Luisa seul doit continuer d'utiliser son prompt d'origine."""
        self.assertIs(L._RERANK_SYSTEM_PROMPTS[(L.SOURCE_LUISA,)],
                      L._RERANK_SYSTEM_PROMPT)
        self.assertIs(L._EXPLAIN_SYSTEM_PROMPTS[(L.SOURCE_LUISA,)],
                      L._EXPLAIN_SYSTEM_PROMPT)

    def test_intitules_luisa_inchanges(self):
        """Le message utilisateur en mode Luisa seul doit rester au mot près.

        Ces intitulés étaient auparavant écrits en dur dans les prompts. Les
        modifier casserait le préfixe stable dont dépend le prompt caching
        d'OpenAI, et changerait le comportement du corpus historique.
        """
        self.assertEqual(
            L._INTITULE_EXTRAITS[(L.SOURCE_LUISA,)],
            "EXTRAITS CANDIDATS DU LIVRE DU CIEL "
            "(chaque extrait porte un ID, un tome et une date)")
        self.assertEqual(L._INTITULE_EXPLICATION[(L.SOURCE_LUISA,)],
                         "EXTRAITS DU LIVRE DU CIEL")

    def test_prompt_mixte_interdit_de_fondre_les_voix(self):
        mixte = L._RERANK_SYSTEM_PROMPTS[(L.SOURCE_LUISA, L.SOURCE_FAUSTINE)]
        self.assertIn("NE JAMAIS FONDRE LES DEUX VOIX", mixte)
        explication = L._EXPLAIN_SYSTEM_PROMPTS[(L.SOURCE_LUISA, L.SOURCE_FAUSTINE)]
        self.assertIn("NE JAMAIS FONDRE LES DEUX VOIX", explication)


if __name__ == "__main__":
    unittest.main()
