import unittest

from graphcite.answer import attribute, parse_refs, verify
from graphcite.models import EvidenceItem, Entity, Evidence, Path, Relation, entity_id, relation_id
from graphcite.retrieval import gather_evidence, match_entities


def e(name):
    return Entity(entity_id(name), name, "ORGANIZATION")


class MatchEntitiesTests(unittest.TestCase):
    def test_full_and_partial_matches_rank_by_score(self):
        entities = [e("Aurora Labs"), e("Volta Systems"), e("Lisbon")]
        matches = match_entities("Where is Aurora headquartered?", entities)
        self.assertEqual([m.entity.name for m in matches], ["Aurora Labs"])
        self.assertEqual(matches[0].score, 0.5)
        full = match_entities("Tell me about Aurora Labs and Lisbon", entities)
        self.assertEqual([m.entity.name for m in full], ["Aurora Labs", "Lisbon"])

    def test_generic_words_do_not_match(self):
        self.assertEqual(match_entities("What is the capital of Portugal?", [e("Aurora Labs"), e("The")]), [])

    def test_possessive_in_question_still_matches(self):
        self.assertEqual(len(match_entities("What did Lisbon's mayor say?", [e("Lisbon")])), 1)


class GatherEvidenceTests(unittest.TestCase):
    def test_orders_short_paths_first_and_deduplicates(self):
        a, b, c = e("A"), e("B"), e("C")
        r1 = Relation(relation_id(a.id, "X", b.id), a.id, b.id, "X", [Evidence("d", "A to B.")])
        r2 = Relation(relation_id(b.id, "Y", c.id), b.id, c.id, "Y", [Evidence("d", "B to C.")])
        paths = [Path([a, b, c], [r1, r2]), Path([a, b], [r1])]
        items = gather_evidence(paths, {a.id})
        self.assertEqual([i.sentence for i in items], ["A to B.", "B to C."])
        self.assertEqual([i.ref for i in items], [1, 2])

    def test_respects_max_items(self):
        a, b = e("A"), e("B")
        rel = Relation("r", a.id, b.id, "X", [Evidence("d", f"Fact {i}.") for i in range(10)])
        self.assertEqual(len(gather_evidence([Path([a, b], [rel])], set(), max_items=4)), 4)


ITEMS = [
    EvidenceItem(1, "d1", "Aurora Labs was founded by Maria Santos in 2018.", ["Aurora Labs -[FOUNDED_BY]-> Maria Santos"]),
    EvidenceItem(2, "d2", "Maria Santos led the power electronics team at Volta Systems.", ["Maria Santos -[LED_TEAM_AT]-> Volta Systems"]),
]


class VerifyTests(unittest.TestCase):
    def status(self, text):
        return [(c.status, c.refs) for c in verify(text, ITEMS)]

    def test_parse_refs_handles_groups(self):
        self.assertEqual(parse_refs("Claim [1] and more [2, 3]."), [1, 2, 3])

    def test_supported_claim(self):
        self.assertEqual(self.status("Maria Santos founded Aurora Labs in 2018 [1]."), [("supported", [1])])

    def test_multiple_citations_are_combined(self):
        claims = verify("Maria Santos founded Aurora Labs and led the power electronics team at Volta Systems [1, 2].", ITEMS)
        self.assertEqual(claims[0].status, "supported")

    def test_uncited_claim(self):
        self.assertEqual(self.status("Maria Santos founded Aurora Labs in 2018."), [("uncited", [])])

    def test_invalid_citation_number(self):
        self.assertEqual(self.status("Maria Santos founded Aurora Labs in 2018 [7]."), [("invalid-citation", [7])])

    def test_weak_when_citation_does_not_contain_the_claim(self):
        claims = verify("Aurora Labs sells quantum pancakes worldwide [1].", ITEMS)
        self.assertEqual(claims[0].status, "weak")
        self.assertLess(claims[0].support, 0.5)

    def test_wrong_citation_is_weak_even_if_number_exists(self):
        claims = verify("Maria Santos led the power electronics team at Volta Systems [1].", ITEMS)
        self.assertEqual(claims[0].status, "weak")

    def test_short_non_claims_are_ignored(self):
        self.assertEqual(verify("Sure. Okay.", ITEMS), [])

    def test_attribute_collects_only_valid_cited_refs_and_ratio(self):
        result = attribute("Maria Santos founded Aurora Labs in 2018 [1]. It is also famous worldwide for pancakes.", ITEMS, [])
        self.assertEqual(result.cited_refs, [1])
        self.assertEqual(result.supported_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
