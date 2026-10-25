import json
import unittest

from graphcite.extraction import parse_extraction

DOC = "Aurora Labs was founded by Maria Santos in 2018. Aurora Labs develops Helios Drive for electric buses."


def payload(nodes, edges):
    return json.dumps({"nodes": nodes, "edges": edges})


GOOD_NODES = [
    {"name": "Aurora Labs", "type": "organization", "summary": "A company.", "evidence": "Aurora Labs was founded by Maria Santos in 2018."},
    {"name": "Maria Santos", "type": "PERSON", "summary": "Founder.", "evidence": "Aurora Labs was founded by Maria Santos in 2018."},
]
GOOD_EDGE = {"source": "Aurora Labs", "target": "Maria Santos", "type": "founded by", "evidence": "Aurora Labs was founded by Maria Santos in 2018."}


class ParseExtractionTests(unittest.TestCase):
    def test_valid_extraction_is_accepted_and_normalised(self):
        result = parse_extraction(payload(GOOD_NODES, [GOOD_EDGE]), "doc-a", DOC)
        self.assertEqual({e.name for e in result.entities}, {"Aurora Labs", "Maria Santos"})
        self.assertEqual(result.entities[0].type, "ORGANIZATION")
        self.assertEqual(result.relations[0].type, "FOUNDED_BY")
        self.assertEqual(result.relations[0].evidence[0].doc, "doc-a")
        self.assertEqual(result.dropped, [])

    def test_entity_not_in_document_is_dropped(self):
        nodes = GOOD_NODES + [{"name": "Acme Corp", "type": "ORGANIZATION", "summary": "", "evidence": "Acme Corp exists."}]
        result = parse_extraction(payload(nodes, []), "d", DOC)
        self.assertNotIn("Acme Corp", {e.name for e in result.entities})
        self.assertTrue(any("Acme Corp" in reason for reason in result.dropped))

    def test_relation_with_fabricated_evidence_is_dropped(self):
        edge = dict(GOOD_EDGE, evidence="Maria Santos secretly owns Aurora Labs outright.")
        result = parse_extraction(payload(GOOD_NODES, [edge]), "d", DOC)
        self.assertEqual(result.relations, [])
        self.assertTrue(any("not found verbatim" in reason for reason in result.dropped))

    def test_relation_with_unknown_endpoint_is_dropped(self):
        edge = dict(GOOD_EDGE, target="Someone Else")
        result = parse_extraction(payload(GOOD_NODES, [edge]), "d", DOC)
        self.assertEqual(result.relations, [])
        self.assertTrue(any("endpoint" in reason for reason in result.dropped))

    def test_self_loop_is_dropped(self):
        edge = dict(GOOD_EDGE, target="Aurora Labs")
        self.assertEqual(parse_extraction(payload(GOOD_NODES, [edge]), "d", DOC).relations, [])

    def test_unknown_entity_type_falls_back_to_concept(self):
        nodes = [dict(GOOD_NODES[0], type="SPACESHIP")]
        self.assertEqual(parse_extraction(payload(nodes, []), "d", DOC).entities[0].type, "CONCEPT")

    def test_invalid_json_and_wrong_shape_are_reported_not_raised(self):
        self.assertTrue(parse_extraction("not json", "d", DOC).dropped)
        self.assertTrue(parse_extraction("[1, 2]", "d", DOC).dropped)
        self.assertEqual(parse_extraction(None, "d", DOC).entities, [])

    def test_malformed_items_are_skipped(self):
        result = parse_extraction(payload([{"type": "PERSON"}, "oops"], ["oops"]), "d", DOC)
        self.assertEqual(result.entities, [])
        self.assertEqual(len(result.dropped), 3)

    def test_duplicate_entities_and_relations_are_merged(self):
        result = parse_extraction(payload(GOOD_NODES + GOOD_NODES, [GOOD_EDGE, GOOD_EDGE]), "d", DOC)
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.relations), 1)


if __name__ == "__main__":
    unittest.main()
