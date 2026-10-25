"""Contract tests run against every GraphStore implementation.

The in-memory store always runs. The Neo4j store runs when NEO4J_URI (and NEO4J_PASSWORD)
are set, as they are in CI; it is skipped otherwise.
"""
import os
import unittest

from graphcite.models import Entity, Evidence, Relation, entity_id, relation_id
from graphcite.store import InMemoryGraphStore, Neo4jGraphStore


def ent(name, doc="d1", sentence=None):
    return Entity(entity_id(name), name, "ORGANIZATION", f"{name} summary", [Evidence(doc, sentence or f"{name} is mentioned.")])


def rel(a, typ, b, doc="d1"):
    return Relation(relation_id(entity_id(a), typ, entity_id(b)), entity_id(a), entity_id(b), typ, [Evidence(doc, f"{a} {typ} {b}.")])


class StoreContract:
    def make_store(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def setUp(self):
        self.store = self.make_store()
        self.addCleanup(self.store.close)
        self.store.clear()
        self.addCleanup(self.store.clear)
        for name in ("A", "B", "C", "D"):
            self.store.upsert_entity(ent(name))
        self.store.upsert_relation(rel("A", "KNOWS", "B"))
        self.store.upsert_relation(rel("B", "KNOWS", "C"))
        self.store.upsert_relation(rel("C", "KNOWS", "D"))

    def names(self, path):
        return [e.name for e in path.entities]

    def test_stats(self):
        self.assertEqual(self.store.stats(), {"entities": 4, "relations": 3, "documents": 1})

    def test_upsert_is_idempotent_and_merges_evidence_across_documents(self):
        self.store.upsert_entity(ent("A", doc="d2", sentence="A appears again."))
        self.store.upsert_entity(ent("A", doc="d2", sentence="A appears again."))
        a = next(e for e in self.store.all_entities() if e.name == "A")
        self.assertEqual({(x.doc, x.sentence) for x in a.evidence}, {("d1", "A is mentioned."), ("d2", "A appears again.")})
        self.assertEqual(self.store.stats()["entities"], 4)
        self.assertEqual(self.store.stats()["documents"], 2)

    def test_relation_requires_existing_entities(self):
        with self.assertRaises(ValueError):
            self.store.upsert_relation(rel("A", "KNOWS", "Z"))

    def test_neighborhood_respects_depth(self):
        one = self.store.neighborhood([entity_id("A")], 1)
        self.assertEqual([self.names(p) for p in one], [["A", "B"]])
        three = {tuple(self.names(p)) for p in self.store.neighborhood([entity_id("A")], 3)}
        self.assertEqual(three, {("A", "B"), ("A", "B", "C"), ("A", "B", "C", "D")})

    def test_neighborhood_is_undirected_and_keeps_direction_in_description(self):
        paths = self.store.neighborhood([entity_id("B")], 1)
        described = sorted(p.describe()[0] for p in paths)
        self.assertEqual(described, ["A -[KNOWS]-> B", "B -[KNOWS]-> C"])

    def test_neighborhood_never_revisits_an_entity(self):
        self.store.upsert_relation(rel("C", "KNOWS", "A"))  # create a cycle A-B-C-A
        for path in self.store.neighborhood([entity_id("A")], 4):
            ids = [e.id for e in path.entities]
            self.assertEqual(len(ids), len(set(ids)))

    def test_neighborhood_rejects_invalid_depth(self):
        for bad in (0, 5, -1, "2", True, 2.0):
            with self.subTest(depth=bad), self.assertRaises(ValueError):
                self.store.neighborhood([entity_id("A")], bad)

    def test_unknown_seed_yields_nothing(self):
        self.assertEqual(self.store.neighborhood(["nope"], 2), [])

    def test_clear_removes_everything(self):
        self.store.clear()
        self.assertEqual(self.store.stats(), {"entities": 0, "relations": 0, "documents": 0})


class InMemoryStoreTests(StoreContract, unittest.TestCase):
    def make_store(self):
        return InMemoryGraphStore()


@unittest.skipUnless(os.environ.get("NEO4J_URI"), "NEO4J_URI is not set; Neo4j integration tests skipped")
class Neo4jStoreTests(StoreContract, unittest.TestCase):
    def make_store(self):
        return Neo4jGraphStore()


if __name__ == "__main__":
    unittest.main()
