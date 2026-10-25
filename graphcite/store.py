"""Graph storage: an in-memory implementation and a Neo4j implementation with one interface."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Protocol

from .models import Entity, Evidence, Path, Relation

MAX_DEPTH = 4


class GraphStore(Protocol):
    def upsert_entity(self, entity: Entity) -> None: ...
    def upsert_relation(self, relation: Relation) -> None: ...
    def all_entities(self) -> list[Entity]: ...
    def all_relations(self) -> list[Relation]: ...
    def neighborhood(self, seed_ids: list[str], depth: int) -> list[Path]: ...
    def stats(self) -> dict[str, int]: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


def _check_depth(depth: int) -> int:
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be an integer between 1 and {MAX_DEPTH}")
    return depth


def _merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    merged = list(existing)
    for item in new:
        if item not in merged:
            merged.append(item)
    return merged


class InMemoryGraphStore:
    """Reference implementation. Also used for the no-services demo and for tests."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}

    def upsert_entity(self, entity: Entity) -> None:
        current = self._entities.get(entity.id)
        if current is None:
            self._entities[entity.id] = Entity(entity.id, entity.name, entity.type, entity.summary, list(entity.evidence))
        else:
            current.evidence = _merge_evidence(current.evidence, entity.evidence)
            current.summary = current.summary or entity.summary

    def upsert_relation(self, relation: Relation) -> None:
        for endpoint in (relation.source_id, relation.target_id):
            if endpoint not in self._entities:
                raise ValueError(f"Unknown entity id {endpoint!r}; upsert entities before relations.")
        current = self._relations.get(relation.id)
        if current is None:
            self._relations[relation.id] = Relation(relation.id, relation.source_id, relation.target_id, relation.type, list(relation.evidence))
        else:
            current.evidence = _merge_evidence(current.evidence, relation.evidence)

    def all_entities(self) -> list[Entity]:
        return sorted(self._entities.values(), key=lambda e: e.name.lower())

    def all_relations(self) -> list[Relation]:
        return sorted(self._relations.values(), key=lambda r: (r.source_id, r.type, r.target_id))

    def neighborhood(self, seed_ids: list[str], depth: int) -> list[Path]:
        depth = _check_depth(depth)
        adjacency: dict[str, list[tuple[str, Relation]]] = defaultdict(list)
        for relation in self.all_relations():
            adjacency[relation.source_id].append((relation.target_id, relation))
            adjacency[relation.target_id].append((relation.source_id, relation))
        paths: list[Path] = []

        def walk(entities: list[str], relations: list[Relation]) -> None:
            if relations:
                paths.append(Path([self._entities[i] for i in entities], list(relations)))
            if len(relations) == depth:
                return
            for neighbour, relation in adjacency[entities[-1]]:
                if neighbour not in entities:
                    walk(entities + [neighbour], relations + [relation])

        for seed in seed_ids:
            if seed in self._entities:
                walk([seed], [])
        return paths

    def stats(self) -> dict[str, int]:
        docs = {e.doc for ent in self._entities.values() for e in ent.evidence}
        return {"entities": len(self._entities), "relations": len(self._relations), "documents": len(docs)}

    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()

    def close(self) -> None:
        return None


def _dump(evidence: list[Evidence]) -> list[str]:
    return [json.dumps({"doc": e.doc, "sentence": e.sentence}, sort_keys=True) for e in evidence]


def _load(items: list[str] | None) -> list[Evidence]:
    out = []
    for raw in items or []:
        data = json.loads(raw)
        out.append(Evidence(data["doc"], data["sentence"]))
    return out


class Neo4jGraphStore:
    """Neo4j-backed store. Only nodes labelled ``Entity`` are ever read, written or deleted."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("The 'neo4j' package is not installed. Run: pip install -r requirements.txt") from exc
        self._driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ.get("NEO4J_PASSWORD", "")),
        )
        self._driver.verify_connectivity()
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE").consume()

    def upsert_entity(self, entity: Entity) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.name = $name, e.type = $type, e.summary = $summary, e.evidence = []
                SET e.summary = CASE WHEN coalesce(e.summary, '') = '' THEN $summary ELSE e.summary END,
                    e.evidence = coalesce(e.evidence, []) + [x IN $evidence WHERE NOT x IN coalesce(e.evidence, [])]
                """,
                id=entity.id, name=entity.name, type=entity.type, summary=entity.summary, evidence=_dump(entity.evidence),
            ).consume()

    def upsert_relation(self, relation: Relation) -> None:
        with self._driver.session() as session:
            found = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids RETURN count(e) AS n", ids=[relation.source_id, relation.target_id]
            ).single()["n"]
            if found != len({relation.source_id, relation.target_id}):
                raise ValueError("Unknown entity id in relation; upsert entities before relations.")
            session.run(
                """
                MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
                MERGE (a)-[r:RELATED {id: $id}]->(b)
                ON CREATE SET r.type = $type, r.evidence = []
                SET r.evidence = coalesce(r.evidence, []) + [x IN $evidence WHERE NOT x IN coalesce(r.evidence, [])]
                """,
                source=relation.source_id, target=relation.target_id, id=relation.id, type=relation.type, evidence=_dump(relation.evidence),
            ).consume()

    @staticmethod
    def _entity(node) -> Entity:
        return Entity(node["id"], node["name"], node["type"], node.get("summary") or "", _load(node.get("evidence")))

    def all_entities(self) -> list[Entity]:
        with self._driver.session() as session:
            rows = session.run("MATCH (e:Entity) RETURN e ORDER BY toLower(e.name)")
            return [self._entity(row["e"]) for row in rows]

    def all_relations(self) -> list[Relation]:
        with self._driver.session() as session:
            rows = session.run(
                "MATCH (a:Entity)-[r:RELATED]->(b:Entity) RETURN r.id AS id, a.id AS s, b.id AS t, r.type AS type, r.evidence AS ev "
                "ORDER BY s, type, t"
            )
            return [Relation(r["id"], r["s"], r["t"], r["type"], _load(r["ev"])) for r in rows]

    def neighborhood(self, seed_ids: list[str], depth: int) -> list[Path]:
        depth = _check_depth(depth)  # validated integer: safe to inline (Cypher cannot parameterise path length)
        query = f"""
            MATCH p = (s:Entity {{id: $seed}})-[:RELATED*1..{depth}]-(n:Entity)
            RETURN [x IN nodes(p) | x] AS nodes,
                   [r IN relationships(p) | {{id: r.id, type: r.type, evidence: r.evidence,
                                              source: startNode(r).id, target: endNode(r).id}}] AS rels
            LIMIT 500
        """
        paths: list[Path] = []
        with self._driver.session() as session:
            for seed in seed_ids:
                for row in session.run(query, seed=seed):
                    entities = [self._entity(n) for n in row["nodes"]]
                    if len({e.id for e in entities}) != len(entities):
                        continue  # skip paths that revisit an entity
                    relations = [Relation(r["id"], r["source"], r["target"], r["type"], _load(r["evidence"])) for r in row["rels"]]
                    paths.append(Path(entities, relations))
        paths.sort(key=lambda p: (len(p.relations), [e.id for e in p.entities]))
        return paths

    def stats(self) -> dict[str, int]:
        with self._driver.session() as session:
            entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
            relations = session.run("MATCH (:Entity)-[r:RELATED]->(:Entity) RETURN count(r) AS n").single()["n"]
        documents = len({ev.doc for ent in self.all_entities() for ev in ent.evidence})
        return {"entities": entities, "relations": relations, "documents": documents}

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (e:Entity) DETACH DELETE e").consume()

    def close(self) -> None:
        self._driver.close()
