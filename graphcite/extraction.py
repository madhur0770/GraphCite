"""LLM-based entity and relation extraction with grounding checks.

The model is asked to quote the sentence that supports every entity and relation.
Anything whose quote cannot be found in the source document, or whose entities do not
appear in the document, is dropped and reported. This keeps hallucinated facts out of the
graph, which is what makes later citations verifiable.
"""
from __future__ import annotations

import json
import re

from .llm import LLM
from .models import Entity, Evidence, ExtractionResult, Relation, entity_id, relation_id
from .text import find_sentence_containing, normalize_name, squash

ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PRODUCT", "EVENT", "CONCEPT"}

EXTRACTION_PROMPT = """You extract a knowledge graph from a document.

Return ONLY a JSON object with this shape:
{{
  "nodes": [{{"name": "...", "type": "PERSON|ORGANIZATION|LOCATION|TECHNOLOGY|PRODUCT|EVENT|CONCEPT", "summary": "one short sentence", "evidence": "exact quote from the document"}}],
  "edges": [{{"source": "node name", "target": "node name", "type": "UPPER_SNAKE_CASE_RELATION", "evidence": "exact quote from the document that states this relation"}}]
}}

Rules:
- Use only facts stated in the document. Do not add outside knowledge.
- "evidence" must be copied verbatim from the document.
- Use the full entity name every time (no pronouns or abbreviations).
- Edge "source" and "target" must be names that appear in "nodes".

DOCUMENT:
{document}
"""


def build_prompt(document: str) -> str:
    return EXTRACTION_PROMPT.format(document=document.strip())


def _clean_relation_type(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(raw).strip()).strip("_").upper()
    return cleaned or "RELATED_TO"


def parse_extraction(raw: str, doc_name: str, document: str) -> ExtractionResult:
    """Validate the model's JSON against the document and build graph objects."""
    dropped: list[str] = []
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return ExtractionResult([], [], [f"Model did not return valid JSON: {exc}"])
    if not isinstance(data, dict):
        return ExtractionResult([], [], ["Model JSON was not an object."])

    doc_squashed = squash(document)
    entities: dict[str, Entity] = {}
    by_normalized: dict[str, Entity] = {}

    for node in data.get("nodes") or []:
        if not isinstance(node, dict) or not str(node.get("name", "")).strip():
            dropped.append(f"Skipped a node without a name: {node!r}")
            continue
        name = str(node["name"]).strip()
        if normalize_name(name) not in normalize_name(document):
            dropped.append(f"Dropped entity '{name}': name does not appear in the document.")
            continue
        sentence = find_sentence_containing(document, str(node.get("evidence", ""))) or find_sentence_containing(document, name)
        if sentence is None:  # pragma: no cover - unreachable because the name was found above
            dropped.append(f"Dropped entity '{name}': no supporting sentence.")
            continue
        etype = str(node.get("type", "CONCEPT")).strip().upper()
        entity = Entity(
            id=entity_id(name),
            name=name,
            type=etype if etype in ENTITY_TYPES else "CONCEPT",
            summary=str(node.get("summary", "")).strip(),
            evidence=[Evidence(doc_name, sentence)],
        )
        if entity.id in entities:
            continue
        entities[entity.id] = entity
        by_normalized[normalize_name(name)] = entity

    relations: dict[str, Relation] = {}
    for edge in data.get("edges") or []:
        if not isinstance(edge, dict):
            dropped.append(f"Skipped a malformed edge: {edge!r}")
            continue
        src = by_normalized.get(normalize_name(str(edge.get("source", ""))))
        dst = by_normalized.get(normalize_name(str(edge.get("target", ""))))
        label = f"{edge.get('source')!s} -[{edge.get('type')!s}]-> {edge.get('target')!s}"
        if src is None or dst is None:
            dropped.append(f"Dropped relation {label}: an endpoint is not an accepted entity.")
            continue
        if src.id == dst.id:
            dropped.append(f"Dropped relation {label}: self-loop.")
            continue
        quote = str(edge.get("evidence", ""))
        if len(squash(quote)) < 8 or squash(quote) not in doc_squashed:
            dropped.append(f"Dropped relation {label}: evidence quote is missing or not found verbatim in the document.")
            continue
        sentence = find_sentence_containing(document, quote) or quote.strip()
        rel_type = _clean_relation_type(edge.get("type", ""))
        rid = relation_id(src.id, rel_type, dst.id)
        relations.setdefault(rid, Relation(rid, src.id, dst.id, rel_type, [Evidence(doc_name, sentence)]))

    return ExtractionResult(list(entities.values()), list(relations.values()), dropped)


def extract(llm: LLM, doc_name: str, document: str) -> ExtractionResult:
    return parse_extraction(llm.complete_json(build_prompt(document)), doc_name, document)
