"""Public entry points: ingest documents, ask questions."""
from __future__ import annotations

from . import answer as answer_mod
from .extraction import extract
from .llm import LLM
from .models import AttributedAnswer, ExtractionResult
from .retrieval import retrieve
from .store import GraphStore

NO_MATCH_ANSWER = "No entity mentioned in the question exists in the knowledge graph, so there is nothing to answer from."
NO_EVIDENCE_ANSWER = "The matching entities have no connected facts within the chosen number of hops."


def ingest(store: GraphStore, llm: LLM, doc_name: str, text: str) -> ExtractionResult:
    result = extract(llm, doc_name, text)
    for entity in result.entities:
        store.upsert_entity(entity)
    for relation in result.relations:
        store.upsert_relation(relation)
    return result


def ask(store: GraphStore, llm: LLM, question: str, depth: int = 2) -> AttributedAnswer:
    steps = [f"Question: {question.strip()}"]
    seeds, items = retrieve(store, question, depth)
    if not seeds:
        steps.append("No entities in the question matched the graph.")
        return AttributedAnswer(NO_MATCH_ANSWER, [], [], [], steps)
    steps.append("Matched entities: " + ", ".join(f"{s.entity.name} ({s.score:.2f})" for s in seeds))
    if not items:
        steps.append(f"No connected evidence within {depth} hop(s).")
        return AttributedAnswer(NO_EVIDENCE_ANSWER, [], [], [], steps)
    steps.append(f"Collected {len(items)} evidence sentence(s) within {depth} hop(s).")
    text = llm.complete(answer_mod.build_prompt(question, items))
    result = answer_mod.attribute(text, items, steps)
    steps.append(
        f"Verified {len(result.claims)} claim(s): "
        f"{sum(c.status == 'supported' for c in result.claims)} supported, "
        f"{sum(c.status == 'weak' for c in result.claims)} weak, "
        f"{sum(c.status in {'uncited', 'invalid-citation'} for c in result.claims)} uncited or mis-cited."
    )
    return result
