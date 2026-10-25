"""Command-line demo that needs no services:  python -m graphcite.demo

Ingests the bundled fictional documents with the scripted stand-in model, then answers the
sample questions and prints the citation check for every claim.
"""
from __future__ import annotations

from . import pipeline
from .demo_data import DOCUMENTS, SAMPLE_QUESTIONS, ScriptedLLM
from .store import InMemoryGraphStore


def main() -> int:
    store, llm = InMemoryGraphStore(), ScriptedLLM()
    print(f"Model: {llm.label}\n")
    for name, text in DOCUMENTS.items():
        result = pipeline.ingest(store, llm, name, text)
        print(f"Ingested {name}: {len(result.entities)} entities, {len(result.relations)} relations, {len(result.dropped)} dropped")
    stats = store.stats()
    print(f"\nGraph: {stats['entities']} entities, {stats['relations']} relations, {stats['documents']} documents")

    for question, depth in SAMPLE_QUESTIONS:
        print(f"\n{'=' * 78}\nQ: {question}   (hops: {depth})")
        answer = pipeline.ask(store, llm, question, depth)
        for step in answer.steps[1:]:
            print(f"  - {step}")
        print(f"\nA: {answer.text}")
        for claim in answer.claims:
            refs = ",".join(map(str, claim.refs)) or "-"
            print(f"  [{claim.status:<16}] cites {refs:<3} overlap {claim.support:.2f}  {claim.sentence[:70]}")
        for item in answer.evidence:
            if item.ref in answer.cited_refs:
                print(f"  [{item.ref}] {item.doc}: \"{item.sentence}\"")
                print(f"      path: {' | '.join(item.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
