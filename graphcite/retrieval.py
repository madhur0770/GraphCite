"""Question to evidence: match entities mentioned in the question, then expand through the graph."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Entity, EvidenceItem, Path
from .store import GraphStore
from .text import STOPWORDS, normalize_name, tokenize

MIN_MATCH = 0.5
MAX_SEEDS = 3


@dataclass(frozen=True)
class SeedMatch:
    entity: Entity
    score: float


def match_entities(question: str, entities: list[Entity]) -> list[SeedMatch]:
    """Entities whose name words appear in the question, best match first.

    Score = share of the entity name's words found in the question. Names made only of
    stop-words never match, and a match must include at least one meaningful word.
    """
    question_tokens = set(tokenize(question))
    matches = []
    for entity in entities:
        name_tokens = tokenize(normalize_name(entity.name))
        meaningful = [t for t in name_tokens if t not in STOPWORDS]
        if not meaningful:
            continue
        hit = [t for t in name_tokens if t in question_tokens]
        if not any(t in question_tokens for t in meaningful):
            continue
        score = len(hit) / len(name_tokens)
        if score >= MIN_MATCH:
            matches.append(SeedMatch(entity, score))
    matches.sort(key=lambda m: (-m.score, -len(m.entity.name), m.entity.name.lower()))
    return matches


def gather_evidence(paths: list[Path], matched_ids: set[str], max_items: int = 12) -> list[EvidenceItem]:
    """Turn paths into numbered, de-duplicated evidence sentences.

    Short paths come first; paths that end at another entity named in the question rank higher.
    """
    ranked = sorted(
        paths,
        key=lambda p: (len(p.relations), 0 if p.entities[-1].id in matched_ids else 1, [e.id for e in p.entities]),
    )
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for path in ranked:
        hops = path.describe()
        for relation in path.relations:
            for ev in relation.evidence:
                key = (ev.doc, ev.sentence)
                if key in seen:
                    continue
                seen.add(key)
                items.append(EvidenceItem(ref=len(items) + 1, doc=ev.doc, sentence=ev.sentence, path=hops))
                if len(items) >= max_items:
                    return items
    return items


def retrieve(store: GraphStore, question: str, depth: int = 2, max_items: int = 12) -> tuple[list[SeedMatch], list[EvidenceItem]]:
    seeds = match_entities(question, store.all_entities())[:MAX_SEEDS]
    if not seeds:
        return [], []
    paths = store.neighborhood([s.entity.id for s in seeds], depth)
    return seeds, gather_evidence(paths, {s.entity.id for s in seeds}, max_items)
