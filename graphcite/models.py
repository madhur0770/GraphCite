"""Plain data structures shared by every part of the pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .text import normalize_name


def entity_id(name: str) -> str:
    """Stable ID derived from the normalised name, so the same entity merges across documents."""
    return hashlib.sha1(normalize_name(name).encode("utf-8")).hexdigest()[:16]


def relation_id(source_id: str, rel_type: str, target_id: str) -> str:
    return hashlib.sha1(f"{source_id}|{rel_type}|{target_id}".encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Evidence:
    """A sentence from a source document that supports an entity or relation."""

    doc: str
    sentence: str


@dataclass
class Entity:
    id: str
    name: str
    type: str
    summary: str = ""
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Relation:
    id: str
    source_id: str
    target_id: str
    type: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Path:
    """A simple path through the graph: entities[i] --relations[i]-- entities[i+1]."""

    entities: list[Entity]
    relations: list[Relation]

    def describe(self) -> list[str]:
        """Human-readable hops, respecting each relation's stored direction."""
        out = []
        for index, relation in enumerate(self.relations):
            left, right = self.entities[index], self.entities[index + 1]
            if relation.source_id == left.id:
                out.append(f"{left.name} -[{relation.type}]-> {right.name}")
            else:
                out.append(f"{right.name} -[{relation.type}]-> {left.name}")
        return out


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    dropped: list[str] = field(default_factory=list)  # reasons, for transparency


@dataclass
class EvidenceItem:
    """A numbered piece of evidence handed to the answer generator."""

    ref: int
    doc: str
    sentence: str
    path: list[str]


@dataclass
class Claim:
    sentence: str
    refs: list[int]
    support: float
    status: str  # "supported" | "weak" | "uncited" | "invalid-citation"


@dataclass
class AttributedAnswer:
    text: str
    claims: list[Claim]
    evidence: list[EvidenceItem]
    cited_refs: list[int]
    steps: list[str]

    @property
    def supported_ratio(self) -> float:
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.status == "supported") / len(self.claims)
