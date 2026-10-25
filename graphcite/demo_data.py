"""Bundled fictional documents and a scripted stand-in for the LLM.

The scripted model lets the whole pipeline run with no Ollama and no Neo4j: it returns
pre-written extraction JSON for the bundled documents and composes answers by restating
the numbered evidence. It exists for tests and the no-services demo and is labelled as
such in the UI. It does not generalise to other text.
"""
from __future__ import annotations

import json
import re

DOCUMENTS = {
    "aurora-labs-profile": (
        "Aurora Labs is a battery technology company headquartered in Lisbon. "
        "Aurora Labs was founded by Maria Santos in 2018. "
        "Aurora Labs develops Helios Drive, a battery management system for electric buses."
    ),
    "maria-santos-bio": (
        "Maria Santos is an electrical engineer and entrepreneur. "
        "Before founding Aurora Labs, Maria Santos led the power electronics team at Volta Systems. "
        "Maria Santos studied at the University of Porto."
    ),
    "volta-systems-news": (
        "Volta Systems is a power electronics manufacturer based in Eindhoven. "
        "Volta Systems acquired Ionex in 2021. "
        "Ionex builds charging controllers for electric buses."
    ),
}

_N = lambda name, typ, summary, quote: {"name": name, "type": typ, "summary": summary, "evidence": quote}  # noqa: E731
_E = lambda src, typ, dst, quote: {"source": src, "type": typ, "target": dst, "evidence": quote}  # noqa: E731

_EXTRACTIONS = {
    "aurora-labs-profile": {
        "nodes": [
            _N("Aurora Labs", "ORGANIZATION", "Battery technology company.", "Aurora Labs is a battery technology company headquartered in Lisbon."),
            _N("Lisbon", "LOCATION", "City where Aurora Labs is headquartered.", "Aurora Labs is a battery technology company headquartered in Lisbon."),
            _N("Maria Santos", "PERSON", "Founder of Aurora Labs.", "Aurora Labs was founded by Maria Santos in 2018."),
            _N("Helios Drive", "PRODUCT", "Battery management system for electric buses.", "Aurora Labs develops Helios Drive, a battery management system for electric buses."),
        ],
        "edges": [
            _E("Aurora Labs", "HEADQUARTERED_IN", "Lisbon", "Aurora Labs is a battery technology company headquartered in Lisbon."),
            _E("Aurora Labs", "FOUNDED_BY", "Maria Santos", "Aurora Labs was founded by Maria Santos in 2018."),
            _E("Aurora Labs", "DEVELOPS", "Helios Drive", "Aurora Labs develops Helios Drive, a battery management system for electric buses."),
        ],
    },
    "maria-santos-bio": {
        "nodes": [
            _N("Maria Santos", "PERSON", "Electrical engineer and entrepreneur.", "Maria Santos is an electrical engineer and entrepreneur."),
            _N("Aurora Labs", "ORGANIZATION", "Company founded by Maria Santos.", "Before founding Aurora Labs, Maria Santos led the power electronics team at Volta Systems."),
            _N("Volta Systems", "ORGANIZATION", "Previous employer of Maria Santos.", "Before founding Aurora Labs, Maria Santos led the power electronics team at Volta Systems."),
            _N("University of Porto", "ORGANIZATION", "University Maria Santos attended.", "Maria Santos studied at the University of Porto."),
        ],
        "edges": [
            _E("Maria Santos", "LED_TEAM_AT", "Volta Systems", "Before founding Aurora Labs, Maria Santos led the power electronics team at Volta Systems."),
            _E("Maria Santos", "STUDIED_AT", "University of Porto", "Maria Santos studied at the University of Porto."),
        ],
    },
    "volta-systems-news": {
        "nodes": [
            _N("Volta Systems", "ORGANIZATION", "Power electronics manufacturer.", "Volta Systems is a power electronics manufacturer based in Eindhoven."),
            _N("Eindhoven", "LOCATION", "City where Volta Systems is based.", "Volta Systems is a power electronics manufacturer based in Eindhoven."),
            _N("Ionex", "ORGANIZATION", "Builder of charging controllers.", "Ionex builds charging controllers for electric buses."),
        ],
        "edges": [
            _E("Volta Systems", "BASED_IN", "Eindhoven", "Volta Systems is a power electronics manufacturer based in Eindhoven."),
            _E("Volta Systems", "ACQUIRED", "Ionex", "Volta Systems acquired Ionex in 2021."),
        ],
    },
}

# (question, hops needed)
SAMPLE_QUESTIONS = [
    ("Who founded the company that develops Helios Drive?", 2),
    ("Which company did the founder of Aurora Labs work for before, and what did that company acquire?", 3),
    ("What is the capital of Portugal?", 2),
]

_EVIDENCE_LINE = re.compile(r"^\[(\d+)\] (.*?) \(source: ", re.MULTILINE)


class ScriptedLLM:
    """Deterministic stand-in for a model; only understands the bundled documents."""

    label = "Scripted demo model (no LLM)"

    def complete_json(self, prompt: str) -> str:
        for name, text in DOCUMENTS.items():
            if text in prompt:
                return json.dumps(_EXTRACTIONS[name])
        return json.dumps({"nodes": [], "edges": []})

    def complete(self, prompt: str) -> str:
        lines = _EVIDENCE_LINE.findall(prompt)
        if not lines:
            return "The evidence does not contain the answer."
        return " ".join(f"{sentence.rstrip('.')} [{ref}]." for ref, sentence in lines)
