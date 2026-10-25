"""Answer generation and citation verification.

The model is told to cite evidence as [n]. Afterwards every sentence of the answer is
checked: does it cite anything, do the cited numbers exist, and do the cited sentences
actually contain the claim's content words? The score is lexical (word containment). It
catches uncited, mis-numbered and unsupported statements, but it cannot judge paraphrase
or logical entailment.
"""
from __future__ import annotations

import re

from .models import AttributedAnswer, Claim, EvidenceItem
from .text import containment, split_sentences, tokenize

SUPPORT_THRESHOLD = 0.5
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

ANSWER_PROMPT = """Answer the question using ONLY the numbered evidence below.
Rules:
- Write short factual sentences that stay close to the wording of the evidence.
- End EVERY sentence with the number(s) of the evidence it uses in square brackets, for example: Ana founded Zeta Corp. [2]
- Do not write phrases such as "as stated in evidence"; give the fact and its [n].
- If the evidence does not contain the answer, say that it does not.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER:"""


def format_evidence(items: list[EvidenceItem]) -> str:
    return "\n".join(f"[{i.ref}] {i.sentence} (source: {i.doc}; path: {' | '.join(i.path)})" for i in items)


def build_prompt(question: str, items: list[EvidenceItem]) -> str:
    return ANSWER_PROMPT.format(evidence=format_evidence(items), question=question.strip())


def parse_refs(sentence: str) -> list[int]:
    refs: list[int] = []
    for group in _CITATION.findall(sentence):
        refs.extend(int(part) for part in re.split(r"\s*,\s*", group))
    return refs


def verify(answer_text: str, items: list[EvidenceItem]) -> list[Claim]:
    by_ref = {item.ref: item for item in items}
    claims: list[Claim] = []
    for sentence in split_sentences(answer_text):
        bare = re.sub(r"\s+([.,;:!?])", r"\1", _CITATION.sub("", sentence)).strip()
        if len(tokenize(bare)) < 3:
            continue  # not a factual statement (for example "Sure." or a bare list marker)
        refs = parse_refs(sentence)
        if not refs:
            claims.append(Claim(bare, [], 0.0, "uncited"))
        elif any(r not in by_ref for r in refs):
            claims.append(Claim(bare, refs, 0.0, "invalid-citation"))
        else:
            support = containment(bare, " ".join(by_ref[r].sentence for r in refs))
            claims.append(Claim(bare, refs, round(support, 2), "supported" if support >= SUPPORT_THRESHOLD else "weak"))
    return claims


def attribute(answer_text: str, items: list[EvidenceItem], steps: list[str]) -> AttributedAnswer:
    claims = verify(answer_text, items)
    valid = {r for c in claims if c.status in {"supported", "weak"} for r in c.refs}
    return AttributedAnswer(answer_text.strip(), claims, items, sorted(valid), steps)
