"""Small, dependency-free text utilities used by extraction, retrieval and verification."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of in on at to for from by with without about as is are was were be been being
    it its this that these those who whom whose which what when where why how do does did has have had not no
    so than then there their they them he she his her him i you your we our us can could should would may might
    will shall also into over under between after before during while per via""".split()
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def normalize_name(name: str) -> str:
    """Case- and punctuation-insensitive form used to merge entities."""
    return " ".join(tokenize(name))


def _strip_possessive(word: str) -> str:
    word = word.lower()
    for suffix in ("'s", "'"):
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_strip_possessive(w) for w in _WORD.findall(text)]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def squash(text: str) -> str:
    """Whitespace- and case-insensitive form for quote matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def containment(claim: str, evidence: str) -> float:
    """Share of the claim's content words that also occur in the evidence (0..1)."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & content_tokens(evidence)) / len(claim_tokens)


def find_sentence_containing(document: str, quote: str) -> str | None:
    """Return the document sentence that contains ``quote`` (case/whitespace-insensitive)."""
    needle = squash(quote)
    if not needle:
        return None
    for sentence in split_sentences(document):
        if needle in squash(sentence):
            return sentence
    # The quote may span a sentence boundary: fall back to a match in the whole document.
    if needle in squash(document):
        return quote.strip()
    return None
