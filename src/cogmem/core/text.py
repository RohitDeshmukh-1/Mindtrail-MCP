"""Deterministic text helpers: normalization, hashing, tokenization, token estimates."""

from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"\w+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_MAX_QUERY_TERMS = 32

STOPWORDS = frozenset(
    """a an and are as at be but by can do does did for from had has have how i if in into is
    it its me my of on or our so than that the their them then there these they this to
    was we were what when where which who why will with you your""".split()  # noqa: SIM905
)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Hash used for exact-duplicate detection; ignores case and whitespace differences."""
    return hashlib.sha256(normalize_whitespace(text).lower().encode()).hexdigest()


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def build_fts_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression (quoted terms joined by OR).

    Quoting every term neutralizes FTS5 operators (AND, NEAR, *, column filters) in user input.
    """
    terms: list[str] = []
    for token in tokenize(query):
        if token in STOPWORDS or token in terms:
            continue
        terms.append(token)
        if len(terms) == _MAX_QUERY_TERMS:
            break
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def estimate_tokens(text: str) -> int:
    """Rough, model-agnostic token estimate (~4 characters per token)."""
    return math.ceil(len(text) / 4)
