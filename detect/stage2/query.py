"""Turning a transcription *prefix* into a structured query.

The input to scoring is never a final sentence but a growing prefix of what has
been heard so far (Stage 1). The parser must therefore degrade gracefully: while
the prefix is incomplete, fields stay empty.

    "the"                       -> Query()                       # no idea yet
    "the cup"                   -> Query(target="cup")           # target only
    "the cup left of"           -> Query(target="cup", relation="to the left of")
    "the cup left of the kettle"-> Query(target="cup", relation="to the left of", anchor="kettle")

Two parsers are provided:

* :class:`HeuristicPrefixParser` - dependency-free, deterministic, fast. This is
  the workhorse for offline replay (Stage 3) and the unit tests.
* :class:`DSPyPromptParser` - wraps the project's optimised ``PromptProcessor``
  (``LLMs/get_prompt.py``) for parity with the live demo. Imported lazily so the
  package stays importable without dspy / a running LLM.

Both satisfy the :class:`QueryParser` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Canonical relation surface forms, matching detect/demo_dyhead.py's RELATION_ALIASES
# and detect/yoloe_dyhead_relation_adapter_common.format_relation_prompt vocabulary.
RELATION_ALIASES: dict[str, str] = {
    "above": "above",
    "over": "above",
    "on": "above",
    "on top of": "above",
    "below": "below",
    "beneath": "below",
    "under": "under",
    "underneath": "under",
    "in front": "in front of",
    "in front of": "in front of",
    "front of": "in front of",
    "behind": "behind",
    "left": "to the left of",
    "left of": "to the left of",
    "to the left": "to the left of",
    "to the left of": "to the left of",
    "right": "to the right of",
    "right of": "to the right of",
    "to the right": "to the right of",
    "to the right of": "to the right of",
}

# Longest phrases first so "to the left of" wins over "left".
_RELATION_PHRASES: list[str] = sorted(RELATION_ALIASES, key=len, reverse=True)

_ARTICLES = {"the", "a", "an", "that", "this", "find", "get", "grab", "pick", "up"}


@dataclass(frozen=True)
class Query:
    """A structured referring expression, possibly partial."""

    target: str = ""
    attributes: tuple[str, ...] = ()
    anchor: str = ""
    relation: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.target

    @property
    def has_relation(self) -> bool:
        """True once both an anchor and a supported relation are known."""
        return bool(self.anchor and self.relation)

    @property
    def target_prompt(self) -> str:
        """The text fed to the detector/embedder, e.g. "blue cup"."""
        attr = " ".join(self.attributes).strip()
        return f"{attr} {self.target}".strip() if attr else self.target

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "attributes": list(self.attributes),
            "anchor": self.anchor,
            "relation": self.relation,
        }


@runtime_checkable
class QueryParser(Protocol):
    def parse(self, prefix_text: str) -> Query: ...


def _clean_noun_phrase(words: list[str]) -> tuple[tuple[str, ...], str]:
    """Split a noun phrase into (attributes, head noun), dropping articles."""
    kept = [w for w in words if w and w not in _ARTICLES]
    if not kept:
        return (), ""
    head = kept[-1]
    attrs = tuple(kept[:-1])
    return attrs, head


class HeuristicPrefixParser:
    """Dependency-free incremental parser.

    Finds the earliest relation phrase; text before it is the target noun phrase,
    text after it is the anchor noun phrase. Anything still being spoken simply
    leaves the corresponding field empty.
    """

    def __init__(self, relation_aliases: dict[str, str] | None = None) -> None:
        self._aliases = dict(relation_aliases or RELATION_ALIASES)
        self._phrases = sorted(self._aliases, key=len, reverse=True)

    def _find_relation(self, tokens: list[str]) -> tuple[int, int, str] | None:
        """Return (start_idx, end_idx, canonical) of the earliest relation phrase."""
        best: tuple[int, int, str] | None = None
        for start in range(len(tokens)):
            for phrase in self._phrases:
                plen = len(phrase.split())
                if start + plen > len(tokens):
                    continue
                window = " ".join(tokens[start : start + plen])
                if window == phrase:
                    canonical = self._aliases[phrase]
                    # Earliest start wins; on a tie the longest phrase wins
                    # (phrases are pre-sorted longest-first, so first hit per start is longest).
                    if best is None or start < best[0]:
                        best = (start, start + plen, canonical)
                    break
        return best

    def parse(self, prefix_text: str) -> Query:
        text = (prefix_text or "").strip().lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        if not tokens:
            return Query()

        match = self._find_relation(tokens)
        if match is None:
            attrs, head = _clean_noun_phrase(tokens)
            return Query(target=head, attributes=attrs)

        start, end, relation = match
        target_attrs, target_head = _clean_noun_phrase(tokens[:start])
        _, anchor_head = _clean_noun_phrase(tokens[end:])

        # Guard: a relation word with no target before it (e.g. "left") is noise.
        if not target_head:
            attrs, head = _clean_noun_phrase(tokens)
            return Query(target=head, attributes=attrs)

        return Query(
            target=target_head,
            attributes=target_attrs,
            anchor=anchor_head,
            relation=relation,
        )


class DSPyPromptParser:
    """Adapter around the project's optimised ``PromptProcessor`` (live-demo parity).

    Heavy deps (dspy + a running LLM endpoint) are imported lazily. ``relation_to_id``
    restricts the parsed relation to those the trained adapter actually supports,
    mirroring ``demo_dyhead.normalize_relation``.
    """

    def __init__(self, prompt_processor=None, relation_to_id: dict[str, int] | None = None) -> None:
        self._processor = prompt_processor
        self._relation_to_id = relation_to_id or {}
        self._fallback = HeuristicPrefixParser()

    def _normalize_relation(self, relation: object) -> str:
        text = " ".join(str(relation or "").strip().lower().split())
        text = RELATION_ALIASES.get(text, text)
        if self._relation_to_id and text not in self._relation_to_id:
            return ""
        return text

    def parse(self, prefix_text: str) -> Query:
        if self._processor is None:
            return self._fallback.parse(prefix_text)
        res = self._processor(speech=prefix_text)
        is_valid = str(getattr(res, "is_valid_command", "false")).strip().lower() in {"true", "yes", "1"}
        target = str(getattr(res, "target", "") or "").strip().lower()
        if not is_valid or target in {"", "none", "null", "unknown"}:
            return Query()
        attributes = _parse_attributes(getattr(res, "attributes", None))
        anchor = str(getattr(res, "anchor_object", "") or "").strip().lower()
        relation = self._normalize_relation(getattr(res, "spatial_relation", ""))
        if not relation:
            anchor = anchor  # keep anchor for plain detection; has_relation stays False
        return Query(target=target, attributes=tuple(attributes), anchor=anchor, relation=relation)


def _parse_attributes(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "[]"}:
        return []
    return [text]
