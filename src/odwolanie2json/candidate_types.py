"""Typed internal records shared by regex extraction and the local ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RegexCandidate:
    """One regex-proposed value anchored to one exact source occurrence.

    ``score`` is raw regex evidence, not a calibrated probability. ``field``
    uses the JSON path notation already used by the hybrid evaluator. ``group_id``
    identifies the source JSON list item; ``passed_minimum_score`` is set only
    when a regex source has an explicit eligibility rule.
    """

    field: str
    value: str
    start: int
    end: int
    score: float = 0.0
    rule_ids: tuple[str, ...] = ()
    group_id: str | None = None
    passed_minimum_score: bool | None = None


@dataclass(frozen=True, slots=True)
class RegexExtraction:
    """Existing JSON output plus its source-grounded regex candidates."""

    data: dict[str, Any]
    candidates: tuple[RegexCandidate, ...]
