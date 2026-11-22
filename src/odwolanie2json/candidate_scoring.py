"""Pure, framework-independent features for ranking regex candidates.

The helper consumes averaged token logits and their character offsets. It does
not load a model, choose an output field, or apply a candidate-level threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from .candidate_types import RegexCandidate


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Immutable model and regex features for one exact regex occurrence.

    ``token_coverage`` is the fraction of the candidate's full character
    interval covered by the union of overlapping token offsets. Whitespace
    gaps count as uncovered because this helper does not receive source text.
    ``boundary_coverage`` contains ``(start, end)`` coverage flags.

    Margins use the existing token threshold: the threshold is subtracted from
    each entity logit, while the ``O`` logit is left unchanged. A missing
    candidate-aligned token span yields ``None`` margins and no predicted role.
    ``best_predicted_role`` is set only when the candidate aligns to token
    boundaries, the model's thresholded per-token argmax labels match one
    complete ``B-``/``I-`` role sequence, and the next token does not continue
    that role. It is ``None`` when the candidate is truncated, the sequence is
    incomplete, ``O`` wins, or the role is tied.
    ``expected_bio_label`` is normalized to the candidate's initial ``B-`` tag.
    """

    candidate: RegexCandidate
    expected_bio_label: str
    role_vs_o_margin: float | None
    role_vs_best_competing_role_margin: float | None
    best_competing_role: str | None
    role_margins_vs_o: tuple[tuple[str, float], ...]
    token_coverage: float
    boundary_coverage: tuple[bool, bool]
    best_predicted_role: str | None
    raw_regex_score: float


def score_candidate_span(
    candidate: RegexCandidate,
    token_logits_by_offset: Mapping[tuple[int, int], Sequence[float]],
    id2label: Mapping[int, str],
    expected_bio_label: str,
    token_threshold: float,
) -> CandidateEvidence:
    """Score one exact candidate occurrence against BIO labels and ``O``.

    This convenience function prepares a one-use scorer. For multiple
    candidates, construct :class:`CandidateSpanScorer` once and reuse it so
    labels and token logits are validated only once.
    """
    return CandidateSpanScorer(
        token_logits_by_offset, id2label, token_threshold
    ).score(candidate, expected_bio_label)


class CandidateSpanScorer:
    """Prevalidated scorer for multiple candidate spans from one document."""

    def __init__(
        self,
        token_logits_by_offset: Mapping[tuple[int, int], Sequence[float]],
        id2label: Mapping[int, str],
        token_threshold: float,
    ) -> None:
        if isinstance(token_threshold, bool) or not isinstance(token_threshold, Real):
            raise TypeError("token_threshold must be a finite real number")
        threshold = float(token_threshold)
        if not math.isfinite(threshold):
            raise ValueError("token_threshold must be finite")
        self.threshold = threshold
        self.labels = _validate_labels(id2label)
        self.label_to_id = {label: index for index, label in self.labels.items()}
        if "O" not in self.label_to_id:
            raise ValueError("id2label must contain the outside label 'O'")
        self.role_names = _bio_roles(self.labels)
        self.logits = _validate_logits(token_logits_by_offset, len(self.labels))
        self.sorted_logits = tuple(sorted(self.logits.items()))

    def score(
        self,
        candidate: RegexCandidate,
        expected_bio_label: str,
    ) -> CandidateEvidence:
        """Score one candidate using the prevalidated document logits."""
        if not isinstance(candidate, RegexCandidate):
            raise TypeError("candidate must be a RegexCandidate")
        if (
            isinstance(candidate.start, bool)
            or not isinstance(candidate.start, int)
            or isinstance(candidate.end, bool)
            or not isinstance(candidate.end, int)
            or candidate.start < 0
            or candidate.end <= candidate.start
        ):
            raise ValueError("candidate must have a non-empty, non-negative span")
        if isinstance(candidate.score, bool) or not isinstance(candidate.score, Real):
            raise TypeError("candidate.score must be a finite real number")
        if not math.isfinite(float(candidate.score)):
            raise ValueError("candidate.score must be finite")
        if not isinstance(expected_bio_label, str) or not expected_bio_label.strip():
            raise ValueError("expected_bio_label must be a non-empty string")

        expected = expected_bio_label.strip()
        expected_role = expected[2:] if expected.startswith(("B-", "I-")) else expected
        if not expected_role or expected_role == "O":
            raise ValueError("expected_bio_label must identify an entity role")
        if f"B-{expected_role}" not in self.label_to_id:
            raise ValueError(f"id2label has no B-{expected_role} class")

        candidate_tokens = [
            (offset, values)
            for offset, values in self.sorted_logits
            if offset[0] < candidate.end and offset[1] > candidate.start
        ]
        coverage, boundary_coverage = _span_coverage(
            candidate.start,
            candidate.end,
            [offset for offset, _ in candidate_tokens],
        )
        regex_score = float(candidate.score)
        if not candidate_tokens:
            return CandidateEvidence(
                candidate=candidate,
                expected_bio_label=f"B-{expected_role}",
                role_vs_o_margin=None,
                role_vs_best_competing_role_margin=None,
                best_competing_role=None,
                role_margins_vs_o=(),
                token_coverage=coverage,
                boundary_coverage=boundary_coverage,
                best_predicted_role=None,
                raw_regex_score=regex_score,
            )

        if expected_role not in self.role_names:
            raise ValueError(f"id2label has no complete BIO labels for {expected_role}")
        o_index = self.label_to_id["O"]
        o_score = _mean(values[o_index] for _, values in candidate_tokens)
        role_scores: dict[str, float] = {}
        for role in self.role_names:
            role_class_ids = _candidate_bio_class_ids(
                role, len(candidate_tokens), self.label_to_id
            )
            if role_class_ids is None:
                continue
            role_scores[role] = _mean(
                values[class_id] - self.threshold
                for (_, values), class_id in zip(candidate_tokens, role_class_ids)
            )

        if expected_role not in role_scores:
            raise ValueError(
                f"id2label lacks the BIO classes needed to score candidate role {expected_role}"
            )
        margins_vs_o = {
            role: score - o_score for role, score in role_scores.items()
        }
        competing = {
            role: margin for role, margin in margins_vs_o.items() if role != expected_role
        }
        if competing:
            best_competing_role = max(competing, key=competing.__getitem__)
            role_vs_competing = (
                margins_vs_o[expected_role] - competing[best_competing_role]
            )
        else:
            best_competing_role = None
            role_vs_competing = None

        def token_prediction(values: Sequence[float]) -> int:
            return max(
                range(len(values)),
                key=lambda class_id: (
                    values[class_id]
                    if self.labels[class_id] == "O"
                    else values[class_id] - self.threshold
                ),
            )

        token_predictions = tuple(
            token_prediction(values) for _, values in candidate_tokens
        )
        span_aligned = (
            candidate_tokens[0][0][0] == candidate.start
            and candidate_tokens[-1][0][1] == candidate.end
        )
        following_prediction = next(
            (
                token_prediction(values)
                for (start, _end), values in self.sorted_logits
                if start >= candidate.end
            ),
            None,
        )
        predicted_roles = []
        if span_aligned:
            for role in self.role_names:
                expected_sequence = _candidate_bio_class_ids(
                    role, len(candidate_tokens), self.label_to_id
                )
                if expected_sequence != token_predictions:
                    continue
                if (
                    following_prediction is not None
                    and self.labels[following_prediction] == f"I-{role}"
                ):
                    continue
                predicted_roles.append(role)
        best_predicted_role = predicted_roles[0] if len(predicted_roles) == 1 else None
        return CandidateEvidence(
            candidate=candidate,
            expected_bio_label=f"B-{expected_role}",
            role_vs_o_margin=margins_vs_o[expected_role],
            role_vs_best_competing_role_margin=role_vs_competing,
            best_competing_role=best_competing_role,
            role_margins_vs_o=tuple(sorted(margins_vs_o.items())),
            token_coverage=coverage,
            boundary_coverage=boundary_coverage,
            best_predicted_role=best_predicted_role,
            raw_regex_score=regex_score,
        )


def _validate_labels(id2label: Mapping[int, str]) -> dict[int, str]:
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("id2label must be a non-empty mapping")
    labels: dict[int, str] = {}
    for raw_index, raw_label in id2label.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError("id2label keys must be non-negative integer class ids")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("id2label values must be non-empty strings")
        label = raw_label.strip()
        if label in labels.values():
            raise ValueError(f"id2label contains duplicate label {label!r}")
        labels[raw_index] = label
    if set(labels) != set(range(len(labels))):
        raise ValueError("id2label class ids must be contiguous from zero")
    malformed = [
        label
        for label in labels.values()
        if label != "O" and not label.startswith(("B-", "I-"))
    ]
    if malformed:
        raise ValueError("non-outside id2label values must use BIO prefixes")
    return labels


def _validate_logits(
    logits_by_offset: Mapping[tuple[int, int], Sequence[float]],
    class_count: int,
) -> dict[tuple[int, int], tuple[float, ...]]:
    if not isinstance(logits_by_offset, Mapping):
        raise TypeError("token_logits_by_offset must be a mapping")
    checked: dict[tuple[int, int], tuple[float, ...]] = {}
    for offset, raw_values in logits_by_offset.items():
        if (
            not isinstance(offset, tuple)
            or len(offset) != 2
            or any(isinstance(point, bool) or not isinstance(point, int) for point in offset)
            or offset[0] < 0
            or offset[1] <= offset[0]
        ):
            raise ValueError("token offsets must be non-empty non-negative (start, end) pairs")
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise TypeError("each token logit vector must be a numeric sequence")
        if len(raw_values) != class_count:
            raise ValueError("each token logit vector must match the id2label class count")
        values: list[float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("token logits must be finite real numbers")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("token logits must be finite")
            values.append(value)
        checked[offset] = tuple(values)
    return checked


def _bio_roles(id2label: Mapping[int, str]) -> tuple[str, ...]:
    prefixes: dict[str, set[str]] = {}
    for label in id2label.values():
        if label == "O":
            continue
        prefix, role = label.split("-", 1)
        prefixes.setdefault(role, set()).add(prefix)
    return tuple(sorted(role for role, found in prefixes.items() if "B" in found))


def _candidate_bio_class_ids(
    role: str,
    token_count: int,
    label_to_id: Mapping[str, int],
) -> tuple[int, ...] | None:
    sequence = [f"B-{role}"] + [f"I-{role}"] * (token_count - 1)
    if any(label not in label_to_id for label in sequence):
        return None
    return tuple(label_to_id[label] for label in sequence)


def _span_coverage(
    start: int,
    end: int,
    offsets: Sequence[tuple[int, int]],
) -> tuple[float, tuple[bool, bool]]:
    intervals = sorted(
        (max(start, token_start), min(end, token_end))
        for token_start, token_end in offsets
        if token_start < end and token_end > start
    )
    covered = 0
    union_start: int | None = None
    union_end: int | None = None
    for left, right in intervals:
        if union_start is None:
            union_start, union_end = left, right
        elif left <= union_end:
            union_end = max(union_end, right)
        else:
            covered += union_end - union_start
            union_start, union_end = left, right
    if union_start is not None and union_end is not None:
        covered += union_end - union_start

    start_covered = any(token_start <= start < token_end for token_start, token_end in offsets)
    end_covered = any(token_start < end <= token_end for token_start, token_end in offsets)
    return covered / (end - start), (start_covered, end_covered)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)

