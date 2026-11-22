#!/usr/bin/env python3
"""Offline candidate evaluator; reads JSONL only and never emits field values or document IDs.

Input schema (one document per JSONL line; scores rank candidates high to low)::

    {
      "document_id": "opaque-id",
      "document_type": "odwolanie",
      "source_kind": "ocr",
      "gold": [
        {"field": "odwolanie.odwolujacy.nazwa", "value": "Example"}
      ],
      "candidates": [
        {
          "field": "odwolanie.odwolujacy.nazwa",
          "value": "Example",
          "score": 0.91,
          "selected": true
        }
      ]
    }

Metric units are document-scoped, exact normalized (JSON path, value) pairs.
Normalization applies Unicode NFKC, case folding, trimming, and whitespace
collapsing; punctuation and diacritics are preserved. Candidate recall@k ranks
unique candidate pairs separately within each document and field, by descending
score (input order breaks ties). Oracle F1 is the candidate-coverage ceiling:
an ideal selector keeps every available gold pair and no false pair.

Candidate-level thresholds must be cross-fitted inside grouped folds: all
documents and synthetic/OCR variants from the same underlying case must remain
in one fold. Do not calibrate and report a threshold on the same held-out fold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO


@dataclass(frozen=True, order=True)
class Pair:
    field: str
    value: str


@dataclass
class Candidate:
    score: float
    selected: bool
    first_seen: int


@dataclass
class Document:
    document_id: str
    document_type: str
    source_kind: str
    gold: set[Pair]
    candidates: dict[Pair, Candidate]


@dataclass
class Counts:
    document_count: int = 0
    gold_pair_count: int = 0
    recall1_hits: int = 0
    recallk_hits: int = 0
    oracle_tp: int = 0
    oracle_fn: int = 0
    selected_tp: int = 0
    selected_fp: int = 0
    selected_fn: int = 0

    def add(self, document: Document, k: int) -> None:
        self.document_count += 1
        self.gold_pair_count += len(document.gold)

        ranked_by_field: dict[str, list[Pair]] = defaultdict(list)
        for pair in document.candidates:
            ranked_by_field[pair.field].append(pair)
        ranks: dict[Pair, int] = {}
        for pairs in ranked_by_field.values():
            pairs.sort(
                key=lambda pair: (
                    -document.candidates[pair].score,
                    document.candidates[pair].first_seen,
                )
            )
            ranks.update({pair: rank for rank, pair in enumerate(pairs, start=1)})

        for pair in document.gold:
            rank = ranks.get(pair)
            if rank is not None and rank <= 1:
                self.recall1_hits += 1
            if rank is not None and rank <= k:
                self.recallk_hits += 1

        available = set(document.candidates)
        oracle_tp = len(document.gold & available)
        self.oracle_tp += oracle_tp
        self.oracle_fn += len(document.gold) - oracle_tp

        selected = {
            pair for pair, candidate in document.candidates.items()
            if candidate.selected
        }
        self.selected_tp += len(document.gold & selected)
        self.selected_fp += len(selected - document.gold)
        self.selected_fn += len(document.gold - selected)

    def as_dict(self, k: int) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "gold_pair_count": self.gold_pair_count,
            "candidate_recall_at_1": _recall(self.recall1_hits, self.gold_pair_count),
            f"candidate_recall_at_{k}": _recall(self.recallk_hits, self.gold_pair_count),
            "oracle_final_field": _f1_counts(self.oracle_tp, 0, self.oracle_fn),
            "selected_final_field": _f1_counts(
                self.selected_tp, self.selected_fp, self.selected_fn
            ),
        }


def _normalize_atom(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _normalize_field(value: str, location: str) -> str:
    # JSON keys are case-sensitive; normalize Unicode/whitespace without folding case.
    parts = [
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", part).strip())
        for part in value.split(".")
    ]
    if not value.strip() or any(not part for part in parts):
        raise ValueError(f"{location} must be a non-empty dotted JSON path")
    return ".".join(parts)


def _pair_from_json(obj: Any, location: str) -> Pair:
    if not isinstance(obj, dict):
        raise ValueError(f"{location} must be an object")
    field_value = obj.get("field")
    value = obj.get("value")
    if not isinstance(field_value, str):
        raise ValueError(f"{location}.field must be a string")
    if not isinstance(value, str):
        raise ValueError(f"{location}.value must be a string")
    normalized_value = _normalize_atom(value)
    if not normalized_value:
        raise ValueError(f"{location}.value must be non-empty after normalization")
    return Pair(_normalize_field(field_value, f"{location}.field"), normalized_value)


def _required_nonempty_string(obj: dict[str, Any], key: str, location: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _parse_document(obj: Any, line_number: int) -> Document:
    where = f"line {line_number}"
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must contain a JSON object")
    document_id = _required_nonempty_string(obj, "document_id", where)
    document_type = _required_nonempty_string(obj, "document_type", where)
    source_kind = _required_nonempty_string(obj, "source_kind", where)

    gold_raw = obj.get("gold")
    candidates_raw = obj.get("candidates")
    if not isinstance(gold_raw, list):
        raise ValueError(f"{where}.gold must be an array of field/value objects")
    if not isinstance(candidates_raw, list):
        raise ValueError(f"{where}.candidates must be an array of candidate objects")

    gold = {
        _pair_from_json(item, f"{where}.gold[{index}]")
        for index, item in enumerate(gold_raw)
    }

    candidates: dict[Pair, Candidate] = {}
    for index, item in enumerate(candidates_raw):
        location = f"{where}.candidates[{index}]"
        pair = _pair_from_json(item, location)
        score = item.get("score") if isinstance(item, dict) else None
        selected = item.get("selected") if isinstance(item, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{location}.score must be a finite number")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"{location}.score must be a finite number")
        if not isinstance(selected, bool):
            raise ValueError(f"{location}.selected must be a boolean")

        existing = candidates.get(pair)
        if existing is None:
            candidates[pair] = Candidate(score, selected, index)
        else:
            existing.score = max(existing.score, score)
            existing.selected = existing.selected or selected

    return Document(document_id, document_type, source_kind, gold, candidates)


def load_jsonl(stream: TextIO) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"line {line_number} contains invalid JSON") from None
        document = _parse_document(obj, line_number)
        if document.document_id in seen_ids:
            raise ValueError(f"line {line_number} repeats a document_id")
        seen_ids.add(document.document_id)
        documents.append(document)
    if not documents:
        raise ValueError("input contains no document records")
    return documents


def _recall(hits: int, total: int) -> dict[str, Any]:
    return {"hits": hits, "total": total, "value": hits / total if total else None}


def _f1_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": (2 * tp / denominator) if denominator else 0.0,
    }


def evaluate(documents: Iterable[Document], k: int) -> dict[str, Any]:
    overall = Counts()
    by_type: dict[str, Counts] = defaultdict(Counts)
    by_source: dict[str, Counts] = defaultdict(Counts)
    for document in documents:
        overall.add(document, k)
        by_type[document.document_type].add(document, k)
        by_source[document.source_kind].add(document, k)

    return {
        "schema_version": 1,
        "k": k,
        "normalization": (
            "Unicode NFKC; casefold; trim and collapse whitespace; preserve "
            "punctuation and diacritics. Dotted JSON path segments are normalized individually."
        ),
        "oracle_definition": (
            "Candidate-coverage ceiling: an ideal selector chooses all available gold pairs "
            "and no false pairs."
        ),
        "threshold_calibration_note": (
            "Calibrate candidate-level thresholds cross-fitted within grouped folds; "
            "keep each underlying case and its synthetic/OCR variants in one fold."
        ),
        "overall": overall.as_dict(k),
        "by_document_type": {
            key: counts.as_dict(k) for key, counts in sorted(by_type.items())
        },
        "by_source_kind": {
            key: counts.as_dict(k) for key, counts in sorted(by_source.items())
        },
    }


def _self_check() -> dict[str, Any]:
    synthetic = [
        {
            "document_id": "synthetic-a",
            "document_type": "synthetic-type-a",
            "source_kind": "synthetic-text",
            "gold": [
                {"field": "party.name", "value": "  ALPHA   BETA "},
                {"field": "party.name", "value": "second"},
                {"field": "party.id", "value": "gold-id"},
                {"field": "meta.code", "value": "gold-code"},
            ],
            "candidates": [
                {"field": "party.name", "value": "wrong", "score": 0.99, "selected": True},
                {"field": "party.name", "value": "alpha beta", "score": 0.80, "selected": False},
                {"field": "party.id", "value": "other-id", "score": 0.50, "selected": False},
                {"field": "party.id", "value": "gold-id", "score": 0.40, "selected": False},
                {"field": "meta.code", "value": "gold-code", "score": 0.90, "selected": True},
            ],
        },
        {
            "document_id": "synthetic-b",
            "document_type": "synthetic-type-b",
            "source_kind": "synthetic-scan",
            "gold": [{"field": "transfer.amount", "value": "123"}],
            "candidates": [
                {"field": "transfer.amount", "value": " 123 ", "score": 0.10, "selected": True}
            ],
        },
    ]
    documents = [_parse_document(item, index) for index, item in enumerate(synthetic, start=1)]
    report = evaluate(documents, k=2)
    overall = report["overall"]

    assert overall["candidate_recall_at_1"] == {"hits": 2, "total": 5, "value": 0.4}
    assert overall["candidate_recall_at_2"] == {"hits": 4, "total": 5, "value": 0.8}
    assert overall["oracle_final_field"] == {
        "tp": 4, "fp": 0, "fn": 1, "micro_f1": 8 / 9
    }
    assert overall["selected_final_field"] == {
        "tp": 2, "fp": 1, "fn": 3, "micro_f1": 0.5
    }
    assert set(report["by_document_type"]) == {"synthetic-type-a", "synthetic-type-b"}
    assert set(report["by_source_kind"]) == {"synthetic-text", "synthetic-scan"}
    return {"self_check": "passed", "synthetic_documents": len(documents)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate offline JSONL candidate predictions. Required record keys: "
            "document_id, document_type, source_kind, gold[{field,value}], "
            "candidates[{field,value,score,selected}]. Candidate scores rank high to low. "
            "No document IDs or field values are printed."
        ),
        epilog=(
            "Example: python evaluate_candidate_predictions.py predictions.jsonl --k 5\n"
            "Self-check: python evaluate_candidate_predictions.py --self-check"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "predictions_jsonl",
        nargs="?",
        help="JSONL input file, or '-' to read JSONL from stdin",
    )
    parser.add_argument("--k", type=int, default=5, help="k for candidate recall@k (default: 5)")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run the small in-memory synthetic metric check and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.k < 2:
        parser.error("--k must be at least 2")

    try:
        self_check_result = _self_check()
        if args.self_check:
            print(json.dumps(self_check_result, ensure_ascii=False))
            return 0
        if not args.predictions_jsonl:
            parser.error("predictions_jsonl is required unless --self-check is used")

        if args.predictions_jsonl == "-":
            documents = load_jsonl(sys.stdin)
        else:
            with Path(args.predictions_jsonl).open("r", encoding="utf-8") as stream:
                documents = load_jsonl(stream)
        report = evaluate(documents, args.k)
        report["self_check"] = self_check_result["self_check"]
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
