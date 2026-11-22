#!/usr/bin/env python3
"""Create privacy-safe group-CV datasets with fold-specific augmentation."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from augment_data import (
    CONTEXT_VARIANTS,
    IdentifierFactory,
    _context_replacements,
    _make_appeal,
    _make_power,
    _make_transfer,
    _used_identifiers,
)

DOCUMENT_TYPES = {"odwolanie", "pelnomocnictwo", "przelew"}


def _load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _fold_assignment(records: list[dict], folds: int, seed: int) -> dict[str, int]:
    types_by_group: dict[str, set[str]] = defaultdict(set)
    source_by_group: dict[str, str] = {}
    for record in records:
        if record["split"] not in {"train", "validation"}:
            continue
        group = record["group"]
        types_by_group[group].add(record["document_type"])
        # OCR variants inherit the parent's group. Preserve the original
        # challenge/scan stratum instead of letting their new source_kind
        # change the seeded fold assignment.
        source_by_group[group] = "scan" if group.startswith("scan-") else "text-pdf"

    # Only challenge cases with all three document types form validation folds.
    groups_by_source: dict[str, list[str]] = defaultdict(list)
    for group, kinds in types_by_group.items():
        if kinds == DOCUMENT_TYPES:
            groups_by_source[source_by_group[group]].append(group)
    if not groups_by_source:
        raise SystemExit("No complete three-document case groups available for CV")

    assignment: dict[str, int] = {}
    for source_index, source_kind in enumerate(sorted(groups_by_source)):
        groups = groups_by_source[source_kind]
        random.Random(seed + source_index).shuffle(groups)
        for index, group in enumerate(groups):
            assignment[group] = index % folds
    return assignment


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Private base JSONL with original group split assignments.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--per-type", type=int, default=782)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument("--context-variant", choices=tuple(CONTEXT_VARIANTS), default="legacy")
    parser.add_argument(
        "--exclude-training-source-kinds", nargs="*", default=[],
        help="Keep these sources in validation folds but remove them from training splits.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.folds < 2 or args.per_type < 1 or args.context_repeats < 1:
        raise SystemExit("folds must be >= 2 and augmentation counts must be positive")
    os.umask(0o077)
    records = _load_records(args.input)
    assignment = _fold_assignment(records, args.folds, args.seed)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    fold_summary = {
        "context_variant": args.context_variant,
        "exclude_training_source_kinds": args.exclude_training_source_kinds,
        "folds": [],
    }

    for fold in range(args.folds):
        fold_records = []
        group_counts = Counter()
        excluded_training_sources = Counter()
        for original in records:
            if original["split"] == "test":
                continue  # Test text and values are absent from every CV dataset.
            record = dict(original)
            group = record["group"]
            if group in assignment:
                record["split"] = "validation" if assignment[group] == fold else "train"
            else:
                # Extra user-supplied documents stay training-only.
                record["split"] = "train"
            if (
                record["split"] == "train"
                and record["source_kind"] in args.exclude_training_source_kinds
            ):
                excluded_training_sources[record["source_kind"]] += 1
                continue
            fold_records.append(record)
            group_counts[record["split"]] += 1

        train_records = [record for record in fold_records if record["split"] == "train"]
        context_examples = []
        context_ids_used = _used_identifiers(train_records)
        for type_index, document_type in enumerate(sorted(DOCUMENT_TYPES)):
            type_records = [
                record for record in train_records
                if record["document_type"] == document_type
            ]
            context_rng = random.Random(
                args.seed + fold * 1_000 + type_index
            )
            context_ids = IdentifierFactory(
                context_rng, set(context_ids_used)
            )
            context_examples.extend(_context_replacements(
                type_records, context_rng, context_ids,
                repeats=args.context_repeats,
                target_labels=CONTEXT_VARIANTS[args.context_variant],
            ))
            context_ids_used.update(context_ids.used)
        # Keep template text and identifiers stable across context variants.
        template_rng = random.Random(args.seed + 100_000 + fold)
        template_ids = IdentifierFactory(template_rng, set(context_ids_used))
        augmented = list(context_examples)
        for index in range(args.per_type):
            serial = index + 1
            augmented.append(_make_appeal(serial, template_rng, template_ids))
            augmented.append(_make_power(serial, template_rng, template_ids))
            augmented.append(_make_transfer(serial, template_rng, template_ids))
        fold_records.extend(augmented)

        output = args.output_dir / f"fold-{fold + 1:02d}.jsonl"
        _write_records(output, fold_records)
        fold_summary["folds"].append({
            "fold": fold + 1,
            "train_records_including_synthetic": len(fold_records)
                - group_counts["validation"],
            "validation_records": group_counts["validation"],
            "synthetic_records": len(augmented),
            "template_records_per_type": args.per_type,
            "context_records": len(augmented) - args.per_type * 3,
            "excluded_training_records_by_source_kind": dict(excluded_training_sources),
            "path": str(output),
        })
        print(
            f"fold {fold + 1}/{args.folds}: validation={group_counts['validation']} "
            f"base records, synthetic={len(augmented)}; wrote private JSONL",
            flush=True,
        )

    summary_path = args.output_dir / "fold-summary.json"
    summary_path.write_text(json.dumps(fold_summary, indent=2) + "\n", encoding="utf-8")
    summary_path.chmod(0o600)


if __name__ == "__main__":
    main()
