#!/usr/bin/env python3
"""Select balanced OCR views for private RoBERTa training experiments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PSM_BY_TYPE = {
    "odwolanie": 3,
    "pelnomocnictwo": 3,
    "przelew": 3,
}
OCR_SOURCE_KINDS = {"ocr-perturbed", "ocr-order-shuffled"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def _select(
    records: list[dict[str, Any]],
    max_order_per_parent: int,
    psm_by_type: dict[str, int],
    order_operations_by_type: dict[str, set[str] | None],
):
    kept: list[dict[str, Any]] = []
    order_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") == "test":
            continue
        source_kind = record.get("source_kind")
        if source_kind not in OCR_SOURCE_KINDS:
            kept.append(record)
            continue

        document_type = record.get("document_type")
        expected_psm = psm_by_type.get(document_type)
        perturbation = record.get("ocr_perturbation") or {}
        if expected_psm is None or perturbation.get("psm") != expected_psm:
            continue
        if source_kind == "ocr-perturbed":
            kept.append(record)
            continue

        operation = perturbation.get("order_operation")
        allowed_operations = order_operations_by_type.get(document_type)
        if allowed_operations is not None and operation not in allowed_operations:
            continue
        parent = record.get("parent_ocr_doc_id")
        if not parent:
            raise ValueError("OCR order variant is missing parent_ocr_doc_id")
        order_by_parent[parent].append(record)

    selected_order = []
    for parent, candidates in order_by_parent.items():
        if max_order_per_parent == 0:
            continue
        if max_order_per_parent == 1:
            # Stable across runs, and independent of document content.
            digest = hashlib.sha256(
                (parent + ":single-order-v1").encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:8], "big") % len(candidates)
            selected_order.append(candidates[index])
            continue
        ranked = sorted(
            candidates,
            key=lambda record: hashlib.sha256(
                f"{parent}:{record['doc_id']}:single-order-v1".encode("utf-8")
            ).digest(),
        )
        selected_order.extend(ranked[:max_order_per_parent])

    kept.extend(selected_order)
    kept.sort(key=lambda record: (
        record.get("group", ""),
        record.get("document_type", ""),
        record.get("source_kind", ""),
        record.get("doc_id", ""),
    ))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-order-per-parent", type=int, default=1,
        help="Maximum selected order variants per OCR parent; zero disables them.",
    )
    parser.add_argument(
        "--odwolanie-psm", type=int, choices=(3, 6, 11),
        default=DEFAULT_PSM_BY_TYPE["odwolanie"],
    )
    parser.add_argument(
        "--pelnomocnictwo-psm", type=int, choices=(3, 6, 11),
        default=DEFAULT_PSM_BY_TYPE["pelnomocnictwo"],
    )
    parser.add_argument(
        "--przelew-psm", type=int, choices=(3, 6, 11),
        default=DEFAULT_PSM_BY_TYPE["przelew"],
    )
    order_choices = ("nearby-lines", "paragraph-blocks", "column-interleave")
    parser.add_argument("--odwolanie-order-operations", nargs="*", choices=order_choices)
    parser.add_argument("--pelnomocnictwo-order-operations", nargs="*", choices=order_choices)
    parser.add_argument("--przelew-order-operations", nargs="*", choices=order_choices)
    args = parser.parse_args()
    if args.max_order_per_parent < 0:
        raise SystemExit("--max-order-per-parent must be >= 0")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    if not args.input.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input}")

    os.umask(0o077)
    psm_by_type = {
        "odwolanie": args.odwolanie_psm,
        "pelnomocnictwo": args.pelnomocnictwo_psm,
        "przelew": args.przelew_psm,
    }
    order_operations_by_type = {
        "odwolanie": (set(args.odwolanie_order_operations)
                      if args.odwolanie_order_operations is not None else None),
        "pelnomocnictwo": (set(args.pelnomocnictwo_order_operations)
                           if args.pelnomocnictwo_order_operations is not None else None),
        "przelew": (set(args.przelew_order_operations)
                    if args.przelew_order_operations is not None else None),
    }
    records = _select(
        _read_jsonl(args.input), args.max_order_per_parent, psm_by_type,
        order_operations_by_type,
    )
    _write_jsonl(args.output, records)
    counts = Counter(
        (record["document_type"], record["source_kind"])
        for record in records
    )
    print(
        f"Selected {len(records)} private records; "
        f"by_type_source={dict(sorted(counts.items()))}; "
        f"PSM_by_type={psm_by_type}; "
        f"order_operations_by_type={order_operations_by_type}; "
        f"max_order_per_parent={args.max_order_per_parent}"
    )
    print(f"Wrote private JSONL to {args.output} (mode 0600).")


if __name__ == "__main__":
    main()
