#!/usr/bin/env python3
"""Replace selected document types' OCR order variants with another profile."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


DOCUMENT_TYPES = {"odwolanie", "pelnomocnictwo", "przelew"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _merge(
    base: list[dict[str, Any]],
    replacement: list[dict[str, Any]],
    replace_types: set[str],
) -> list[dict[str, Any]]:
    base = [record for record in base if record.get("split") != "test"]
    replacement = [record for record in replacement if record.get("split") != "test"]
    parent_info = {
        record["doc_id"]: (
            record.get("group"), record.get("split"), record.get("document_type")
        )
        for record in base if record.get("source_kind") == "ocr-perturbed"
    }
    selected_replacements = [
        record for record in replacement
        if record.get("source_kind") == "ocr-order-shuffled"
        and record.get("document_type") in replace_types
    ]
    seen_parents = set()
    for record in selected_replacements:
        parent = record.get("parent_ocr_doc_id")
        if parent not in parent_info:
            raise ValueError("Replacement order row has no matching OCR parent")
        if parent_info[parent] != (
            record.get("group"), record.get("split"), record.get("document_type")
        ):
            raise ValueError("Replacement order row does not match parent group/split/type")
        if parent in seen_parents:
            raise ValueError("Replacement profile has multiple rows for one OCR parent")
        seen_parents.add(parent)

    kept = [
        record for record in base
        if not (
            record.get("source_kind") == "ocr-order-shuffled"
            and record.get("document_type") in replace_types
        )
    ]
    merged = kept + selected_replacements
    merged.sort(key=lambda record: (
        record.get("group", ""),
        record.get("document_type", ""),
        record.get("source_kind", ""),
        record.get("doc_id", ""),
    ))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace-types", nargs="+", choices=sorted(DOCUMENT_TYPES), required=True,
    )
    args = parser.parse_args()
    if args.base.resolve() == args.output.resolve() or args.replacement.resolve() == args.output.resolve():
        raise SystemExit("Output path must differ from both inputs")
    if not args.base.is_file() or not args.replacement.is_file():
        raise SystemExit("Both input JSONL files must exist")

    os.umask(0o077)
    merged = _merge(
        _read_jsonl(args.base), _read_jsonl(args.replacement), set(args.replace_types)
    )
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.parent.chmod(0o700)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in merged:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.chmod(0o600)
    counts = Counter(
        (record["document_type"], record["source_kind"])
        for record in merged
    )
    print(
        f"Merged {len(merged)} private records; replaced order variants for "
        f"{sorted(set(args.replace_types))}; by_type_source={dict(sorted(counts.items()))}"
    )
    print(f"Wrote private JSONL to {args.output} (mode 0600).")


if __name__ == "__main__":
    main()
