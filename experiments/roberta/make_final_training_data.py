#!/usr/bin/env python3
"""Build a full train+validation corpus, excluding the held-out test groups."""
from __future__ import annotations

import argparse
import json
import os
import random
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=782)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument("--context-variant", choices=tuple(CONTEXT_VARIANTS), default="legacy")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.per_type < 1 or args.context_repeats < 1:
        raise SystemExit("augmentation counts must be positive")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    os.umask(0o077)

    with args.input.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    excluded_test = sum(record["split"] == "test" for record in records)
    records = [dict(record) for record in records if record["split"] != "test"]
    for record in records:
        record["split"] = "train"

    synthetic = []
    context_ids_used = _used_identifiers(records)
    document_types = ("odwolanie", "pelnomocnictwo", "przelew")
    for type_index, document_type in enumerate(document_types):
        type_records = [
            record for record in records
            if record["document_type"] == document_type
        ]
        context_rng = random.Random(args.seed + type_index)
        context_ids = IdentifierFactory(context_rng, set(context_ids_used))
        synthetic.extend(_context_replacements(
            type_records, context_rng, context_ids,
            repeats=args.context_repeats,
            target_labels=CONTEXT_VARIANTS[args.context_variant],
        ))
        context_ids_used.update(context_ids.used)
    template_rng = random.Random(args.seed + 100_000)
    template_ids = IdentifierFactory(template_rng, set(context_ids_used))
    for index in range(args.per_type):
        serial = index + 1
        synthetic.append(_make_appeal(serial, template_rng, template_ids))
        synthetic.append(_make_power(serial, template_rng, template_ids))
        synthetic.append(_make_transfer(serial, template_rng, template_ids))

    output_records = records + synthetic
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.chmod(0o600)
    print(
        f"Wrote {len(output_records)} private training records; "
        f"base={len(records)}, synthetic={len(synthetic)}, "
        f"excluded_test_records={excluded_test}, context_variant={args.context_variant}",
        flush=True,
    )


if __name__ == "__main__":
    main()
