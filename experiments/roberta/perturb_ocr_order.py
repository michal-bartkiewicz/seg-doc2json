#!/usr/bin/env python3
"""Create local OCR text variants with bounded reading-order errors."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from make_ocr_variants import _project_entities, _read_jsonl


_PARAGRAPH_SEPARATOR = re.compile(r"(?:\r?\n[ \t]*){2,}")


def _shuffle_nearby_lines(text: str, entities: list[dict[str, Any]], seed: int):
    lines = text.splitlines(keepends=True)
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in lines:
        ranges.append((offset, offset + len(line)))
        offset += len(line)

    entity_lines = set()
    multiline_entity_lines = set()
    for entity in entities:
        covered = {
            index for index, (start, end) in enumerate(ranges)
            if start < entity["end"] and end > entity["start"]
        }
        entity_lines.update(covered)
        if len(covered) > 1:
            multiline_entity_lines.update(covered)

    eligible = [
        index for index, line in enumerate(lines)
        if line.strip() and index not in multiline_entity_lines
    ]
    pairs = [
        (left, right)
        for position, left in enumerate(eligible)
        for right in eligible[position + 1:]
        if right - left <= 3
    ]
    if not pairs:
        return text, 0

    rng = random.Random(seed)
    rng.shuffle(pairs)
    # Prefer moves that relocate labelled lines too. Entity text remains
    # intact within its line and labels are projected again after reordering.
    pairs.sort(key=lambda pair: not (set(pair) & entity_lines))
    target_swaps = min(3, max(1, len(eligible) // 20))
    used: set[int] = set()
    swaps = 0
    for left, right in pairs:
        if left in used or right in used:
            continue
        lines[left], lines[right] = lines[right], lines[left]
        used.update((left, right))
        swaps += 1
        if swaps >= target_swaps:
            break
    return "".join(lines), swaps


def _paragraph_blocks(text: str) -> list[tuple[int, int, str, str]]:
    """Return (start, end, content, following separator) for nonempty blocks."""
    blocks = []
    cursor = 0
    for match in _PARAGRAPH_SEPARATOR.finditer(text):
        content = text[cursor:match.start()]
        if content.strip():
            blocks.append((cursor, match.end(), content, match.group()))
        cursor = match.end()
    content = text[cursor:]
    if content.strip():
        blocks.append((cursor, len(text), content, ""))
    return blocks


def _reorder_paragraph_blocks(text: str, seed: int):
    """Swap one nearby pair of complete OCR paragraphs, retaining separators."""
    blocks = _paragraph_blocks(text)
    pairs = [
        index for index in range(len(blocks) - 1)
        if blocks[index][3]
    ]
    if not pairs:
        return text, None

    rng = random.Random(seed)
    index = rng.choice(pairs)
    start, _, left, separator = blocks[index]
    _, end, right, right_separator = blocks[index + 1]
    reordered = (
        text[:start] + right + separator + left + right_separator + text[end:]
    )
    if reordered == text:
        return text, None
    return reordered, {"block_pair": [index, index + 1]}


def _column_like(left: str, right: str) -> bool:
    """Conservatively recognize adjacent, similarly sized multi-line blocks."""
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    if min(len(left_lines), len(right_lines)) < 3:
        return False
    if max(len(left_lines), len(right_lines)) > 1.5 * min(
        len(left_lines), len(right_lines)
    ):
        return False
    left_chars = sum(len(line.strip()) for line in left_lines)
    right_chars = sum(len(line.strip()) for line in right_lines)
    if min(left_chars, right_chars) < 90:
        return False
    if max(left_chars, right_chars) > 2.0 * min(left_chars, right_chars):
        return False
    return min(
        sum(len(line.strip()) for line in left_lines) / len(left_lines),
        sum(len(line.strip()) for line in right_lines) / len(right_lines),
    ) >= 15


def _interleave_column_blocks(left: str, right: str, trailing_separator: str) -> str:
    """Convert two column-major blocks into a plausible row-wise OCR reading."""
    newline = "\r\n" if "\r\n" in left + right else "\n"

    def lines_and_trailing_newline(block: str) -> tuple[list[str], str]:
        trailing = ""
        if block.endswith("\r\n"):
            trailing, block = "\r\n", block[:-2]
        elif block.endswith(("\n", "\r")):
            trailing, block = block[-1], block[:-1]
        return block.splitlines(), trailing

    left_lines, left_trailing = lines_and_trailing_newline(left)
    right_lines, right_trailing = lines_and_trailing_newline(right)
    interleaved = []
    for index in range(max(len(left_lines), len(right_lines))):
        if index < len(left_lines):
            interleaved.append(left_lines[index])
        if index < len(right_lines):
            interleaved.append(right_lines[index])
    trailing = right_trailing or left_trailing
    return newline.join(interleaved) + trailing + trailing_separator


def _reorder_column_blocks(text: str, seed: int):
    """Interleave one conservative pair of adjacent column-like OCR blocks."""
    blocks = _paragraph_blocks(text)
    candidates = [
        index for index in range(len(blocks) - 1)
        if blocks[index][3]
        and _column_like(blocks[index][2], blocks[index + 1][2])
    ]
    if not candidates:
        return text, None

    rng = random.Random(seed)
    index = rng.choice(candidates)
    start, _, left, _ = blocks[index]
    _, end, right, trailing_separator = blocks[index + 1]
    interleaved = _interleave_column_blocks(left, right, trailing_separator)
    reordered = text[:start] + interleaved + text[end:]
    if reordered == text:
        return text, None
    return reordered, {
        "block_pair": [index, index + 1],
        "left_lines": len(left.splitlines()),
        "right_lines": len(right.splitlines()),
    }


def _operation_seed(seed: int, operation: str) -> int:
    payload = f"{seed}:{operation}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _source_entity_values(source: dict[str, Any]) -> Counter:
    text = source["text"]
    return Counter(
        (entity["label"], text[entity["start"]:entity["end"]])
        for entity in source["entities"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    if not args.input.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input}")

    os.umask(0o077)
    source_records = _read_jsonl(args.input)
    records = [record for record in source_records if record.get("split") != "test"]
    variants = []
    rejected: Counter = Counter()
    generated_by_operation: Counter = Counter()
    for source in records:
        if source.get("source_kind") != "ocr-perturbed":
            continue
        seed_payload = f"{args.seed}:{source['doc_id']}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        proposals = []
        seen_variant_texts = {source["text"]}
        line_text, swaps = _shuffle_nearby_lines(
            source["text"], source["entities"], _operation_seed(seed, "nearby-lines")
        )
        if swaps:
            proposals.append(("nearby-lines", line_text, {"line_swaps": swaps}))
        block_text, block_stats = _reorder_paragraph_blocks(
            source["text"], _operation_seed(seed, "paragraph-blocks")
        )
        if block_stats:
            proposals.append(("paragraph-blocks", block_text, block_stats))
        column_text, column_stats = _reorder_column_blocks(
            source["text"], _operation_seed(seed, "column-blocks")
        )
        if column_stats:
            proposals.append(("column-interleave", column_text, column_stats))

        for operation, reordered_text, operation_stats in proposals:
            if reordered_text in seen_variant_texts:
                rejected[f"{operation}:duplicate_or_no_change"] += 1
                continue
            # These edits only move OCR text. Requiring exact original surfaces
            # prevents the alignment helper's OCR-confusion fallback from
            # silently changing an annotation during a reading-order transform.
            if any(
                source["text"][entity["start"]:entity["end"]] not in reordered_text
                for entity in source["entities"]
            ):
                rejected[f"{operation}:entity_surface_missing"] += 1
                continue
            entities, ignore_regions, unmatched, incomplete = _project_entities(
                source, reordered_text
            )
            if (
                unmatched or incomplete or ignore_regions
                or _source_entity_values(source)
                != Counter(
                    (entity["label"], reordered_text[entity["start"]:entity["end"]])
                    for entity in entities
                )
            ):
                rejected[f"{operation}:entity_alignment_after_reordering"] += 1
                continue
            if not entities:
                rejected[f"{operation}:no_aligned_entities"] += 1
                continue

            suffix = {
                "nearby-lines": "nearby-lines",
                "paragraph-blocks": "paragraph-blocks",
                "column-interleave": "column-interleave",
            }[operation]
            variant = dict(source)
            variant.update({
                "doc_id": f"{source['doc_id']}/order-{suffix}",
                # Keep this stable so existing CV/source-kind aggregation works.
                "source_kind": "ocr-order-shuffled",
                "extraction_method": "local-tesseract-5.5.0+post-ocr-reading-order-perturbation",
                "augmentation_kind": f"post-ocr-{operation}",
                "parent_ocr_doc_id": source["doc_id"],
                "ocr_perturbation": {
                    **source.get("ocr_perturbation", {}),
                    "order_seed": args.seed,
                    "order_operation": operation,
                    **operation_stats,
                },
                "text": reordered_text,
                "entities": entities,
                "ignore_regions": ignore_regions,
            })
            variants.append(variant)
            seen_variant_texts.add(reordered_text)
            generated_by_operation[operation] += 1

    all_records = records + variants
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.chmod(0o600)
    print(
        f"Post-OCR reading-order variants: generated={len(variants)}, "
        f"by_operation={dict(sorted(generated_by_operation.items()))}, "
        f"rejected={sum(rejected.values())}, source_ocr={sum(1 for r in records if r.get('source_kind') == 'ocr-perturbed')}; "
        f"reasons={dict(sorted(rejected.items()))}"
    )
    print(f"Wrote {len(all_records)} private records to {args.output} (mode 0600).")


if __name__ == "__main__":
    main()
