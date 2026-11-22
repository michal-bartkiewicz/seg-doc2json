#!/usr/bin/env python3
"""Create local, scan-like OCR training variants without sending text anywhere."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prepare_data import (  # noqa: E402
    _document_sources,
    _find_spans,
    _normalise_with_offsets,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

DPI = 300
LANGUAGE = "pol+eng"
DEFAULT_PSM_MODES = (3,)
NUMERIC_LABELS = {
    "APPELLANT_NIP", "AUTHORITY_NIP", "PRINCIPAL_NIP", "APPELLANT_REGON",
    "AUTHORITY_REGON", "APPELLANT_KRS", "AUTHORITY_KRS", "PRINCIPAL_KRS",
    "ATTORNEY_PESEL", "SOURCE_ACCOUNT", "DESTINATION_ACCOUNT", "TRANSACTION_ID",
}
OCR_DIGIT_CONFUSIONS = {
    "o": "0", "q": "0", "d": "0", "i": "1", "l": "1",
    "z": "2", "s": "5", "g": "6", "b": "8",
}


def _run(args: list[str], env: dict[str, str], *, text: bool = False):
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=text,
        env=env,
    )


def _version_and_language(
    tesseract: str, tessdata_dir: Path, env: dict[str, str]
) -> str:
    version = _run([tesseract, "--version"], env, text=True).stdout.splitlines()[0]
    if not version.startswith("tesseract 5.5.0"):
        raise SystemExit(f"Expected local Tesseract 5.5.0, found: {version}")
    languages = _run(
        [tesseract, "--list-langs", "--tessdata-dir", str(tessdata_dir)],
        env, text=True,
    ).stdout.splitlines()
    if not {"pol", "eng"}.issubset(set(languages)):
        raise SystemExit("Local Tesseract must have both pol and eng language data")
    return version


def _page_count(pdf: Path, pdfinfo: str, env: dict[str, str]) -> int:
    result = _run([pdfinfo, str(pdf)], env, text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("Poppler did not report a PDF page count")
    return int(match.group(1))


def _perturb(image: Image.Image, rng: random.Random) -> Image.Image:
    """Mild scan-like skew, resampling, blur and JPEG artifacts at 300 DPI."""
    image = image.convert("L")
    width, height = image.size
    image = image.rotate(
        rng.uniform(-0.8, 0.8),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        translate=(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)),
        fillcolor=255,
    )
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.93, 1.07))
    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.08, 0.35)))
    if rng.random() < 0.8:
        scale = rng.uniform(0.82, 0.97)
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        image = small.resize((width, height), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(68, 92), optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("L")


def _ocr_pdf(
    pdf: Path,
    seed: int,
    psm: int,
    temp_root: Path,
    pdftoppm: str,
    pdfinfo: str,
    tesseract: str,
    tessdata_dir: Path,
    env: dict[str, str],
) -> str:
    pages = _page_count(pdf, pdfinfo, env)
    page_texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ocr-variant-", dir=temp_root) as temp:
        folder = Path(temp)
        for page_number in range(1, pages + 1):
            prefix = folder / "page"
            _run(
                [
                    pdftoppm, "-f", str(page_number), "-l", str(page_number),
                    "-singlefile", "-r", str(DPI), "-gray", "-png",
                    str(pdf), str(prefix),
                ],
                env,
            )
            rendered = prefix.with_suffix(".png")
            with Image.open(rendered) as image:
                rng = random.Random(seed + page_number * 1_000_003)
                perturbed = _perturb(image, rng)
            perturbed_path = folder / "ocr-input.png"
            perturbed.save(perturbed_path, format="PNG", optimize=False)
            output = _run(
                [
                    tesseract, str(perturbed_path), "stdout",
                    "-l", LANGUAGE, "--psm", str(psm),
                    "--tessdata-dir", str(tessdata_dir),
                ],
                env,
                text=True,
            ).stdout
            page_texts.append(output.strip())
            rendered.unlink(missing_ok=True)
            perturbed_path.unlink(missing_ok=True)
    return "\n\n".join(text for text in page_texts if text)


def _confusion_matches(text: str, surface: str, label: str) -> list[tuple[int, int]]:
    if label not in NUMERIC_LABELS:
        return []
    expected = "".join(char for char in surface if char.isdigit())
    if len(expected) < 8:
        return []
    normalized, starts, ends = _normalise_with_offsets(text)
    max_confusions = max(1, len(expected) // 8)
    found = set()
    for index in range(0, len(normalized) - len(expected) + 1):
        substitutions = 0
        for offset, digit in enumerate(expected):
            char = normalized[index + offset]
            mapped = char if char.isdigit() else OCR_DIGIT_CONFUSIONS.get(char)
            if mapped != digit:
                break
            substitutions += char != digit
        else:
            if substitutions == 0 or substitutions > max_confusions:
                continue
            start = starts[index]
            end = ends[index + len(expected) - 1]
            if start and text[start - 1].isalnum():
                continue
            if end < len(text) and text[end].isalnum():
                continue
            found.add((start, end))
    return sorted(found)


def _project_entities(source: dict[str, Any], target_text: str):
    source_text = source["text"]
    candidates: set[tuple[int, int, str]] = set()
    unmatched_by_label: Counter = Counter()
    for entity in source["entities"]:
        surface = source_text[entity["start"]:entity["end"]]
        matches = _find_spans(target_text, surface, entity["label"])
        if not matches:
            matches = _confusion_matches(target_text, surface, entity["label"])
        if not matches:
            unmatched_by_label[entity["label"]] += 1
            continue
        candidates.update((start, end, entity["label"]) for start, end in matches)

    ordered = sorted(candidates)
    ambiguous: set[tuple[int, int, str]] = set()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if right[0] >= left[1]:
                break
            if left[0] < right[1] and right[0] < left[1]:
                ambiguous.update((left, right))
    candidates.difference_update(ambiguous)
    ignore_regions = [
        {"start": start, "end": end}
        for start, end in sorted({(start, end) for start, end, _ in ambiguous})
    ]
    entities = [
        {"start": start, "end": end, "label": label}
        for start, end, label in sorted(candidates)
    ]
    source_counts = Counter(entity["label"] for entity in source["entities"])
    target_counts = Counter(entity["label"] for entity in entities)
    incomplete_labels = {
        label: {"source": count, "ocr": target_counts[label]}
        for label, count in source_counts.items()
        if target_counts[label] < count
    }
    return entities, ignore_regions, unmatched_by_label, incomplete_labels


def _load_extra_pdf_map(paths: list[Path]) -> dict[str, Path]:
    return {
        f"user-transfer-{index:04d}.pdf": path.resolve()
        for index, path in enumerate(paths, start=1)
    }


def _process_record(
    index: int,
    source: dict[str, Any],
    pdf: Path,
    tesseract: str,
    pdftoppm: str,
    pdfinfo: str,
    tessdata_dir: Path,
    temp_root: Path,
    seed: int,
    variants: int,
    psm_modes: tuple[int, ...],
    env: dict[str, str],
):
    variants_out: list[dict[str, Any]] = []
    rejected = Counter()
    for variant in range(variants):
        psm = psm_modes[variant % len(psm_modes)]
        digest = hashlib.sha256(
            f"{seed}:{source['doc_id']}:{variant}".encode("utf-8")
        ).digest()
        local_seed = int.from_bytes(digest[:8], "big")
        ocr_text = _ocr_pdf(
            pdf, local_seed, psm, temp_root, pdftoppm, pdfinfo,
            tesseract, tessdata_dir, env,
        )
        if not ocr_text.strip():
            rejected["empty_ocr"] += 1
            continue
        entities, ignore_regions, unmatched, incomplete = _project_entities(
            source, ocr_text
        )
        rejected.update({f"unmatched:{key}": value for key, value in unmatched.items()})
        if unmatched or incomplete:
            rejected["incomplete_entity_coverage"] += 1
            continue
        if not entities:
            rejected["no_aligned_entities"] += 1
            continue
        record = dict(source)
        record.update({
            "doc_id": f"{source['doc_id']}/ocr-perturbed-{variant + 1:02d}",
            "source_kind": "ocr-perturbed",
            "extraction_method": f"tesseract-5.5.0-pol+eng-psm{psm}-300dpi-perturbed",
            "augmentation_kind": "pdf-render-scan-perturb-local-ocr",
            "parent_doc_id": source["doc_id"],
            "ocr_perturbation": {
                "seed": seed,
                "variant": variant + 1,
                "dpi": DPI,
                "language": LANGUAGE,
                "psm": psm,
            },
            "text": ocr_text,
            "entities": entities,
            "ignore_regions": ignore_regions,
        })
        variants_out.append(record)
    return index, variants_out, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/upstream/dane")
    parser.add_argument("--extra-transfer-pdf", type=Path, action="append", default=[])
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--pdfinfo", default="pdfinfo")
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--variants", type=int, default=len(DEFAULT_PSM_MODES))
    parser.add_argument(
        "--psm-modes", type=int, nargs="+", choices=(3, 4, 6, 11),
        default=DEFAULT_PSM_MODES,
        help="Tesseract page segmentation modes cycled across variants (default: PSM 3).",
    )
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()
    if args.variants < 1:
        raise SystemExit("--variants must be positive")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    if not args.input.is_file():
        raise SystemExit(f"Input JSONL not found: {args.input}")
    if any(not path.is_file() for path in args.extra_transfer_pdf):
        raise SystemExit("An extra transfer PDF path does not exist")

    tesseract = shutil.which(args.tesseract)
    pdftoppm = shutil.which(args.pdftoppm)
    pdfinfo = shutil.which(args.pdfinfo)
    if not tesseract or not pdftoppm or not pdfinfo:
        raise SystemExit("Local Tesseract and Poppler (pdftoppm/pdfinfo) are required")
    args.tessdata_dir = args.tessdata_dir.resolve()
    if not args.tessdata_dir.is_dir():
        raise SystemExit("The local Tesseract language-data directory does not exist")

    os.umask(0o077)
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    version = _version_and_language(tesseract, args.tessdata_dir, env)
    records = _read_jsonl(args.input)
    source_to_pdf = {}
    for _, _, pdf, _, _ in _document_sources(args.data_root, Path("/nonexistent-ocr-cache")):
        source_to_pdf[pdf.relative_to(args.data_root).as_posix()] = pdf.resolve()
    source_to_pdf.update(_load_extra_pdf_map(args.extra_transfer_pdf))

    jobs = []
    for record in records:
        if record["split"] == "test":
            continue
        pdf = source_to_pdf.get(record["source_file"])
        if pdf is not None:
            jobs.append((record, pdf))

    if not jobs:
        raise SystemExit("No train/validation PDF sources matched the input JSONL")
    temp_root = args.output.parent / "ocr-tmp"
    temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_root.chmod(0o700)

    ocr_records: list[list[dict[str, Any]] | None] = [None] * len(jobs)
    rejected_totals: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_record,
                index,
                record,
                pdf,
                tesseract,
                pdftoppm,
                pdfinfo,
                args.tessdata_dir,
                temp_root,
                args.seed,
                args.variants,
                tuple(args.psm_modes),
                env,
            ): index
            for index, (record, pdf) in enumerate(jobs)
        }
        completed = 0
        accepted = 0
        for future in as_completed(futures):
            index, variants_out, rejected = future.result()
            ocr_records[index] = variants_out
            rejected_totals.update(rejected)
            accepted += len(variants_out)
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(
                    f"Local OCR progress {completed}/{len(jobs)} PDFs; "
                    f"accepted OCR variants={accepted}",
                    flush=True,
                )

    all_records = list(records)
    for variants_out in ocr_records:
        if variants_out:
            all_records.extend(variants_out)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.chmod(0o600)
    accepted_by_psm = Counter(
        str(record["ocr_perturbation"]["psm"])
        for variants_out in ocr_records for record in (variants_out or [])
    )
    print(
        f"Local OCR variants: Tesseract 5.5.0, pol+eng, Poppler {DPI} DPI, "
        f"PSM modes={list(args.psm_modes)}, workers={args.workers}; "
        f"processed={len(jobs)}, accepted={sum(accepted_by_psm.values())}, "
        f"accepted_by_psm={dict(sorted(accepted_by_psm.items()))}"
    )
    print(f"Unmatched/alignment rejections: {dict(sorted(rejected_totals.items()))}")
    print(f"Wrote {len(all_records)} private records to {args.output} (mode 0600).")


if __name__ == "__main__":
    main()
