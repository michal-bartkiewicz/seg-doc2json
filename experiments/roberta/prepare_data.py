#!/usr/bin/env python3
"""Prepare local, span-labelled data for the ModernBERT token-classification run.

The script reads PDFs and gold JSONs locally. Scanned PDFs use the existing
local Tesseract OCR cache when available; no document is sent to a hosted API.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from odwolanie2json.extract import extract_text  # noqa: E402


TYPE_PATHS = {
    "odwolanie": ("odwolanie.pdf", "odwolania", "odwolanie"),
    "pelnomocnictwo": ("pelnomocnictwo.pdf", "pelnomocnictwa", "pelnomocnictwo"),
    "przelew": ("potwierdzenie_przelewu.pdf", "potwierdzenia", "przelew"),
}

# JSON paths are deliberately mapped to role-aware NER classes. The output
# models are an experiment and do not replace the current deterministic path.
FIELD_LABELS = {
    "odwolanie": {
        "OdwolanieKIO.data": "APPEAL_DATE",
        "OdwolanieKIO.czynnosc_dzien": "ACTION_DATE",
        "OdwolanieKIO.miejscowosc": "APPEAL_CITY",
        "Odwolujacy[].nazwa": "APPELLANT_ORG",
        "Odwolujacy[].nip": "APPELLANT_NIP",
        "Odwolujacy[].regon": "APPELLANT_REGON",
        "Odwolujacy[].krs": "APPELLANT_KRS",
        "Odwolujacy[].pelnomocnik.imie_nazwisko": "COUNSEL_PERSON",
        "Odwolujacy[].pelnomocnik.nazwa": "COUNSEL_ORG",
        "Odwolujacy[].pelnomocnik.mail": "COUNSEL_EMAIL",
        "Zamawiajacy[].nazwa": "CONTRACTING_AUTHORITY_ORG",
        "Zamawiajacy[].nip": "AUTHORITY_NIP",
        "Zamawiajacy[].regon": "AUTHORITY_REGON",
        "Zamawiajacy[].krs": "AUTHORITY_KRS",
        "ZamowieniePubliczne.numer_bzp": "BZP_REFERENCE",
        "ZamowieniePubliczne.numer_postepowania": "PROCUREMENT_REFERENCE",
        "ZamowieniePubliczne.numer_ted": "TED_REFERENCE",
        "ZamowieniePubliczne.tytul": "PROCUREMENT_TITLE",
    },
    "pelnomocnictwo": {
        "mocodawca[].data.imie_nazwisko": "PRINCIPAL_PERSON",
        "mocodawca[].data.krs": "PRINCIPAL_KRS",
        "mocodawca[].data.nazwa": "PRINCIPAL_ORG",
        "mocodawca[].data.nip": "PRINCIPAL_NIP",
        "mocodawca[].podpisujacy[].imie_nazwisko": "SIGNATORY_PERSON",
        "pelnomocnicy[].imie_nazwisko": "ATTORNEY_PERSON",
        "pelnomocnicy[].pesel": "ATTORNEY_PESEL",
    },
    "przelew": {
        "przelew_bankowy.data_operacji": "OPERATION_DATE",
        "przelew_bankowy.kwota": "TRANSFER_AMOUNT",
        "przelew_bankowy.na_rachunek": "DESTINATION_ACCOUNT",
        "przelew_bankowy.numer_transakcji": "TRANSACTION_ID",
        "przelew_bankowy.tytul": "TRANSFER_TITLE",
        "przelew_bankowy.z_rachunku": "SOURCE_ACCOUNT",
    },
}

NON_SPAN_FIELDS = {
    "OdwolanieKIO.czynnosc_powod_id",  # ontology/classification value
    "OdwolanieKIO.czynnosc_opis",  # normalized summary, usually not extractive
    "Zamawiajacy[].prowadzacy_postepowanie",  # categorical/flag value
    "mocodawca[].data.typ",  # categorical party type
}

# Very long free-text fields do not fit reliably into a token-classification
# window and are retained in the coverage report rather than partially tagged.
MAX_SPAN_CHARS = 700


def _flatten(value: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(_flatten(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for child in value:
            out.extend(_flatten(child, path + "[]"))
    else:
        out.append((path, value))
    return out


def _candidates(value: Any, label: str) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    candidates = [str(value).strip()]
    if label.endswith("DATE"):
        raw = str(value).strip()
        parsed = None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                parsed = date.fromisoformat(raw[:10])
            except ValueError:
                pass
        if parsed:
            day, month, year = parsed.day, parsed.month, parsed.year
            genitive_months = (
                "", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
                "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
            )
            nominative_months = (
                "", "styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
                "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień",
            )
            candidates.extend((
                parsed.isoformat(),
                f"{day:02d}.{month:02d}.{year}", f"{day}.{month}.{year}",
                f"{day:02d}/{month:02d}/{year}", f"{day}/{month}/{year}",
                f"{day:02d}-{month:02d}-{year}", f"{day}-{month}-{year}",
                f"{day} {genitive_months[month]} {year}",
                f"{day} {nominative_months[month]} {year}",
                f"{day} {genitive_months[month]} {year} r.",
            ))
    if isinstance(value, (int, float, Decimal)):
        try:
            amount = Decimal(str(value))
            if label == "TRANSFER_AMOUNT":
                candidates.extend((f"{amount:.2f}", f"{amount:.2f}".replace(".", ",")))
                if amount == amount.to_integral_value():
                    candidates.extend((str(int(amount)), f"{int(amount)},00"))
            elif amount == amount.to_integral_value():
                candidates.append(str(int(amount)))
        except (InvalidOperation, ValueError):
            pass
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _normalise_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Drop punctuation/spacing while preserving maps to source char offsets."""
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, original in enumerate(text):
        char = {"ł": "l", "Ł": "L"}.get(original, original)
        for decomposed in unicodedata.normalize("NFKD", char):
            if unicodedata.combining(decomposed) or not decomposed.isalnum():
                continue
            folded = decomposed.casefold()
            for unit in folded:
                chars.append(unit)
                starts.append(index)
                ends.append(index + 1)
    return "".join(chars), starts, ends


def _find_spans(text: str, value: Any, label: str) -> list[tuple[int, int]]:
    normalized, starts, ends = _normalise_with_offsets(text)
    found: set[tuple[int, int]] = set()
    for candidate in _candidates(value, label):
        needle, _, _ = _normalise_with_offsets(candidate)
        if len(needle) < (5 if label.endswith(("NIP", "REGON", "KRS", "PESEL", "ACCOUNT")) else 3):
            continue
        offset = 0
        while needle and (index := normalized.find(needle, offset)) >= 0:
            offset = index + 1
            start, end = starts[index], ends[index + len(needle) - 1]
            # Avoid matching a short field value inside a larger alphanumeric token.
            if start and text[start - 1].isalnum():
                continue
            if end < len(text) and text[end].isalnum():
                continue
            if end - start <= MAX_SPAN_CHARS:
                found.add((start, end))
    return sorted(found)


def _entities(
    document_type: str, gold: Any, text: str
) -> tuple[list[dict[str, Any]], list[dict[str, int]], Counter]:
    candidates: set[tuple[int, int, str]] = set()
    stats: Counter = Counter()
    for path, value in _flatten(gold):
        label = FIELD_LABELS[document_type].get(path)
        if path in NON_SPAN_FIELDS and value is not None and value != "":
            stats["non_span_values"] += 1
            continue
        if label is None or value is None or value == "":
            continue
        stats["gold_values"] += 1
        stats[f"gold:{label}"] += 1
        matches = _find_spans(text, value, label)
        if not matches:
            stats["unmatched_values"] += 1
            stats[f"unmatched:{label}"] += 1
            continue
        stats["matched_values"] += 1
        stats[f"matched:{label}"] += 1
        stats["matched_occurrences"] += len(matches)
        for span in matches:
            candidates.add((span[0], span[1], label))

    # BIO cannot represent overlapping spans. Drop every span in an overlap
    # instead of allowing field traversal order to overwrite token labels.
    ordered = sorted(candidates)
    ambiguous: set[tuple[int, int, str]] = set()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if right[0] >= left[1]:
                break
            if left[0] < right[1] and right[0] < left[1]:
                ambiguous.update((left, right))
    stats["overlapping_spans_dropped"] = len(ambiguous)
    ignored_regions = [
        {"start": start, "end": end}
        for start, end in sorted({(start, end) for start, end, _ in ambiguous})
    ]
    stats["ignored_overlap_regions"] = len(ignored_regions)
    candidates.difference_update(ambiguous)
    entities = [
        {"start": start, "end": end, "label": label}
        for start, end, label in sorted(candidates)
    ]
    return entities, ignored_regions, stats


def _document_sources(data_root: Path, ocr_cache: Path):
    regular = sorted(data_root.glob("przyklad_*/"))
    for case_dir in regular:
        case_id = case_dir.name.removeprefix("przyklad_")
        for document_type, (pdf_name, _, _) in TYPE_PATHS.items():
            pdf = case_dir / pdf_name
            gold = case_dir / f"{document_type}.json"
            if pdf.is_file() and gold.is_file():
                yield f"case-{case_id}", document_type, pdf, gold, None

    scans_root = data_root / "skany"
    for document_type, (_, scan_dirname, cache_type) in TYPE_PATHS.items():
        for scan_dir in sorted((scans_root / scan_dirname).glob("*/")):
            scan_id = scan_dir.name
            pdf = scan_dir / f"{scan_id}.pdf"
            gold = scan_dir / f"{scan_id}.json"
            cache = ocr_cache / cache_type / f"{scan_id}.txt"
            if pdf.is_file() and gold.is_file():
                yield f"scan-{scan_id}", document_type, pdf, gold, cache


def _split_groups(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups_by_kind: dict[str, list[str]] = defaultdict(list)
    for record in records:
        group = record["group"]
        kind = "scan" if group.startswith("scan-") else "case"
        if group not in groups_by_kind[kind]:
            groups_by_kind[kind].append(group)
    result: dict[str, str] = {}
    for kind, groups in groups_by_kind.items():
        random.Random(seed + (1000 if kind == "scan" else 0)).shuffle(groups)
        test_count = max(1, round(len(groups) * 0.2)) if len(groups) > 1 else 0
        validation_count = max(1, round(len(groups) * 0.1)) if len(groups) > 2 else 0
        test_groups = set(groups[:test_count])
        validation_groups = set(groups[test_count:test_count + validation_count])
        for group in groups:
            result[group] = (
                "test" if group in test_groups else
                "validation" if group in validation_groups else
                "train"
            )
    return result


def _prepare_record(source, data_root: Path, ocr_cache: Path, split: str):
    group, document_type, pdf_path, gold_path, cached_ocr = source
    if cached_ocr is not None and cached_ocr.is_file():
        text = cached_ocr.read_text(encoding="utf-8", errors="replace")
        extraction_method = "local-ocr-cache"
    else:
        text, ocr_pages = extract_text(pdf_path)
        extraction_method = "local-ocr" if ocr_pages else "pdftotext"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    entities, ignore_regions, stats = _entities(document_type, gold, text)
    source_kind = "scan" if group.startswith("scan-") else "text-pdf"
    record = {
        "doc_id": f"{group}/{document_type}",
        "group": group,
        "document_type": document_type,
        "source_kind": source_kind,
        "extraction_method": extraction_method,
        "source_file": pdf_path.relative_to(data_root).as_posix(),
        "split": split,
        "text": text,
        "entities": entities,
        "ignore_regions": ignore_regions,
    }
    return record, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/upstream/dane")
    parser.add_argument(
        "--ocr-cache", type=Path,
        default=Path("/tmp/odwolanie2json-scan-ocr"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/odwolanie2json-roberta/dataset.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ocr-workers", type=int, choices=(1, 2), default=2,
        help="Maximum concurrent local PDF/OCR jobs (at most two OCR processes).",
    )
    parser.add_argument(
        "--include-test-text", action="store_true",
        help="Explicitly OCR/extract held-out test groups; excluded by default.",
    )
    args = parser.parse_args()
    os.umask(0o077)

    sources = list(_document_sources(args.data_root, args.ocr_cache))
    if not sources:
        raise SystemExit(f"No PDF/JSON pairs found under {args.data_root}")
    assignments = _split_groups(
        [{"group": source[0]} for source in sources], args.seed
    )
    excluded_test_records = sum(
        assignments[source[0]] == "test" for source in sources
    )
    sources_to_prepare = [
        (index, source) for index, source in enumerate(sources)
        if args.include_test_text or assignments[source[0]] != "test"
    ]
    prepared: list[tuple[dict[str, Any], Counter] | None] = [
        None
    ] * len(sources_to_prepare)
    with ThreadPoolExecutor(max_workers=args.ocr_workers) as executor:
        futures = {
            executor.submit(
                _prepare_record,
                source,
                args.data_root,
                args.ocr_cache,
                assignments[source[0]],
            ): output_index
            for output_index, (_, source) in enumerate(sources_to_prepare)
        }
        completed = 0
        for future in as_completed(futures):
            prepared[futures[future]] = future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(
                    f"Prepared {completed}/{len(futures)} local PDF records "
                    f"(OCR concurrency <= {args.ocr_workers})",
                    flush=True,
                )

    records: list[dict[str, Any]] = []
    totals: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_type_source: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for item in prepared:
        if item is None:
            raise RuntimeError("A local PDF preparation job did not produce a record")
        record, stats = item
        totals.update(stats)
        by_type[record["document_type"]].update(stats)
        by_type_source[(record["document_type"], record["source_kind"])].update(stats)
        records.append(record)

    matched_pdf_paths = {source[2].resolve() for source in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            record["split"] = assignments[record["group"]]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.chmod(0o600)

    print(f"Wrote {len(records)} labelled documents to {args.output}")
    print(f"Gold values aligned: {totals['matched_values']}/{totals['gold_values']}")
    print(
        f"Occurrences: {totals['matched_occurrences']}; "
        f"overlapping spans dropped: {totals['overlapping_spans_dropped']}; "
        f"non-span values skipped: {totals['non_span_values']}"
    )
    all_pdfs = {path.resolve() for path in args.data_root.rglob("*.pdf")}
    print(f"PDFs without a matching gold pair: {len(all_pdfs - matched_pdf_paths)}")
    for document_type in TYPE_PATHS:
        subset = [r for r in records if r["document_type"] == document_type]
        all_type_sources = [s for s in sources if s[1] == document_type]
        n_train = sum(assignments[s[0]] == "train" for s in all_type_sources)
        n_validation = sum(assignments[s[0]] == "validation" for s in all_type_sources)
        n_test = sum(assignments[s[0]] == "test" for s in all_type_sources)
        n_test_processed = sum(r["split"] == "test" for r in subset)
        stat = by_type[document_type]
        print(
            f"{document_type}: docs train={n_train} validation={n_validation} "
            f"test={n_test_processed} excluded_test={n_test - n_test_processed}; "
            f"aligned={stat['matched_values']}/{stat['gold_values']}; "
            f"unmatched={stat['unmatched_values']}"
        )
        for label in sorted(key.removeprefix("gold:") for key in stat if key.startswith("gold:")):
            print(
                f"  {label}: matched={stat[f'matched:{label}']}/"
                f"{stat[f'gold:{label}']}"
            )
        for source_kind in ("text-pdf", "scan"):
            source_stat = by_type_source[(document_type, source_kind)]
            print(
                f"  {source_kind}: aligned={source_stat['matched_values']}/"
                f"{source_stat['gold_values']}"
            )
    print("Split groups are shared across all three document types.")


if __name__ == "__main__":
    main()
