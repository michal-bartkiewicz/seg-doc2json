#!/usr/bin/env python3
"""Append locally extracted, manually span-labelled transfer PDFs to train JSONL."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from odwolanie2json.regex_extraction import (  # noqa: E402
    RegexExtractor,
    _digit_candidates,
    _parse_dates,
    validate_nrb,
)


DATE_LABEL = re.compile(
    r"data\s+(?:i\s+godzina\s+)?operacji|data\s+transakcji|"
    r"data\s+księgowania|data\s+zlecenia",
    re.IGNORECASE,
)
AMOUNT_PATTERNS = (
    re.compile(
        r"(?<!\d)(\d{1,3}(?:[ .\u00a0]\d{3})*(?:,\d{2}|\.\d{2})|"
        r"\d+(?:,\d{2}|\.\d{2}))\s*(?:PLN|zł|PLN\.)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:kwota|suma|wartość\s+transakcji)[^\d]{0,30}"
        r"(\d{1,3}(?:[ .\u00a0]\d{3})*(?:,\d{2}|\.\d{2})|"
        r"\d+(?:,\d{2}|\.\d{2}))",
        re.IGNORECASE,
    ),
)


def _amount_number(value: str) -> float:
    compact = value.replace(" ", "").replace("\u00a0", "")
    return float(compact.replace(".", "").replace(",", "."))


def _label_pdf(pdf: Path, index: int) -> dict[str, Any]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
    )
    text = result.stdout.decode("utf-8", errors="replace")
    entities: list[dict[str, Any]] = []
    extracted = RegexExtractor().extract("przelew", text)["przelew_bankowy"]

    accounts = _digit_candidates(text, (26,))
    if len(accounts) != 2:
        raise ValueError(f"PDF {index}: expected exactly two account spans")
    if not all(validate_nrb(text[start:end]) for _, start, end in accounts):
        raise ValueError(f"PDF {index}: an account did not pass NRB validation")
    if extracted["z_rachunku"] != accounts[0][0] or extracted["na_rachunek"] != accounts[1][0]:
        raise ValueError(f"PDF {index}: account-role heuristic disagrees with the extracted spans")
    for label, (_, start, end) in zip(
        ("SOURCE_ACCOUNT", "DESTINATION_ACCOUNT"), accounts
    ):
        entities.append({"start": start, "end": end, "label": label})

    operation_date = None
    for marker in DATE_LABEL.finditer(text):
        dates = _parse_dates(text[marker.end():marker.end() + 120])
        if dates:
            date_value, start, end = dates[0]
            operation_date = (marker.end() + start, marker.end() + end)
            if extracted["data_operacji"] != date_value:
                raise ValueError(f"PDF {index}: date-role heuristic disagrees with the extracted span")
            break
    if operation_date is None or extracted["data_operacji"] is None:
        raise ValueError(f"PDF {index}: no labelled operation date found")
    entities.append({"start": operation_date[0], "end": operation_date[1], "label": "OPERATION_DATE"})

    extractor_amount = extracted["kwota"]
    amount_span = None
    for pattern in AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            if _amount_number(match.group(1)) == extractor_amount:
                amount_span = (match.start(1), match.end(1))
                break
        if amount_span:
            break
    if amount_span is None:
        raise ValueError(f"PDF {index}: no confidently aligned amount found")
    entities.append({"start": amount_span[0], "end": amount_span[1], "label": "TRANSFER_AMOUNT"})

    entities.sort(key=lambda item: (item["start"], item["end"]))
    if any(not 0 <= item["start"] < item["end"] <= len(text) for item in entities):
        raise ValueError(f"PDF {index}: invalid entity span")

    doc_id = f"user-transfer/extra-{index:04d}"
    return {
        "doc_id": doc_id,
        "group": doc_id,
        "document_type": "przelew",
        "source_kind": "text-pdf",
        "extraction_method": "pdftotext-layout-local",
        "source_file": f"user-transfer-{index:04d}.pdf",
        "split": "train",
        "text": text,
        "entities": entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Private base train/val/test JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Private JSONL with added train records")
    parser.add_argument("--pdf", type=Path, action="append", required=True, help="Local transfer confirmation; repeat per PDF")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    if not args.input.is_file():
        raise SystemExit(f"Base JSONL not found: {args.input}")
    if any(not pdf.is_file() for pdf in args.pdf):
        raise SystemExit("One or more transfer PDF paths do not exist")

    original_lines = args.input.read_bytes().splitlines(keepends=True)
    records = [json.loads(line.decode("utf-8")) for line in original_lines if line.strip()]
    existing_ids = {record["doc_id"] for record in records}
    appended = [_label_pdf(pdf, index) for index, pdf in enumerate(args.pdf, start=1)]
    if any(record["doc_id"] in existing_ids for record in appended):
        raise SystemExit("An extra transfer record already exists in the base JSONL")

    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("wb") as stream:
        for line in original_lines:
            stream.write(line)
        if original_lines and original_lines[-1] and not original_lines[-1].endswith((b"\n", b"\r")):
            stream.write(b"\n")
        for record in appended:
            stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    args.output.chmod(0o600)

    print(f"Added {len(appended)} train-only transfer records with {sum(len(r['entities']) for r in appended)} aligned spans.")
    print(f"Wrote {len(records) + len(appended)} private records to {args.output} (mode 0600).")


if __name__ == "__main__":
    main()
