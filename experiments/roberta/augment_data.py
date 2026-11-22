#!/usr/bin/env python3
"""Append synthetic, checksum-valid identifier examples to TRAIN only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from odwolanie2json.regex_extraction import (  # noqa: E402
    validate_nip,
    validate_nrb,
    validate_pesel,
    validate_regon,
)


class SyntheticDocument:
    def __init__(self, doc_id: str, document_type: str, index: int):
        self.doc_id = doc_id
        self.document_type = document_type
        self.index = index
        self.parts: list[str] = []
        self.length = 0
        self.entities: list[dict[str, Any]] = []

    def text(self, value: str) -> None:
        self.parts.append(value)
        self.length += len(value)

    def entity(self, label: str, value: str) -> None:
        start = self.length
        self.parts.append(value)
        self.length += len(value)
        self.entities.append({"start": start, "end": self.length, "label": label})

    def finish(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "group": self.doc_id,
            "document_type": self.document_type,
            "source_kind": "synthetic",
            "extraction_method": "synthetic-template",
            "source_file": self.doc_id,
            "split": "train",
            "text": "".join(self.parts),
            "entities": sorted(self.entities, key=lambda item: (item["start"], item["end"])),
        }


class IdentifierFactory:
    def __init__(self, rng: random.Random, used: set[str]):
        self.rng = rng
        self.used = used

    def _unique(self, generator, validator) -> str:
        for _ in range(10000):
            value = generator()
            digits = re.sub(r"\D", "", value)
            if digits not in self.used and validator(value):
                self.used.add(digits)
                return digits
        raise RuntimeError("Unable to create a unique valid synthetic identifier")

    def pesel(self) -> str:
        def generate():
            # Future birth dates keep these valid-looking examples unassigned.
            year = self.rng.randint(2027, 2039)
            month = self.rng.randint(1, 12)
            day = self.rng.randint(1, 28)
            sequence = [self.rng.randrange(10) for _ in range(3)]
            sequence.append(self.rng.choice((0, 2, 4, 6, 8)))
            month_code = month + 20  # 2000–2099 century encoding.
            first_ten = f"{year % 100:02d}{month_code:02d}{day:02d}" + "".join(map(str, sequence))
            weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
            check = (10 - sum(int(digit) * weight for digit, weight in zip(first_ten, weights)) % 10) % 10
            return first_ten + str(check)

        return self._unique(generate, validate_pesel)

    def nip(self) -> str:
        def generate():
            first = [self.rng.randrange(10) for _ in range(9)]
            weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
            check = sum(digit * weight for digit, weight in zip(first, weights)) % 11
            if check == 10:
                return ""
            return "".join(map(str, first)) + str(check)

        return self._unique(generate, validate_nip)

    def regon(self, local_unit: bool | None = None) -> str:
        def check_digit(digits: str, weights: tuple[int, ...]) -> str:
            check = sum(int(digit) * weight for digit, weight in zip(digits, weights)) % 11
            return str(0 if check == 10 else check)

        def generate():
            use_fourteen = self.rng.choice((False, True)) if local_unit is None else local_unit
            first_eight = "".join(str(self.rng.randrange(10)) for _ in range(8))
            first_nine = first_eight + check_digit(first_eight, (8, 9, 2, 3, 4, 5, 6, 7))
            if not use_fourteen:
                return first_nine
            first_thirteen = first_nine + "".join(str(self.rng.randrange(10)) for _ in range(4))
            return first_thirteen + check_digit(
                first_thirteen, (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
            )

        return self._unique(generate, validate_regon)

    def nrb(self) -> str:
        def generate():
            # Use the public NBP clearing identifier and a random synthetic BBAN.
            bban = "10100071" + "".join(str(self.rng.randrange(10)) for _ in range(16))
            check = 98 - int(bban + "252100") % 97
            return f"{check:02d}" + bban

        return self._unique(generate, validate_nrb)

    def krs(self) -> str:
        return self._unique(
            lambda: f"{self.rng.randrange(10**10):010d}",
            lambda value: bool(re.fullmatch(r"\d{10}", value)),
        )


LEGACY_CONTEXT_LABELS = {
    "ATTORNEY_PESEL", "SOURCE_ACCOUNT", "DESTINATION_ACCOUNT",
}
APPEAL_ID_CONTEXT_LABELS = {
    "APPELLANT_NIP", "AUTHORITY_NIP", "APPELLANT_REGON", "AUTHORITY_REGON",
    "APPELLANT_KRS", "AUTHORITY_KRS",
}
PRINCIPAL_ID_CONTEXT_LABELS = {"PRINCIPAL_NIP", "PRINCIPAL_KRS"}
CONTEXT_VARIANTS = {
    "legacy": LEGACY_CONTEXT_LABELS,
    "appeal-ids": LEGACY_CONTEXT_LABELS | APPEAL_ID_CONTEXT_LABELS,
    "all-ids": LEGACY_CONTEXT_LABELS | APPEAL_ID_CONTEXT_LABELS | PRINCIPAL_ID_CONTEXT_LABELS,
}


def format_nip(digits: str, style: int) -> str:
    if style == 1:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:8]}-{digits[8:]}"
    if style == 2:
        return f"{digits[:3]} {digits[3:6]} {digits[6:8]} {digits[8:]}"
    return digits


def format_regon(digits: str, style: int) -> str:
    if style == 0:
        return digits
    chunks = (3, 3, 3, 5) if len(digits) == 14 else (3, 3, 3)
    values, index = [], 0
    for width in chunks:
        values.append(digits[index:index + width])
        index += width
    return (" " if style == 1 else "-").join(values)


def format_pesel(digits: str, style: int) -> str:
    return f"{digits[:6]} {digits[6:]}" if style else digits


def format_nrb(digits: str, style: int) -> str:
    if style == 0:
        return digits
    chunks = (2, 4, 4, 4, 4, 4, 4)
    values, index = [], 0
    for width in chunks:
        values.append(digits[index:index + width])
        index += width
    return (" " if style == 1 else "-").join(values)


def future_date(rng: random.Random) -> str:
    value = date(rng.randint(2027, 2032), rng.randint(1, 12), rng.randint(1, 28))
    return value.strftime("%d.%m.%Y")


def _make_appeal(index: int, rng: random.Random, ids: IdentifierFactory) -> dict[str, Any]:
    doc = SyntheticDocument(f"synthetic/odwolanie/{index:04d}", "odwolanie", index)
    suffix = f"{index:04d}"
    doc.text("Odwołujący: ")
    doc.entity("APPELLANT_ORG", f"Fikcyjny Wykonawca {suffix} sp. z o.o.")
    doc.text(" (NIP ")
    doc.entity("APPELLANT_NIP", format_nip(ids.nip(), rng.randrange(3)))
    doc.text(", REGON ")
    doc.entity("APPELLANT_REGON", format_regon(ids.regon(), rng.randrange(2)))
    doc.text(", KRS ")
    doc.entity("APPELLANT_KRS", ids.krs())
    doc.text("). Przez pełnomocnika ")
    doc.entity("COUNSEL_PERSON", rng.choice(("Ada Testowa", "Marek Fikcyjny", "Lena Wzorcowa", "Olek Przykładowy")))
    doc.text(", kancelaria ")
    doc.entity("COUNSEL_ORG", f"Kancelaria Testowa {suffix}")
    doc.text(", e-mail: ")
    doc.entity("COUNSEL_EMAIL", f"pelnomocnik{suffix}@example.test")
    doc.text(". Zamawiający: ")
    doc.entity("CONTRACTING_AUTHORITY_ORG", f"Fikcyjny Zamawiający {suffix} S.A.")
    doc.text(" (NIP ")
    doc.entity("AUTHORITY_NIP", format_nip(ids.nip(), rng.randrange(3)))
    doc.text(", REGON ")
    doc.entity("AUTHORITY_REGON", format_regon(ids.regon(), rng.randrange(2)))
    doc.text(", KRS ")
    doc.entity("AUTHORITY_KRS", ids.krs())
    doc.text("). Postępowanie ")
    doc.entity("PROCUREMENT_REFERENCE", f"ZP.{2027 + index % 6}.{suffix}")
    doc.text(", BZP ")
    doc.entity("BZP_REFERENCE", f"{2027 + index % 6}/BZP 000000{index:04d}")
    doc.text(", TED ")
    doc.entity("TED_REFERENCE", f"{2027 + index % 6}/S 001-{index:08d}")
    doc.text(", przedmiot: ")
    doc.entity("PROCUREMENT_TITLE", f"Dostawa fikcyjnych materiałów biurowych, część {index % 9 + 1}")
    doc.text(". Miejscowość ")
    doc.entity("APPEAL_CITY", rng.choice(("Warszawa", "Kraków", "Gdańsk", "Poznań", "Łódź", "Wrocław")))
    doc.text(", data odwołania ")
    doc.entity("APPEAL_DATE", future_date(rng))
    doc.text("; zaskarżona czynność z dnia ")
    doc.entity("ACTION_DATE", future_date(rng))
    doc.text(".")
    return doc.finish()


def _make_power(index: int, rng: random.Random, ids: IdentifierFactory) -> dict[str, Any]:
    doc = SyntheticDocument(f"synthetic/pelnomocnictwo/{index:04d}", "pelnomocnictwo", index)
    doc.text("Pełnomocnictwo. Mocodawca ")
    personal_principal = index % 4 == 0
    if personal_principal:
        principal_name = rng.choice(("Alicja Testowa", "Jan Przykładowy", "Mira Wzorcowa"))
        doc.entity("PRINCIPAL_PERSON", principal_name)
    else:
        doc.entity("PRINCIPAL_ORG", f"Fikcyjna Spółka Mocodawcy {index:04d} sp. z o.o.")
        doc.text(" (NIP ")
        doc.entity("PRINCIPAL_NIP", format_nip(ids.nip(), rng.randrange(3)))
        doc.text(", KRS ")
        doc.entity("PRINCIPAL_KRS", ids.krs())
        doc.text(")")
    doc.text(" działa przez podpisującego ")
    signatory = rng.choice(("Oskar Testowy", "Ewa Przykładowa", "Marek Wzorcowy", "Nina Fikcyjna"))
    doc.entity("SIGNATORY_PERSON", signatory)
    doc.text(" i udziela pełnomocnictwa ")
    attorney = rng.choice(("Iga Testowa", "Piotr Przykładowy", "Klara Wzorcowa", "Leon Fikcyjny"))
    while attorney == signatory:
        attorney = rng.choice(("Iga Testowa", "Piotr Przykładowy", "Klara Wzorcowa", "Leon Fikcyjny"))
    doc.entity("ATTORNEY_PERSON", attorney)
    doc.text(" (PESEL ")
    doc.entity("ATTORNEY_PESEL", format_pesel(ids.pesel(), rng.randrange(2)))
    doc.text(") do reprezentowania mocodawcy przed Krajową Izbą Odwoławczą.")
    return doc.finish()


def _make_transfer(index: int, rng: random.Random, ids: IdentifierFactory) -> dict[str, Any]:
    doc = SyntheticDocument(f"synthetic/przelew/{index:04d}", "przelew", index)
    doc.text("Potwierdzenie przelewu. Data operacji: ")
    doc.entity("OPERATION_DATE", future_date(rng) + f" {rng.randrange(8, 18):02d}:{rng.choice(("00", "15", "30", "45"))}:00")
    doc.text(". Kwota transakcji: ")
    amount = f"{rng.randint(100, 50000)},{rng.randrange(100):02d}"
    doc.entity("TRANSFER_AMOUNT", amount)
    doc.text(" PLN. Z rachunku ")
    doc.entity("SOURCE_ACCOUNT", format_nrb(ids.nrb(), rng.randrange(3)))
    doc.text(" na rachunek ")
    doc.entity("DESTINATION_ACCOUNT", format_nrb(ids.nrb(), rng.randrange(3)))
    doc.text(". Numer transakcji: ")
    doc.entity("TRANSACTION_ID", f"TST-{index:04d}-{rng.randrange(100000, 999999)}")
    doc.text(". Tytuł: ")
    doc.entity("TRANSFER_TITLE", f"Wadium KIO, postępowanie ZP-{index:04d}")
    doc.text(".")
    return doc.finish()


def _replacement_for(label: str, old_value: str, rng: random.Random, ids: IdentifierFactory) -> str:
    if label == "ATTORNEY_PESEL":
        return format_pesel(ids.pesel(), int(bool(re.search(r"\s", old_value))))
    if label in {"SOURCE_ACCOUNT", "DESTINATION_ACCOUNT"}:
        style = 2 if "-" in old_value else 1 if re.search(r"\s", old_value) else 0
        return format_nrb(ids.nrb(), style)
    style = 2 if "-" in old_value else 1 if re.search(r"\s", old_value) else 0
    if label.endswith("_NIP"):
        return format_nip(ids.nip(), style)
    if label.endswith("_REGON"):
        old_digits = re.sub(r"\D", "", old_value)
        return format_regon(ids.regon(local_unit=len(old_digits) == 14), style)
    if label.endswith("_KRS"):
        return ids.krs()
    raise ValueError(f"Unsupported contextual replacement label: {label}")


def _context_replacements(
    records: list[dict[str, Any]], rng: random.Random, ids: IdentifierFactory,
    repeats: int = 1, target_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    target_labels = LEGACY_CONTEXT_LABELS if target_labels is None else target_labels
    variant_seed = rng.getrandbits(64)
    augmented = []
    for record in records:
        if record["split"] != "train" or record["source_kind"] != "text-pdf" and record["source_kind"] != "scan":
            continue
        text = record["text"]
        for target_index, target in enumerate(record["entities"]):
            if target["label"] not in target_labels:
                continue
            old_value = text[target["start"]:target["end"]]
            for repeat_index in range(repeats):
                seed_payload = (
                    f"{variant_seed}:{record['doc_id']}:{target_index}:"
                    f"{target['label']}:{repeat_index}"
                ).encode("utf-8")
                target_seed = int.from_bytes(
                    hashlib.sha256(seed_payload).digest()[:8], "big"
                )
                target_rng = random.Random(target_seed)
                replacement = _replacement_for(
                    target["label"], old_value, target_rng, ids
                )
                start = max(0, target["start"] - 180)
                end = min(len(text), target["end"] + 180)
                # Keep all labelled spans wholly inside the copied snippet.
                changed = True
                while changed:
                    changed = False
                    for entity in record["entities"]:
                        if entity["start"] < end and entity["end"] > start:
                            new_start = min(start, entity["start"])
                            new_end = max(end, entity["end"])
                            if new_start != start or new_end != end:
                                start, end = new_start, new_end
                                changed = True
                local_start = target["start"] - start
                local_end = target["end"] - start
                snippet = text[start:end]
                delta = len(replacement) - (local_end - local_start)
                synthetic_text = snippet[:local_start] + replacement + snippet[local_end:]
                synthetic_entities = []
                for entity_index, entity in enumerate(record["entities"]):
                    if entity["start"] < start or entity["end"] > end:
                        continue
                    entity_start = entity["start"] - start
                    entity_end = entity["end"] - start
                    if entity_index == target_index:
                        entity_end = entity_start + len(replacement)
                    elif entity_start >= local_end:
                        entity_start += delta
                        entity_end += delta
                    synthetic_entities.append({
                        "start": entity_start,
                        "end": entity_end,
                        "label": entity["label"],
                    })
                synthetic_ignore_regions = []
                for region in record.get("ignore_regions", []):
                    if region["start"] >= end or region["end"] <= start:
                        continue
                    region_start = max(region["start"], start) - start
                    region_end = min(region["end"], end) - start
                    if region_start >= local_end:
                        region_start += delta
                        region_end += delta
                    elif region_end > local_end:
                        region_end += delta
                    synthetic_ignore_regions.append({
                        "start": region_start,
                        "end": region_end,
                    })
                serial = len(augmented) + 1
                doc_id = f"synthetic-context/{record['document_type']}/{target['label'].lower()}/{serial:05d}"
                augmented.append({
                    "doc_id": doc_id,
                    "group": doc_id,
                    "document_type": record["document_type"],
                    "source_kind": "synthetic-context",
                    "extraction_method": "synthetic-id-in-train-context",
                    "source_file": doc_id,
                    "split": "train",
                    "text": synthetic_text,
                    "entities": sorted(synthetic_entities, key=lambda item: (item["start"], item["end"])),
                    "ignore_regions": synthetic_ignore_regions,
                    "synthetic_target": {
                        "label": target["label"],
                        "start": local_start,
                        "end": local_start + len(replacement),
                    },
                })
    return augmented


def _used_identifiers(records: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for record in records:
        # Keep augmentation independent of validation/test values.
        if record["split"] != "train":
            continue
        text = record["text"]
        for entity in record["entities"]:
            label = entity["label"]
            if label.endswith(("NIP", "REGON", "KRS", "PESEL", "ACCOUNT")):
                digits = re.sub(r"\D", "", text[entity["start"]:entity["end"]])
                if digits:
                    ids.add(digits)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=64)
    parser.add_argument("--context-repeats", type=int, default=1)
    parser.add_argument("--context-variant", choices=tuple(CONTEXT_VARIANTS), default="legacy")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.per_type < 1:
        raise SystemExit("--per-type must be positive")
    if args.context_repeats < 1:
        raise SystemExit("--context-repeats must be positive")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must be different")
    os.umask(0o077)

    original_lines = args.input.read_bytes().splitlines(keepends=True)
    records = [
        json.loads(line.decode("utf-8"))
        for line in original_lines
        if line.strip()
    ]
    train_types = {record["document_type"] for record in records if record["split"] == "train"}
    if train_types != {"odwolanie", "pelnomocnictwo", "przelew"}:
        raise SystemExit("Expected all three document types in the train split")
    augmented = list(records)
    rng = random.Random(args.seed)
    train_records = [record for record in records if record["split"] == "train"]
    # Identifier generation depends only on the training split.
    ids = IdentifierFactory(rng, _used_identifiers(train_records))
    context_examples = _context_replacements(
        train_records, rng, ids, repeats=args.context_repeats,
        target_labels=CONTEXT_VARIANTS[args.context_variant],
    )
    for index in range(args.per_type):
        augmented.append(_make_appeal(index + 1, rng, ids))
        augmented.append(_make_power(index + 1, rng, ids))
        augmented.append(_make_transfer(index + 1, rng, ids))
    augmented.extend(context_examples)

    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open("wb") as stream:
        for line in original_lines:
            stream.write(line)
        if original_lines and original_lines[-1] and not original_lines[-1].endswith((b"\n", b"\r")):
            stream.write(b"\n")
        for record in augmented[len(records):]:
            stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    args.output.chmod(0o600)

    counts = Counter(
        (record["document_type"], entity["label"])
        for record in augmented
        if record["source_kind"].startswith("synthetic")
        for entity in record["entities"]
    )
    print(
        f"Added {args.per_type} template documents per type "
        f"and {len(context_examples)} context-preserving identifier replacements; "
        f"all original JSONL lines were copied byte-for-byte."
    )
    print(f"Wrote {len(augmented)} private records to {args.output} (mode 0600)")
    for key, value in sorted(counts.items()):
        print(f"synthetic {key[0]} {key[1]}: {value}")


if __name__ == "__main__":
    main()
