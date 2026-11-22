from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .candidate_scoring import CandidateEvidence, CandidateSpanScorer
from .candidate_types import RegexCandidate
from .regex_extraction import (
    RegexExtractor,
    _DATE_ISO,
    _DATE_NUMERIC,
    _DATE_WORD,
    _MONTHS,
    _fold,
)
from .scan_extraction import fuse_scan_model, repair_ocr_text, repair_scan_fields


_ROBERTA_FIELDS: dict[str, dict[str, str]] = {
    "odwolanie": {
        "APPEAL_DATE": "OdwolanieKIO.data",
        "ACTION_DATE": "OdwolanieKIO.czynnosc_dzien",
        "APPEAL_CITY": "OdwolanieKIO.miejscowosc",
        "APPELLANT_ORG": "Odwolujacy[].nazwa",
        "APPELLANT_NIP": "Odwolujacy[].nip",
        "APPELLANT_REGON": "Odwolujacy[].regon",
        "APPELLANT_KRS": "Odwolujacy[].krs",
        "COUNSEL_PERSON": "Odwolujacy[].pelnomocnik.imie_nazwisko",
        "COUNSEL_ORG": "Odwolujacy[].pelnomocnik.nazwa",
        "COUNSEL_EMAIL": "Odwolujacy[].pelnomocnik.mail",
        "CONTRACTING_AUTHORITY_ORG": "Zamawiajacy[].nazwa",
        "AUTHORITY_NIP": "Zamawiajacy[].nip",
        "AUTHORITY_REGON": "Zamawiajacy[].regon",
        "AUTHORITY_KRS": "Zamawiajacy[].krs",
        "BZP_REFERENCE": "ZamowieniePubliczne.numer_bzp",
        "PROCUREMENT_REFERENCE": "ZamowieniePubliczne.numer_postepowania",
        "TED_REFERENCE": "ZamowieniePubliczne.numer_ted",
        "PROCUREMENT_TITLE": "ZamowieniePubliczne.tytul",
    },
    "pelnomocnictwo": {
        "PRINCIPAL_PERSON": "mocodawca[].data.imie_nazwisko",
        "PRINCIPAL_KRS": "mocodawca[].data.krs",
        "PRINCIPAL_ORG": "mocodawca[].data.nazwa",
        "PRINCIPAL_NIP": "mocodawca[].data.nip",
        "SIGNATORY_PERSON": "mocodawca[].podpisujacy[].imie_nazwisko",
        "ATTORNEY_PERSON": "pelnomocnicy[].imie_nazwisko",
        "ATTORNEY_PESEL": "pelnomocnicy[].pesel",
    },
    "przelew": {
        "OPERATION_DATE": "przelew_bankowy.data_operacji",
        "TRANSFER_AMOUNT": "przelew_bankowy.kwota",
        "DESTINATION_ACCOUNT": "przelew_bankowy.na_rachunek",
        "TRANSACTION_ID": "przelew_bankowy.numer_transakcji",
        "TRANSFER_TITLE": "przelew_bankowy.tytul",
        "SOURCE_ACCOUNT": "przelew_bankowy.z_rachunku",
    },
}
_FIELD_LABELS = {
    document_type: {field: label for label, field in mapping.items()}
    for document_type, mapping in _ROBERTA_FIELDS.items()
}
_MONTH_PATTERN = "|".join(
    sorted((re.escape(month) for month in _MONTHS), key=len, reverse=True)
)
_DATE_WORD_VALUE = re.compile(
    rf"(?<!\d)(?P<d>0?[1-9]|[12]\d|3[01])\s+(?P<month>{_MONTH_PATTERN})\s+"
    r"(?P<y>18\d{2}|19\d{2}|20\d{2}|21\d{2}|22\d{2})(?!\d)",
    re.IGNORECASE,
)
_AMOUNT_TOKEN = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[ \u00a0.]\d{3})+|\d+)(?:[,.]\d{1,2})?(?!\d)"
)


@dataclass(frozen=True)
class HybridResult:
    data: dict[str, Any]
    metrics: dict[str, int]


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _flatten_scalars(value: Any, path: str = "") -> list[tuple[str, str]]:
    flattened: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            flattened.extend(_flatten_scalars(child, child_path))
    elif isinstance(value, list):
        for child in value:
            flattened.extend(_flatten_scalars(child, path + "[]"))
    elif isinstance(value, str) and value.strip():
        flattened.append((path, value.strip()))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened.append((path, str(value)))
    return flattened


def _literal_occurrences(value: str, text: str) -> list[tuple[int, int]]:
    parts = value.split()
    if not parts:
        return []
    pattern = r"\s+".join(re.escape(part) for part in parts)
    if value[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if value[-1].isalnum():
        pattern += r"(?!\w)"
    return [
        (match.start(), match.end())
        for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
    ]


def _compact_occurrences(value: str, text: str, *, digits_only: bool) -> list[tuple[int, int]]:
    positions: list[int] = []
    compact: list[str] = []
    for index, character in enumerate(text):
        if character.isdigit() if digits_only else character.isalnum():
            positions.append(index)
            compact.append(character)
    if digits_only:
        needle = "".join(character for character in value if character.isdigit())
    else:
        needle = "".join(character for character in value if character.isalnum())
    needle = needle.casefold()
    haystack = "".join(compact).casefold()
    if len(needle) < (6 if digits_only else 10):
        return []
    output = []
    start = 0
    while (index := haystack.find(needle, start)) >= 0:
        output.append((positions[index], positions[index + len(needle) - 1] + 1))
        start = index + 1
    return output


def _date_key(value: str) -> date | None:
    candidate = value.strip()
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        pass
    for pattern in (_DATE_ISO, _DATE_NUMERIC, _DATE_WORD_VALUE):
        match = pattern.fullmatch(candidate)
        if not match:
            continue
        groups = match.groupdict()
        month = groups["m"] if "m" in groups else _MONTHS.get(_fold(groups["month"]))
        try:
            return date(int(groups["y"]), int(month), int(groups["d"]))
        except (TypeError, ValueError):
            return None
    return None


def _date_occurrences(value: str, text: str) -> list[tuple[int, int]]:
    target = _date_key(value)
    if target is None:
        return []
    found: list[tuple[int, int]] = []
    for pattern in (_DATE_ISO, _DATE_NUMERIC, _DATE_WORD):
        for match in pattern.finditer(text):
            groups = match.groupdict()
            month = groups["m"] if "m" in groups else _MONTHS.get(_fold(groups["month"]))
            try:
                parsed = date(int(groups["y"]), int(month), int(groups["d"]))
            except (TypeError, ValueError):
                continue
            if parsed == target:
                found.append((match.start(), match.end()))
    for match in _DATE_WORD_VALUE.finditer(text):
        groups = match.groupdict()
        month = _MONTHS.get(_fold(groups["month"]))
        try:
            parsed = date(int(groups["y"]), int(month), int(groups["d"]))
        except (TypeError, ValueError):
            continue
        if parsed == target:
            found.append((match.start(), match.end()))
    return list(dict.fromkeys(found))


def _decimal_key(value: str) -> Decimal | None:
    candidate = re.sub(r"[^\d,.+-]", "", value).replace("\u00a0", "")
    if "," in candidate and "." in candidate:
        if candidate.rfind(",") > candidate.rfind("."):
            candidate = candidate.replace(".", "").replace(",", ".")
        else:
            candidate = candidate.replace(",", "")
    elif "," in candidate:
        candidate = candidate.replace(",", ".")
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def _amount_occurrences(value: str, text: str) -> list[tuple[int, int]]:
    target = _decimal_key(value)
    if target is None:
        return []
    found = []
    for match in _AMOUNT_TOKEN.finditer(text):
        if _decimal_key(match.group()) == target:
            found.append((match.start(), match.end()))
    return found


def _source_occurrences(field: str, value: str, text: str) -> list[tuple[int, int]]:
    matches = _literal_occurrences(value, text)
    if matches:
        return matches
    key = field.rsplit(".", 1)[-1].casefold()
    if key in {"data", "data_operacji", "czynnosc_dzien"}:
        matches = _date_occurrences(value, text)
        if matches:
            return matches
    if key == "kwota":
        matches = _amount_occurrences(value, text)
        if matches:
            return matches
    if any(token in key for token in ("nip", "regon", "krs", "pesel", "rachunku")):
        return _compact_occurrences(value, text, digits_only=True)
    if key in {"na_rachunek", "z_rachunku"}:
        return _compact_occurrences(value, text, digits_only=False)
    return []


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return bool(union) and intersection / union >= 0.8


def _poa_attorney_has_pesel(data: dict[str, Any], value: str) -> bool:
    return any(
        _normalise(str(person.get("imie_nazwisko") or "")) == _normalise(value)
        and bool(person.get("pesel"))
        for person in (data.get("pelnomocnicy") or [])
    )


def _move_poa_person(
    data: dict[str, Any],
    source: str | None,
    target: str,
    value: str,
    group_id: str | None = None,
) -> None:
    """Move one POA person, using the source array index when it is known."""
    principal_field = _ROBERTA_FIELDS["pelnomocnictwo"]["PRINCIPAL_PERSON"]
    attorney_field = _ROBERTA_FIELDS["pelnomocnictwo"]["ATTORNEY_PERSON"]

    if source == principal_field:
        parties = data.get("mocodawca") or []
        prefix = group_id.split("/", 1)[0] if group_id else ""
        try:
            source_index = int(prefix.split(":", 1)[1]) if prefix.startswith("mocodawca:") else None
        except (IndexError, ValueError):
            source_index = None
        selected_parties = (
            [parties[source_index]]
            if source_index is not None and 0 <= source_index < len(parties)
            else parties if group_id is None else []
        )
        for party in selected_parties:
            person = (party.get("data") or {}).get("imie_nazwisko")
            if isinstance(person, str) and _normalise(person) == _normalise(value):
                party["data"]["imie_nazwisko"] = None
    elif source == attorney_field:
        attorneys = data.get("pelnomocnicy") or []
        prefix = group_id.split("/", 1)[0] if group_id else ""
        try:
            source_index = int(prefix.split(":", 1)[1]) if prefix.startswith("pelnomocnicy:") else None
        except (IndexError, ValueError):
            source_index = None
        if source_index is not None and 0 <= source_index < len(attorneys):
            if _normalise(str(attorneys[source_index].get("imie_nazwisko") or "")) == _normalise(value):
                del attorneys[source_index]
        elif group_id is None:
            data["pelnomocnicy"] = [
                person
                for person in attorneys
                if _normalise(str(person.get("imie_nazwisko") or "")) != _normalise(value)
            ]

    if target == attorney_field:
        attorneys = data.get("pelnomocnicy")
        if not isinstance(attorneys, list):
            attorneys = []
            data["pelnomocnicy"] = attorneys
        if not any(
            _normalise(str(person.get("imie_nazwisko") or "")) == _normalise(value)
            for person in attorneys
        ):
            attorneys.append({"imie_nazwisko": value, "pesel": None})
    elif target == principal_field:
        parties = data.get("mocodawca")
        if not isinstance(parties, list):
            parties = []
            data["mocodawca"] = parties
        for party in parties:
            party_data = party.setdefault("data", {})
            person = party_data.get("imie_nazwisko")
            if isinstance(person, str) and _normalise(person) == _normalise(value):
                return
        parties.append(
            {
                "data": {
                    "imie_nazwisko": value,
                    "krs": None,
                    "nazwa": None,
                    "nip": None,
                    "typ": None,
                },
                "podpisujacy": [{"imie_nazwisko": None}],
            }
        )


class HybridExtractor:
    """Combine document rules and local ModernBERT over selectable or OCR text."""

    name = "local-regex-modernbert"

    def __init__(
        self,
        *,
        roberta_bundle: Path | None = None,
        roberta_config: Path | None = None,
        device: str = "cpu",
        policy: str = "rank",
    ) -> None:
        if policy != "rank":
            raise ValueError("ModernBERT policy must be 'rank'.")
        self.policy = policy
        self.roberta_bundle = roberta_bundle.resolve() if roberta_bundle else None
        self.roberta_config_path = roberta_config.resolve() if roberta_config else None
        self.device_name = device
        self._regex = RegexExtractor()
        self._torch: Any = None
        self._model_class: Any = None
        self._tokenizer: Any = None
        self._models: dict[str, tuple[Any, dict[int, str], float, int, int]] = {}
        self._config: dict[str, Any] = {}

        if self.roberta_bundle is not None and not self.roberta_bundle.is_dir():
            raise ValueError(f"Local ModernBERT bundle does not exist: {self.roberta_bundle}")
        if self.roberta_bundle is not None:
            missing_types = [
                document_type
                for document_type in self._regex.document_types
                if not (self.roberta_bundle / document_type).is_dir()
            ]
            if missing_types:
                raise ValueError(
                    "ModernBERT bundle is missing checkpoints for: "
                    + ", ".join(sorted(missing_types))
                )
            try:
                import torch
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    PreTrainedTokenizerFast,
                )
            except ImportError as exc:
                raise ValueError(
                    "Local ModernBERT inference needs optional dependencies; run "
                    "WITH_LOCAL_ROBERTA=1 ./setup.sh."
                ) from exc
            self._torch = torch
            self._model_class = AutoModelForTokenClassification
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested for ModernBERT but is unavailable.")
            tokenizer_dir = self.roberta_bundle / "odwolanie"
            try:
                tokenizer_config = json.loads(
                    (tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8")
                )
                if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
                    special_tokens = {}
                    for key in (
                        "unk_token",
                        "cls_token",
                        "sep_token",
                        "pad_token",
                        "mask_token",
                        "bos_token",
                        "eos_token",
                    ):
                        value = tokenizer_config.get(key)
                        if isinstance(value, dict):
                            value = value.get("content")
                        if value is not None:
                            special_tokens[key] = value
                    self._tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
                        **special_tokens,
                    )
                else:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_dir,
                        use_fast=True,
                        local_files_only=True,
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Cannot load the ModernBERT tokenizer locally from bundle/odwolanie."
                ) from exc
        if self.roberta_bundle is not None and self.roberta_config_path is not None:
            if not self.roberta_config_path.is_file():
                raise ValueError(
                    f"Local ModernBERT inference config does not exist: {self.roberta_config_path}"
                )
            try:
                self._config = json.loads(
                    self.roberta_config_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Cannot read local ModernBERT inference config.") from exc

    @property
    def document_types(self) -> frozenset[str]:
        return self._regex.document_types

    @property
    def model_enabled(self) -> bool:
        return self.roberta_bundle is not None

    @property
    def model_name(self) -> str | None:
        return str(self._config.get("model", "PKOBP/embed-modernbert-68m")) if self.model_enabled else None

    def extract(self, document_type: str, text: str, *, ocr: bool = False) -> HybridResult:
        if document_type not in self._regex.document_types:
            raise ValueError(f"No local regex extractor for document type: {document_type}")
        if ocr:
            return self._extract_scan(document_type, text)
        extraction = self._regex.extract_with_candidates(document_type, text)
        data = extraction.data
        regex_candidates: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        output_values = {
            (field, _normalise(value))
            for field, value in _flatten_scalars(data)
        }

        def candidate_row(
            field: str,
            value: str,
            group_id: str | None,
        ) -> dict[str, Any]:
            key = (field, _normalise(value), group_id)
            return regex_candidates.setdefault(
                key,
                {
                    "field": field,
                    "value": value,
                    "locations": [],
                    "occurrence_candidates": {},
                    "regex_score": 0.0,
                    "passed_minimum_score": False,
                    "rule_ids": set(),
                    "selected_by_regex": (field, key[1]) in output_values,
                    "group_id": group_id,
                },
            )

        for candidate in extraction.candidates:
            row = candidate_row(candidate.field, candidate.value, candidate.group_id)
            span = (candidate.start, candidate.end)
            if span not in row["occurrence_candidates"]:
                row["occurrence_candidates"][span] = candidate
            row["locations"].append(span)
            row["regex_score"] = max(row["regex_score"], candidate.score)
            row["passed_minimum_score"] = (
                row["passed_minimum_score"] or candidate.passed_minimum_score is True
            )
            row["rule_ids"].update(candidate.rule_ids)

        # Some regex values are normalized (for example, dates or account
        # numbers) and therefore cannot be represented by a literal candidate
        # span. Keep the existing normalization-aware locator as a fallback.
        for field, value in _flatten_scalars(data):
            matching_rows = [
                row
                for (row_field, row_value, _group_id), row in regex_candidates.items()
                if row_field == field and row_value == _normalise(value)
            ]
            if not matching_rows:
                matching_rows = [candidate_row(field, value, None)]
            source_locations = _source_occurrences(field, value, text)
            if document_type == "pelnomocnictwo" and field.endswith(".imie_nazwisko"):
                source_locations = source_locations[:1]
            for item in matching_rows:
                item["locations"].extend(source_locations)
        for item in regex_candidates.values():
            item["locations"] = list(dict.fromkeys(item["locations"]))

        metrics = {
            "regex_candidates": len(regex_candidates),
            "roberta_spans": 0,
            "scored_candidates": 0,
            "supported": 0,
            "contradicted": 0,
            "unconfirmed": 0,
            "unlocated": 0,
            "ambiguous_location": 0,
            "unmapped": 0,
            "regex_only": 0,
            "removed": 0,
            "reassigned": 0,
            "fallback_retained": 0,
            "poa_role_ambiguous": 0,
            "poa_candidates_selected": 0,
        }
        if not self.model_enabled:
            metrics["regex_only"] = len(regex_candidates)
            return HybridResult(data, metrics)

        predicted_spans, token_logits, id2label, token_threshold = (
            self._predict_spans_with_logits(document_type, text)
        )
        fields_by_label = _ROBERTA_FIELDS[document_type]
        labels_by_field = _FIELD_LABELS[document_type]
        indexed_spans = [
            (start, end, label, fields_by_label[label])
            for start, end, label in predicted_spans
            if label in fields_by_label
        ]
        metrics["roberta_spans"] = len(indexed_spans)
        role_fields = {
            fields_by_label[label]: label
            for label in ("PRINCIPAL_PERSON", "ATTORNEY_PERSON")
        } if document_type == "pelnomocnictwo" else {}

        # Score exact regex spans using averaged BIO logits. For POA role
        # choices, preserve the source list item and use only the first mention.
        role_occurrences: dict[
            tuple[str, str | None], dict[tuple[int, int], CandidateEvidence]
        ] = defaultdict(dict)
        first_role_occurrence: dict[str, tuple[int, int]] = {}
        for key, candidate in regex_candidates.items():
            if key[0] not in role_fields or not (
                candidate["selected_by_regex"] or candidate["passed_minimum_score"]
            ):
                continue
            for location in candidate["locations"]:
                current = first_role_occurrence.get(key[1])
                if current is None or location < current:
                    first_role_occurrence[key[1]] = location

        candidate_scorer = CandidateSpanScorer(
            token_logits,
            id2label,
            token_threshold,
        )
        for key, candidate in regex_candidates.items():
            field = candidate["field"]
            expected_label = labels_by_field.get(field)
            locations = candidate["locations"]
            if expected_label is None:
                metrics["unmapped"] += 1
                continue
            if not locations:
                metrics["unlocated"] += 1
                continue

            for location in locations:
                proposed = candidate["occurrence_candidates"].get(location)
                if proposed is None:
                    proposed = RegexCandidate(
                        field=field,
                        value=candidate["value"],
                        start=location[0],
                        end=location[1],
                        score=float(candidate["regex_score"]),
                        rule_ids=tuple(sorted(candidate["rule_ids"])),
                    )
                evidence = candidate_scorer.score(proposed, expected_label)
                metrics["scored_candidates"] += 1
                location_labels = {
                    predicted_field
                    for start, end, _label, predicted_field in indexed_spans
                    if _overlaps(location, (start, end))
                }
                if location_labels == {field}:
                    metrics["supported"] += 1
                elif location_labels and field not in location_labels:
                    metrics["contradicted"] += 1
                else:
                    metrics["unconfirmed"] += 1

                if field in role_fields and (
                    candidate["selected_by_regex"] or candidate["passed_minimum_score"]
                ) and location == first_role_occurrence.get(key[1]):
                    role_occurrences[(key[1], candidate["group_id"])][location] = evidence

        assignments: list[tuple[str, str, str, str | None, bool]] = []
        assignment_by_group: dict[tuple[str, str | None], str] = {}
        if role_fields:
            role_field_by_label = {label: field for field, label in role_fields.items()}
            selected_groups_by_value: dict[str, set[str]] = defaultdict(set)
            for (field, value_key, group_id), row in regex_candidates.items():
                if field in role_fields and group_id and row["selected_by_regex"]:
                    selected_groups_by_value[value_key].add(group_id)

            for (value_key, group_id), occurrences in role_occurrences.items():
                rows = [
                    row
                    for (field, row_value, row_group), row in regex_candidates.items()
                    if field in role_fields
                    and row_value == value_key
                    and row_group == group_id
                ]
                eligible_rows = [
                    row
                    for row in rows
                    if row["selected_by_regex"] or row["passed_minimum_score"]
                ]
                if not eligible_rows:
                    continue
                row = max(eligible_rows, key=lambda item: item["regex_score"])
                if group_id is None:
                    if row["selected_by_regex"] or selected_groups_by_value.get(value_key):
                        # Without a structural source group, a repeated name
                        # cannot safely mutate a particular JSON list item.
                        metrics["poa_role_ambiguous"] += 1
                        continue
                    if not row["passed_minimum_score"]:
                        continue

                predicted_fields: set[str] = set()
                for evidence in occurrences.values():
                    if not all(evidence.boundary_coverage):
                        continue
                    predicted_role = evidence.best_predicted_role
                    target_field = role_field_by_label.get(predicted_role or "")
                    if target_field is not None:
                        predicted_fields.add(target_field)
                if len(predicted_fields) != 1:
                    metrics["poa_role_ambiguous"] += 1
                    continue
                target_field = next(iter(predicted_fields))
                if (
                    target_field == _ROBERTA_FIELDS["pelnomocnictwo"]["PRINCIPAL_PERSON"]
                    and _poa_attorney_has_pesel(data, row["value"])
                ):
                    target_field = _ROBERTA_FIELDS["pelnomocnictwo"]["ATTORNEY_PERSON"]
                assignments.append(
                    (
                        row["field"],
                        target_field,
                        row["value"],
                        group_id,
                        row["selected_by_regex"],
                    )
                )
                assignment_by_group[(value_key, group_id)] = target_field

        for key, candidate in regex_candidates.items():
            field = candidate["field"]
            locations = list(dict.fromkeys(candidate["locations"]))
            location_labels = []
            for location in locations:
                location_labels.append(
                    {
                        predicted_field
                        for start, end, _label, predicted_field in indexed_spans
                        if _overlaps(location, (start, end))
                    }
                )
            if not candidate["selected_by_regex"]:
                continue
            if field not in labels_by_field or not locations:
                metrics["fallback_retained"] += 1
                continue
            supported_locations = [labels == {field} for labels in location_labels]
            contradicted_locations = [bool(labels) and field not in labels for labels in location_labels]
            assigned = assignment_by_group.get((key[1], candidate["group_id"]))
            if any(supported_locations) or assigned == field:
                continue
            if assigned and field in role_fields:
                continue
            if contradicted_locations and all(contradicted_locations):
                metrics["fallback_retained"] += 1
            else:
                if len(locations) > 1:
                    metrics["ambiguous_location"] += 1
                metrics["fallback_retained"] += 1

        if role_fields:
            principal_field = _ROBERTA_FIELDS["pelnomocnictwo"]["PRINCIPAL_PERSON"]
            assignments.sort(
                key=lambda item: (
                    item[0] == _ROBERTA_FIELDS["pelnomocnictwo"]["ATTORNEY_PERSON"]
                    and item[1] == principal_field,
                    int(item[3].split(":", 1)[1].split("/", 1)[0])
                    if item[3]
                    and item[3].startswith("pelnomocnicy:")
                    and item[0] == _ROBERTA_FIELDS["pelnomocnictwo"]["ATTORNEY_PERSON"]
                    else 0,
                ),
                reverse=True,
            )
            for source_field, target_field, value, group_id, selected in assignments:
                if selected and source_field == target_field:
                    continue
                if selected and group_id is not None:
                    _move_poa_person(data, source_field, target_field, value, group_id)
                    metrics["reassigned"] += 1
                elif not selected:
                    # These alternatives passed the regex's original minimum
                    # score and are admitted only when the BIO model assigns
                    # one role above O at the existing per-type threshold.
                    already_present = any(
                        _normalise(str((party.get("data") or {}).get("imie_nazwisko") or ""))
                        == _normalise(value)
                        for party in data.get("mocodawca") or []
                    ) or any(
                        _normalise(str(person.get("imie_nazwisko") or "")) == _normalise(value)
                        for person in data.get("pelnomocnicy") or []
                    )
                    if already_present:
                        continue
                    _move_poa_person(data, None, target_field, value)
                    metrics["poa_candidates_selected"] += 1

        return HybridResult(data, metrics)

    def _extract_scan(self, document_type: str, text: str) -> HybridResult:
        repaired = repair_ocr_text(text)
        data = self._regex.extract(document_type, repaired)
        repair_scan_fields(document_type, data, repaired, text)
        metrics = {"ocr_documents": 1, "roberta_spans": 0, "ocr_model_selected": 0}
        if self.model_enabled:
            spans, logits, labels, threshold = self._predict_spans_with_logits(document_type, text)
            scorer = CandidateSpanScorer(logits, labels, threshold)
            metrics["roberta_spans"] = len(spans)
            metrics.update(fuse_scan_model(document_type, data, text, spans, scorer))
        return HybridResult(data, metrics)

    def _load_model(self, document_type: str) -> tuple[Any, dict[int, str], float, int, int]:
        cached = self._models.get(document_type)
        if cached is not None:
            return cached
        assert self.roberta_bundle is not None
        model_dir = self.roberta_bundle / document_type
        if not model_dir.is_dir():
            raise ValueError(f"No fine-tuned ModernBERT checkpoint for {document_type}.")
        try:
            model = self._model_class.from_pretrained(
                model_dir,
                local_files_only=True,
            ).to(self.device_name)
        except OSError as exc:
            raise ValueError(
                f"Cannot load the local ModernBERT checkpoint for {document_type}."
            ) from exc
        model.eval()

        labels_file = model_dir / "labels.json"
        if labels_file.is_file():
            try:
                raw_labels = json.loads(labels_file.read_text(encoding="utf-8"))
                id2label = {int(index): str(label) for index, label in raw_labels.items()}
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid ModernBERT label map: {labels_file}") from exc
        else:
            id2label = {
                int(index): str(label)
                for index, label in model.config.id2label.items()
            }
        if not id2label or any(index not in id2label for index in range(model.config.num_labels)):
            raise ValueError(f"ModernBERT label map is incomplete for {document_type}.")

        type_config = self._config.get("selected_by_type", {}).get(document_type, {})
        threshold = float(
            type_config.get(
                "median_validation_threshold",
                type_config.get("entity_logit_threshold", 0.0),
            )
        )
        window_size = int(type_config.get("window_size", 512))
        stride = int(type_config.get("stride", 96))
        loaded = (model, id2label, threshold, window_size, stride)
        self._models[document_type] = loaded
        return loaded

    def _predict_spans(self, document_type: str, text: str) -> list[tuple[int, int, str]]:
        """Compatibility wrapper returning decoded model spans only."""
        return self._predict_spans_with_logits(document_type, text)[0]

    def _predict_spans_with_logits(
        self,
        document_type: str,
        text: str,
    ) -> tuple[
        list[tuple[int, int, str]],
        dict[tuple[int, int], tuple[float, ...]],
        dict[int, str],
        float,
    ]:
        """Return decoded spans and averaged raw BIO logits by source offset."""
        model, id2label, threshold, window_size, stride = self._load_model(document_type)
        tokenizer = self._tokenizer
        if not text.strip():
            return [], {}, id2label, threshold
        try:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=window_size,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError("ModernBERT could not tokenize the source text.") from exc

        windows = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offsets": [tuple(map(int, offset)) for offset in offsets],
            }
            for input_ids, attention_mask, offsets in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["offset_mapping"],
            )
        ]
        torch = self._torch
        score_sums: dict[tuple[int, int], list[float]] = {}
        score_counts: dict[tuple[int, int], int] = defaultdict(int)
        with torch.inference_mode():
            for batch_start in range(0, len(windows), 8):
                batch = windows[batch_start : batch_start + 8]
                inputs = tokenizer.pad(
                    [
                        {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
                        for item in batch
                    ],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device_name) for key, value in inputs.items()}
                logits = model(**inputs).logits.float().cpu().tolist()
                for window, token_logits in zip(batch, logits):
                    for offset, values in zip(window["offsets"], token_logits):
                        if offset == (0, 0):
                            continue
                        if offset not in score_sums:
                            score_sums[offset] = [float(value) for value in values]
                        else:
                            score_sums[offset] = [
                                prior + float(value)
                                for prior, value in zip(score_sums[offset], values)
                            ]
                        score_counts[offset] += 1

        offsets = sorted(score_sums)
        averaged_logits = {
            offset: tuple(value / score_counts[offset] for value in score_sums[offset])
            for offset in offsets
        }
        tag_ids: list[int] = []
        for offset in offsets:
            scores = list(averaged_logits[offset])
            for label_id in range(1, len(scores)):
                scores[label_id] -= threshold
            tag_ids.append(max(range(len(scores)), key=scores.__getitem__))

        spans: list[tuple[int, int, str]] = []
        active_label: str | None = None
        active_start = 0
        active_end = 0

        def close_span() -> None:
            nonlocal active_label
            if active_label is not None:
                spans.append((active_start, active_end, active_label))
            active_label = None

        for (start, end), tag_id in zip(offsets, tag_ids):
            tag = id2label[tag_id]
            if tag == "O":
                close_span()
                continue
            prefix, label = tag.split("-", 1)
            if prefix == "B" or active_label != label:
                close_span()
                active_label, active_start, active_end = label, start, end
            else:
                active_end = end
        close_span()
        return spans, averaged_logits, id2label, threshold
