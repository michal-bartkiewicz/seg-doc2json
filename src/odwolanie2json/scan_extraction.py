"""Repair OCR artifacts and combine text rules with local entity predictions."""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .candidate_scoring import CandidateSpanScorer
from .candidate_types import RegexCandidate
from .regex_extraction import (
    _CAUSES,
    _bounded_person_candidates,
    _candidate_span,
    _clean_org_name,
    _fold,
    _header_date_city,
    _identifier_hits,
    _normalize_poa_company_name,
    _ocr_literal_pattern,
    _parse_amount_value,
    _parse_dates,
    _poa_action_match,
    _poa_attorney_segment,
    validate_nrb,
)


def _join(text: str) -> str:
    return " ".join(text.split())


def repair_ocr_text(text: str) -> str:
    text = re.sub(r"(?i)\bsp\.?[ \t]+z[ \t]+[o0]\.[ \t]*[o0]\b\.?", "sp. z o.o.", text)
    text = re.sub(r"(?<=\d)(?=r\.)", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(\d{1,2})[-. ](0?[1-9]|1[0-2])[-. ](20\d{2})\b",
        r"\1.\2.\3",
        text,
    )
    text = re.sub(
        r"\b(20\d{2})[./-](0\d|1[0-2])[./-]([0-3]\d)\b",
        r"\1-\2-\3",
        text,
    )
    return re.sub(r"(?<=\w)/[ \t]+(?=\d{2,}(?:/|[ \t]*[,;]))", "/", text)


_RECEIPT_LABEL = re.compile(
    r"(?im)^\s*[^\w\n]*?(?:"
    + _ocr_literal_pattern("tytuł")
    + r"(?:em|[ \t]+(?:transakcji|płatności|przelewu))?\b"
    + r"|opis(?=[ \t]*[:：=]|[ \t]*$)"
    + r"|" + _ocr_literal_pattern("szczegóły płatności") + r"\b)"
    + r"[ \t:：=–|]*"
)
_RECEIPT_END = re.compile(
    r"(?im)^[ \t]*[^\w\n]*(?:kwota\b|data\b|status\b|typ\s+(?:transakcji|przelewu)\b|"
    r"(?:z|na)\s+rachun\w*\b|dane\s+(?:nadawcy|odbiorcy)\b|"
    r"(?:numer|nr\.?|identyfikator)\b|dokument\b|niniejsze\b|"
    r"wydruk\b|potwierdzenie\b|opłata\s+za\s+przelew\b|bank\b|"
    r"strona\b|nie\s+wymaga\b|s[.:]?i?gnature\s+not\b|"
    r"operacj\w*\s*:|identyfikator\b)"
)


def _receipt_title(text: str) -> str | None:
    for marker in _RECEIPT_LABEL.finditer(text):
        following = text[marker.end():marker.end() + 850]
        stop = _RECEIPT_END.search(following)
        if stop:
            following = following[:stop.start()]
        rows = [
            line.strip(" \t'‘’«»\"„”|:;_")
            for line in following.splitlines()
            if line.strip(" \t'‘’«»\"„”|:;.-")
        ]
        if not rows or len(rows) > 4:
            continue
        # OCR of some banking tables repeats a truncated title immediately
        # before the full value in the neighboring column.
        if len(rows) >= 2 and rows[1].casefold().startswith(rows[0].casefold()):
            rows.pop(0)
        value = _join(" ".join(rows)).strip(" \t,;:.\"'‘’„”«»|")
        if 3 <= len(value) <= 300 and re.search(r"[^\W\d_]", value):
            return value
    return None


def _repair_receipt(data: dict[str, Any], text: str, original: str) -> None:
    fields = data["przelew_bankowy"]
    title = _receipt_title(original)
    if title and (
        not fields["tytul"]
        or re.search(r"(?i)\b(?:wpis|op[łl]at[ay])\b", title)
    ):
        fields["tytul"] = title
    iban_fields = []
    for field in ("z_rachunku", "na_rachunek"):
        account = fields[field]
        if not account or not re.fullmatch(r"\d{26}", account):
            continue
        grouped = r"[ \t\u00a0]*".join(account)
        if re.search(rf"(?i)(?<!\w)PL[ \t]*{grouped}(?!\d)", original):
            iban_fields.append(field)
    # Receipts often prefix both domestic accounts with PL as decoration.
    # Keep an explicit prefix when the source uses different formats by role.
    if len(iban_fields) == 1 and fields["z_rachunku"] and fields["na_rachunek"]:
        field = iban_fields[0]
        fields[field] = "PL" + fields[field]
    if not fields["numer_transakcji"]:
        match = re.search(r"(?im)^[ \t]*numer[ \t:|]+(\d{5,22})[ \t]*\n\s*transakcji\b", text)
        match = match or re.search(
            r"(?im)^[ \t]*(?:identyfikator(?:\s+\w+)?|nr\.?\s+referencyjn\w*|numer\s+referencyjn\w*"
            r"(?:\s+transakcji)?)\s*[:|–-]?\s*([A-Z0-9][A-Z0-9./_-]{4,45})\b",
            text,
        )
        if match and re.search(r"\d", match.group(1)):
            fields["numer_transakcji"] = match.group(1)


def _repair_poa(data: dict[str, Any], text: str) -> None:
    action = _poa_action_match(text)
    if not action:
        return
    prefix = text[:action.start()]
    heading = re.search(r"(?im)^\s*PE[ŁL]NOMOCNICTWO\s*$", prefix)
    if heading and len(data.get("mocodawca") or []) == 1:
        grantor = data["mocodawca"][0]
        fields = grantor["data"]
        block = prefix[heading.end():]
        person = re.search(
            r"(?is)\bni[żz]ej\s+podpisan\w*(?:\s*/\s*[aey])?\s+([^,\n]{4,75})\s*,?\s*"
            r"prowadz\w*(?:\s*/\s*a)?\s+dzia[łl]alno[śs][ćc]",
            block,
        )
        names = _bounded_person_candidates(person.group(1)) if person else []
        if names:
            fields.update(imie_nazwisko=names[0], nazwa=None, typ="osoba_fizyczna")
            if not any(item["imie_nazwisko"] for item in grantor["podpisujacy"]):
                grantor["podpisujacy"] = [{"imie_nazwisko": names[0]}]
        else:
            org = re.search(r"(?im)^\s*([^\n]{4,150}?)\s+z\s+siedzib\w*\b", block)
            if org and (fields["krs"] or fields["typ"] == "osoba_prawna"):
                name = _normalize_poa_company_name(_clean_org_name(org.group(1)))
                if name:
                    fields["nazwa"] = name

    attorneys = data["pelnomocnicy"]
    segment = _poa_attorney_segment(text, action)
    roles = {
        _fold(item["data"]["imie_nazwisko"])
        for item in (data.get("mocodawca") or [])
        if item["data"]["imie_nazwisko"]
    }
    repaired = []
    for entry in attorneys:
        span = _candidate_span(segment, entry["imie_nazwisko"])
        if span and "\n" in span.group():
            # If each line already has its own name, discard matches that
            # concatenate the end of one entry with the next person's name.
            start = segment.rfind("\n", 0, span.start()) + 1
            end = segment.find("\n", span.end())
            covered = segment[start:end if end >= 0 else len(segment)]
            if len([line for line in covered.splitlines() if _bounded_person_candidates(line)]) >= 2:
                continue
        if _fold(entry["imie_nazwisko"]) not in roles:
            repaired.append(entry)
    data["pelnomocnicy"] = repaired or attorneys


_CITY_SPELLINGS = {
    _fold(city): city for city in (
        "Białystok", "Bydgoszcz", "Gdańsk", "Gdynia", "Gorzów Wielkopolski",
        "Jelenia Góra", "Kraków", "Łódź", "Olsztyn", "Opole", "Poznań",
        "Rzeszów", "Szczecin", "Toruń", "Warszawa", "Wrocław",
        "Zielona Góra", "Żyrardów",
    )
}
_ROLE_LINE = re.compile(r"(?im)^[ \t]*(?:Odwo[łl]uj[ąa]cy|Zamawiaj[ąa]cy|Dotyczy)\s*:")
_EMAIL_LINE = re.compile(r"(?im)^[^\n]{0,18}\be[ -]?mail\s*[:;]\s*([^\n]+)")
_COUNSEL_FIRM = re.compile(r"(?im)^[ \t]*([^\n]{0,160}\bkancelari[aią]\b[^\n]{0,160})$")
_AUTHORITY = re.compile(
    r"(?i)^(?:powiatow|miejski|gminn|zesp[oó][łl]|urz[ąa]d|gmina|miasto|"
    r"zarz[ąa]d|zak[łl]ad|uniwersytet|politechnika|komenda|muzeum|wojew[oó]dz|szpital)"
)


def _mail_from_ocr(value: str, name: str | None) -> str | None:
    value = _fold(value.replace("©", "@").replace("€", "@")).replace("|", "l")
    left, separator, right = value.partition("@")
    if not separator:
        split = re.fullmatch(r"\s*([a-z0-9._ +'-]+?)(?:q|\(a\)?)\s*([a-z0-9.-]+\.[a-z]{2,})(?:\s.*)?", value)
        if split:
            left, right = split.groups()
        elif name:
            first, *rest = _fold(name).split()
            surname = rest[-1] if rest else ""
            if not surname:
                return None
            embedded = re.match(
                rf"\s*([a-z0-9._+'-]*{re.escape(surname)})a\s*([a-z0-9.-]+\.[a-z]{{2,}})",
                value,
            )
            if not embedded:
                return None
            left, right = embedded.groups()
        else:
            return None
    host = re.match(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?![a-z0-9-])", right.lstrip())
    if not host:
        return None
    left = re.sub(r"(?<=\w)\s+(?=\w)", ".", left.strip())
    if name and re.fullmatch(r"[a-z]\.[a-z-]+", left):
        given, *family = _fold(name).split()
        expected = f"{given[0]}.{family[-1]}" if family else left
        if SequenceMatcher(None, left, expected).ratio() >= 0.86:
            left = expected
    return f"{left}@{host.group()}" if re.fullmatch(r"[a-z0-9._+'-]+", left) else None


def _repair_appeal_cause(fields: dict[str, Any], text: str) -> None:
    intro = re.search(r"(?is)\bwnosz\w*\s+odwo[łl]ani\w*\s+wobec\s*:", text[:6500])
    if not intro:
        return
    clause = text[intro.end():intro.end() + 1300]
    stop = re.search(r"(?im)^\s*(?:zaskar[żz]onym|zarzucam|wnosz\w*\s+o\b|uzasadnienie)\b", clause)
    if stop:
        clause = clause[:stop.start()]
    bullet = re.search(r"(?m)^\s*(?:\d{1,2}|[I|l])\s*[.)]\s+([^;]+)", clause)
    first = bullet.group(1) if bullet else clause.split(";", 1)[0]
    first = _fold(_join(first))[:500]
    if not first.strip():
        return
    chosen = None
    # Explicit acts in the scope of the first bullet outrank matching phrases
    # in the statutory citations and in the requested remedy.
    if re.match(r"^tresci\s+(?:wyjasnien|odpowiedzi)\b", first):
        chosen = 6
    elif re.match(r"^tresci\s+specyfikacji\b", first) and re.search(r"opublik|udostep", first):
        chosen = 4
    elif re.match(r"^(?:czynnosci\s+)?uniewazn\w*\s+postepowan", first):
        chosen = 9
    elif re.match(r"^(?:czynnosci\s+)?wyklucz\w*\s+odwolujac", first):
        chosen = 3
    else:
        matches = [
            (match.start(), cause)
            for cause, _, patterns in _CAUSES
            for pattern in patterns
            if (match := re.search(_fold(pattern), first))
        ]
        if matches:
            chosen = min(matches)[1]
    if chosen:
        fields["czynnosc_powod_id"] = chosen
        fields["czynnosc_opis"] = next(description for cause, description, _ in _CAUSES if cause == chosen)
    dates = _parse_dates(bullet.group(1) if bullet else clause.split(";", 1)[0])
    if dates and dates[0][0][:4] >= "2010":
        filing = fields["data"]
        if not filing or dates[0][0] <= filing:
            fields["czynnosc_dzien"] = dates[0][0]


def _repair_appeal(data: dict[str, Any], text: str) -> None:
    filing = data["OdwolanieKIO"]
    header_end = re.search(r"(?im)^[ \t]*ODWO[ŁL]ANIE[ \t]*$", text)
    header = text[:header_end.start()] if header_end else text[:4300]
    if not filing["data"] or not filing["miejscowosc"]:
        city_lines = re.sub(r"(?im)^([\w-]+(?:[ \t]+[A-ZĄĆĘŁŃÓŚŹŻ][\w-]+){0,2})[ \t]+(?=\d{1,2}[./-]\d{1,2}[./-]\d{4}\b)", r"\1, ", header)
        found_date, city = _header_date_city(city_lines)
        filing["data"] = filing["data"] or found_date
        filing["miejscowosc"] = filing["miejscowosc"] or city
    if filing["miejscowosc"]:
        city = filing["miejscowosc"]
        filing["miejscowosc"] = _CITY_SPELLINGS.get(_fold(city), city)
    _repair_appeal_cause(filing, text)

    appellant = data["Odwolujacy"][0]
    counsel = appellant["pelnomocnik"]
    representative = re.search(
        r"(?i)\b(?:reprezentowan\w*|dzia[łl]aj\w*)\s+przez\b[^\n]*?(?:pe[łl]nomocni\w*)?[ \t]*[:–-]?",
        header,
    )
    role_lines = list(_ROLE_LINE.finditer(header))
    authority_label = next(
        (role for role in role_lines if _fold(role.group()).startswith("zamawiaj") and (
            not representative or representative.start() < role.start()
        )),
        None,
    )
    if representative:
        attorney_header = header[representative.end():authority_label.start() if authority_label else len(header)]
        if not counsel["imie_nazwisko"]:
            names = _bounded_person_candidates(attorney_header[:220])
            if names:
                counsel["imie_nazwisko"] = names[0]
        firms = list(_COUNSEL_FIRM.finditer(attorney_header))
        if not firms:
            firms = list(_COUNSEL_FIRM.finditer(header[:representative.start()]))
        for firm in firms:
            value = _clean_org_name(firm.group(1).strip(" \t©|_?4"))
            if value:
                counsel["nazwa"] = value
                break
        if not counsel["mail"]:
            for email in _EMAIL_LINE.finditer(attorney_header):
                value = _mail_from_ocr(email.group(1), counsel["imie_nazwisko"])
                if value and (counsel["nazwa"] or counsel["imie_nazwisko"]):
                    counsel["mail"] = value
                    break
        if not appellant["nazwa"] or (
            counsel["nazwa"] and _fold(appellant["nazwa"]).startswith(_fold(counsel["nazwa"])[:14])
        ):
            explicit = re.search(
                r"(?is)\bw\s+imieniu\s+Odwo[łl]uj[ąa]cego\s*[-–—:]\s*([^()\n]{4,150})\s*\(dalej",
                text[:6500],
            )
            if explicit and (name := _clean_org_name(explicit.group(1))):
                appellant["nazwa"] = name

    buyers = data["Zamawiajacy"]
    for buyer in buyers:
        if buyer["nazwa"]:
            buyer["nazwa"] = buyer["nazwa"].lstrip(" _|:–")
    if len(buyers) == 1 and (
        buyers[0]["nazwa"] is None or _fold(buyers[0]["nazwa"]) in {"dotyczy", "zamawiajacy"}
    ):
        scope = header[representative.end():] if representative else header
        candidates = [
            _clean_org_name(row.strip(" _|:–"))
            for row in scope.splitlines()
            if _AUTHORITY.match(row.strip(" _|:–"))
        ]
        if names := [name for name in candidates if name]:
            buyers[0]["nazwa"] = names[0]

    shared = re.search(r"(?i)wsp[oó]lnie\s+udzielaj[ąa]c\w*\s+zam[oó]wieni\w*\s*:", header)
    if shared:
        scope = header[shared.end():]
        stop = re.search(r"(?im)^\s*Dotyczy\s*:", scope)
        scope = scope[:stop.start()] if stop else scope
        rows = list(re.finditer(r"(?m)^\s*\d{1,2}[.)]\s+([^\n]+)", scope))
        if 2 <= len(rows) <= 5:
            records = []
            for i, row in enumerate(rows):
                block = scope[row.end():rows[i+1].start() if i + 1 < len(rows) else len(scope)]
                name = _clean_org_name(row.group(1).strip())
                if not name or not _AUTHORITY.search(name):
                    break
                ids = _identifier_hits(block)
                records.append(dict(nazwa=name, **ids, prowadzacy_postepowanie=int(bool(
                    re.search(r"(?i)prowadz[ąa]cy\s+post[ęe]powanie", block)
                ))))
            if len(records) == len(rows) and sum(row["prowadzacy_postepowanie"] for row in records) == 1:
                data["Zamawiajacy"] = records

    procurement = data["ZamowieniePubliczne"]
    title = re.search(r'(?is)\bpn\.[ \t\n]*(?:[„"«]|,,)\s*([^”"»]{5,350})[”"»]', header)
    if title:
        procurement["tytul"] = _join(title.group(1)).strip(" ,.;")
    if not procurement["numer_postepowania"] or len(procurement["numer_postepowania"]) <= 3:
        number = re.search(
            r"(?i)(?:nr\.?\s+referencyjn\w*|numer\s+post[ęe]powania)\s*:\s*"
            r"([A-Z0-9]+(?:[./_-][ \t]*[A-Z0-9]+){1,8})",
            header,
        )
        if number:
            procurement["numer_postepowania"] = re.sub(r"\s+", "", number.group(1))


def repair_scan_fields(
    document_type: str, data: dict[str, Any], text: str, original: str,
) -> None:
    if document_type == "odwolanie":
        _repair_appeal(data, text)
    elif document_type == "pelnomocnictwo":
        _repair_poa(data, text)
    elif document_type == "przelew":
        _repair_receipt(data, text, original)


def fuse_scan_model(
    document_type: str,
    data: dict[str, Any],
    text: str,
    spans: list[tuple[int, int, str]],
    scorer: CandidateSpanScorer,
) -> dict[str, int]:
    """Rank source spans by local token logits and fill missing fields."""
    candidates = defaultdict(list)
    for start, end, label in spans:
        raw = text[start:end]
        start += len(raw) - len(raw.lstrip())
        end -= len(raw) - len(raw.rstrip())
        if start >= end or end - start > 350:
            continue
        if (start and text[start - 1].isalnum() and text[start].isalnum()) or (
            end < len(text) and text[end - 1].isalnum() and text[end].isalnum()
        ):
            continue
        value = _join(text[start:end]).strip(" ,;:„“\"")
        if not value:
            continue
        evidence = scorer.score(RegexCandidate(label, value, start, end, 0.0, ()), label)
        margin = evidence.role_vs_o_margin
        if margin is not None and margin >= 1.0:
            candidates[label].append((margin, start, end, value))
    metrics = {"ocr_model_candidates": sum(map(len, candidates.values())), "ocr_model_selected": 0}

    def fill(fields, field, label, convert, *, accept=lambda _start, _end: True):
        if fields[field] is not None:
            return
        values = {}
        for margin, start, end, value in candidates[label]:
            if accept(start, end) and (normalised := convert(value)) is not None:
                values[normalised] = max(values.get(normalised, 0.0), margin)
        ranked = sorted(values, key=values.__getitem__, reverse=True)
        if ranked and (len(ranked) == 1 or values[ranked[0]] - values[ranked[1]] >= 1.0):
            fields[field] = ranked[0]
            metrics["ocr_model_selected"] += 1

    def date_value(value):
        dates = _parse_dates(repair_ocr_text(value))
        return dates[0][0] if len(dates) == 1 and dates[0][1:] == (0, len(value)) else None

    def account_value(value):
        if not re.fullmatch(r"(?:PL\s*)?\d(?:[ -]*\d){25}", value, re.IGNORECASE):
            return None
        digits = re.sub(r"\D", "", value)
        return digits if validate_nrb(digits) else None

    def person_value(value):
        return value if any(_fold(candidate) == _fold(value) for candidate in _bounded_person_candidates(value)) else None

    if document_type == "przelew":
        transfer = data["przelew_bankowy"]
        fill(transfer, "data_operacji", "OPERATION_DATE", date_value)
        for field, other, label in (
            ("z_rachunku", "na_rachunek", "SOURCE_ACCOUNT"),
            ("na_rachunek", "z_rachunku", "DESTINATION_ACCOUNT"),
        ):
            other_account = re.sub(r"\D", "", transfer[other] or "")
            fill(transfer, field, label, lambda value: (
                account if (account := account_value(value)) != other_account else None
            ))
        if transfer["kwota"] is None:
            def confirmed_amount(start, end):
                return bool(re.match(r"\s*(?:PLN|z[łl])\b", text[end:end + 12], re.IGNORECASE))

            fill(transfer, "kwota", "TRANSFER_AMOUNT", lambda value: (
                float(amount) if (amount := _parse_amount_value(value)) is not None else None
            ), accept=confirmed_amount)
    elif document_type == "odwolanie":
        appeal = data["OdwolanieKIO"]
        heading = re.search(r"(?im)^[ \t]*ODWO[ŁL]ANIE[ \t]*$", text)
        end_header = heading.start() if heading else min(2000, len(text))
        fill(appeal, "data", "APPEAL_DATE", date_value, accept=lambda _, end: end < end_header)
        fill(appeal, "miejscowosc", "APPEAL_CITY", lambda value: _CITY_SPELLINGS.get(_fold(value)),
             accept=lambda _, end: end < end_header)
        appellant = data["Odwolujacy"][0]
        counsel = appellant["pelnomocnik"]
        cue = re.search(r"(?i)\breprezentowan\w*\s+przez\b", text[:end_header])
        if cue:
            fill(counsel, "imie_nazwisko", "COUNSEL_PERSON", person_value,
                 accept=lambda start, end: cue.end() <= start < end <= min(cue.end() + 240, end_header))
    elif document_type == "pelnomocnictwo":
        action = _poa_action_match(text)
        if action is None:
            return metrics
        attorneys = data["pelnomocnicy"]
        start_range = action.end()
        end_range = start_range + len(_poa_attorney_segment(text, action))
        principal_names = {
            _fold(grantor["data"]["imie_nazwisko"])
            for grantor in (data.get("mocodawca") or [])
            if grantor["data"]["imie_nazwisko"]
        }
        if not attorneys:
            attorney = {"imie_nazwisko": None, "pesel": None}
            fill(attorney, "imie_nazwisko", "ATTORNEY_PERSON", person_value,
                 accept=lambda start, end: start_range <= start < end <= end_range
                 and _fold(_join(text[start:end])) not in principal_names)
            if attorney["imie_nazwisko"]:
                attorneys.append(attorney)
        for attorney in attorneys:
            name = attorney["imie_nazwisko"]
            if not name or attorney["pesel"]:
                continue
            segment = text[start_range:end_range]
            occurrence = _candidate_span(segment, name)
            if occurrence:
                after = start_range + occurrence.end()
                fill(attorney, "pesel", "ATTORNEY_PESEL", lambda value: (
                    value if re.fullmatch(r"\d{11}", value) else None
                ), accept=lambda start, end: after <= start < end <= min(after + 80, end_range)
                and bool(re.fullmatch(r"(?is)[^(\d]*(?:\([^\d(]*)?\bPESEL\s*:\s*", text[after:start]))
                )
    return metrics
