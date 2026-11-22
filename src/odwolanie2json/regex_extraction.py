from __future__ import annotations

import csv
import gzip
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .candidate_types import RegexCandidate, RegexExtraction


_RESOURCE = Path(__file__).with_name("resources") / "pesel_name_lexicon_2026.csv.gz"
_DIGITS = re.compile(r"(?<!\d)\d(?:[\s\u00a0-]*\d){7,25}(?!\d)")
_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?![\w-])"
)
_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)
_DATE_NUMERIC = re.compile(
    r"(?<!\d)(?P<d>0?[1-9]|[12]\d|3[01])[ \t]*[./-][ \t]*"
    r"(?P<m>0?[1-9]|1[0-2])[ \t]*[./-][ \t]*"
    r"(?P<y>19\d{2}|20\d{2}|21\d{2}|22\d{2})(?!\d)"
)
_DATE_ISO = re.compile(
    r"(?<!\d)(?P<y>18\d{2}|19\d{2}|20\d{2}|21\d{2}|22\d{2})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])(?!\d)"
)
_MONTHS = {
    "stycznia": 1, "styczen": 1, "stycz": 1,
    "lutego": 2, "luty": 2,
    "marca": 3, "marzec": 3,
    "kwietnia": 4, "kwiecien": 4,
    "maja": 5, "maj": 5,
    "czerwca": 6, "czerwiec": 6,
    "lipca": 7, "lipiec": 7,
    "sierpnia": 8, "sierpien": 8,
    "wrzesnia": 9, "wrzesien": 9,
    "pazdziernika": 10, "pazdziernik": 10,
    "listopada": 11, "listopad": 11,
    "grudnia": 12, "grudzien": 12,
}
_DATE_WORD = re.compile(
    r"(?<!\d)(?P<d>0?[1-9]|[12]\d|3[01])\s+"
    r"(?P<month>stycznia|styczeń|styczen|stycz|lutego|luty|marca|marzec|kwietnia|kwiecień|kwiecien|"
    r"maja|maj|czerwca|czerwiec|lipca|lipiec|sierpnia|sierpień|sierpien|września|wrzesnia|wrzesień|wrzesien|"
    r"października|pazdziernika|październik|pazdziernik|listopada|listopad|grudnia|grudzien)\s+"
    r"(?P<y>18\d{2}|19\d{2}|20\d{2}|21\d{2}|22\d{2})(?!\d)",
    re.IGNORECASE,
)

_TEMPLATES: dict[str, dict[str, Any]] = {
    "odwolanie": {
        "OdwolanieKIO": {
            "data": None,
            "czynnosc_dzien": None,
            "czynnosc_powod_id": None,
            "czynnosc_opis": None,
            "miejscowosc": None,
        },
        "Odwolujacy": [{
            "krs": None,
            "nazwa": None,
            "nip": None,
            "regon": None,
            "przez-pelnomocnika": None,
            "pelnomocnik": {"imie_nazwisko": None, "nazwa": None, "mail": None},
        }],
        "Zamawiajacy": [{
            "krs": None,
            "nazwa": None,
            "nip": None,
            "regon": None,
            "prowadzacy_postepowanie": None,
        }],
        "ZamowieniePubliczne": {
            "numer_bzp": None,
            "numer_postepowania": None,
            "numer_ted": None,
            "tytul": None,
        },
    },
    "pelnomocnictwo": {
        "mocodawca": [{
            "data": {"imie_nazwisko": None, "krs": None, "nazwa": None, "nip": None, "typ": None},
            "podpisujacy": [{"imie_nazwisko": None}],
        }],
        "pelnomocnicy": [{"imie_nazwisko": None, "pesel": None}],
    },
    "przelew": {
        "przelew_bankowy": {
            "data_operacji": None,
            "kwota": None,
            "na_rachunek": None,
            "numer_transakcji": None,
            "tytul": None,
            "z_rachunku": None,
        }
    },
}

_CAUSES: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "Decyzja o odrzuceniu oferty", (r"odrzuc\w*.{0,80}\bofert\w*",)),
    (2, "Ogłoszenie wyboru oferty", (r"wyb\w*.{0,100}\bofert\w*", r"najkorzystniejsz\w*.{0,50}\bofert\w*")),
    (3, "Decyzja o wykluczeniu wykonawcy", (r"wyklucz\w*.{0,100}(?:\bwykonawc\w*|odwołując\w*|odwolujac\w*|podmiot\w*)", r"poinformow\w*.{0,80}wyklucz\w*")),
    (4, "Publikacja specyfikacji warunków zamówienia (SWZ)", (r"(?:publikac\w*|opublik\w*|udostępn\w*|udostepn\w*).{0,100}\b(?:swz|specyfikacj\w*)",)),
    (5, "Modyfikacja treści SWZ", (r"(?:modyfikac\w*|zmodyfik\w*|zmian\w*).{0,100}\b(?:swz|specyfikacj\w*)",)),
    (6, "Odpowiedzi/wyjaśnienia do SWZ", (r"(?:odpowied\w*|wyjaśn\w*).{0,100}\b(?:swz|specyfikacj\w*)",)),
    (7, "Zaniechanie odrzucenia oferty", (r"zaniecha\w*.{0,100}odrzuc\w*.{0,80}\bofert\w*", r"\bnie\s+(?:odrzuc\w*|dokonał\w*.{0,40}odrzuc\w*).{0,100}\bofert\w*")),
    (8, "Zaniechanie wykluczenia wykonawcy", (r"zaniecha\w*.{0,100}wyklucz\w*.{0,100}\b(?:wykonawc\w*|podmiot\w*)", r"\bnie\s+wyklucz\w*.{0,100}\b(?:wykonawc\w*|podmiot\w*)")),
    (9, "Decyzja o unieważnieniu postępowania", (r"unieważn\w*.{0,100}\bpostępowan\w*", r"uniewazn\w*.{0,100}\bpostepowan\w*")),
    (10, "Zaniechanie unieważnienia postępowania", (r"zaniecha\w*.{0,100}unieważn\w*.{0,100}\bpostępowan\w*", r"zaniecha\w*.{0,100}uniewazn\w*.{0,100}\bpostepowan\w*", r"\bnie\s+unieważn\w*.{0,100}\bpostępowan\w*", r"\bnie\s+uniewazn\w*.{0,100}\bpostepowan\w*")),
    (11, "Opis przedmiotu zamówienia (OPZ)", (r"opis\w*.{0,100}\bprzedmiot\w*.{0,50}\bzamów\w*", r"\bopz\b")),
    (12, "Projektowane postanowienia umowy", (r"projektowan\w*.{0,100}\bpostanow\w*.{0,50}\bumow\w*", r"\bppu\b")),
)


@dataclass(frozen=True)
class Party:
    name: str | None
    nip: str | None
    regon: str | None
    krs: str | None
    start: int
    end: int


def _fold(value: str) -> str:
    value = value.replace("ł", "l").replace("Ł", "L")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()


_OCR_LETTER_VARIANTS = {
    "a": "aą",
    "c": "cć",
    "e": "eę",
    "l": "lł",
    "n": "nń",
    "o": "oó",
    "s": "sś",
    "z": "zźż",
}


def _ocr_literal_pattern(value: str) -> str:
    """Compile a Polish literal tolerant of lost accents and OCR line-wrap hyphens."""
    parts: list[str] = []
    for index, character in enumerate(value):
        if character.isspace():
            parts.append(r"\s+")
            continue
        folded = _fold(character)
        variants = _OCR_LETTER_VARIANTS.get(folded)
        parts.append(f"[{variants}]" if variants else re.escape(character))
        if index + 1 < len(value) and not value[index + 1].isspace():
            parts.append(r"(?:-[ \t]*\r?\n[ \t]*)?")
    return "".join(parts)


def _ocr_acronym_pattern(value: str) -> str:
    """Match short identifier labels OCR may split with spaces or periods."""
    separator = r"[.\s]*"
    body = separator.join(re.escape(character) for character in value)
    return rf"(?<!\w){body}{separator}(?!\w)"


_OCR_REGISTRY_LABELS = "(?:" + "|".join(
    _ocr_acronym_pattern(label) for label in ("KRS", "NIP", "REGON")
) + ")"
_OCR_IDENTIFIER_LABELS = "(?:" + "|".join(
    _ocr_acronym_pattern(label) for label in ("KRS", "NIP", "REGON", "PESEL")
) + ")"


@lru_cache(maxsize=1)
def _name_lexicon() -> tuple[frozenset[str], frozenset[str]]:
    given: set[str] = set()
    surnames: set[str] = set()
    try:
        with gzip.open(_RESOURCE, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                name = _fold(row["name"].strip())
                if name:
                    (given if row["kind"] == "given" else surnames).add(name)
    except OSError:
        return frozenset(), frozenset()
    return frozenset(given), frozenset(surnames)


def validate_pesel(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 11:
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    if (sum(int(digits[i]) * weights[i] for i in range(10)) % 10) != (10 - int(digits[10])) % 10:
        return False
    month_code = int(digits[2:4])
    century_offsets = ((80, 1800), (0, 1900), (20, 2000), (40, 2100), (60, 2200))
    year = month = 0
    for offset, century in century_offsets:
        if offset <= month_code - 1 <= offset + 11:
            month = month_code - offset
            year = century + int(digits[:2])
            break
    if not month:
        return False
    try:
        date(year, month, int(digits[4:6]))
    except ValueError:
        return False
    return True


def validate_nip(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    check = sum(int(digits[i]) * weights[i] for i in range(9)) % 11
    return check != 10 and check == int(digits[9])


def validate_regon(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 9:
        weights = (8, 9, 2, 3, 4, 5, 6, 7)
        check = sum(int(digits[i]) * weights[i] for i in range(8)) % 11
        return (0 if check == 10 else check) == int(digits[8])
    if len(digits) == 14:
        if not validate_regon(digits[:9]):
            return False
        weights = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
        check = sum(int(digits[i]) * weights[i] for i in range(13)) % 11
        return (0 if check == 10 else check) == int(digits[13])
    return False


def validate_krs(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 1 <= len(digits) <= 10 and set(digits) != {"0"}


def validate_nrb(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value).upper()
    if compact.startswith("PL"):
        compact = compact[2:]
    if not re.fullmatch(r"\d{26}", compact):
        return False
    return int(compact[2:] + "2521" + compact[:2]) % 97 == 1


def _digit_candidates(text: str, lengths: Iterable[int]) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for length in lengths:
        pattern = re.compile(rf"(?<!\d)\d(?:[ \t\r\n\u00a0-]*\d){{{length - 1}}}(?!\d)")
        for match in pattern.finditer(text):
            digits = re.sub(r"\D", "", match.group())
            if len(digits) == length:
                found.append((digits, match.start(), match.end()))
    return sorted(found, key=lambda item: item[1])


def _labeled_number(
    text: str,
    label: str,
    validator: Any,
    lengths: tuple[int, ...],
    window: int = 100,
) -> str | None:
    for marker in re.finditer(label, text, flags=re.IGNORECASE | re.UNICODE):
        suffix = text[marker.end(): marker.end() + window]
        candidates = _digit_candidates(suffix, lengths)
        if candidates:
            # The explicit field label outweighs a failed check digit: challenge
            # data may be synthetic and OCR can corrupt one digit. Checksum
            # validity is reported separately from extraction.
            nearest = min(candidate[1] for candidate in candidates)
            at_nearest_label = [candidate for candidate in candidates if candidate[1] == nearest]
            return max(at_nearest_label, key=lambda candidate: len(candidate[0]))[0]
    return None


def _labeled_identifier_number(
    text: str,
    label: str,
    validator: Any,
    lengths: tuple[int, ...],
) -> str | None:
    """Extract an identifier after a contiguous or OCR-split acronym label."""
    prefix = _ocr_acronym_pattern(label) + r"\s*[:№#]?\s*"
    return _labeled_number(text, prefix, validator, lengths)


def _labeled_pesels(text: str) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    for marker in re.finditer(_ocr_acronym_pattern("PESEL"), text, flags=re.IGNORECASE):
        for digits, start, end in _digit_candidates(text[marker.end(): marker.end() + 64], (11,)):
            results.append((digits, marker.end() + start, marker.end() + end))
    return results


def _parse_dates(text: str) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    patterns = (_DATE_ISO, _DATE_NUMERIC, _DATE_WORD)
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if "month" in groups:
                month = _MONTHS[_fold(groups["month"])]
            else:
                month = int(groups["m"])
            try:
                parsed = date(int(groups["y"]), month, int(groups["d"]))
            except ValueError:
                continue
            results.append((parsed.isoformat(), match.start(), match.end()))
    return sorted(set(results), key=lambda item: item[1])


def _header_date_city(text: str) -> tuple[str | None, str | None]:
    # Party details can precede the city/date line, and some PDFs place that
    # line after the ODWOŁANIE heading. The anchored city-plus-date form keeps
    # this wider local search specific.
    header = text[:5000]
    filing_date: str | None = None
    city: str | None = None
    date_label = re.search(
        r"\bdata\s+(?:odwołania|odwolania|wniesienia|złożenia|zlozenia)\b\s*[:\-–]?\s*",
        header,
        re.IGNORECASE,
    )
    if date_label:
        parsed = _parse_dates(header[date_label.end():date_label.end() + 100])
        if parsed:
            filing_date = parsed[0][0]
    city_label = re.search(
        r"\bmiejscowoś(?:ć|c)\b\s*[:\-–]?\s*(.{2,60}?)"
        r"(?=\s*(?:[,;]|\bdata\b|\n|$))",
        header,
        re.IGNORECASE | re.UNICODE,
    )
    if city_label:
        candidate = re.sub(r"\s+", " ", city_label.group(1)).strip(" ,;.-")
        city_word = r"[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*(?:-[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*)*"
        if re.fullmatch(rf"{city_word}(?:\s+{city_word}){{0,2}}", candidate):
            city = candidate
    date_patterns = (
        re.compile(r"(?im)^\s*(?P<city>[A-ZĄĆĘŁŃÓŚŹŻ][^\n,]{1,60}?)\s*(?:,\s*(?:dnia\s*)?|\s+dnia\s+)(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{4})\b"),
        re.compile(r"(?im)^\s*(?P<city>[A-ZĄĆĘŁŃÓŚŹŻ][^\n,]{1,60}?)\s*(?:,\s*(?:dnia\s*)?|\s+dnia\s+)(?P<date>\d{1,2}\s+[A-Za-ząćęłńóśźż]+\s+\d{4})\b"),
    )
    for pattern in date_patterns:
        match = pattern.search(header)
        if match:
            parsed_city = re.sub(r"\s+", " ", match.group("city")).strip(" ,.-")
            city_word = r"[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*(?:-[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*)*"
            if not re.fullmatch(rf"{city_word}(?:\s+{city_word}){{0,2}}", parsed_city):
                continue
            date_row = _parse_dates(match.group("date"))
            if date_row:
                return filing_date or date_row[0][0], city or parsed_city or None
    return filing_date, city


def _first_claim(text: str) -> str:
    matches = list(re.finditer(r"\bZarzut\s*(?:nr\s*)?(?:1|I)\b", text, re.IGNORECASE))
    if not matches:
        return text[:5000]
    start = matches[0].start()
    end_match = re.search(
        r"\bZarzut\s*(?:nr\s*)?(?:2|II)\b|\bWnoszę\s+o\s*:|\bUzasadnienie\b",
        text[matches[0].end():],
        re.IGNORECASE,
    )
    end = matches[0].end() + end_match.start() if end_match else min(len(text), matches[0].end() + 3500)
    return text[start:end]


def _cause_id(text: str) -> int | None:
    heading = re.search(r"\bODWOŁANIE\b", text, re.IGNORECASE)
    scope_start = heading.end() if heading else 0
    priority = (7, 8, 10, 1, 2, 3, 9, 6, 5, 4, 12, 11)

    def first_action(scope: str, *, direct: bool = False) -> int | None:
        normalized = _fold(scope)
        matches: list[tuple[int, int, int]] = []
        for cause_id, _, patterns in _CAUSES:
            for pattern in patterns:
                match = re.search(_fold(pattern), normalized, re.IGNORECASE | re.DOTALL)
                if match:
                    matches.append((match.start(), priority.index(cause_id), cause_id))
                    break
        omissions = [match for match in matches if match[2] in (7, 8, 10)]
        if omissions:
            return min(omissions)[2]
        if direct and re.search(
            r"\bopz\b|opis\w*.{0,100}przedmiot\w*.{0,50}zamow\w*",
            normalized,
            re.IGNORECASE | re.DOTALL,
        ) and re.search(r"opublik\w*|sporządz\w*|sporzadz\w*|opis\w*", normalized):
            return 11
        return min(matches)[2] if matches else None

    body = text[scope_start:]
    # Prefer a dated factual statement by the contracting authority. It is
    # usually the first concise description of the challenged act; allegation
    # lists later in the appeal often mention several competing causes.
    for subject in re.finditer(r"\bZamawiając(?:y|a)\b", body, re.IGNORECASE):
        tail = body[subject.start():subject.start() + 1200]
        if not _parse_dates(tail[:260]):
            continue
        cuts: list[int] = []
        semicolon = re.search(r";", tail)
        if semicolon:
            cuts.append(semicolon.start())
        for boundary in re.finditer(r"\.(?:\s+|\n+)(?=[A-ZĄĆĘŁŃÓŚŹŻ])", tail):
            if re.search(r"\b(?:r|nr|ust|pkt|poz|Dz|S|sp|ul)\.$", tail[:boundary.start() + 1], re.IGNORECASE):
                continue
            cuts.append(boundary.start())
            break
        section = re.search(
            r"\n\s*(?:Uprzednio|Zaskarżonym|Zarzut\b|Strona\s+\d|Wobec|Na\s+tej\s+podstawie)",
            tail,
            re.IGNORECASE,
        )
        if section:
            cuts.append(section.start())
        clause = tail[:min(cuts)] if cuts else tail
        direct_cause = first_action(clause, direct=True)
        if direct_cause is not None:
            return direct_cause

    body = text[scope_start:]
    filing_cue = re.search(r"\bwnosz\w*\s+o\s*[:\-–]?", body, re.IGNORECASE)
    if filing_cue:
        intro = body[filing_cue.end():]
    else:
        action_start = re.search(
            r"\bZamawiając\w*\s+(?:w\s+dniu|zaniecha\w*|opublik\w*|zmodyfik\w*|odrzuc\w*|wyklucz\w*|unieważn\w*)",
            body,
            re.IGNORECASE,
        )
        intro = body[action_start.start():] if action_start else ""
    first_allegation = re.search(r"\bZarzut\s*(?:nr\s*)?(?:1|I)\b", intro, re.IGNORECASE)
    intro = intro[:first_allegation.start()] if first_allegation else intro[:1800]
    cause_id = first_action(intro[:1800])
    if cause_id is not None:
        return cause_id
    claim_cause = first_action(_first_claim(text))
    return claim_cause


def _cause_date(text: str) -> str | None:
    explicit_action = re.search(
        r"\b(?:zaskarżon\w*\s+)?czynnoś\w*.{0,80}?\b(?:z\s+dnia|w\s+dniu)\b\s*",
        text[:5000],
        re.IGNORECASE | re.DOTALL,
    )
    if explicit_action:
        action_dates = _parse_dates(text[explicit_action.end():explicit_action.end() + 100])
        if action_dates:
            return action_dates[0][0]
    heading = re.search(r"\bODWOŁANIE\b", text, re.IGNORECASE)
    scope_start = heading.end() if heading else 0
    body = text[scope_start:]
    filing_cue = re.search(r"\bwnosz\w*\s+o\s*[:\-–]?", body, re.IGNORECASE)
    if filing_cue:
        intro = body[filing_cue.end():]
    else:
        intro = body
    first_allegation = re.search(r"\bZarzut\s*(?:nr\s*)?(?:1|I)\b", intro, re.IGNORECASE)
    if first_allegation:
        intro = intro[:first_allegation.start()]
    else:
        intro = intro[:1800]
    intro_dates = _parse_dates(intro)
    for date_value, start, end in intro_dates:
        before = _fold(intro[max(0, start - 160):start])
        after = _fold(intro[end:min(len(intro), end + 160)])
        if re.search(r"\b(?:w\s+dniu|z\s+dnia)\b", before) and re.search(
            r"zamawiajac|czynnos|odrzuc|wyklucz|uniewazn|wybor|publik|zmodyfik|zaniech",
            before + " " + after,
        ):
            return date_value
    if intro_dates:
        return intro_dates[0][0]

    claim = _first_claim(text)
    dates = _parse_dates(claim)
    if dates:
        return dates[0][0]
    normalized = _fold(text[:7000])
    date_by_pos = _parse_dates(text[:7000])
    for date_value, start, end in date_by_pos:
        before = _fold(text[max(0, start - 180):start])
        after = _fold(text[end:min(len(text), end + 180)])
        context = before + " " + after
        if re.search(r"(czynnos|zaniechan|decyzj|zamawiajac|odrzuc|wyklucz|uniewazn|wybor|ogloszen).{0,100}(w dniu|z dnia)|(?:w dniu|z dnia).{0,100}(czynnos|zamawiajac|odrzuc|wyklucz|uniewazn|wybor)", context):
            return date_value
    return None


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, _fold(text), flags=re.IGNORECASE | re.DOTALL) is not None


def _normalize_name_candidate(value: str) -> str | None:
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n,;:.()[]{}\"'„”«»")
    value = re.sub(r"^(?:r\s*\.\s*pr\s*\.?|adw\s*\.?|mec\s*\.?)\s*", "", value, flags=re.IGNORECASE)
    return value or None


def _person_candidates(text: str) -> list[str]:
    given_names, surnames = _name_lexicon()
    if not given_names or not surnames:
        return []
    tokens = list(_TOKEN.finditer(text))
    found: list[tuple[int, int, str]] = []
    for i in range(len(tokens)):
        for given_len in (1, 2):
            if i + given_len >= len(tokens):
                continue
            given = " ".join(_fold(m.group()) for m in tokens[i:i + given_len])
            if given not in given_names:
                continue
            for surname_len in (1, 2, 3):
                j = i + given_len
                if j + surname_len > len(tokens):
                    continue
                surname = " ".join(_fold(m.group()) for m in tokens[j:j + surname_len])
                if surname not in surnames:
                    continue
                start = tokens[i].start()
                end = tokens[j + surname_len - 1].end()
                candidate = _normalize_name_candidate(text[start:end])
                if candidate:
                    found.append((start, end, candidate))
    # Prefer the longest lexical match at a start position, then preserve text order.
    best: dict[int, tuple[int, str]] = {}
    for start, end, name in found:
        if start not in best or end > best[start][0]:
            best[start] = (end, name)
    return list(dict.fromkeys(name for _, (_, name) in sorted(best.items())))


def _people_near_cues(text: str, cues: tuple[str, ...], limit: int = 320) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for cue in cues:
        for match in re.finditer(cue, text, re.IGNORECASE | re.UNICODE):
            window = text[match.end():match.end() + limit]
            for name in _person_candidates(window):
                position = window.find(name)
                candidates.append((match.end() + max(position, 0), name))
    return list(dict.fromkeys(name for _, name in sorted(candidates)))


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        rows.append((offset, end, line.strip()))
        offset = end
    return rows


def _identifier_hits(line: str) -> dict[str, str]:
    return {
        "krs": _labeled_identifier_number(line, "KRS", validate_krs, (10,)),
        "nip": _labeled_identifier_number(line, "NIP", validate_nip, (10,)),
        "regon": _labeled_identifier_number(line, "REGON", validate_regon, (9, 14)),
    }


def _clean_org_name(line: str) -> str | None:
    value = re.sub(
        rf"{_OCR_REGISTRY_LABELS}\s*[:№#]?.*$",
        "",
        line,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\s*(?:\d{1,2}|[a-z])[.)]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)^\s*pe[łl]nomocnictwo\s*[.:,–-]?\s*", "", value)
    role_label = re.match(
        r"(?i)^(?:Zamawiając\w*|Odwołując\w*|Wykonawc\w*)\s*[–—-]\s*"
        r"(?:Lider(?:\s+konsorcjum)?|Partner)(?:\s*,\s*prowadzący\s+postępowanie)?"
        r"\s*(?::\s*(.*))?$",
        value,
    )
    if role_label:
        value = role_label.group(1) or ""
    if re.fullmatch(
        r"(?i)\s*(?:(?:Zamawiając\w*|Odwołując\w*|Wykonawc\w*)\s*[–—-]\s*)?"
        r"(?:Lider(?:\s+konsorcjum)?|Partner)(?:\s*,\s*prowadzący\s+postępowanie)?\s*:?\s*",
        value,
    ):
        return None
    if re.search(r"\bzamowienia\s+w\s+skladzie\b", _fold(value)):
        return None
    value = re.sub(r"^Mocodawc\w*\s+jest\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:Odwołujący|Zamawiający|Wykonawca|Mocodawc\w*)\s*[:\-–]?\s*", "", value, flags=re.IGNORECASE)
    value = re.split(
        r"\s+(?:z\s+siedzib\w*|zarejestrowan\w*|wpisan\w*|reprezentowan\w*|"
        r"adres\b|ul\.?\b|ulica\b|al\.?\b|aleja\b|tel\.?\b|fax\b|"
        r"nip\b|regon\b|krs\b)",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r",?\s*\d{2}-\d{3}\b.*$", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;:\t\r\n\"'„”«»")
    if value.count("(") > value.count(")"):
        value = value.rstrip(" (")
    if value.count("[") > value.count("]"):
        value = value.rstrip(" [")
    if not re.search(
        r"(?i)(?:\bS\.?\s*A|\bsp\.?\s*z\s*o\.?\s*o|\bsp\.?\s*[kjp]|\bs\.?\s*c)\.$",
        value,
    ):
        value = value.rstrip(".")
    if len(value) < 3 or len(value) > 180:
        return None
    folded = _fold(value)
    if re.search(r"\b(?:ul|ulica|al|aleja|pl|plac|tel|telefon|fax|adres|e-mail|email|reprezentowan|prezes krajowej izby|krajowej izby odwolawczej)\b", folded):
        return None
    if re.search(r"\b\d{2}-\d{3}\b|@", value):
        return None
    if not re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]", value):
        return None
    return value


def _name_score(value: str, distance: int) -> int:
    folded = _fold(value)
    score = max(0, 24 - distance)
    if re.search(r"\b(sp[oó]łk\w*|sp\s*\.\s*z\s*\.\s*o\s*\.\s*o|s\s*\.\s*a|s\s*\.\s*k|s\s*\.\s*j|spółdzielni\w*|gmina|miasto|powiat|urząd|szpital|szkoła|filharmonia|uniwersytet|politechnika|województwo|starostwo|zarząd|fundacj\w*|stowarzyszen\w*)\b", value, re.IGNORECASE):
        score += 16
    if re.search(r"\b(?:wykonawca|odwolujacy|zamawiajacy|kancelaria)\b", folded):
        score += 2
    return score


def _inline_party_records(text: str) -> list[Party]:
    """Parse role-labeled party blocks even when PDF text extraction joins lines."""
    header = text[:8000]
    marker_pattern = re.compile(
        r"(?P<role>\b(?:Odwołując\w*|Odwolujac\w*|Zamawiając\w*|Zamawiajac\w*)\b)"
        r"\s*[:\-–]\s*",
        re.IGNORECASE | re.UNICODE,
    )
    markers = list(marker_pattern.finditer(header))
    if not any(_fold(match.group("role")).startswith("odwoluj") for match in markers):
        return []
    if not any(_fold(match.group("role")).startswith("zamawiaj") for match in markers):
        return []
    appellant_marker = next(
        match for match in markers if _fold(match.group("role")).startswith("odwoluj")
    )
    buyer_marker = next(
        match for match in markers if _fold(match.group("role")).startswith("zamawiaj")
    )
    stop_pattern = re.compile(
        r"\b(?:postępowani\w*|postepowani\w*|BZP|TED|przedmiot|miejscowoś\w*|"
        r"data\s+odwołani\w*|data\s+odwolani\w*|zaskarżon\w*|zaskarzon\w*)\b",
        re.IGNORECASE | re.UNICODE,
    )
    appellant: Party | None = None
    buyers: list[Party] = []
    for index, marker in enumerate(markers):
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(header)
        block = header[marker.end():block_end]
        stop = stop_pattern.search(block)
        if stop:
            block = block[:stop.start()]
            block_end = marker.end() + stop.start()
        identifiers = _identifier_hits(block)
        party = Party(
            _clean_org_name(block),
            identifiers["nip"],
            identifiers["regon"],
            identifiers["krs"],
            marker.start(),
            block_end,
        )
        if _fold(marker.group("role")).startswith("odwoluj"):
            if appellant is None:
                appellant = party
        else:
            buyers.append(party)
    return [appellant, *buyers] if appellant is not None and buyers else []


def _party_records(text: str) -> list[Party]:
    heading = re.search(r"(?im)^\s*ODWOŁANIE\b", text)
    header_end = heading.start() if heading else min(len(text), 5000)
    header = text[:header_end]
    inline_records = [] if heading else _inline_party_records(header)
    if inline_records:
        return inline_records
    lines = _line_offsets(header)
    id_rows: list[int] = []
    for i, (_, _, line) in enumerate(lines):
        hits = _identifier_hits(line)
        if any(hits.values()):
            id_rows.append(i)
    groups: list[list[int]] = []
    for row in id_rows:
        if groups and row - groups[-1][-1] <= 2:
            between = " ".join(lines[i][2] for i in range(groups[-1][-1] + 1, row))
            if not re.search(r"\b(?:Odwołujący|Zamawiający|Wykonawca)\b", between, re.IGNORECASE):
                groups[-1].append(row)
                continue
        groups.append([row])

    role_markers: list[tuple[int, int, str]] = []
    numbered_prefix = r"(?:(?:\d{1,2}|[a-z])[.)]\s*)?"
    role_patterns = (
        ("appellant", rf"(?im)^\s*{numbered_prefix}(?:Odwołując|Odwolujac)\w*\s*[:\-–]?(?:\s*Lider\b)?"),
        ("buyer", rf"(?im)^\s*{numbered_prefix}(?:Zamawiając|Zamawiajac)\w*\s*[:\-–]?(?:[^\n]{{0,100}})?"),
    )
    for role, pattern in role_patterns:
        role_markers.extend((m.start(), m.end(), role) for m in re.finditer(pattern, header))
    role_markers.sort()
    non_role_stops = [
        match.start()
        for match in re.finditer(
            r"(?im)^\s*(?:Wykonawca\s+wybrany|dotyczy\s*:|ODWOŁANIE\b)",
            header,
        )
    ]

    records: list[tuple[Party, str | None]] = []
    previous_end = 0
    for rows in groups:
        lo, hi = rows[0], rows[-1]
        start, end = lines[lo][0], lines[hi][1]
        prior_markers = [marker for marker in role_markers if previous_end <= marker[0] < start]
        role = prior_markers[-1][2] if prior_markers else None
        lower_bound = max(previous_end, prior_markers[-1][1] if prior_markers else max(0, start - 1200))
        candidates: list[tuple[int, str]] = []
        pre_identifier_lines = [
            row for row in lines
            if lower_bound <= row[0] < start and row[2]
        ]
        for line_start, line_end, line in pre_identifier_lines:
            candidate = _clean_org_name(line)
            if candidate:
                distance = max(0, (start - line_end) // 80)
                candidates.append((_name_score(candidate, distance), candidate))

        # OCR can split the standard company suffix across adjacent lines.
        for first in range(len(pre_identifier_lines)):
            wrapped: list[tuple[int, int, str]] = []
            for line_start, line_end, line in pre_identifier_lines[first:first + 4]:
                if wrapped and wrapped[-1][1] != line_start:
                    break
                wrapped.append((line_start, line_end, line))
                if len(wrapped) < 2:
                    continue
                raw_candidate = " ".join(row[2] for row in wrapped)
                company_suffix = re.search(
                    r"\bsp\s*\.?\s*z\s*o\s*\.?\s*o\b",
                    raw_candidate,
                    re.IGNORECASE,
                )
                if not company_suffix:
                    continue
                line_boundaries: list[int] = []
                boundary = 0
                for row in wrapped[:-1]:
                    boundary += len(row[2])
                    line_boundaries.append(boundary)
                    boundary += 1
                if not any(
                    company_suffix.start() < split < company_suffix.end()
                    for split in line_boundaries
                ):
                    continue
                candidate = _clean_org_name(raw_candidate)
                if not candidate:
                    continue
                distance = max(0, (start - wrapped[-1][1]) // 80)
                score = _name_score(candidate, distance)
                cue_floor = max(0, 24 - distance) + 16
                candidates.append((max(score, cue_floor) + 1, candidate))
        for i in rows:
            candidate = _clean_org_name(lines[i][2])
            if candidate:
                candidates.append((_name_score(candidate, 0), candidate))
        name = max(candidates, default=(0, None), key=lambda item: item[0])[1]

        identifiers: dict[str, str | None] = {"krs": None, "nip": None, "regon": None}
        for row in rows:
            hits = _identifier_hits(lines[row][2])
            for key, value in hits.items():
                if value and identifiers[key] is None:
                    identifiers[key] = value
        party = Party(name, identifiers["nip"], identifiers["regon"], identifiers["krs"], start, end)
        records.append((party, role))
        previous_end = end

    if not records:
        return []
    if not role_markers:
        return [party for party, _ in records]

    assigned: dict[str, list[Party]] = {"appellant": [], "buyer": []}
    for marker_index, (marker_start, marker_end, role) in enumerate(role_markers):
        boundaries = [len(header)]
        if marker_index + 1 < len(role_markers):
            boundaries.append(role_markers[marker_index + 1][0])
        boundaries.extend(stop for stop in non_role_stops if marker_end < stop)
        block_end = min(boundaries)
        in_block = [
            party for party, _ in records
            if marker_end <= party.start < block_end
        ]
        if in_block:
            # A labeled role block may list multiple consortium members.
            assigned[role].extend(in_block)

    appellants = sorted(dict.fromkeys(assigned["appellant"]), key=lambda party: party.start)
    buyers = sorted(dict.fromkeys(assigned["buyer"]), key=lambda party: party.start)
    selected_appellant = appellants[0] if appellants else records[0][0]
    if not buyers:
        buyers = [party for party, _ in records if party is not selected_appellant]
    return [selected_appellant, *buyers]


def _party_role_name(text: str, role: str) -> str | None:
    label = r"(?:Zamawiający|Zamawiajacy)" if role == "buyer" else r"(?:Odwołujący|Odwołująca|Wykonawca)"
    for match in re.finditer(rf"\b{label}\b\s*[:\-–]?\s*", text[:5000], re.IGNORECASE):
        window = text[match.end():match.end() + 240]
        lines = [line.strip() for line in window.splitlines()]
        for line in lines:
            candidate = _clean_org_name(line)
            if candidate:
                return candidate
    return None


def _empty_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _empty_like(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_empty_like(child) for child in value]
    return None


def _appeal_counsel_fields(text: str) -> tuple[str | None, str | None, str | None]:
    heading = re.search(r"(?im)^\s*ODWOŁANIE\b", text)
    header = text[:heading.start()] if heading else text[:8000]
    appellant = re.search(
        r"\b(?:Odwołując\w*|Odwolujac\w*|Wykonawc\w*)\s*[:\-–]?",
        header,
        re.IGNORECASE,
    )
    if appellant:
        next_role = re.search(
            r"\b(?:Zamawiając\w*|Zamawiajac\w*)\s*[:\-–]?",
            header[appellant.end():],
            re.IGNORECASE,
        )
        appellant_end = appellant.end() + next_role.start() if next_role else len(header)
        appellant_block = header[appellant.start():appellant_end]
    else:
        appellant_block = header

    name: str | None = None
    representative = re.search(
        r"\b(?:(?:reprezent\w*|działając\w*|dzialajac\w*)\s+przez|"
        r"przez\s+pe[łl]nomocnik\w*)\s*[:\-–]?",
        appellant_block,
        re.IGNORECASE,
    )
    if representative:
        end = re.search(
            r"\b(?:kancelaria|e-?mail|adres\s+do\s+korespondencji|"
            r"Zamawiając\w*|Zamawiajac\w*|ODWOŁANIE)\b",
            appellant_block[representative.end():],
            re.IGNORECASE,
        )
        segment = appellant_block[representative.end():representative.end() + end.start()] if end else appellant_block[representative.end():]
        names = _bounded_person_candidates(segment) + _capitalized_person_candidates(segment)
        unique = { _fold(candidate): candidate for candidate in names if candidate }
        ordered = sorted(
            unique.values(),
            key=lambda candidate: (_candidate_span(segment, candidate).start() if _candidate_span(segment, candidate) else len(segment)),
        )
        name = ordered[0] if ordered else None

    if name is None:
        explicit = re.search(r"\bPe[łl]nomocnik\w*\s*[:\-–]\s*", appellant_block, re.IGNORECASE)
        if explicit:
            segment = appellant_block[explicit.end():]
            names = _bounded_person_candidates(segment) + _capitalized_person_candidates(segment)
            name = names[0] if names else None

    mailing_match = re.search(
        r"adres\s+do\s+korespondencji\s*[:\-–]?\s*(.+?)(?=\b(?:Zamawiając\w*|Wykonawca\s+wybrany|dotyczy|ODWOŁANIE)\b|$)",
        appellant_block,
        re.IGNORECASE | re.DOTALL,
    )
    mailing = mailing_match.group(1) if mailing_match else appellant_block

    firm = None
    firm_match = re.search(
        r"\bkancelaria\b\s*[:\-–]?\s*(.{2,180}?)"
        r"(?=\s*(?:[,;.]?\s*(?:e-?mail|adres|NIP|REGON|KRS)\b|$))",
        appellant_block,
        re.IGNORECASE | re.DOTALL,
    )
    if firm_match and heading is None:
        firm = _clean_org_name(firm_match.group(1))
    firm_boundary = re.search(
        r"(?i)(?:,\s*|\s+)(?:ul\.?\b|ulica\b|al\.?\b|aleja\b|pl\.?\b|plac\b|adres\b|tel\.?\b|fax\b|e-?mail\b)",
        mailing,
    )
    firm_source = mailing[:firm_boundary.start()] if firm_boundary else mailing
    if firm is None and re.search(r"\bkancelaria\b", firm_source, re.IGNORECASE):
        firm = _clean_org_name(firm_source)

    email_match = _EMAIL.search(mailing)
    if not email_match:
        email_match = _EMAIL.search(appellant_block)
    return name, firm, email_match.group() if email_match else None


def _build_party_data(template: dict[str, Any], party: Party | None, text: str, role: str) -> dict[str, Any]:
    output = _empty_like(template)
    if party:
        output.update({key: getattr(party, key) for key in ("krs", "nip", "regon") if key in output})
        if "nazwa" in output:
            output["nazwa"] = party.name
    if "nazwa" in output and output["nazwa"] is None:
        output["nazwa"] = _party_role_name(text, role)
    if role == "buyer" and "prowadzacy_postepowanie" in output:
        output["prowadzacy_postepowanie"] = None
        if party:
            role_pattern = re.compile(
                r"(?im)^\s*(?:(?:\d{1,2}|[a-z])[.)]\s*)?Zamawiając\w*"
                r"[^\n]{0,100}?\b(?P<role>Lider|Partner|prowadzący\s+postępowanie)\b[^\n]*$"
            )
            role_cues = list(role_pattern.finditer(text[:party.start]))
            if role_cues:
                output["prowadzacy_postepowanie"] = int(_fold(role_cues[-1].group("role")) != "partner")
    if role == "appellant" and "przez-pelnomocnika" in output:
        represented = _has(
            text,
            r"(?:reprezentowan\w*.{0,80}przez|działając\w*.{0,80}pełnomocnik|"
            r"dzialajac\w*.{0,80}pelnomocnik|przez\s+pe[łl]nomocnik\w*|"
            r"w oparciu o załączone pełnomocnictwo|w oparciu o zalaczone pelnomocnictwo)",
        )
        output["przez-pelnomocnika"] = int(represented)
    if role == "appellant" and "pelnomocnik" in output and isinstance(output["pelnomocnik"], dict):
        counsel = output["pelnomocnik"]
        counsel_name, counsel_firm, counsel_email = _appeal_counsel_fields(text)
        if "imie_nazwisko" in counsel:
            counsel["imie_nazwisko"] = counsel_name
        if "mail" in counsel:
            counsel["mail"] = counsel_email
        if "nazwa" in counsel:
            counsel["nazwa"] = counsel_firm
    return output


def _procurement_id_scope(text: str) -> str:
    """Keep notice IDs in the case-identification section, not cited old tenders."""
    scope = text[:10000]
    section = re.search(r"\b(?:Zarzut\s*(?:nr\s*)?1\b|Uzasadnienie\b)", scope, re.IGNORECASE)
    return scope[:section.start()] if section else scope


def _extract_bzp(text: str) -> str | None:
    scope = _procurement_id_scope(text)
    match = re.search(
        r"\b(20\d{2})\s*/\s*BZP\s*(\d{8,10})(?:\s*/\s*(\d{2}))?\b",
        scope,
        re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1)}/BZP{match.group(2)}" + (f"/{match.group(3)}" if match.group(3) else "")


def _extract_ted(text: str) -> str | None:
    scope = _procurement_id_scope(text)
    patterns = (
        re.compile(
            r"\b(?P<year>20\d{2})\s*/\s*(?P<section>[A-RT-Z])\s*"
            r"(?P<issue>\d{1,3})\s*[-/]\s*(?P<num>\d{6,8})\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?P<num>\d{6})\s*[-/]\s*(?P<year>20\d{2})\b"),
        re.compile(r"\b(?P<year>20\d{2})\s*/\s*S\s*\d{1,3}\s*[-–]\s*(?P<num>\d{6,8})\b", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(scope)
        if match:
            if "section" in match.groupdict():
                return (
                    f"{match.group('year')}/{match.group('section').upper()} "
                    f"{match.group('issue')}-{match.group('num')}"
                )
            return f"{match.group('num')}-{match.group('year')}"
    return None


def _extract_proceeding_number(text: str) -> str | None:
    patterns = (
        r"\bpostępowani\w*\s*[:№#–-]?\s*([A-Z0-9][A-Z0-9./_-]{2,50})(?=\s*[,;]|\s*$)",
        r"(?:nr\.?\s*referencyjn\w*\s*(?:sprawy|postępowania)?|numer\s+referencyjn\w*|numer\s+postępowania|nr\.?\s+postępowania|nr\.?\s+sprawy)\s*[:№#]?\s*([A-Z0-9][A-Z0-9./_-]{2,50})",
        r"\bpostępowania\b.{0,45}\bpn\.?\s*[:№#]?\s*([A-Z0-9][A-Z0-9./_-]{2,50})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ./-_")
            if not re.fullmatch(r"20\d{2}/BZP\d+", value, re.IGNORECASE) and not re.fullmatch(r"\d{6}-20\d{2}", value):
                return value
    return None


def _extract_title(text: str) -> str | None:
    labelled_subject = re.search(
        r"\bprzedmiot(?:\s+zamówienia)?\s*[:\-–]\s*(.{5,400}?)"
        r"(?=\s*(?:[.;,]\s*)?\b(?:miejscowość|miejscowosc|data\s+odwołania|"
        r"data\s+odwolania|zaskarżon\w*|zaskarzon\w*)\b|[.;]?\s*$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if labelled_subject:
        value = re.sub(r"\s+", " ", labelled_subject.group(1)).strip(" \t\r\n,;:.\"'„”«»")
        if len(value) >= 5:
            return value
    pn = re.search(r"\bPN\b\.?", text, re.IGNORECASE)
    if pn:
        quote = re.search(
            r'["„“«](.{5,500}?)["”»]',
            text[pn.end():pn.end() + 500],
            re.DOTALL,
        )
        if quote:
            value = re.sub(r"\s+", " ", quote.group(1)).strip(" \t\r\n,;:.\"'„”»")
            if len(value) >= 5:
                return value
    patterns = (
        r"\bpn\.?\s*[\"„«]?(.{5,600}?)(?:[\"”»])?\s*(?=\b(?:nr\.?\s+referencyjn|numer\s+ogłoszenia|numer\s+TED|numer\s+postępowania|ogłoszenie\s+o\s+zamówieniu)\b)",
        r"\bprzedmiot\s+zamówienia\s*[:\-]\s*(.{5,400}?)(?=\n|\b(?:nr\.?\s+postępowania|wartość|termin)\b)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n,;:.\"'„”«»")
            if len(value) >= 5:
                return value
    return None


def _extract_appeal(text: str, template: dict[str, Any]) -> dict[str, Any]:
    output = {key: None for key in template}
    kio_template = template.get("OdwolanieKIO", {})
    if isinstance(kio_template, dict):
        kio = {key: None for key in kio_template}
        filing_date, city = _header_date_city(text)
        kio["data"] = filing_date
        kio["miejscowosc"] = city
        cause_id = _cause_id(text)
        kio["czynnosc_powod_id"] = cause_id
        kio["czynnosc_opis"] = next((label for value, label, _ in _CAUSES if value == cause_id), None)
        kio["czynnosc_dzien"] = _cause_date(text)
        output["OdwolanieKIO"] = kio

    parties = _party_records(text)
    appellant = parties[0] if parties else None
    buyers = parties[1:]
    if "Odwolujacy" in template and isinstance(template["Odwolujacy"], list) and template["Odwolujacy"]:
        output["Odwolujacy"] = [_build_party_data(template["Odwolujacy"][0], appellant, text, "appellant")]
    if "Zamawiajacy" in template and isinstance(template["Zamawiajacy"], list) and template["Zamawiajacy"]:
        output["Zamawiajacy"] = [
            _build_party_data(template["Zamawiajacy"][0], buyer, text, "buyer")
            for buyer in buyers
        ]
        if not output["Zamawiajacy"]:
            output["Zamawiajacy"] = [_build_party_data(template["Zamawiajacy"][0], None, text, "buyer")]
    if "ZamowieniePubliczne" in template and isinstance(template["ZamowieniePubliczne"], dict):
        procurement = {key: None for key in template["ZamowieniePubliczne"]}
        procurement["numer_bzp"] = _extract_bzp(text)
        procurement["numer_ted"] = _extract_ted(text)
        procurement["numer_postepowania"] = _extract_proceeding_number(text)
        procurement["tytul"] = _extract_title(text)
        output["ZamowieniePubliczne"] = procurement
    return output


def _extract_grantor_name(text: str) -> str | None:
    parties = _party_records(text)
    if parties and parties[0].name:
        return parties[0].name
    for marker in re.finditer(r"\bMocodawc\w*\s*[:\-–]?", text[:6000], re.IGNORECASE):
        section = text[marker.end():marker.end() + 700]
        stop = re.search(
            r"\b(?:Pełnomocnik\w*|ustanawiam|udzielam|udziela)\b|§\s*2\b",
            section,
            re.IGNORECASE,
        )
        if stop:
            section = section[:stop.start()]
        for line in section.splitlines():
            candidate = _clean_org_name(line)
            if candidate:
                return candidate
    patterns = (
        r"(?is)(?:§\s*1\s*[.)]?\s*)(.{3,220}?)\s+jest\s+",
        r"(?is)\b(?:ja[, ]+)?(.{3,220}?)\s+(?:udzielam|udziela|udzielamy)\s+(?:niniejszym\s+)?pełnomocnictwa\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text[:5000])
        if match:
            name = _clean_org_name(match.group(1))
            if name:
                return name
    return None


def _extract_company_type(text: str, name: str | None, identifiers: dict[str, str | None]) -> str:
    folded = _fold((name or "") + " " + text[:1000])
    if any(identifiers.values()) or re.search(r"\b(spolka|sp z oo|s a|s\.a|krs|nip|regon|fundacja|stowarzyszenie|gmina|miasto|powiat|wojewodztwo)\b", folded):
        return "osoba_prawna"
    return "osoba_fizyczna"


def _candidate_span(text: str, name: str) -> re.Match[str] | None:
    parts = name.split()
    if not parts:
        return None
    pattern = r"[\s.,]+".join(re.escape(part) for part in parts)
    return re.search(pattern, text, re.IGNORECASE)


def _bounded_person_candidates(text: str) -> list[str]:
    """Find names within a role-bounded clause, rejecting OCR and list artifacts."""
    connectors = {"a", "i", "oraz", "lub", "albo", "do", "w", "we", "z", "ze", "na", "za", "przez", "pod", "od", "dla"}
    role_words = {
        "mocodawca", "pelnomocnik", "pelnomocnictwo", "adwokat", "radca", "prawny",
        "pesel", "regon", "krs", "nip", "wpisu", "prezes", "zarzadu", "czlonek",
        "prokurent", "pracownik", "dalej", "podpisany", "podpisana",
    }
    candidates: list[tuple[int, int, str]] = []
    for name in _person_candidates(text):
        span = _candidate_span(text, name)
        if span is None or re.search(r"\d", name):
            continue
        candidate_tokens = list(_TOKEN.finditer(text[span.start():span.end()]))
        if any(
            len(text[span.start() + left.end():span.start() + right.start()]) > 8
            for left, right in zip(candidate_tokens, candidate_tokens[1:])
        ):
            continue
        tokens = [_fold(match.group()) for match in _TOKEN.finditer(name)]
        if len(tokens) < 2 or tokens[0] in connectors or tokens[-1] in connectors:
            continue
        if any(token in role_words for token in tokens):
            continue
        cleaned = re.sub(r"\s+[.,]\s+", " ", name).strip()
        candidates.append((span.start(), span.end(), cleaned))

    # Names can overlap when a surname is also a given name in the lexicon
    # (for example, "First Last Next Surname"). Keep the first non-overlapping
    # complete match in the clause.
    result: list[str] = []
    end = -1
    for start, stop, name in sorted(candidates, key=lambda row: (row[0], -(row[1] - row[0]))):
        if start < end:
            continue
        result.append(name)
        end = stop
    return list(dict.fromkeys(result))


def _capitalized_person_candidates(text: str) -> list[str]:
    """Find title-case person-shaped names when the public name lexicon has gaps."""
    tokens = list(_TOKEN.finditer(text))
    excluded = {
        "w", "imieniu", "mocodawca", "pelnomocnik", "pelnomocnictwo", "adw", "radca", "prawny",
        "czlonek", "zarzadu", "prezes", "wiceprezes", "dyrektor", "prokurent", "wspolnik",
        "sp", "sa", "krs", "nip", "regon", "pesel", "lista", "wpisu", "numer", "dane",
        "rejestrowe", "zakres", "postepowanie", "zamawiajacy", "wykonawca", "podpisujacy",
    }
    found: list[tuple[int, int, str]] = []
    for start in range(len(tokens)):
        if not tokens[start].group()[0].isupper():
            continue
        for size in (3, 2):
            stop = start + size
            if stop > len(tokens):
                continue
            row = tokens[start:stop]
            if any(not token.group()[0].isupper() for token in row):
                continue
            if any(_fold(token.group()) in excluded for token in row):
                continue
            if any(
                len(gap) > 8 or not re.fullmatch(r"\s+", gap)
                for left, right in zip(row, row[1:])
                for gap in [text[left.end():right.start()]]
            ):
                continue
            candidate = " ".join(token.group() for token in row)
            found.append((row[0].start(), row[-1].end(), candidate))
            break

    result: list[str] = []
    end = -1
    for start, stop, candidate in sorted(found, key=lambda row: (row[0], -(row[1] - row[0]))):
        if start < end:
            continue
        result.append(candidate)
        end = stop
    return list(dict.fromkeys(result))


def _normalize_poa_company_name(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(?i)odpowiedzia[łl]nością", "odpowiedzialnością", value)
    value = re.sub(r"(?i)\bsp\.?\s*z\s*[o0]\s*\.?\s*[o0]\s*\.?$", "sp. z o.o.", value)
    value = re.sub(r"(?i)\bsp(\s*\.\s*k)\.?$", r"sp\1.", value)
    value = re.sub(r"(?i)\bsp(\s*\.\s*j)\.?$", r"sp\1.", value)
    value = re.sub(r"(?i)\bs\.?\s*a\.?$", "S.A.", value)
    return value


def _poa_action_match(text: str) -> re.Match[str] | None:
    return re.search(
        r"\b(?:"
        rf"udziel\w*\s+(?:niniejszego\s+)?{_ocr_literal_pattern('pełnomocnictwa')}"
        rf"|ustanaw\w*\s+(?:niniejszym\s+)?{_ocr_literal_pattern('pełnomocnikiem')}"
        rf"|ustanow\w*\s+(?:niniejszym\s+)?{_ocr_literal_pattern('pełnomocnikiem')}"
        rf"|{_ocr_literal_pattern('upoważni')}\w*\s+(?:(?:niniejszym|wskazanym|wskazanemu)\s+)?"
        rf"{_ocr_literal_pattern('pełnomocnik')}\w*"
        rf"|{_ocr_literal_pattern('upoważni')}(?:am|amy)\b"
        r")\b\s*[:\-–]?",
        text,
        re.IGNORECASE,
    )


def _poa_attorney_segment(text: str, action: re.Match[str]) -> str:
    tail = text[action.end():]
    stops = (
        r"\bdo\s+(?:reprezentowan\w*|reprezentacj\w*)",
        r"\bdo\s*[:：]",
        rf"\bw\s+{_ocr_literal_pattern('postępowan')}\w*",
        r"\bw\s+spraw\w*",
        rf"\bzakres\s+(?:{_ocr_literal_pattern('pełnomocnictwa')}|umocowania)\b",
        rf"\b{_ocr_literal_pattern('pełnomocnictwo')}\s+dotycz\w*",
        r"§\s*\d+\b",
    )
    boundaries = [re.search(pattern, tail, re.IGNORECASE) for pattern in stops]
    end = min((match.start() for match in boundaries if match), default=min(len(tail), 1200))
    return tail[:end]


def _merge_name_candidates(
    text: str,
    sources: list[tuple[str, int, list[str]]],
    *,
    minimum_score: int = 3,
    candidate_sink: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Merge role-specific rules and boost names independently confirmed by them."""
    merged: dict[str, dict[str, Any]] = {}
    for source, weight, names in sources:
        seen: set[str] = set()
        for name in names:
            key = _fold(name)
            if not key or key in seen:
                continue
            seen.add(key)
            row = merged.setdefault(
                key,
                {
                    "name": name,
                    "score": 0,
                    "sources": set(),
                    "position": len(text),
                    "start": None,
                    "end": None,
                },
            )
            row["score"] += weight
            row["sources"].add(source)
            span = _candidate_span(text, name)
            if span and span.start() < row["position"]:
                row["position"] = min(row["position"], span.start())
                row["start"] = span.start()
                row["end"] = span.end()

    selected = []
    for row in merged.values():
        independent_boost = min(2, max(0, len(row["sources"]) - 1)) * 2
        if candidate_sink is not None:
            candidate_sink.append({
                "value": row["name"],
                "score": row["score"] + independent_boost,
                "rule_ids": tuple(sorted(row["sources"])),
                "passed_minimum_score": row["score"] + independent_boost >= minimum_score,
                "start": row["start"],
                "end": row["end"],
            })
        if row["score"] + independent_boost >= minimum_score:
            selected.append(row)
    selected.sort(key=lambda row: row["position"])
    return [row["name"] for row in selected]


def _poa_role_line_values(text: str, role: str) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    pattern = re.compile(rf"(?im)^\s*{role}\b\s+(.+?)\s*$")
    for match in pattern.finditer(text):
        value = match.group(1).strip()
        if re.match(
            r"(?i)(?:upoważniam|ustanawiam|jest\b|są\b|pełnomocnik\b|mocodawca\b|"
            r"umocowan\w*|do\s+reprezentowania\b)",
            value,
        ):
            continue
        if value:
            values.append((match.start(1), value))
    return values


def _poa_table_attorneys(text: str) -> list[str]:
    candidates: list[str] = []
    for role in (r"Pe[łl]nomocnicy", r"Pe[łl]nomocnik"):
        for offset, _ in _poa_role_line_values(text, role):
            boundary = re.search(
                r"(?im)^\s*(?:Zamawiając\w*|Postępowanie|Zakres|Data|Dane\b|Mocodawc\w*|"
                r"Pe[łl]nomocnictwo\b)\s*[:\-–]?",
                text[offset:],
            )
            section = text[offset:offset + boundary.start()] if boundary else text[offset:offset + 900]
            candidates.extend(_bounded_person_candidates(section))
            candidates.extend(_capitalized_person_candidates(section))

    header = re.search(r"(?im)^\s*Mocodawca\s{2,}Pe[łl]nomocnik\s*$", text)
    if header:
        tail = text[header.end():]
        for line in tail.splitlines()[:14]:
            columns = re.split(r"\s{2,}", line.strip())
            if len(columns) > 1:
                candidates.extend(_bounded_person_candidates(columns[1]))
                candidates.extend(_capitalized_person_candidates(columns[1]))
            if re.match(r"(?i)^\s*(?:Pełnomocnictwo|Zamawiający|Postępowanie|Zakres)", line):
                break

    return list(dict.fromkeys(candidates))


def _poa_accepting_attorneys(text: str) -> list[str]:
    candidates: list[str] = []
    for marker in re.finditer(r"Pe[łl]nomocnictwo\s+przyjmuj\w*\s*[:\-–]?", text, re.IGNORECASE):
        line_start = text.rfind("\n", 0, marker.start()) + 1
        line_end = text.find("\n", marker.end())
        line_end = len(text) if line_end < 0 else line_end
        header_line = text[line_start:line_end]
        if re.search(r"\bMocodawca\s*:", header_line, re.IGNORECASE):
            for line in text[line_end + 1:].splitlines()[:12]:
                columns = re.split(r"\s{2,}", line.strip())
                if len(columns) > 1:
                    candidates.extend(_bounded_person_candidates(columns[-1]))
                    candidates.extend(_capitalized_person_candidates(columns[-1]))
                if re.match(r"(?i)^\s*(?:W\s+imieniu|Zakres|Pełnomocnictwo\s+dotycz)", line):
                    break
        else:
            tail = text[marker.end():marker.end() + 450]
            next_section = re.search(r"(?im)^\s*(?:W\s+imieniu|Zakres|Pełnomocnictwo\s+dotycz)", tail)
            if next_section:
                tail = tail[:next_section.start()]
            candidates.extend(_bounded_person_candidates(tail))
            candidates.extend(_capitalized_person_candidates(tail))
    return list(dict.fromkeys(candidates))


def _poa_grantor_blocks(text: str, action: re.Match[str] | None) -> list[str]:
    end = action.start() if action else len(text)
    if action is None:
        heading = re.search(r"PE[ŁL]NOMOCNICTWO", text, re.IGNORECASE)
        search_start = heading.end() if heading else 0
        next_role = re.search(r"\bPe[łl]nomocnic\w*\b", text[search_start:], re.IGNORECASE)
        if next_role:
            end = search_start + next_role.start()
        else:
            end = min(len(text), 5000)
    prefix = text[:end]
    item_starts = list(re.finditer(
        r"(?:^|[;:\n])\s*(?P<item>(?:\d{1,2}|[a-z])\s*[.)]\s+)",
        prefix,
        re.IGNORECASE,
    ))
    if _has(prefix, r"konsorcj\w*|wsp[oó]lnie\s+ubiegaj\w*") and len(item_starts) >= 2:
        return [
            prefix[match.start("item"):item_starts[index + 1].start("item") if index + 1 < len(item_starts) else len(prefix)]
            for index, match in enumerate(item_starts)
        ]
    return [prefix]


def _poa_grantor_from_block(block: str, text: str, attorney_names: list[str]) -> dict[str, Any]:
    block = re.sub(r"^\s*(?:\d{1,2}|[a-z])\s*[.)]\s*", "", block, flags=re.IGNORECASE)
    individual_match = re.search(
        r"(?is)(?:ja\s*,?\s*)?niżej\s+podpisan\w*(?:\s*\([^)]*\))?\s+(.+?)\s*,?\s*prowadz\w*\s+(?:działalność|dzialalnosc)",
        block,
    )
    individual_name = None
    if individual_match:
        names = _bounded_person_candidates(individual_match.group(1))
        individual_name = names[0] if names else None

    company_candidates: list[tuple[int, str]] = []
    table_company_candidates: list[str] = []
    inline_grantor = re.search(
        r"\bMocodawc\w*\s*[:\-–]?\s*(?P<name>.+?)"
        rf"(?=\s*(?:\(|,?\s*{_OCR_IDENTIFIER_LABELS}|"
        r",?\s*(?:działa|dziala|działając|dzialajac|podpisuj\w*|udziel\w*|reprezentuj\w*)\b))",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if inline_grantor:
        candidate = _clean_org_name(inline_grantor.group("name"))
        if candidate:
            local_ids = _identifier_hits(block[inline_grantor.end():inline_grantor.end() + 120])
            legal_entity = bool(any(local_ids.values())) or re.search(
                r"\b(?:sp[oó]łk\w*|sp\s*\.\s*z\s*o\.?\s*o|s\.?\s*a\.?|fundacj\w*|"
                r"stowarzyszen\w*|gmina|miasto|powiat|wojew[oó]dztwo)\b",
                candidate,
                re.IGNORECASE,
            )
            if not legal_entity and individual_name is None:
                person_names = (
                    _bounded_person_candidates(candidate)
                    + _capitalized_person_candidates(candidate)
                )
                if person_names:
                    individual_name = person_names[0]
            if individual_name is None or legal_entity:
                company_candidates.append(
                    (
                        9 if legal_entity else 1,
                        _normalize_poa_company_name(candidate) or candidate,
                    )
                )

    representation_pattern = (
        rf"(?is)\bdo\s+reprezentacji\s+(.+?)"
        rf"(?=\s*\(|,?\s+{_OCR_IDENTIFIER_LABELS}|,?\s+ustanawiam\b|$)"
    )
    org_patterns = (
        (5, rf"(?is)\bMocodawc\w*\s+jest\s+(.+?)(?=,?\s+(?:ul\.?\b|{_OCR_IDENTIFIER_LABELS})|[,;\n]|$)"),
        (4, r"(?im)^\s*Mocodawca\s{2,}(.+?)\s*$"),
        (4, representation_pattern),
        (3, rf"(?is)\bdziałając\s+w\s+imieniu\s+(.+?)(?=,?\s+(?:dalej|{_OCR_IDENTIFIER_LABELS})|[,;\n]|$)"),
        (3, rf"(?is)\bpod\s+firm\w*\s+(.+?)(?=,?\s+{_OCR_IDENTIFIER_LABELS}|[,;\n]|$)"),
    )
    representation_company_candidates: list[str] = []
    for weight, pattern in org_patterns:
        for match in re.finditer(pattern, block, re.IGNORECASE):
            candidate = _normalize_poa_company_name(_clean_org_name(match.group(1)))
            if candidate:
                company_candidates.append((weight, candidate))
                if pattern == representation_pattern:
                    representation_company_candidates.append(candidate)

    # Some source PDFs put the registry labels in a separate line after the name.
    rows = block.splitlines()
    for index, line in enumerate(rows):
        marker = re.search(rf"(?:{_OCR_IDENTIFIER_LABELS})", line, re.IGNORECASE)
        if not marker:
            continue
        same_line = _clean_org_name(line[:marker.start()])
        if same_line:
            combined = None
            if index:
                previous = _clean_org_name(rows[index - 1])
                if previous and len(previous.split()) <= 2 and not re.search(
                    r"(?i)\b(?:ul|ulica|adres|nip|krs|regon|mocodawc|wykonawc|konsorcj)\b",
                    previous,
                ) and re.search(r"(?i)\b(?:sp\.?\s*(?:z\s*o\.?\s*o\.?|k\.?|j\.?)|s\.?\s*a\.?)\s*$", same_line):
                    combined = _normalize_poa_company_name(f"{previous} {same_line}")
                    company_candidates.append((8, combined))
            if not combined:
                company_candidates.append((3, _normalize_poa_company_name(same_line)))
        elif index:
            previous = _clean_org_name(rows[index - 1])
            if previous:
                company_candidates.append((2, _normalize_poa_company_name(previous)))

    table_header = re.search(r"(?im)^\s*Mocodawca\s{2,}Pe[łl]nomocnik\s*$", block)
    if table_header:
        for line in block[table_header.end():].splitlines()[:14]:
            columns = re.split(r"\s{2,}", line.strip())
            if len(columns) > 1:
                candidate = _normalize_poa_company_name(_clean_org_name(columns[0]))
                if candidate and not re.match(r"(?i)^(?:pe[łl]nomocnik|mocodawca)\b", candidate):
                    company_candidates.append((6, candidate))
                    table_company_candidates.append(candidate)

    for _, value in _poa_role_line_values(block, r"Mocodawca"):
        candidate = _normalize_poa_company_name(_clean_org_name(value))
        if candidate:
            company_candidates.append((5, candidate))

    records = _party_records(block)
    for party in records:
        if party.name and not re.match(r"(?i)^(?:pe[łl]nomocnik|mocodawca|zamawiający)\b", party.name):
            weight = 4 if any((party.krs, party.nip, party.regon)) else 2
            company_candidates.append((weight, _normalize_poa_company_name(party.name)))

    identifiers = {
        "krs": _labeled_identifier_number(block, "KRS", validate_krs, (10,)),
        "nip": _labeled_identifier_number(block, "NIP", validate_nip, (10,)),
        "regon": _labeled_identifier_number(block, "REGON", validate_regon, (9, 14)),
    }

    def _org_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", _fold(value))

    scores: dict[str, tuple[int, str]] = {}
    for weight, value in company_candidates:
        key = _org_key(value)
        if not key:
            continue
        previous = scores.get(key)
        scores[key] = ((previous[0] if previous else 0) + weight, previous[1] if previous else value)
    # A two-column Mocodawca/Pełnomocnik table gives an explicit role for its
    # first column. Prefer its first legal-entity name over generic candidates
    # accumulated from nearby prose and repeated party mentions.
    table_company_name = next(
        (
            candidate
            for candidate in table_company_candidates
            if _extract_company_type(block, candidate, identifiers) == "osoba_prawna"
        ),
        None,
    )
    representation_company_name = next(
        (
            candidate
            for candidate in representation_company_candidates
            if _extract_company_type(block, candidate, identifiers) == "osoba_prawna"
        ),
        None,
    )
    declared_company = None
    if identifiers["krs"] and not individual_name:
        heading = re.search(r"(?im)^[ \t]*PE[ŁL]NOMOCNICTWO(?:[ \t]+PROCESOWE)?[ \t]*$", block)
        prefix = block[heading.end():] if heading else block
        for pattern in (
            r"(?is)\bdzia[łl]aj[ąa]c\s+w\s+imieniu\s+([^()]{4,160}?)\s*\(dalej\b",
            r"(?im)^[ \t]*([^\n]{4,150}?)\s+z\s+siedzib\w*\b",
        ):
            match = re.search(pattern, prefix)
            if match:
                declared_company = _normalize_poa_company_name(_clean_org_name(match.group(1)))
                if declared_company:
                    break
        if heading and not declared_company:
            paragraph = re.split(r"\n[ \t]*\n", prefix.lstrip(), maxsplit=1)[0]
            address = re.match(
                r"(?i)([^\n,]{4,150}?),\s+(?:ul\.|ulica\b|al\.|aleja\b|pl\.|plac\b)",
                paragraph,
            )
            if address and _identifier_hits(paragraph)["krs"] == identifiers["krs"]:
                declared_company = _normalize_poa_company_name(_clean_org_name(address.group(1)))
    authoritative_company_name = table_company_name or representation_company_name or declared_company
    company_name = authoritative_company_name or max(
        scores.values(), default=(0, None), key=lambda row: row[0]
    )[1]

    trade_name = None
    firm_match = re.search(rf"(?is)\bpod\s+firm\w*\s+(.+?)(?=,?\s+{_OCR_IDENTIFIER_LABELS}|[,;\n]|$)", block)
    if firm_match and authoritative_company_name is None:
        trade_name = _normalize_poa_company_name(_clean_org_name(firm_match.group(1)))
        if trade_name:
            if individual_name and not _fold(trade_name).endswith(_fold(individual_name)):
                owner = r"[ \t]+".join(re.escape(part) for part in individual_name.split())
                continuation = re.match(
                    rf"\r?\n[ \t]*({owner})(?=[ \t]*[,;])",
                    block[firm_match.end():],
                    re.IGNORECASE,
                )
                if continuation:
                    trade_name = f"{trade_name} {continuation.group(1)}"
            company_name = trade_name

    signer_names: list[str] = []
    for match in re.finditer(r"reprezentowan\w*\s+przez\s*[:\-–]?", block, re.IGNORECASE):
        following = block[match.end():]
        stop = re.search(r"\(\s*dalej\s*[:\-–]?\s*[\"„]?mocodawc\w*", following, re.IGNORECASE)
        signer_clause = following[:stop.start()] if stop else following[:500]
        signer_names.extend(_bounded_person_candidates(signer_clause))
    for match in re.finditer(r"\bpodpisuj\w*\s*[:\-–]?\s*", block, re.IGNORECASE):
        following = block[match.end():]
        stop = re.search(
            rf"\b(?:i\s+)?udziel\w*\b|{_ocr_acronym_pattern('PESEL')}|\bPe[łl]nomocnictw\w*\b",
            following,
            re.IGNORECASE,
        )
        signer_clause = following[:stop.start()] if stop else following[:180]
        signer_names.extend(
            _bounded_person_candidates(signer_clause)
            + _capitalized_person_candidates(signer_clause)
        )
    signer_names = _merge_name_candidates(
        text,
        [("representation", 3, signer_names), ("signature", 3, _poa_signature_names(text, company_name, attorney_names))],
    )

    return {
        "imie_nazwisko": individual_name,
        "nazwa": company_name,
        **identifiers,
        "typ": "osoba_fizyczna" if individual_name else _extract_company_type(block, company_name, identifiers),
        "podpisujacy": signer_names,
    }


def _poa_signature_names(text: str, company_name: str | None, attorney_names: list[str]) -> list[str]:
    attorney_keys = {_fold(name) for name in attorney_names}
    mapped: list[str] = []
    sections = list(re.finditer(r"(?im)^\s*W\s+imieniu\s+(.+?)\s*:\s*$", text))
    for index, marker in enumerate(sections):
        party_label = _fold(_clean_org_name(marker.group(1)) or marker.group(1))
        if company_name and _fold(company_name) not in party_label and party_label not in _fold(company_name):
            continue
        end = sections[index + 1].start() if index + 1 < len(sections) else min(len(text), marker.end() + 900)
        section = text[marker.end():end]
        names = _merge_name_candidates(
            text,
            [
                ("signature-lexicon", 3, _bounded_person_candidates(section)),
                ("signature-case", 3, _capitalized_person_candidates(section)),
            ],
        )
        mapped.extend(name for name in names if _fold(name) not in attorney_keys and not _fold(name).startswith("za "))
    if mapped:
        return list(dict.fromkeys(mapped))

    mocodawca_header = re.search(r"(?im)^\s*Mocodawca\s*:\s*.*$", text)
    if mocodawca_header:
        end = min(len(text), mocodawca_header.end() + 1400)
        next_heading = re.search(r"(?im)^\s*(?:Pełnomocnictwo\s+przyjmuj\w*|W\s+imieniu\b|Zakres\b)", text[mocodawca_header.end():end])
        if next_heading:
            end = mocodawca_header.end() + next_heading.start()
        section = text[mocodawca_header.end():end]
        names = _merge_name_candidates(
            text,
            [
                ("signature-lexicon", 3, _bounded_person_candidates(section)),
                ("signature-case", 3, _capitalized_person_candidates(section)),
            ],
        )
        names = [name for name in names if _fold(name) not in attorney_keys and not _fold(name).startswith("za ")]
        if names:
            return list(dict.fromkeys(names))

    separators = list(re.finditer(r"(?:\.{5,}|_{5,})", text[-3000:]))
    if separators:
        signature_start = max(0, len(text) - 3000) + separators[0].start()
        section = text[signature_start:]
        names = _merge_name_candidates(
            text,
            [
                ("signature-lexicon", 3, _bounded_person_candidates(section)),
                ("signature-case", 3, _capitalized_person_candidates(section)),
            ],
        )
        return list(dict.fromkeys(name for name in names if _fold(name) not in attorney_keys and not _fold(name).startswith("za ")))
    return []


def _poa_grantors(
    text: str,
    action: re.Match[str] | None,
    attorney_names: list[str],
) -> list[dict[str, Any]]:
    records = [
        _poa_grantor_from_block(block, text, attorney_names)
        for block in _poa_grantor_blocks(text, action)
    ]
    # Use the old broad parser only if the POA has no recognizable declaration.
    if records and any(row["nazwa"] or row["imie_nazwisko"] or row["krs"] or row["nip"] for row in records):
        return records
    parties = _party_records(text)[:2]
    if parties:
        return [{
            "imie_nazwisko": None,
            "nazwa": party.name,
            "krs": party.krs,
            "nip": party.nip,
            "regon": party.regon,
            "typ": _extract_company_type(text[:1800], party.name, {"krs": party.krs, "nip": party.nip, "regon": party.regon}),
            "podpisujacy": [],
        } for party in parties]
    return records


def _rank_person_candidates(text: str, cue_pattern: str) -> list[str]:
    names = _person_candidates(text[:12000])
    positions = {name: text.casefold().find(name.casefold()) for name in names}
    cues = [match.start() for match in re.finditer(cue_pattern, text, re.IGNORECASE)]
    if not cues:
        return names
    return sorted(
        names,
        key=lambda name: (
            min(abs(positions[name] - cue) for cue in cues),
            positions[name],
        ),
    )


def _extract_poa(
    text: str,
    template: dict[str, Any],
    *,
    candidate_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = {key: None for key in template}
    moc_template = template.get("mocodawca", [{}])
    moc_item_template = moc_template[0] if isinstance(moc_template, list) and moc_template else {}
    data_template = moc_item_template.get("data", {}) if isinstance(moc_item_template, dict) else {}
    action = _poa_action_match(text)
    attorney_segment = _poa_attorney_segment(text, action) if action else ""
    attorney_sources = [
        ("action-window-lexicon", 4, _bounded_person_candidates(attorney_segment)),
        ("action-window-case", 3, _capitalized_person_candidates(attorney_segment)),
        ("role-table", 4, _poa_table_attorneys(text)),
        ("acceptance-signature", 3, _poa_accepting_attorneys(text)),
    ]
    attorney_names = _merge_name_candidates(
        text, attorney_sources, candidate_sink=candidate_sink
    )
    grantors = _poa_grantors(text, action, attorney_names)

    excluded_people = {
        _fold(name)
        for grantor in grantors
        for name in [grantor.get("imie_nazwisko"), *grantor.get("podpisujacy", [])]
        if name
    }
    attorney_names = [name for name in attorney_names if _fold(name) not in excluded_people]
    if not attorney_names:
        cue_names = _people_near_cues(
            text[:12000],
            (r"pełnomocnik\w*\s*[:\-–]", r"ustanawiam\w*.{0,80}", r"upoważniam\w*.{0,80}"),
        )
        attorney_names = _merge_name_candidates(
            text,
            [("explicit-role-cue", 3, cue_names)],
            candidate_sink=candidate_sink,
        )
        attorney_names = [name for name in attorney_names if _fold(name) not in excluded_people]

    sign_template = moc_item_template.get("podpisujacy", [{}]) if isinstance(moc_item_template, dict) else [{}]
    sign_item = sign_template[0] if sign_template else {"imie_nazwisko": None}
    grantor_items = []
    for grantor in grantors:
        data = {key: None for key in data_template}
        if "nazwa" in data:
            data["nazwa"] = grantor["nazwa"]
        for key in ("krs", "nip"):
            if key in data:
                data[key] = grantor[key]
        if "typ" in data:
            data["typ"] = grantor["typ"]
        if "imie_nazwisko" in data:
            data["imie_nazwisko"] = grantor["imie_nazwisko"]

        grantor_item = {key: None for key in moc_item_template} if isinstance(moc_item_template, dict) else {"data": data}
        grantor_item["data"] = data
        grantor_item["podpisujacy"] = [
            {key: (name if key == "imie_nazwisko" else None) for key in sign_item}
            for name in grantor["podpisujacy"]
        ]
        if not grantor_item["podpisujacy"]:
            grantor_item["podpisujacy"] = [{key: None for key in sign_item}]
        grantor_items.append(grantor_item)
    if grantor_items:
        output["mocodawca"] = grantor_items

    attorney_template = template.get("pelnomocnicy", [{}])
    attorney_item_template = attorney_template[0] if attorney_template else {"imie_nazwisko": None, "pesel": None}
    output["pelnomocnicy"] = []
    attorney_spans = [_candidate_span(attorney_segment, name) for name in attorney_names]
    for index, name in enumerate(attorney_names):
        item = {key: None for key in attorney_item_template}
        item["imie_nazwisko"] = name if "imie_nazwisko" in item else None
        span = attorney_spans[index]
        if "pesel" in item and span:
            next_start = next(
                (following.start() for following in attorney_spans[index + 1:] if following),
                len(attorney_segment),
            )
            local_pesels = _labeled_pesels(attorney_segment[span.end():next_start])
            if local_pesels:
                item["pesel"] = local_pesels[0][0]
        output["pelnomocnicy"].append(item)
    return output


def _extract_accounts(text: str) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for candidate in _digit_candidates(text, (26,)):
        if candidate[0] not in seen:
            candidates.append(candidate)
            seen.add(candidate[0])
    if not candidates:
        return None, None

    role_patterns = {
        "sender": r"\b(?:z\s+rachunku|rachunek\s+nadawc\w*|nadawc\w*)\b",
        "recipient": r"\b(?:na\s+rachunek|rachunek\s+odbiorc\w*|odbiorc\w*)\b",
    }

    def nearby_role(candidate: tuple[str, int, int]) -> str | None:
        _, start, end = candidate
        hits: list[tuple[int, str]] = []
        for role, pattern in role_patterns.items():
            for marker in re.finditer(pattern, text, re.IGNORECASE):
                distance = (
                    start - marker.end()
                    if marker.end() <= start
                    else marker.start() - end
                    if marker.start() >= end
                    else 0
                )
                if 0 <= distance <= 180:
                    hits.append((distance, role))
        hits.sort()
        if not hits:
            return None
        closest_distance, closest_role = hits[0]
        next_other_distance = min(
            (distance for distance, role in hits if role != closest_role),
            default=None,
        )
        if next_other_distance is not None and next_other_distance - closest_distance < 16:
            return None
        return closest_role

    if len(candidates) >= 2:
        roles = [nearby_role(candidate) for candidate in candidates]
        sender_indexes = [index for index, role in enumerate(roles) if role == "sender"]
        recipient_indexes = [index for index, role in enumerate(roles) if role == "recipient"]
        if (
            len(sender_indexes) == 1
            and len(recipient_indexes) == 1
            and sender_indexes[0] != recipient_indexes[0]
        ):
            return candidates[sender_indexes[0]][0], candidates[recipient_indexes[0]][0]
        # Some bank PDFs flatten two-column tables and place both role labels
        # close to both accounts. Use labels only when they unambiguously assign
        # one candidate to each role; otherwise preserve the document order.
        return candidates[0][0], candidates[1][0]

    only = candidates[0]
    role = nearby_role(only)
    return (None, only[0]) if role == "recipient" else (only[0], None)


def _parse_amount_value(value: str) -> Decimal | None:
    compact = re.sub(r"[^\d,.+\-]", "", value).lstrip("+")
    if not compact or not re.fullmatch(r"-?\d[\d,.]*", compact):
        return None
    sign = "-" if compact.startswith("-") else ""
    body = compact.lstrip("-")
    if not re.search(r"\d", body):
        return None

    if "." in body and "," in body:
        decimal_separator = "." if body.rfind(".") > body.rfind(",") else ","
        integer, fraction = body.rsplit(decimal_separator, 1)
        integer = re.sub(r"[.,]", "", integer)
    elif "." in body or "," in body:
        separator = "." if "." in body else ","
        parts = body.split(separator)
        fraction = parts[-1]
        grouped = len(parts) > 2 and all(len(part) == 3 for part in parts[1:])
        if grouped or (len(parts) == 2 and len(fraction) == 3 and len(parts[0]) <= 3):
            integer = "".join(parts)
            fraction = ""
        else:
            integer = "".join(parts[:-1])
    else:
        integer, fraction = body, ""

    if not integer.isdigit() or (fraction and not fraction.isdigit()):
        return None
    try:
        return Decimal(f"{sign}{integer}" + (f".{fraction}" if fraction else ""))
    except InvalidOperation:
        return None


def _extract_amount(text: str) -> float | None:
    amount_core = (
        r"(?<!\d)(?P<amount>(?:\d{1,3}(?:[ .,\u00a0]\d{3})+|\d+)"
        r"(?:[,.]\d{1,2})?)"
    )
    amount_pattern = re.compile(amount_core + r"(?![\d,.])")
    labels = (
        (r"\bkwota\s+(?:transakcji|przelewu)\b", 12),
        (r"\bw\s+kwocie\b", 12),
        (rf"\b{_ocr_literal_pattern('wartość transakcji')}\b", 11),
        (rf"(?im)^\s*{_ocr_literal_pattern('wartość')}\b", 6),
        (r"\bkwota\b", 8),
        (r"\bsuma\b", 7),
    )
    candidates: list[tuple[float, int, Decimal]] = []

    def add_candidate(raw: str, position: int, score: float) -> None:
        parsed = _parse_amount_value(raw)
        if parsed is not None:
            candidates.append((score, position, parsed))

    paired_header = re.compile(
        r"(?im)^[ \t]*kwota(?:[ \t]+(?:operacji|transakcji|przelewu))?"
        r"[ \t]{2,}data\b[^\n]*$"
    )
    for header in paired_header.finditer(text):
        following = text[header.end():header.end() + 500]
        row = next((line.strip() for line in following.splitlines()[:4] if line.strip()), "")
        columns = re.split(r"[ \t]{2,}", row, maxsplit=1)
        if len(columns) != 2 or not _parse_dates(columns[1]):
            continue
        amount = re.fullmatch(
            rf"(?:PLN[ \t]+)?{amount_core}(?:[ \t]*(?:PLN|z[łl]))?",
            columns[0],
            re.IGNORECASE,
        )
        if amount:
            add_candidate(amount.group("amount"), header.end(), 13)

    for label, priority in labels:
        for marker in re.finditer(label, text, re.IGNORECASE):
            suffix = text[marker.end():marker.end() + 90]
            for match in amount_pattern.finditer(suffix):
                gap = suffix[:match.start()]
                if len(re.sub(r"[\s.:…]+", "", gap)) <= 45:
                    add_candidate(
                        match.group("amount"),
                        marker.end() + match.start("amount"),
                        priority - min(len(gap) / 500, 0.9),
                    )
                    break

    currency_pattern = re.compile(
        rf"{amount_core}(?![\d,.])\s*"
        rf"(?:P\s*L\s*N\.?|{_ocr_literal_pattern('zł')}\.?|zl\.?)(?!\w)",
        re.IGNORECASE,
    )
    for match in currency_pattern.finditer(text):
        amount = amount_pattern.search(match.group())
        if amount:
            add_candidate(amount.group("amount"), match.start(), 5)
    for match in re.finditer(rf"\bPLN[ \t]+{amount_core}(?![\d,.])", text, re.IGNORECASE):
        add_candidate(match.group("amount"), match.start("amount"), 5)

    if not candidates:
        return None
    _, _, amount = min(candidates, key=lambda row: (-row[0], row[1]))
    return float(amount)


def _extract_transaction_date(text: str) -> str | None:
    for label in (
        r"data\s+(?:i\s+godzina\s+)?operacji",
        r"data\s+transakcji",
        r"data\s+(?:wykonania|realizacji)",
        r"bank\s+potwierdza\W+że\s+w\s+dniu",
        rf"data\s+{_ocr_literal_pattern('księgowania')}",
        r"data\s+zlecenia",
    ):
        for marker in re.finditer(label, text, re.IGNORECASE):
            window = text[marker.end():marker.end() + 100]
            dates = _parse_dates(window)
            if dates:
                return dates[0][0]
    dates = _parse_dates(text[:2500])
    return dates[0][0] if dates else None


def _extract_transaction_title(text: str) -> str | None:
    specific_labels = (
        "tytuł transakcji",
        "tytuł płatności",
        "tytuł przelewu",
        "tytułem przelewu",
        "tytułem",
        "tytuł",
    )
    label_matches = [
        match
        for label in specific_labels
        for match in re.finditer(
            rf"(?<!\w){_ocr_literal_pattern(label)}(?!\w)",
            text,
            re.IGNORECASE,
        )
    ]
    # Some confirmations put the description under "Opis" or in the next
    # column. Within prose, only an explicit separator marks a field.
    label_matches.extend(
        re.finditer(
            rf"(?<!\w){_ocr_literal_pattern('opis')}[ \t]*[:：=]|"
            rf"(?m:^[ \t]*{_ocr_literal_pattern('opis')}(?:[ \t]*$|[ \t]{{2,}}(?=\S)))",
            text,
            re.IGNORECASE,
        )
    )
    label_matches.sort(key=lambda match: (match.start(), -(match.end() - match.start())))

    date_field = (
        r"data\s+(?:i\s+godzina\s+)?(?:operacji|transakcji|"
        + _ocr_literal_pattern("księgowania")
        + r"|zlecenia)"
    )
    number_field = (
        r"(?:numer|nr\.?|id|identyfikator)\s+"
        r"(?:transakcji|operacji|referenc\w*|ref\.?)"
    )
    field_pattern = (
        rf"(?:{date_field}|kwota(?:\s+(?:transakcji|przelewu))?|"
        rf"(?:z|na)\s+rachunek|{number_field}|nadawca|odbiorca|status)"
    )
    line_field = re.compile(
        rf"^\s*(?:{field_pattern}\b\s*(?:[:：=]\s*)?|"
        r"operacj\w*\s*[:：=]\s*|nr\.?\s+ref\b|"
        r"typ\s+(?:operacji|transakcji|przelewu)\b|"
        r"(?:kanał|waluta|wartość|tryb)\b|opłata\b(?!\s+skarbowa)|"
        r"(?:bank|dokument|niniejsze)\b|signature\s+not\b)",
        re.IGNORECASE,
    )
    inline_field = re.compile(
        rf"(?:[,;.]?\s+)(?P<field>{field_pattern})\b\s*(?:[:：=]\s*)?|"
        r"(?:[,;.]?\s+)operacj\w*\s*[:：=]\s*",
        re.IGNORECASE,
    )

    for marker in label_matches:
        tail = text[marker.end():marker.end() + 1200]
        rows: list[str] = []
        for raw_line in tail.splitlines():
            line = raw_line.strip(" \t:：=–-")
            if not line:
                continue
            if line_field.match(line):
                break
            next_field = inline_field.search(line)
            if next_field:
                line = line[:next_field.start()]
            line = line.strip(" \t\r\n,;:.\"'„”«»|-–")
            if line:
                rows.append(line)
            if sum(len(row) for row in rows) >= 300 or len(rows) >= 4:
                break
        value = re.sub(r"\s+", " ", " ".join(rows)).strip()
        if 3 <= len(value) <= 300:
            return value

    patterns = (
        rf"(?:{_ocr_literal_pattern('tytuł transakcji')}|"
        rf"{_ocr_literal_pattern('tytuł przelewu')}|"
        rf"{_ocr_literal_pattern('tytuł płatności')}|"
        rf"{_ocr_literal_pattern('tytułem przelewu')}|tytułem|tytuł)\b\s*[:\-]?\s*"
        rf"(.{{3,300}}?)(?=\n\s*{field_pattern}\b|$)",
        rf"\b({_ocr_literal_pattern('wpłata za')}\b.{{3,180}}?)(?=\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n,;:.\"'„”«»")
            if len(value) >= 3:
                return value
    return None


def _extract_transaction_number(text: str) -> str | None:
    explicit = re.compile(
        r"\b(?:numer|nr\.?|id|identyfikator)\s*[:№#-]?\s*"
        r"(?:transakcj\w*|operacj\w*|referenc\w*|ref\.?)"
        r"\s*[:№#]?\s*([A-Z0-9][A-Z0-9./_-]{3,60})",
        re.IGNORECASE,
    )
    for match in explicit.finditer(text):
        value = match.group(1).strip(" ./-_")
        if re.search(r"\d", value):
            return value

    repeated_codes: dict[str, int] = {}
    for candidate in re.finditer(
        r"(?<![A-Z0-9])([A-Z0-9]{12,32})(?![A-Z0-9])", text, re.IGNORECASE
    ):
        value = candidate.group(1).upper()
        if not (re.search(r"[A-Z]", value) and re.search(r"\d", value)):
            continue
        # Long country-prefixed account numbers can also be repeated in a receipt.
        if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", value):
            continue
        repeated_codes[value] = repeated_codes.get(value, 0) + 1

    repeated = [value for value, count in repeated_codes.items() if count >= 2]
    if len(repeated) == 1:
        return repeated[0]
    return None


def _extract_transfer(text: str, template: dict[str, Any]) -> dict[str, Any]:
    output = {key: None for key in template}
    account_template = template.get("przelew_bankowy", {})
    if not isinstance(account_template, dict):
        return output
    transfer = {key: None for key in account_template}
    sender, recipient = _extract_accounts(text)
    transfer["data_operacji"] = _extract_transaction_date(text)
    transfer["kwota"] = _extract_amount(text)
    transfer["z_rachunku"] = sender
    transfer["na_rachunek"] = recipient
    transfer["numer_transakcji"] = _extract_transaction_number(text)
    transfer["tytul"] = _extract_transaction_title(text)
    output["przelew_bankowy"] = transfer
    return output


ExtractorHandler = Callable[[str, dict[str, Any]], dict[str, Any]]

_EXTRACTOR_HANDLERS: dict[str, ExtractorHandler] = {
    "odwolanie": _extract_appeal,
    "pelnomocnictwo": _extract_poa,
    "przelew": _extract_transfer,
}


def _exact_source_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Return exact source spans for a value, allowing whitespace variation only."""
    needle = value.strip()
    if not needle:
        return []
    pattern_parts: list[str] = []
    for part in re.split(r"(\s+)", needle):
        pattern_parts.append(r"\s+" if part.isspace() else re.escape(part))
    pattern = "".join(pattern_parts)
    if needle[0].isalnum() or needle[0] == "_":
        pattern = r"(?<!\w)" + pattern
    if needle[-1].isalnum() or needle[-1] == "_":
        pattern += r"(?!\w)"
    return [(match.start(), match.end()) for match in re.finditer(pattern, text, re.IGNORECASE)]


class RegexExtractor:
    """Deterministic extractors registered by document type."""

    name = "regex"
    model = "deterministic-rules"
    thinking_level = None

    def __init__(self) -> None:
        self._templates = deepcopy(_TEMPLATES)
        self._handlers = dict(_EXTRACTOR_HANDLERS)

    @property
    def document_types(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def register(
        self,
        document_type: str,
        handler: ExtractorHandler,
        template: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        """Register a type-specific handler and output template."""
        document_type = document_type.strip()
        if not document_type:
            raise ValueError("Document type must not be empty.")
        if document_type in self._handlers and not replace:
            raise ValueError(f"Document type is already registered: {document_type}")
        if not callable(handler):
            raise TypeError("Extractor handler must be callable.")
        if not isinstance(template, dict):
            raise TypeError("Extractor template must be a dictionary.")
        self._handlers[document_type] = handler
        self._templates[document_type] = deepcopy(template)

    def extract(
        self,
        document_type: str,
        text: str,
        template: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        handler = self._handlers.get(document_type)
        if handler is None:
            raise ValueError(f"No regex extractor is configured for document type: {document_type}")
        selected_template = deepcopy(
            template if template is not None else self._templates[document_type]
        )
        return handler(text, selected_template)

    def extract_with_candidates(
        self,
        document_type: str,
        text: str,
    ) -> RegexExtraction:
        """Return the unchanged JSON result and source-grounded regex candidates.

        Scores are raw, uncalibrated rule weights; fields without an existing
        rule score use 0.0. They are not probabilities. Candidate offsets are
        emitted only when the value can be located in the supplied source text.
        """
        handler = self._handlers.get(document_type)
        if handler is None:
            raise ValueError(f"No regex extractor is configured for document type: {document_type}")

        candidate_sink: list[dict[str, Any]] = []
        if handler is _extract_poa:
            data = _extract_poa(
                text,
                deepcopy(self._templates[document_type]),
                candidate_sink=candidate_sink,
            )
        else:
            data = self.extract(document_type, text)

        # First include every locatable string value in the existing output.
        records: dict[tuple[str, int, int, str | None], dict[str, Any]] = {}

        def add_candidate(
            field: str,
            value: str,
            start: int,
            end: int,
            *,
            score: float = 0.0,
            rule_ids: tuple[str, ...] = (),
            group_id: str | None = None,
            passed_minimum_score: bool | None = None,
        ) -> None:
            key = (field, start, end, group_id)
            row = records.get(key)
            if row is None:
                records[key] = {
                    "field": field,
                    "value": value,
                    "start": start,
                    "end": end,
                    "score": score,
                    "rule_ids": set(rule_ids),
                    "group_id": group_id,
                    "passed_minimum_score": passed_minimum_score,
                }
            else:
                row["score"] = max(row["score"], score)
                row["rule_ids"].update(rule_ids)
                if row["group_id"] is None:
                    row["group_id"] = group_id
                if passed_minimum_score is not None:
                    row["passed_minimum_score"] = (
                        bool(row["passed_minimum_score"]) or passed_minimum_score
                    )

        def visit(value: Any, path: str, group_id: str | None = None) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}" if path else str(key), group_id)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    container = path[:-2] if path.endswith("[]") else path
                    component = container.rsplit(".", 1)[-1]
                    child_group = (
                        f"{group_id}/{component}:{index}"
                        if group_id
                        else f"{component}:{index}"
                    )
                    visit(child, f"{container}[]", child_group)
            elif isinstance(value, str) and value.strip():
                occurrences = _exact_source_occurrences(text, value)
                if document_type == "pelnomocnictwo" and path.endswith(".imie_nazwisko"):
                    if not occurrences:
                        span = _candidate_span(text, value)
                        if span:
                            occurrences = [(span.start(), span.end())]
                    occurrences = occurrences[:1]
                for start, end in occurrences:
                    add_candidate(path, value, start, end, group_id=group_id)

        visit(data, "")

        # Preserve the POA role-merge alternatives, including candidates that
        # the JSON extractor later filtered out. The source names and weights
        # come directly from _merge_name_candidates.
        for row in candidate_sink:
            start, end = row.get("start"), row.get("end")
            if not (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(text)
            ):
                matches = _exact_source_occurrences(text, row["value"])
                if not matches:
                    continue
                start, end = matches[0]
            if start is not None and end is not None:
                add_candidate(
                    "pelnomocnicy[].imie_nazwisko",
                    row["value"],
                    start,
                    end,
                    score=float(row["score"]),
                    rule_ids=tuple(row["rule_ids"]),
                    passed_minimum_score=bool(row["passed_minimum_score"]),
                )

        candidates = []
        for row in records.values():
            candidates.append(RegexCandidate(
                field=row["field"],
                value=row["value"],
                start=row["start"],
                end=row["end"],
                score=float(row["score"]),
                rule_ids=tuple(sorted(row["rule_ids"])),
                group_id=row["group_id"],
                passed_minimum_score=row["passed_minimum_score"],
            ))
        return RegexExtraction(data=data, candidates=tuple(candidates))
