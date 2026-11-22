from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from typing import Any


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _string_leaves(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _string_leaves(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _string_leaves(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.strip():
        yield path, value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, str(value)


def validate_result(result: Any, source_text: str) -> dict[str, Any]:
    """Baseline provenance check: regex-search every scalar string/number in source."""
    source = _normalize(source_text)
    checked: list[str] = []
    missing: list[str] = []
    for path, value in _string_leaves(result):
        candidate = _normalize(value)
        if len(candidate) < 2:
            missing.append(path)
            continue
        # Escaping the value turns it into a literal regex while whitespace can
        # vary between OCR output and the model's normalized JSON.
        pattern = r"\s+".join(re.escape(part) for part in candidate.split())
        if re.search(pattern, source, flags=re.IGNORECASE | re.UNICODE):
            checked.append(path)
        else:
            missing.append(path)
    return {
        "fields_checked": len(checked) + len(missing),
        "fields_found_in_source": len(checked),
        "fields_not_found": missing,
        "method": "case-insensitive literal regex on whitespace-normalized extracted text",
    }
