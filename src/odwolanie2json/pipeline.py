from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classify import ClassificationError, classify_document_type
from .extract import extract_text
from .hybrid import HybridExtractor, HybridResult
from .validation import validate_result


DOCUMENT_WORKERS = 2


@dataclass
class SourceDocument:
    path: Path
    relative_path: str
    document_type: str | None
    text: str
    ocr_pages: list[int]
    type_assignment: str
    classification_score: int | None
    classification_signals: tuple[str, ...]
    classification_error: str | None = None

    @property
    def uses_ocr(self) -> bool:
        return bool(self.ocr_pages)


def _load_documents(input_dir: Path, known_types: set[str]) -> list[SourceDocument]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    allowed = {".pdf", ".txt", ".text"}
    files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )

    def load_document(path: Path) -> SourceDocument:
        relative = path.relative_to(input_dir)
        text, ocr_pages = extract_text(path)
        folder_type = relative.parts[0] if len(relative.parts) > 1 else ""
        classification_error = None
        if folder_type in known_types:
            document_type = folder_type
            assignment = "directory"
            score = None
            signals: tuple[str, ...] = ()
        else:
            try:
                classification = classify_document_type(text)
            except ClassificationError as exc:
                document_type = None
                assignment = "unclassified"
                score = None
                signals = ()
                classification_error = str(exc)
            else:
                document_type = classification.document_type
                assignment = "regex"
                score = classification.score
                signals = classification.signals
        return SourceDocument(
            path,
            relative.as_posix(),
            document_type,
            text,
            ocr_pages,
            assignment,
            score,
            signals,
            classification_error,
        )

    if not files:
        return []
    with ThreadPoolExecutor(max_workers=min(DOCUMENT_WORKERS, len(files))) as pool:
        return list(pool.map(load_document, files))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _make_output_item(
    document: SourceDocument,
    result: HybridResult,
) -> dict[str, Any]:
    return {
        "source_file": document.relative_path,
        "document_type": document.document_type,
        "type_assignment": document.type_assignment,
        "classification_score": document.classification_score,
        "classification_signals": list(document.classification_signals),
        "classification_error": document.classification_error,
        "extraction_route": (
            None if document.document_type is None else "ocr" if document.uses_ocr else "text"
        ),
        "candidate_decisions": result.metrics,
        "source_validation": (
            validate_result(result.data, document.text)
            if document.document_type is not None
            else None
        ),
        "ocr_pages": document.ocr_pages,
    }


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    extractor: HybridExtractor,
) -> dict[str, Any]:
    """Run local OCR, regex extraction, and optional local ModernBERT adjudication."""
    started = time.perf_counter()
    known_types = set(extractor.document_types)
    documents = _load_documents(input_dir, known_types)
    if not documents:
        raise ValueError(f"No PDF or text documents found under {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        (output_dir / Path(document.relative_path).with_suffix(".json")).unlink(
            missing_ok=True
        )
    for filename in ("_report.json", "_validation.json", "_state.json"):
        (output_dir / filename).unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    unclassified_count = 0
    for document in documents:
        if document.document_type is None:
            hybrid_result = HybridResult(data={}, metrics={})
            unclassified_count += 1
            status = "Unclassified"
        else:
            hybrid_result = extractor.extract(
                document.document_type, document.text, ocr=document.uses_ocr
            )
            status = f"Extracted {document.document_type}"
        output_path = output_dir / Path(document.relative_path).with_suffix(".json")
        _atomic_write_json(output_path, hybrid_result.data)
        item = _make_output_item(document, hybrid_result)
        results.append(item)
        for key, value in hybrid_result.metrics.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"{status}: {document.relative_path}"
            + (f" (local OCR pages: {len(document.ocr_pages)})" if document.ocr_pages else ""),
            flush=True,
        )

    elapsed = round(time.perf_counter() - started, 3)
    report = {
        "engine": extractor.name,
        "model": extractor.model_name,
        "model_enabled": extractor.model_enabled,
        "roberta_policy": extractor.policy,
        "document_count": len(documents),
        "result_count": len(results),
        "unclassified_count": unclassified_count,
        "complete": True,
        "processing_seconds": elapsed,
        "ocr_page_count": sum(len(document.ocr_pages) for document in documents),
        "candidate_decisions": totals,
        "results": sorted(results, key=lambda item: item["source_file"]),
    }
    _atomic_write_json(
        output_dir / "_report.json",
        {key: value for key, value in report.items() if key != "results"},
    )
    _atomic_write_json(output_dir / "_validation.json", report["results"])
    return report
