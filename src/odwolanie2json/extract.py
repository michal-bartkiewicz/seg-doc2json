from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import os
from pathlib import Path


class ExtractionError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNTIME = PROJECT_ROOT / ".runtime" / "root"


def _tool_path(name: str) -> str | None:
    local_tool = LOCAL_RUNTIME / "usr" / "bin" / name
    if local_tool.is_file():
        return str(local_tool)
    return shutil.which(name)


def _runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    local_libs = LOCAL_RUNTIME / "usr" / "lib" / "x86_64-linux-gnu"
    if local_libs.is_dir():
        prior = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{local_libs}:{prior}" if prior else str(local_libs)
    local_tessdata = (
        LOCAL_RUNTIME / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"
    )
    if local_tessdata.is_dir():
        env["TESSDATA_PREFIX"] = str(local_tessdata)
    return env


def _run(args: list[str]) -> str:
    executable = _tool_path(args[0])
    if executable is None:
        raise ExtractionError(f"Missing local executable: {args[0]}")
    try:
        result = subprocess.run(
            [executable, *args[1:]],
            check=True,
            capture_output=True,
            text=True,
            env=_runtime_environment(),
        )
    except FileNotFoundError as exc:
        raise ExtractionError(f"Missing local executable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        raise ExtractionError(
            f"{args[0]} failed (exit {exc.returncode})"
            + (f": {detail[-1][:240]}" if detail else "")
        ) from exc
    return result.stdout


def _has_selectable_text(page: str) -> bool:
    meaningful = re.sub(r"[^\w]", "", page, flags=re.UNICODE)
    return len(meaningful) >= 35


def _ocr_page(pdf: Path, page_number: int) -> str:
    if not _tool_path("pdftoppm") or not _tool_path("tesseract"):
        raise ExtractionError(
            "A scanned PDF page needs local pdftoppm and tesseract; run ./setup.sh first."
        )
    with tempfile.TemporaryDirectory(prefix="odwolanie2json-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        _run(
            [
                "pdftoppm", "-f", str(page_number), "-l", str(page_number),
                "-singlefile", "-r", "300", "-png", str(pdf), str(prefix),
            ]
        )
        image = prefix.with_suffix(".png")
        return _run(["tesseract", str(image), "stdout", "-l", "pol+eng", "--psm", "3"])


def extract_text(path: Path) -> tuple[str, list[int]]:
    """Return local text and 1-based page numbers that required local OCR."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return path.read_text(encoding="utf-8", errors="replace"), []
    if suffix != ".pdf":
        raise ExtractionError(f"Unsupported input format: {suffix or '(none)'}")

    raw = _run(["pdftotext", "-layout", str(path), "-"])
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not pages:
        pages = [""]

    output: list[str] = []
    ocr_pages: list[int] = []
    for page_number, page in enumerate(pages, start=1):
        if _has_selectable_text(page):
            output.append(page.strip())
        else:
            text = _ocr_page(path, page_number).strip()
            output.append(text)
            ocr_pages.append(page_number)
    return "\n\n".join(part for part in output if part), ocr_pages
