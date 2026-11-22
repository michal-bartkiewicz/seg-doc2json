#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This setup script supports Debian/Ubuntu in WSL (apt-get required)." >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 13))'; then
  echo "Python 3.13 or newer is required (found: $(python3 --version 2>&1))." >&2
  exit 1
fi

packages=()
command -v pdftotext >/dev/null 2>&1 || packages+=(poppler-utils)
command -v pdftoppm >/dev/null 2>&1 || packages+=(poppler-utils)
command -v tesseract >/dev/null 2>&1 || packages+=(tesseract-ocr)
if command -v tesseract >/dev/null 2>&1; then
  languages="$(tesseract --list-langs 2>/dev/null || true)"
  grep -q '^pol$' <<<"$languages" || packages+=(tesseract-ocr-pol)
  grep -q '^eng$' <<<"$languages" || packages+=(tesseract-ocr-eng)
else
  packages+=(tesseract-ocr-pol tesseract-ocr-eng)
fi

if ! python3 -c 'import venv' >/dev/null 2>&1; then
  packages+=(python3-venv)
fi

if ((${#packages[@]})); then
  if [[ "$EUID" -eq 0 ]]; then
    apt-get update
    apt-get install -y "${packages[@]}"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${packages[@]}"
  else
    echo "Install these system packages, then rerun setup.sh: ${packages[*]}" >&2
    exit 1
  fi
fi

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
if [[ "${WITH_LOCAL_ROBERTA:-0}" == "1" ]]; then
  .venv/bin/python -m pip install -r requirements-roberta.txt
fi
mkdir -p data/documents outputs
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Place PDFs or UTF-8 text files in data/documents/<document-type>/."
