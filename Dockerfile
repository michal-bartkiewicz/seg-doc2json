FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app/src \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    'torch>=2.5,<3' \
    'transformers>=4.50,<5'

WORKDIR /opt/app
COPY src/ /opt/app/src/
COPY experiments/roberta/results/transfer-order-candidate-inference-config.json \
    /opt/app/transfer-order-candidate-inference-config.json

ENTRYPOINT ["python", "-m", "odwolanie2json.main"]
