# Uruchomienie rozwiązania na VM

Wszystkie wejściowe PDF-y i pliki tekstowe umieść w `/opt/solution/input`. Wyniki JSON pojawią się w `/opt/solution/output`.

Obraz Dockera i lokalny bundle ModernBERT są przygotowane wcześniej. Z katalogu `/opt/solution` uruchom:

```bash
docker run --rm \
  --cpus=2 --memory=12g \
  -v /opt/solution/input:/input:ro \
  -v /opt/solution/output:/output \
  -v /opt/solution/model_bundle:/model_bundle:ro \
  hacknation2026-odwolanie2json:latest \
  --input /input \
  --output /output \
  --roberta-bundle /model_bundle \
  --roberta-config /opt/app/transfer-order-candidate-inference-config.json \
  --roberta-device cpu
```

OCR (Poppler + Tesseract z językami polskim i angielskim), regexy i ModernBERT działają lokalnie. Uruchomienie nie używa Gemini, zewnętrznego API ani nie pobiera modeli. Rozwiązanie oczekuje jednego JSON-a na dokument w `/opt/solution/output` oraz raportów `_report.json` i `_validation.json`.
