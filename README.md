# hacknation2026-odwolanie2json

Local extraction of Polish procurement documents into structured JSON. The pipeline uses document rules and an optional local ModernBERT token classifier. Scanned pages are recognized by Tesseract on the same machine.

## Privacy and repository size

- PDFs and OCR text stay local. Text PDFs use `pdftotext`; image pages use local `pdftoppm` and Polish/English Tesseract.
- Regex extraction, type classification, and optional ModernBERT inference run locally. The model loader uses `local_files_only=True` and does not fetch weights from Hugging Face.
- Source documents, labels, outputs, OCR images, and model checkpoints are excluded from Git. The repository stays well under 2 GB.

## Install in WSL (Ubuntu/Debian)

The single setup script installs Poppler, Tesseract, and a Python virtual environment:

```bash
./setup.sh
source .venv/bin/activate
```

The default install is regex-only and does not need a key or network model access. To install the optional CPU runtime for ModernBERT, use the same script:

```bash
WITH_LOCAL_ROBERTA=1 ./setup.sh
```

## Inputs

```text
data/
  documents/
    odwolanie/
    pelnomocnictwo/
    przelew/
```

Put PDFs or UTF-8 text files under `data/documents/<type>/`. The three supported types are `odwolanie`, `pelnomocnictwo`, and `przelew`. Unknown or top-level files are classified locally with weighted regex rules. No labeled samples or `expected.json` are needed for inference.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m odwolanie2json.main
```

By default the regex extractor writes the result JSON directly. To verify candidates with a local fine-tuned ModernBERT bundle, supply the directory containing `odwolanie/`, `pelnomocnictwo/`, and `przelew/` checkpoints:

```bash
PYTHONPATH=src .venv/bin/python -m odwolanie2json.main \
  --roberta-bundle /path/to/local/transfer-order-candidate/bundle
```

Keep the bundle in a local directory. CPU is the default and `--roberta-device cuda` selects CUDA. On PDFs with selectable text, `--roberta-policy rank` lets ModernBERT reassign a principal or attorney candidate when its exact span decodes as a complete BIO role sequence. The pipeline retains the regex value when evidence is uncertain.

For PDFs with pages processed by Tesseract, the extractor repairs common OCR artifacts before applying document rules. It also checks receipt fields, role boundaries in powers of attorney, and headings and first claims in appeals. With the bundle enabled, the model can supply a missing field from a source span with sufficient model evidence and valid field format. The model reads the original OCR text so its offsets stay aligned with the source.

For powers of attorney, the model uses distinct BIO labels for principal, signatory, and attorney. Candidate support is field-aware, so a principal span does not validate an attorney field. Fields that do not have a trained span label remain under the regex rules.

For principal and attorney names, regex candidates carry exact source offsets and the JSON list item that produced them. ModernBERT compares the candidate span against its BIO role scores and `O`, using the existing per-type token threshold. If a name repeats, role scoring uses its first source occurrence; later mentions cannot change that decision. The assignment is applied only to the JSON list item that proposed the name. Regex attorney alternatives that passed their source rule's minimum score can fill a role when the model gives one unambiguous assignment. A PESEL-linked attorney is not moved to principal. The same candidate evidence is exposed through `RegexExtractor.extract_with_candidates()` for future field-specific ranking.

## Outputs and evaluation

Each source receives one JSON file under `outputs/` with the same relative path. Inputs the classifier rejects receive `{}`. `_report.json` contains aggregate decision counts and timing. `_validation.json` records source-presence checks, document-type signals, OCR page numbers, and per-document decision counts. Its `extraction_route` is `ocr` when Tesseract processed at least one page, `text` for documents extracted from text, and `null` for rejected documents.

### Local field extraction benchmark

The scores compare exact normalized JSON path/value pairs. The scan benchmark reuses saved Tesseract text and includes classification. Six original scans from two cases were reserved from rule tuning and model fitting. The 30-scan aggregate also includes the 24 scans used during development.

| Dataset | Documents | Before | After |
| --- | ---: | ---: | ---: |
| Original scans | 30 | 0.7872 | 0.9740 |
| Reserved original scans | 6 | 0.8182 | 0.9552 |
| PDFs with selectable text, regex | 180 | 0.9818 | 1.0000 |

On the 30 scans, the new pipeline scored 300 true positives, 7 false positives, and 9 false negatives. The document rules supplied the values on this benchmark. ModernBERT proposed candidates but made no additional selections. Starting from the six reserved PDFs reproduced 0.9552 through the CLI with local OCR and the model enabled. The run processed 15 pages in 79.4 seconds on the development laptop. These figures use the public development documents and their internal split.

The last text layout changes raised the score from 0.9980 to 1.0000 on the same 180 development PDFs (2,032 true positives, no false positives or false negatives). The extractor reads bank descriptions in adjacent columns, takes amounts from their column beside the transaction date, and preserves grantor names split across lines.

The OCR-aware ModernBERT experiment reported pooled cross-fitted exact-span micro-F1 of 0.7513, versus 0.7101 for its no-OCR control. A later transfer-order candidate reported per-type cross-fitted scores of 0.7486 for appeals, 0.7074 for powers of attorney, and 0.8133 for transfers; those per-type figures come from separate profiles and are not a pooled score for the mixed bundle. None of these results measure regex-candidate verification or final JSON micro-F1. The inference config's per-type thresholds decode BIO spans; they are not calibrated candidate-acceptance thresholds. Tune candidate decisions on out-of-fold outputs using pooled TP/FP/FN and exact output-level micro-F1; keep all variants of a source case in the same fold and reserve the held-out test set.

The local candidate-evaluation utility is [`experiments/roberta/evaluate_candidate_predictions.py`](experiments/roberta/evaluate_candidate_predictions.py). It reports candidate recall@k, the candidate-coverage oracle, and final-field micro-F1 from JSONL. Its input contains extracted field values and must remain local; the report omits IDs and values. The utility does not generate predictions or claim a new F1 result by itself.

OCR remains local and is performed only for pages without adequate selectable text. No OCR or model checkpoint is included in this repository.
