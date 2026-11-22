# ModernBERT token-classification experiment

The selected model setup on the `roberta` branch is OCR-aware ModernBERT. It
fine-tunes one `PKOBP/embed-modernbert-68m` token classifier per document type,
with role-aware BIO labels, using local OCR and line-order variants in training.

The latest OCR-aware fit is on W1 at
`/home/michal/odwolanie2json-roberta-run/final-ocr-order-20261122/bundle`.
It uses a 512-token window, stride 96, learning rate `7e-5`, batch 16, BF16
training and FP32 checkpoints. Per-type epochs and entity weights were selected
by grouped CV. The held-out test split was excluded from CV, final fitting and
evaluation. See [`results/ocr-order-ablation.json`](results/ocr-order-ablation.json)
for aggregate-only comparisons with a matched no-OCR-training control and
[`results/final-ocr-order-training.json`](results/final-ocr-order-training.json)
for the final fit summary. The previous `selected-group-cv-final-7e5` bundle is
the historical no-OCR baseline.

## Data and privacy

`prepare_data.py` reads the gold JSON and PDF files locally. Ordinary PDFs use
`pdftotext`; the 30 scans use local Tesseract OCR. An optional text cache lives
at `/tmp/odwolanie2json-scan-ocr`; if it is absent, OCR runs locally again.
The temporary cache was deleted after training. The generated JSONL contains document text,
uses mode `0600`, and is written under a mode `0700` directory. It should stay
outside Git. The current checkout links `data/upstream/dane` read-only to the
existing local challenge data.

The split is by whole case/scan group: 70% train, 10% validation and 20% test.
All three document types from one group stay in the same split. Four extra PDFs
have no paired gold JSON and are excluded.

Two additional user-supplied transfer confirmations were extracted with local
`pdftotext` and appended to the private training JSONL. Each contributes only
the four confidently aligned spans: operation date, amount, source account and
destination account. They are train-only and are not stored in Git.

To add such local PDFs to a freshly prepared base, use
`add_transfer_pdfs.py`. It extracts text with `pdftotext` and appends only those
four validated span types; it never sends the PDFs to a remote service.

The preparer only aligns text-extractive labels. Categories such as the action
reason code, normalized action description and party type are not NER spans.
Overlapping gold spans are omitted because BIO cannot encode overlaps. It also
reports gold values that could not be aligned to source text.

## Training on local OCR output

`make_ocr_variants.py` creates one private OCR view for each train/validation
PDF by default, using Tesseract PSM 3. It renders pages with Poppler `pdftoppm` at 300 DPI, applies
mild local scan-like skew/resampling/blur/JPEG perturbations, and reads each
view with Tesseract 5.5.0 using `pol+eng`. PSM 6 and 11 remain available for
controlled experiments, but multi-PSM training did not beat the PSM 3 baseline.
At most two PDF/OCR jobs run at once
and Tesseract's OpenMP thread limit is one. PDFs are rendered and OCRed locally;
temporary page images are removed after each document. OCR text and labels stay
in private JSONL files until training. No hosted OCR, Gemini, or other inference
API is used.

OCR variants inherit the parent document's group and split. The held-out test
groups are not extracted or rendered. A variant is discarded if an aligned
source entity is missing from OCR or if the OCR alignment is incomplete; a
small set of plausible digit confusions is allowed for long numeric
identifiers. `perturb_ocr_order.py` then creates up to three views from each OCR
text: nearby line swaps, adjacent paragraph swaps and conservative interleaving
of column-like blocks. Span labels are projected to their new offsets, and a
variant is discarded if it changes any entity surface or alignment.
The fold builder keeps parent
groups together and preserves the original seeded fold assignment. Only
train/validation fold JSONL files are transferred to W1 over SSH for GPU
training; no PDFs or test rows are copied.

Private local workflow (all files under `/tmp/odwolanie2json-roberta-ocr`,
mode `0600`, parent directory mode `0700`):

```bash
python3 experiments/roberta/prepare_data.py \
  --ocr-workers 2 \
  --ocr-cache /tmp/odwolanie2json-roberta-ocr/no-cache \
  --output /tmp/odwolanie2json-roberta-ocr/base.jsonl

python3 experiments/roberta/add_transfer_pdfs.py \
  --input /tmp/odwolanie2json-roberta-ocr/base.jsonl \
  --output /tmp/odwolanie2json-roberta-ocr/base-with-transfers.jsonl \
  --pdf /private/path/transfer-1.pdf \
  --pdf /private/path/transfer-2.pdf

python3 experiments/roberta/make_ocr_variants.py \
  --input /tmp/odwolanie2json-roberta-ocr/base-with-transfers.jsonl \
  --output /tmp/odwolanie2json-roberta-ocr/base-with-ocr.jsonl \
  --tessdata-dir /path/to/tesseract-5.5.0/tessdata \
  --workers 2 --variants 1 --psm-modes 3 --seed 90210

python3 experiments/roberta/perturb_ocr_order.py \
  --input /tmp/odwolanie2json-roberta-ocr/base-with-ocr.jsonl \
  --output /tmp/odwolanie2json-roberta-ocr/base-with-ocr-order.jsonl \
  --seed 24680

python3 experiments/roberta/make_group_cv.py \
  --input /tmp/odwolanie2json-roberta-ocr/base-with-ocr-order.jsonl \
  --output-dir /tmp/odwolanie2json-roberta-ocr/cv \
  --folds 3 --per-type 782 --context-variant legacy --seed 2027
```

For an OCR ablation, use the same input and add
`--exclude-training-source-kinds ocr-perturbed ocr-order-shuffled`. This keeps
OCR views in validation while removing them from each fold's training split.

The OCR-only results are added by source kind to the aggregate CV report so
they can be compared with the original selectable-text and scan records.

`select_ocr_training_views.py` defaults to PSM 3 and keeps at most one
line-order variant per OCR parent. The transfer-only candidate described below
uses `merge_ocr_order_variants.py` to replace order variants for `przelew`
while retaining the existing appeal and power-of-attorney variants.

This is the default training path: build folds from `base-with-ocr-order.jsonl`
without `--exclude-training-source-kinds`. The CV runner uses the selected
entity weights 4/1/4 for appeals/powers/transfers, batch 16, BF16 validation,
batch 32 evaluation, up to 24 epochs and patience 4. Set `ROBERTA_WEIGHTS="1 4"`
to rerun the full weight comparison. The selected full fit uses epochs 14/4/8.
On W1, a synthetic-token throughput probe
measured about 174k tokens/s and 3.7 GiB peak allocation at batch 16; batch 32
was slower at about 159k tokens/s. Training uses fused AdamW, pinned input
batches and one loss synchronization per epoch. CV reports include real-window
throughput and peak GPU memory.

## Synthetic augmentation

`augment_data.py` adds synthetic templates and context-preserving identifier
replacements only from training records. It is the historical non-OCR path;
the selected OCR-aware fit uses `make_final_training_data.py` on
`base-with-ocr-order.jsonl` and excludes test rows. That earlier private base
had 212 records (the 210 paired challenge documents plus two additional
transfer confirmations). Its large augmentation contained 2,460 synthetic
records: 782 templates per type (2,346 total) and 114 context examples. This adds 2,000
synthetic examples over the previous 460-record augmentation. Every original
JSONL line, including validation and test, is copied byte-for-byte before the
synthetic train-only records are appended. Generated identifier values are
unique against training gold identifiers and one another. The JSONL stays mode
`0600` in a mode `0700` directory.

To reproduce the 2,460-record augmentation from the private base JSONL:

```bash
python3 experiments/roberta/augment_data.py \
  --input /tmp/odwolanie2json-roberta/dataset-base-with-user-transfers.jsonl \
  --output /tmp/odwolanie2json-roberta/dataset-augmented-2460.jsonl \
  --per-type 782 --context-repeats 3 --seed 2027
```

Prepare that private base locally with:

```bash
python3 experiments/roberta/prepare_data.py \
  --output /tmp/odwolanie2json-roberta/dataset-base.jsonl
python3 experiments/roberta/add_transfer_pdfs.py \
  --input /tmp/odwolanie2json-roberta/dataset-base.jsonl \
  --output /tmp/odwolanie2json-roberta/dataset-base-with-user-transfers.jsonl \
  --pdf /private/path/transfer-1.pdf \
  --pdf /private/path/transfer-2.pdf
```

The previous 460-record augmentation can be reproduced with `--per-type 128
--context-repeats 2 --seed 1337` on the same base. The current data-generation
script keeps synthetic outputs out of the repository.

Generated PESEL values encode future birth dates (2027–2039) and pass the
project checksum/date validator; future dates reduce the chance of representing
an existing person. NIP, REGON and NRB values are generated with
the project's checksum algorithms, checked against those validators, and
unique against the training identifiers. They are syntactically valid test
values; the script does not query government registers to check assignment.
KRS has no checksum, so synthetic KRS values only follow its digit format.

The construction rules were checked against [Gov.pl's PESEL explanation](https://www.gov.pl/web/gov/czym-jest-numer-pesel),
[GUS's REGON definition](https://stat.gov.pl/metainformacje/slownik-pojec/pojecia-stosowane-w-statystyce-publicznej/2963%2Cpojecie.html),
and the [legal NRB structure](https://eli.gov.pl/api/acts/DU/2022/1181/text.html).
The official [KSeF guide](https://ksef.podatki.gov.pl/media/i1ubiukl/aplikacja-podatnika-ksef-20-podrecznik-uzytkownika_25042026.pdf)
specifies the 10-digit NIP format.

For the original 210 paired documents, alignment matched 1,899 of 2,046 mapped
extractive gold values before overlap removal and dropped 370 conflicting or
overlapping spans. Only 22 of 70 source-account values and 22 of 70
destination-account values aligned with transfer text; the two extra user
confirmations add two manually aligned spans to each account class. Only 18 of
54 TED references aligned.

## OCR-aware CV and final fit

The train/validation set contains 170 original records, including 24 scan
records. Local Tesseract 5.5.0 accepted 108 OCR-rendered variants; 106 of those
also received a line-order shuffle that can place labelled lines before or
after their usual context. Failed span projections were discarded. These
views remain in the parent case's fold. PDFs and OCR were processed locally;
no inference API was used. For GPU fitting, only the private train/validation
JSONL (including its OCR text) was copied over SSH to the authorized W1
training host. No PDFs or held-out test rows were copied there.

We ran matched 3-fold group CV with and without OCR views in training. Both
variants used identical validation data, including OCR and line-order variants.
The control removed those two source kinds from training only. The selected
weights were 4/1/4 for appeals/powers/transfers. Pooled cross-fitted exact-span
micro-F1 rose from `0.7101` to `0.7513` (`+0.0411`) with OCR training. By source,
F1 was `0.7738` vs `0.6966` on OCR text, `0.7323` vs `0.6596` on line-order
variants, `0.5366` vs `0.4876` on original scans, and `0.7871` vs `0.7866` on
selectable-text PDFs. On original non-OCR records as a whole, it was `0.7495`
vs `0.7417`. These are grouped cross-validation results, not an untouched test
score; synthetic variants of the same source stay in the same fold.

| Type | OCR-trained CV F1 | No-OCR control | OCR text | Shuffled OCR lines | Original non-OCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Odwołanie | 0.7486 | 0.7383 | 0.7670 | 0.7252 | 0.7505 |
| Pełnomocnictwo | 0.7074 | 0.6981 | 0.7252 | 0.6475 | 0.7299 |
| Przelew | 0.8051 | 0.6542 | 0.8231 | 0.8092 | 0.7793 |

The line-order augmentation helped appeals and transfers, but slightly lowered
the power-of-attorney line-shuffle score (`0.6475` vs `0.6569`). That case
needs more varied, realistic OCR layout examples. It did not reduce the
original non-OCR score for any document type in this run.

The full fit contains 2,862 records: the 170 original train/validation records,
108 OCR views, 106 line-order variants, 2,346 generated templates and 132
context replacements. It ran for 14/4/8 epochs for appeals/powers/transfers,
with entity weights 4/1/4 and CV thresholds `0.75/0.0/1.25`. The total GPU
training time was about 69 seconds; the largest per-fit allocation was 3.81
GiB on the RTX 4090. Checkpoints are FP32 and total about 794 MiB. The bundle's
`inference_config.json` contains selected thresholds and window settings; an
aggregate copy is in [`results/ocr-order-inference-config.json`](results/ocr-order-inference-config.json).

Full aggregate reports are in [`results/ocr-order-cv-24ep.json`](results/ocr-order-cv-24ep.json),
[`results/ocr-order-control-cv-24ep.json`](results/ocr-order-control-cv-24ep.json),
and [`results/ocr-order-ablation.json`](results/ocr-order-ablation.json).
These contain no document text or identifier values.

## Transfer-only order candidate

We fixed the fold builder so synthetic context generation for one document
type cannot change when another type's OCR variants change. With that matched
setup, three 3-fold group-CV runs (seeds 2027, 2028 and 2029) tested PSM 3 and
replaced only transfer order variants. Mean cross-fitted transfer F1 was
`0.8133` (`0.8113`, `0.8103`, `0.8185`) versus `0.8051` for the historical
selected profile. In the directly paired seed-2027 comparison, it was `0.8113`
versus `0.7862` for the baseline. Test groups were excluded from every run.

The gain is not uniform by source. Across the three candidate runs, OCR text,
shuffled OCR and scans averaged F1 `0.8351`, `0.8301` and `0.6469`, compared
with `0.8231`, `0.8092` and `0.6286` in the historical baseline. Selectable-text
PDF and original non-OCR means were lower (`0.8092` vs `0.8260`, and `0.7690`
vs `0.7793`). This is a promising candidate, not an untouched-test result.
The paired and multiseed metrics are in
[`results/transfer-order-multiseed.json`](results/transfer-order-multiseed.json).

A separate transfer checkpoint was then fit for 7 epochs on the complete
train+validation corpus, with the candidate order rows and no test records. On
the RTX 4090 the fit took about 18 seconds, used 3.80 GiB peak allocated VRAM,
and ran at about 68.7k tokens/s. The candidate bundle is kept separately on W1
at `/home/michal/odwolanie2json-roberta-run/final-transfer-order-20261122/candidate/bundle`;
the currently selected bundle remains intact. Its inference threshold is the
median candidate CV threshold (`0.25`).
The candidate's inference config and training summary are recorded in
[`results/transfer-order-candidate-inference-config.json`](results/transfer-order-candidate-inference-config.json)
and [`results/transfer-order-candidate-training-summary.json`](results/transfer-order-candidate-training-summary.json).

## Earlier grouped-CV selection without OCR variants

This historical selection used exact-span micro-F1 with thresholds cross-fitted between the
other two folds. The 42 test records are absent from every fold. Each fold
keeps all document types from a case together. The final full fit uses the 170
original train+validation records and 2,478 synthetic records (2,346 templates
plus 132 legacy context replacements); the 42 test records were excluded.

| Learning rate | Odwołanie | Pełnomocnictwo | Przelew |
| ---: | ---: | ---: | ---: |
| `3e-5` | 0.691 | 0.661 | 0.730 |
| `5e-5` | 0.740 | 0.732 | 0.763 |
| `7e-5` | **0.765** | **0.797** | **0.775** |

The previous baseline fit uses `7e-5`, batch size 8, 512-token windows with stride
96, and entity loss weights 4/4/1 for appeals/powers/transfers. Median epochs
from the folds are 8/6/6. Inference thresholds are 0.0/−1.0/−0.5 respectively.
The final checkpoints are FP32 and total about 794 MiB (832 MB). The file
`inference_config.json` beside them contains the CV thresholds and full-fit
configuration. `training_summary.json` records training counts and loss only;
no document text is included.
Direct `train_eval.py` runs default to learning rate `7e-5`, batch size 16,
up to 24 validation-selected epochs with patience 4, FP32 saving, and entity
weights 4/1/4 by document type. CV scripts save their temporary checkpoints as
BF16.

Context augmentation that added NIP/REGON/KRS examples did not help: appeal
cross-fitted F1 changed from 0.7398 to 0.7404, while power-of-attorney F1
changed from 0.7317 to 0.7273. The selected final data therefore keeps the
legacy contexts. In a historical comparison, 512-token windows also beat
1024-token windows for appeals and powers of attorney; transfer F1 was tied.

The comparison JSON includes fold supports, thresholds, learning-rate and
context ablations, the historical window comparison, and final training
metadata. All values are aggregate-only; it contains no extracted text or
identifier values.

## Historical exploratory test rechecks

These exact-span micro-F1 values are from earlier experiments on the same
14-document-per-type test split. The test split was reused during experiment
selection and rechecks, so the scores are exploratory, not an untouched final
benchmark and do not evaluate the current `7e-5` bundle.

| Synthetic train samples | Odwołanie | Pełnomocnictwo | Przelew | Pooled micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 460, 8 epochs | 0.787 | 0.655 | 0.530 | 0.713 |
| 2,460, initial 3e-5 run | 0.895 | 0.717 | 0.661 | 0.815 |
| 2,460, learning-rate and seed sweep | **0.914** | **0.764** | **0.685** | **0.842** |

The historical bundle selected per type by validation F1 across eight candidates:
learning rates 2e-5, 3e-5 and 5e-5; entity weights 1 and 4; seed 42, plus a
second seed 123 at 5e-5. The selected runs are seed 123, weight 4 for appeals;
seed 123, weight 1 for powers of attorney; and seed 42, weight 4 for transfers.
Validation F1 is 0.818, 0.854 and 0.931 respectively. The selection report
contains no test metrics; newly tuned candidates used `--skip-test-eval`, and
the saved BF16 bundle was re-evaluated after selection.

Its pooled test score is 0.842 over 364 gold spans (331 exact matches, 422
predictions). Against the 460-sample run on the same base and split, every
document type improved. The test split has only 14 documents per type; PESEL
has one aligned test span, while each transfer account has four and the
transfer-title label has two. Final transfer F1 is 0.50 for source accounts,
0.545 for destination accounts and 0.00 for titles. These small supports make
per-label scores noisy. The test split was observed in earlier experiment
rounds, so none of these scores is an untouched benchmark or a production
estimate.

On scans, the selected bundle has test F1 0.667 for appeals, 0.737 for powers
of attorney and 0.414 for transfers. There are only two scanned test documents
per type, so these scores are especially noisy.

Reloaded BF16 checkpoints reproduced the reported historical test scores.
That old selected bundle is stored privately on W1 at
`/home/michal/odwolanie2json-roberta-run/selected-multiseed-validation`; the
three models total about 402 MB. FP8 was not evaluated. Aggregate tuning and
recheck metrics are in
`experiments/roberta/results/learning-rate-multiseed-sweep.json`.

## W1 training

The experiment was run on W1 with the existing Conda Python as the base for
the `roberta-segmentation` venv. It reuses the installed CUDA/PyTorch stack and
does not alter the project install requirements. The venv was created with
`--system-site-packages` so it can see the existing Conda CUDA packages.

```bash
/home/michal/.conda/envs/cdlm_wm_bench/bin/python -m venv \
  --system-site-packages /home/michal/roberta-segmentation
```

```bash
python3 experiments/roberta/prepare_data.py \
  --output /tmp/odwolanie2json-roberta/dataset-base.jsonl
python3 experiments/roberta/add_transfer_pdfs.py \
  --input /tmp/odwolanie2json-roberta/dataset-base.jsonl \
  --output /tmp/odwolanie2json-roberta/dataset-base-with-user-transfers.jsonl \
  --pdf /private/path/transfer-1.pdf \
  --pdf /private/path/transfer-2.pdf
python3 experiments/roberta/augment_data.py \
  --input /tmp/odwolanie2json-roberta/dataset-base-with-user-transfers.jsonl \
  --output /tmp/odwolanie2json-roberta/dataset-augmented-2460.jsonl \
  --per-type 782 --context-repeats 3 --seed 2027

# Copy the private JSONL and the four training scripts to W1 over SSH.
bash /tmp/odwolanie2json-roberta-sweep/run_sweep.sh \
  /tmp/odwolanie2json-roberta-sweep/dataset-augmented-2460.jsonl \
  /home/michal/odwolanie2json-roberta-run/run-v11-full-sweep
```

In the historical W1 sweep, each epoch was selected using validation
exact-span micro-F1. The eight-run sweep covered three learning rates, two
entity weights and two seeds for the best learning rate. It took about 529
seconds of GPU training on the RTX 4090.
The six new learning-rate/seed runs skipped test evaluation; the earlier
3e-5 candidates already had exploratory test metrics, which the validation-only
selector ignored. A clean `run_sweep.sh` execution skips test for every
candidate and evaluates the saved BF16 bundle once at the end. Raw training
JSONL was removed from W1 after re-evaluation; checkpoints and aggregate
metrics remain in private run directories.

PESEL has extremely limited aligned supervision, bank-account gold strings do
not align in most transfer PDFs, and several appeal references are missing
from extracted text. Those classes should retain deterministic fallbacks.

## Local regex–ModernBERT integration check

The final three-type bundle was copied to a local cache and run on CPU against
the nine labeled reference samples. Regex and ranked hybrid both scored
0.9744 exact normalized field-value micro-F1 (95 TP, 2 FP, 3 FN); the model
reassigned no candidates on this small set. No OCR was needed for these nine
PDFs. This is a smoke evaluation on reference samples, not a substitute for
grouped cross-validation. Aggregate details and runtime are in
`results/local-hybrid-sample-evaluation-20261122.json`.
