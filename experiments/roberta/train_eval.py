#!/usr/bin/env python3
"""Fine-tune ModernBERT per document type and evaluate exact BIO spans."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "PKOBP/embed-modernbert-68m"
MODEL_REVISION = "5c76fc72d527c991fcff9265945649a5b3c8e1d2"
IGNORE_INDEX = -100
WINDOW = 512
STRIDE = 96
LABEL_INVENTORY = {
    "odwolanie": (
        "APPEAL_DATE", "ACTION_DATE", "APPEAL_CITY", "APPELLANT_ORG",
        "APPELLANT_NIP", "APPELLANT_REGON", "APPELLANT_KRS", "COUNSEL_PERSON",
        "COUNSEL_ORG", "COUNSEL_EMAIL", "CONTRACTING_AUTHORITY_ORG", "AUTHORITY_NIP",
        "AUTHORITY_REGON", "AUTHORITY_KRS", "BZP_REFERENCE", "PROCUREMENT_REFERENCE",
        "TED_REFERENCE", "PROCUREMENT_TITLE",
    ),
    "pelnomocnictwo": (
        "PRINCIPAL_PERSON", "PRINCIPAL_KRS", "PRINCIPAL_ORG", "PRINCIPAL_NIP",
        "SIGNATORY_PERSON", "ATTORNEY_PERSON", "ATTORNEY_PESEL",
    ),
    "przelew": (
        "OPERATION_DATE", "TRANSFER_AMOUNT", "DESTINATION_ACCOUNT", "TRANSACTION_ID",
        "TRANSFER_TITLE", "SOURCE_ACCOUNT",
    ),
}
DEFAULT_ENTITY_WEIGHTS = {
    "odwolanie": 4.0,
    "pelnomocnictwo": 1.0,
    "przelew": 4.0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_label_map(records: list[dict[str, Any]], document_type: str):
    del records  # The class inventory comes from the schema, never the held-out labels.
    labels = list(LABEL_INVENTORY[document_type])
    tags = ["O"]
    for label in labels:
        tags.extend((f"B-{label}", f"I-{label}"))
    return labels, {tag: index for index, tag in enumerate(tags)}, {index: tag for index, tag in enumerate(tags)}


def encode_documents(
    records, tokenizer, label2id, mask_ignore_regions=False,
    window_size=WINDOW, stride=STRIDE,
):
    windows = []
    doc_windows: dict[str, list[int]] = defaultdict(list)
    gold_by_doc: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    coverage = Counter()
    for record in records:
        if not record["text"].strip():
            coverage["raw_entities"] += len(record["entities"])
            coverage["unrepresentable_entities"] += len(record["entities"])
            continue
        encoded = tokenizer(
            record["text"],
            truncation=True,
            max_length=window_size,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        record_windows = []
        for input_ids, attention_mask, offsets in zip(
            encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
        ):
            labels = [IGNORE_INDEX if start == end == 0 else label2id["O"] for start, end in offsets]
            if mask_ignore_regions:
                ignored_regions = record.get("ignore_regions", [])
                for token_index, (start, end) in enumerate(offsets):
                    if start == end:
                        continue
                    if any(
                        start < region["end"] and end > region["start"]
                        for region in ignored_regions
                    ):
                        labels[token_index] = IGNORE_INDEX
            record_windows.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "offsets": offsets,
                "doc_id": record["doc_id"],
            })

        coverage["raw_entities"] += len(record["entities"])
        for entity in record["entities"]:
            full_windows = []
            partial_tokens = []
            for window_index, window in enumerate(record_windows):
                offsets = window["offsets"]
                token_indices = [
                    index for index, (start, end) in enumerate(offsets)
                    if end > start and start < entity["end"] and end > entity["start"]
                ]
                if not token_indices:
                    continue
                first, last = token_indices[0], token_indices[-1]
                token_start, token_end = offsets[first][0], offsets[last][1]
                boundary_chars_are_punct = (
                    not any(char.isalnum() for char in record["text"][token_start:entity["start"]])
                    and not any(char.isalnum() for char in record["text"][entity["end"]:token_end])
                )
                complete = (
                    token_start <= entity["start"]
                    and token_end >= entity["end"]
                    and boundary_chars_are_punct
                )
                if complete:
                    full_windows.append((window_index, token_indices, token_start, token_end))
                else:
                    partial_tokens.append((window_index, token_indices))

            if not full_windows:
                coverage["unrepresentable_entities"] += 1
                for window_index, token_indices in partial_tokens:
                    for token_index in token_indices:
                        record_windows[window_index]["labels"][token_index] = IGNORE_INDEX
                continue

            coverage["represented_entities"] += 1
            aligned = (full_windows[0][2], full_windows[0][3], entity["label"])
            gold_by_doc[record["doc_id"]].add(aligned)
            complete_indices = {item[0] for item in full_windows}
            for window_index, token_indices in partial_tokens:
                if window_index not in complete_indices:
                    for token_index in token_indices:
                        record_windows[window_index]["labels"][token_index] = IGNORE_INDEX
            begin_id = label2id[f"B-{entity['label']}"]
            inside_id = label2id[f"I-{entity['label']}"]
            for window_index, token_indices, _, _ in full_windows:
                labels = record_windows[window_index]["labels"]
                labels[token_indices[0]] = begin_id
                for token_index in token_indices[1:]:
                    labels[token_index] = inside_id

        for window in record_windows:
            window_index = len(windows)
            windows.append(window)
            doc_windows[record["doc_id"]].append(window_index)
    return windows, doc_windows, gold_by_doc, coverage


def collate(batch, tokenizer, include_labels=True):
    model_inputs = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    if include_labels:
        width = model_inputs["input_ids"].shape[1]
        labels = [item["labels"] + [IGNORE_INDEX] * (width - len(item["labels"])) for item in batch]
        model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return model_inputs


def decode_bio(offsets, tag_ids, id2label):
    entities = set()
    active_label = None
    active_start = None
    active_end = None

    def close():
        nonlocal active_label, active_start, active_end
        if active_label is not None:
            entities.add((active_start, active_end, active_label))
        active_label = active_start = active_end = None

    for (start, end), tag_id in zip(offsets, tag_ids):
        if start == end == 0:
            close()
            continue
        tag = id2label[int(tag_id)]
        if tag == "O":
            close()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or active_label != label:
            close()
            active_label, active_start, active_end = label, start, end
        else:
            active_end = end
    close()
    return entities


def aggregate_token_logits(model, windows, tokenizer, device, dtype, batch_size=32):
    model.eval()
    sums: dict[str, dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
    counts: dict[str, Counter] = defaultdict(Counter)
    autocast_enabled = dtype in (torch.bfloat16, torch.float16)
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            inputs = collate(batch, tokenizer, include_labels=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
                logits = model(**inputs).logits
            cpu_logits = logits.float().cpu().numpy()
            for item, token_logits in zip(batch, cpu_logits):
                for (char_start, char_end), token_score in zip(item["offsets"], token_logits):
                    if char_start == char_end == 0:
                        continue
                    key = (char_start, char_end)
                    previous = sums[item["doc_id"]].get(key)
                    if previous is None:
                        sums[item["doc_id"]][key] = token_score.copy()
                    else:
                        previous += token_score
                    counts[item["doc_id"]][key] += 1
    return {
        doc_id: {
            offset: token_scores[offset] / counts[doc_id][offset]
            for offset in token_scores
        }
        for doc_id, token_scores in sums.items()
    }


def decode_score_maps(score_maps, id2label, entity_logit_threshold=0.0):
    result: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    for doc_id, token_scores in score_maps.items():
        ordered = sorted(token_scores)
        tag_ids = []
        for offset in ordered:
            scores = token_scores[offset].copy()
            scores[1:] -= entity_logit_threshold
            tag_ids.append(int(scores.argmax()))
        result[doc_id].update(decode_bio(ordered, tag_ids, id2label))
    return result


def predict(
    model, windows, tokenizer, id2label, device, dtype,
    entity_logit_threshold=0.0, batch_size=32,
):
    score_maps = aggregate_token_logits(
        model, windows, tokenizer, device, dtype, batch_size=batch_size
    )
    return decode_score_maps(score_maps, id2label, entity_logit_threshold)


def score_spans(records, predictions, gold_by_doc):
    gold_by_type: dict[str, set] = defaultdict(set)
    pred_by_type: dict[str, set] = defaultdict(set)
    doc_type = {}
    source_kind = {}
    ocr_variant_kind = {}
    for record in records:
        doc_id = record["doc_id"]
        kind = record["document_type"]
        doc_type[doc_id] = kind
        source_kind[doc_id] = record["source_kind"]
        psm = record.get("ocr_perturbation", {}).get("psm")
        if psm is not None:
            ocr_variant_kind[doc_id] = f"psm{psm}:{record['source_kind']}"
        gold = gold_by_doc.get(doc_id, set())
        for span in gold:
            gold_by_type[kind].add((doc_id, *span))
        for span in predictions.get(doc_id, set()):
            pred_by_type[kind].add((doc_id, *span))

    def metrics(gold, pred):
        tp = len(gold & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"gold": len(gold), "predicted": len(pred), "correct": tp,
                "precision": precision, "recall": recall, "f1": f1}

    output = {"overall": metrics(set().union(*gold_by_type.values()), set().union(*pred_by_type.values()))}
    output["by_document_type"] = {}
    output["by_source_kind"] = {}
    output["by_ocr_psm_source_kind"] = {}
    output["by_label"] = {}
    for kind in sorted(set(doc_type.values())):
        output["by_document_type"][kind] = metrics(gold_by_type[kind], pred_by_type[kind])
    for source in sorted(set(source_kind.values())):
        gold = {item for kind_set in gold_by_type.values() for item in kind_set if source_kind[item[0]] == source}
        pred = {item for kind_set in pred_by_type.values() for item in kind_set if source_kind[item[0]] == source}
        output["by_source_kind"][source] = metrics(gold, pred)
    for psm_source in sorted(set(ocr_variant_kind.values())):
        gold = {
            item for kind_set in gold_by_type.values() for item in kind_set
            if ocr_variant_kind.get(item[0]) == psm_source
        }
        pred = {
            item for kind_set in pred_by_type.values() for item in kind_set
            if ocr_variant_kind.get(item[0]) == psm_source
        }
        output["by_ocr_psm_source_kind"][psm_source] = metrics(gold, pred)
    all_labels = sorted({span[3] for values in gold_by_type.values() for span in values}
                        | {span[3] for values in pred_by_type.values() for span in values})
    for label in all_labels:
        gold = {item for kind_set in gold_by_type.values() for item in kind_set if item[3] == label}
        pred = {item for kind_set in pred_by_type.values() for item in kind_set if item[3] == label}
        output["by_label"][label] = metrics(gold, pred)
    return output


def train_one(document_type, records, tokenizer, args, output_root, device):
    started = time.perf_counter()
    all_type = [record for record in records if record["document_type"] == document_type]
    train_records = [record for record in all_type if record["split"] == "train"]
    validation_records = (
        [] if args.fixed_epochs else
        [record for record in all_type if record["split"] == "validation"]
    )
    test_records = [record for record in all_type if record["split"] == "test"]
    labels, label2id, id2label = make_label_map(all_type, document_type)
    if not labels:
        raise RuntimeError(f"No schema labels for {document_type}")
    train_windows, _, train_gold, train_coverage = encode_documents(
        train_records, tokenizer, label2id,
        mask_ignore_regions=args.mask_ambiguous_overlaps,
        window_size=args.window_size,
        stride=args.stride,
    )
    if args.fixed_epochs:
        validation_windows, validation_gold, validation_coverage = [], {}, Counter()
    else:
        validation_windows, _, validation_gold, validation_coverage = encode_documents(
            validation_records, tokenizer, label2id,
            window_size=args.window_size,
            stride=args.stride,
        )
    if args.skip_test_eval:
        test_windows, test_gold, test_coverage = [], {}, Counter()
    else:
        test_windows, _, test_gold, test_coverage = encode_documents(
            test_records, tokenizer, label2id,
            window_size=args.window_size,
            stride=args.stride,
        )
    if (not train_windows or (not args.fixed_epochs and not validation_windows)
            or (not args.skip_test_eval and not test_windows)):
        raise RuntimeError(
            f"Missing windows for {document_type}: train={len(train_windows)}, "
            f"validation={len(validation_windows)}, test={len(test_windows)}"
        )
    train_counts = Counter(label for spans in train_gold.values() for _, _, label in spans)
    validation_counts = Counter(label for spans in validation_gold.values() for _, _, label in spans)
    test_counts = Counter(label for spans in test_gold.values() for _, _, label in spans)

    config_maps = {
        "label2id": label2id,
        "id2label": {str(index): label for index, label in id2label.items()},
        "num_labels": len(id2label),
    }
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
        fused=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_windows,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer, include_labels=True),
        pin_memory=device.type == "cuda",
    )
    history = []
    best_state = None
    best_validation_f1 = None if args.fixed_epochs else -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_tokens = sum(sum(window["attention_mask"]) for window in train_windows)
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        for batch in train_loader:
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            token_labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
                class_weights = torch.ones(len(id2label), dtype=torch.float32, device=device)
                class_weights[1:] = args.entity_weight
                loss = F.cross_entropy(
                    logits.reshape(-1, len(id2label)),
                    token_labels.reshape(-1),
                    weight=class_weights,
                    ignore_index=IGNORE_INDEX,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            epoch_loss_sum += loss.detach().float()
        mean_train_loss = (epoch_loss_sum / len(train_loader)).item()
        history.append({"epoch": epoch + 1, "mean_train_loss": mean_train_loss})
        if args.fixed_epochs:
            if epoch + 1 == args.epochs:
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"{document_type}: epoch {epoch + 1}/{args.epochs} "
                f"loss={mean_train_loss:.4f}",
                flush=True,
            )
            continue
        validation_predictions = predict(
            model, validation_windows, tokenizer, id2label, device,
            torch.bfloat16 if args.validation_dtype == "bf16" else torch.float32,
            batch_size=args.eval_batch_size,
        )
        validation_metrics = score_spans(validation_records, validation_predictions, validation_gold)
        validation_f1 = validation_metrics["overall"]["f1"]
        history[-1]["validation_exact_span_f1"] = validation_f1
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        print(
            f"{document_type}: epoch {epoch + 1}/{args.epochs} "
            f"loss={mean_train_loss:.4f} val_f1={validation_f1:.4f}",
            flush=True,
        )
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"{document_type}: early stop after {epoch + 1} epochs; "
                f"best epoch={best_epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(f"No validation checkpoint selected for {document_type}")

    # Restore the selected FP32 state before each precision so quantization
    # comparisons start from identical trained weights.
    precision_results = {}
    if not args.skip_test_eval:
        for precision, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
            model.to(dtype=torch.float32)
            model.load_state_dict(best_state)
            model.to(dtype=dtype)
            predictions = predict(
                model, test_windows, tokenizer, id2label, device, dtype,
                batch_size=args.eval_batch_size,
            )
            precision_results[precision] = score_spans(test_records, predictions, test_gold)
            print(f"{document_type}: {precision} exact-span F1={precision_results[precision]['overall']['f1']:.4f}", flush=True)
    model.to(dtype=torch.float32)
    model.load_state_dict(best_state)

    model_dir = output_root / document_type
    model_dir.mkdir(parents=True, exist_ok=True)
    model.config.update(config_maps)
    save_dtype = torch.bfloat16 if args.save_dtype == "bf16" else torch.float32
    model.to(dtype=save_dtype)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (model_dir / "labels.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    training_elapsed = time.perf_counter() - training_started
    report = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "document_type": document_type,
        "split": {
            "train_documents": len(train_records),
            "validation_documents": len(validation_records),
            "test_documents": len(test_records),
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "test_windows": len(test_windows),
        },
        "labels": list(id2label.values()),
        "training_label_support": {label: train_counts[label] for label in labels},
        "validation_label_support": {label: validation_counts[label] for label in labels},
        "test_gold_support": {label: test_counts[label] for label in labels},
        "span_alignment_coverage": {
            "train": train_coverage,
            "validation": validation_coverage,
            "test": test_coverage,
        },
        "epochs": history,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_exact_span_f1": best_validation_f1,
        "fixed_epochs": args.fixed_epochs,
        "test_evaluation": "skipped" if args.skip_test_eval else "evaluated",
        "hyperparameters": {"learning_rate": args.learning_rate, "batch_size": args.batch_size,
                            "eval_batch_size": args.eval_batch_size,
                            "validation_dtype": args.validation_dtype,
                            "early_stopping_patience": args.early_stopping_patience,
                            "epoch_limit": args.epochs,
                            "entity_loss_weight": args.entity_weight, "seed": args.seed,
                            "mask_ambiguous_overlaps": args.mask_ambiguous_overlaps,
                            "window_size": args.window_size, "stride": args.stride},
        "inference_precision_metrics": precision_results,
        "saved_checkpoint_dtype": args.save_dtype,
        "elapsed_seconds": elapsed,
        "training_elapsed_seconds": training_elapsed,
        "training_tokens_per_second": training_tokens * len(history) / max(training_elapsed, 1e-9),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "model_dir": str(model_dir),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--early-stopping-patience", type=int, default=4,
                        help="Stop after this many validation epochs without improvement; 0 disables it.")
    parser.add_argument(
        "--fixed-epochs", type=int,
        help="Train exactly this many epochs without validation-based checkpoint selection.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--window-size", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--entity-weight", type=float, default=None,
                        help="Cross-entropy weight for BIO classes compared with O; "
                             "defaults by document type to 4/1/4.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dtype", choices=("bf16", "fp32"), default="fp32",
        help="Checkpoint storage dtype; FP32 avoids weight-rounding at the cost of a larger file.",
    )
    parser.add_argument(
        "--mask-ambiguous-overlaps", action="store_true",
        help="Ignore tokens for spans dropped because conflicting gold spans overlap.",
    )
    parser.add_argument(
        "--document-types", nargs="+",
        choices=("odwolanie", "pelnomocnictwo", "przelew"),
        default=("odwolanie", "pelnomocnictwo", "przelew"),
        help="Train only the requested document types.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Do not encode, score, or otherwise inspect test examples during hyperparameter search.",
    )
    args = parser.parse_args()
    if args.window_size < 2 or not 0 <= args.stride < args.window_size:
        raise SystemExit("Require window-size >= 2 and 0 <= stride < window-size")
    if args.early_stopping_patience < 0 or args.eval_batch_size < 1:
        raise SystemExit("Early-stopping patience must be >= 0 and eval batch size >= 1")
    if args.fixed_epochs is not None:
        if args.fixed_epochs < 1:
            raise SystemExit("--fixed-epochs must be positive")
        if not args.skip_test_eval:
            raise SystemExit("--fixed-epochs requires --skip-test-eval")
        args.epochs = args.fixed_epochs
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("This experiment expects the authorized CUDA host w1.")
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_fast=True
    )
    records = read_jsonl(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "runs": [],
    }
    for document_type in args.document_types:
        type_args = argparse.Namespace(**vars(args))
        if type_args.entity_weight is None:
            type_args.entity_weight = DEFAULT_ENTITY_WEIGHTS[document_type]
        run = train_one(
            document_type, records, tokenizer, type_args, args.output_dir, device
        )
        report["runs"].append(run)
        path = args.output_dir / "metrics.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {path}", flush=True)
    report["total_training_seconds"] = sum(run["elapsed_seconds"] for run in report["runs"])
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "total_training_seconds": report["total_training_seconds"],
        "best_validation_f1_by_type": {
            run["document_type"]: run["best_validation_exact_span_f1"]
            for run in report["runs"]
        },
    }
    if not args.skip_test_eval:
        summary["test_f1_by_type_precision"] = {
            run["document_type"]: {
                precision: scores["overall"]["f1"]
                for precision, scores in run["inference_precision_metrics"].items()
            }
            for run in report["runs"]
        }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
