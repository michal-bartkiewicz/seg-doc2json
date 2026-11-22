#!/usr/bin/env python3
"""Tune one global BIO-entity logit threshold per type on validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train_eval import (
    aggregate_token_logits,
    decode_score_maps,
    encode_documents,
    make_label_map,
    read_jsonl,
    score_spans,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-min", type=float, default=-4.0)
    parser.add_argument("--threshold-max", type=float, default=4.0)
    parser.add_argument("--threshold-step", type=float, default=0.25)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--document-types", nargs="+",
        choices=("odwolanie", "pelnomocnictwo", "przelew"),
        default=("odwolanie", "pelnomocnictwo", "przelew"),
    )
    args = parser.parse_args()
    if args.window_size < 2 or not 0 <= args.stride < args.window_size:
        raise SystemExit("Require window-size >= 2 and 0 <= stride < window-size")
    if args.eval_batch_size < 1:
        raise SystemExit("eval-batch-size must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for validation calibration.")
    device = torch.device("cuda")
    records = read_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir / args.document_types[0], use_fast=True
    )
    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )
    results = []
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    for document_type in args.document_types:
        validation_records = [
            record for record in records
            if record["document_type"] == document_type and record["split"] == "validation"
        ]
        _, label2id, id2label = make_label_map(validation_records, document_type)
        windows, _, gold_by_doc, _ = encode_documents(
            validation_records, tokenizer, label2id,
            window_size=args.window_size,
            stride=args.stride,
        )
        model_path = args.model_dir / document_type
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        model.to(dtype=dtype)
        token_scores = aggregate_token_logits(
            model, windows, tokenizer, device, dtype,
            batch_size=args.eval_batch_size,
        )
        del model
        torch.cuda.empty_cache()

        curve = []
        for threshold in thresholds:
            predictions = decode_score_maps(token_scores, id2label, float(threshold))
            metrics = score_spans(validation_records, predictions, gold_by_doc)
            overall = metrics["overall"]
            curve.append({
                "threshold": float(threshold),
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "gold": overall["gold"],
                "predicted": overall["predicted"],
                "correct": overall["correct"],
                "by_label": metrics["by_label"],
                "by_source_kind": metrics["by_source_kind"],
                "by_ocr_psm_source_kind": metrics["by_ocr_psm_source_kind"],
            })

        baseline = min(curve, key=lambda row: abs(row["threshold"]))
        best = max(curve, key=lambda row: (row["f1"], -abs(row["threshold"])))
        results.append({
            "document_type": document_type,
            "validation_documents": len(validation_records),
            "baseline_threshold": baseline["threshold"],
            "baseline": baseline,
            "selected_threshold": best["threshold"],
            "selected": best,
            "curve": curve,
        })
        print(
            f"{document_type}: val F1 {baseline['f1']:.4f} at threshold 0; "
            f"best {best['f1']:.4f} at {best['threshold']:+.2f} "
            f"(P={best['precision']:.4f}, R={best['recall']:.4f}, "
            f"{best['correct']}/{best['gold']} spans)",
            flush=True,
        )

    report = {
        "selection_metric": "validation exact-span micro-F1, separately by document type",
        "validation_only": True,
        "model_dtype": args.dtype,
        "window_size": args.window_size,
        "stride": args.stride,
        "eval_batch_size": args.eval_batch_size,
        "calibration": "subtract a scalar from every BIO entity logit before argmax",
        "threshold_grid": {
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
