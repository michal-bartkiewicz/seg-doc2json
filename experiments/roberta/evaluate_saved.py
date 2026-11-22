#!/usr/bin/env python3
"""Re-evaluate the saved BF16 per-type checkpoints from disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train_eval import (
    encode_documents,
    make_label_map,
    predict,
    read_jsonl,
    score_spans,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path(__file__).with_name("results") / "ocr-order-inference-config.json",
        help="CV config with per-type entity logit thresholds; defaults to the selected OCR setup.",
    )
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=96)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this evaluation.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "odwolanie", use_fast=True)
    records = read_jsonl(args.dataset)
    inference_config = (
        json.loads(args.inference_config.read_text(encoding="utf-8"))
        if args.inference_config else {}
    )
    selected = inference_config.get("selected_by_type", {})
    results = []
    for document_type in ("odwolanie", "pelnomocnictwo", "przelew"):
        split_records = [record for record in records
                         if record["document_type"] == document_type
                         and record["split"] == args.split]
        if not split_records:
            continue
        type_config = selected.get(document_type, {})
        window_size = int(type_config.get("window_size", args.window_size))
        stride = int(type_config.get("stride", args.stride))
        _, label2id, id2label = make_label_map(
            [record for record in records if record["document_type"] == document_type], document_type
        )
        windows, _, gold_by_doc, _ = encode_documents(
            split_records, tokenizer, label2id,
            window_size=window_size,
            stride=stride,
        )
        model_path = args.model_dir / document_type
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        dtype = (
            torch.bfloat16 if args.dtype == "bf16" else
            torch.float32 if args.dtype == "fp32" else model.dtype
        )
        model.to(dtype=dtype)
        threshold = float(type_config.get("median_validation_threshold", 0.0))
        pred = predict(
            model, windows, tokenizer, id2label, device, dtype,
            entity_logit_threshold=threshold,
        )
        metrics = score_spans(split_records, pred, gold_by_doc)
        results.append({
            "document_type": document_type,
            "split": args.split,
            "documents": len(split_records),
            "windows": len(windows),
            "dtype": str(dtype).removeprefix("torch."),
            "entity_logit_threshold": threshold,
            "window_size": window_size,
            "stride": stride,
            "metrics": metrics,
        })
        print(
            f"{document_type}: {args.split} F1={metrics['overall']['f1']:.4f} "
            f"(P={metrics['overall']['precision']:.4f}, R={metrics['overall']['recall']:.4f})",
            flush=True,
        )
    report = {
        "device": torch.cuda.get_device_name(0),
        "split": args.split,
        "inference_config": str(args.inference_config) if args.inference_config else None,
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
