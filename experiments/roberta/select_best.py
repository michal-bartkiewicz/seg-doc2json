#!/usr/bin/env python3
"""Select one model per document type using validation F1 and bundle it."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Run directories containing models/metrics.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Select on validation F1 only and omit test metrics/support from the selection report.",
    )
    args = parser.parse_args()

    candidates = {}
    for run_dir in args.runs:
        report_path = run_dir / "models" / "metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for result in report["runs"]:
            candidates.setdefault(result["document_type"], []).append((run_dir, result))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for document_type in ("odwolanie", "pelnomocnictwo", "przelew"):
        options = candidates.get(document_type, [])
        if not options:
            raise SystemExit(f"No candidate checkpoint for {document_type}")
        run_dir, result = max(options, key=lambda item: item[1]["best_validation_exact_span_f1"])
        source = run_dir / "models" / document_type
        destination = args.output_dir / "models" / document_type
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        selection = {
            "document_type": document_type,
            "selected_run": run_dir.name,
            "best_epoch": result["best_epoch"],
            "validation_exact_span_f1": result["best_validation_exact_span_f1"],
            "training_label_support": result["training_label_support"],
        }
        if not args.validation_only:
            selection["test_gold_support"] = result.get("test_gold_support", {})
        test_metrics = None if args.validation_only else result.get("inference_precision_metrics")
        if test_metrics:
            selection["test_metrics"] = test_metrics
        selected.append(selection)
        message = (
            f"{document_type}: {run_dir.name}, epoch={result['best_epoch']}, "
            f"validation F1={result['best_validation_exact_span_f1']:.4f}"
        )
        if test_metrics and "bf16" in test_metrics:
            message += f", test BF16 F1={test_metrics['bf16']['overall']['f1']:.4f}"
        print(message, flush=True)

    summary = {"selection_metric": "validation exact-span micro-F1", "selected": selected}
    (args.output_dir / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
