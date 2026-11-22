#!/usr/bin/env python3
"""Aggregate fold metrics without reading or emitting source document text."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DOCUMENT_TYPES = ("odwolanie", "pelnomocnictwo", "przelew")


def _micro(rows: list[dict]) -> dict:
    gold = sum(row["gold"] for row in rows)
    predicted = sum(row["predicted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gold": gold, "predicted": predicted, "correct": correct,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-types", nargs="+", choices=DOCUMENT_TYPES,
                        default=DOCUMENT_TYPES)
    parser.add_argument("--context-variant", choices=("legacy", "appeal-ids", "all-ids"))
    args = parser.parse_args()

    summaries = []
    candidates_by_type = defaultdict(list)
    folds = sorted(
        path for path in args.cv_root.glob("fold-*") if path.is_dir()
    )
    if not folds:
        raise SystemExit(f"No CV folds found under {args.cv_root}")
    fold_metadata_path = args.cv_root / "fold-summary.json"
    context_variant = args.context_variant or "unspecified"
    if fold_metadata_path.is_file():
        fold_metadata = json.loads(fold_metadata_path.read_text(encoding="utf-8"))
        if isinstance(fold_metadata, dict) and args.context_variant is None:
            context_variant = fold_metadata.get("context_variant", context_variant)
    for document_type in args.document_types:
        available_weights = sorted({
            int(run_dir.name.rsplit("-w", 1)[1])
            for run_dir in folds[0].glob(f"{document_type}-w*")
            if "-w" in run_dir.name
        })
        if not available_weights:
            raise SystemExit(f"No CV candidates found for {document_type}")
        for weight in available_weights:
            fold_reports, calibrations = [], []
            for fold_dir in folds:
                run_dir = fold_dir / f"{document_type}-w{weight}"
                train_report = json.loads(
                    (run_dir / "models" / "metrics.json").read_text(encoding="utf-8")
                )["runs"][0]
                calibration = json.loads(
                    (run_dir / "thresholds.json").read_text(encoding="utf-8")
                )["results"][0]
                fold_reports.append(train_report)
                calibrations.append(calibration)

            raw_rows = [calibration["baseline"] for calibration in calibrations]
            same_fold_tuned_rows = [calibration["selected"] for calibration in calibrations]
            thresholds = [calibration["selected_threshold"] for calibration in calibrations]
            cross_fitted_rows = []
            for index, calibration in enumerate(calibrations):
                external_threshold = statistics.median(
                    value for other_index, value in enumerate(thresholds)
                    if other_index != index
                )
                row = min(
                    calibration["curve"],
                    key=lambda value: abs(value["threshold"] - external_threshold),
                )
                cross_fitted_rows.append(row)

            source_kinds = sorted({
                source
                for calibration in calibrations
                for source in calibration["baseline"].get("by_source_kind", {})
            })
            by_source_kind = {}
            for source in source_kinds:
                source_raw = [
                    calibration["baseline"]["by_source_kind"][source]
                    for calibration in calibrations
                    if source in calibration["baseline"].get("by_source_kind", {})
                ]
                source_cross_fitted = [
                    row["by_source_kind"][source]
                    for row in cross_fitted_rows
                    if source in row.get("by_source_kind", {})
                ]
                by_source_kind[source] = {
                    "out_of_fold_exact_span_micro": _micro(source_raw),
                    "cross_fitted_threshold_micro": _micro(source_cross_fitted),
                }
            ocr_psm_source_kinds = sorted({
                source
                for calibration in calibrations
                for source in calibration["baseline"].get("by_ocr_psm_source_kind", {})
            })
            by_ocr_psm_source_kind = {}
            for source in ocr_psm_source_kinds:
                source_raw = [
                    calibration["baseline"]["by_ocr_psm_source_kind"][source]
                    for calibration in calibrations
                    if source in calibration["baseline"].get("by_ocr_psm_source_kind", {})
                ]
                source_cross_fitted = [
                    row["by_ocr_psm_source_kind"][source]
                    for row in cross_fitted_rows
                    if source in row.get("by_ocr_psm_source_kind", {})
                ]
                by_ocr_psm_source_kind[source] = {
                    "out_of_fold_exact_span_micro": _micro(source_raw),
                    "cross_fitted_threshold_micro": _micro(source_cross_fitted),
                }
            original_sources = [
                by_source_kind[source]
                for source in ("text-pdf", "scan")
                if source in by_source_kind
            ]
            if original_sources:
                by_source_kind["original-non-ocr"] = {
                    metric: _micro([source[metric] for source in original_sources])
                    for metric in (
                        "out_of_fold_exact_span_micro",
                        "cross_fitted_threshold_micro",
                    )
                }

            best_epochs = [report["best_epoch"] for report in fold_reports]
            epochs_ran = [report.get("epochs_ran", report["best_epoch"])
                          for report in fold_reports]
            training_seconds = [report.get("training_elapsed_seconds", 0.0)
                                for report in fold_reports]
            tokens_per_second = [report.get("training_tokens_per_second", 0.0)
                                 for report in fold_reports]
            peak_gpu_memory = [report.get("peak_gpu_memory_gib", 0.0)
                               for report in fold_reports]
            fold_f1 = [
                report["best_validation_exact_span_f1"] for report in fold_reports
            ]
            window_sizes = {
                report["hyperparameters"].get("window_size", 512)
                for report in fold_reports
            }
            strides = {
                report["hyperparameters"].get("stride", 96) for report in fold_reports
            }
            learning_rates = {
                report["hyperparameters"].get("learning_rate") for report in fold_reports
            }
            batch_sizes = {
                report["hyperparameters"].get("batch_size") for report in fold_reports
            }
            eval_batch_sizes = {
                report["hyperparameters"].get("eval_batch_size", 8)
                for report in fold_reports
            }
            validation_dtypes = {
                report["hyperparameters"].get("validation_dtype", "fp32")
                for report in fold_reports
            }
            early_stopping_patience = {
                report["hyperparameters"].get("early_stopping_patience", 0)
                for report in fold_reports
            }
            epoch_limits = {
                report["hyperparameters"].get("epoch_limit", len(report["epochs"]))
                for report in fold_reports
            }
            if any(len(values) != 1 for values in
                   (window_sizes, strides, learning_rates, batch_sizes,
                    eval_batch_sizes, validation_dtypes, early_stopping_patience,
                    epoch_limits)):
                raise SystemExit(
                    "Learning rate, batch size, window size and stride must stay "
                    "fixed across CV folds, as must eval batch, validation dtype "
                    "early-stopping patience and epoch limit"
                )

            candidate = {
                "document_type": document_type,
                "entity_loss_weight": weight,
                "folds": len(folds),
                "out_of_fold_exact_span_micro": _micro(raw_rows),
                "cross_fitted_threshold_micro": _micro(cross_fitted_rows),
                "by_source_kind": by_source_kind,
                "by_ocr_psm_source_kind": by_ocr_psm_source_kind,
                "same_fold_tuned_micro_diagnostic": _micro(same_fold_tuned_rows),
                "fold_raw_f1_mean": statistics.mean(fold_f1),
                "fold_raw_f1_stddev": statistics.pstdev(fold_f1),
                "best_epochs": best_epochs,
                "epochs_ran": epochs_ran,
                "median_best_epoch": int(statistics.median(best_epochs)),
                "total_training_seconds": sum(training_seconds),
                "mean_training_tokens_per_second": statistics.mean(tokens_per_second),
                "peak_gpu_memory_gib": max(peak_gpu_memory),
                "validation_thresholds": thresholds,
                "median_validation_threshold": statistics.median(thresholds),
                "learning_rate": learning_rates.pop(),
                "batch_size": batch_sizes.pop(),
                "eval_batch_size": eval_batch_sizes.pop(),
                "validation_dtype": validation_dtypes.pop(),
                "early_stopping_patience": early_stopping_patience.pop(),
                "epoch_limit": epoch_limits.pop(),
                "window_size": window_sizes.pop(),
                "stride": strides.pop(),
            }
            summaries.append(candidate)
            candidates_by_type[document_type].append(candidate)

    selected = {}
    for document_type, candidates in candidates_by_type.items():
        winner = max(candidates, key=lambda row: (
            row["cross_fitted_threshold_micro"]["f1"],
            row["out_of_fold_exact_span_micro"]["f1"],
        ))
        selected[document_type] = {
            "entity_loss_weight": winner["entity_loss_weight"],
            "median_best_epoch": winner["median_best_epoch"],
            "median_validation_threshold": winner["median_validation_threshold"],
            "epoch_limit": winner["epoch_limit"],
            "window_size": winner["window_size"],
            "stride": winner["stride"],
            "out_of_fold_f1": winner["out_of_fold_exact_span_micro"]["f1"],
            "cross_fitted_threshold_f1": winner["cross_fitted_threshold_micro"]["f1"],
        }

    report = {
        "model": "PKOBP/embed-modernbert-68m",
        "model_revision": "5c76fc72d527c991fcff9265945649a5b3c8e1d2",
        "validation_design": "3-fold group CV over original train+validation cases; document types from one case stay together",
        "test_documents_included": False,
        "context_variant": context_variant,
        "training_selection_metric": "pooled exact-span micro-F1 with threshold cross-fitted from the other two folds",
        "threshold_calibration_note": "for each fold, apply the median threshold selected on the other folds; same-fold tuned metrics are diagnostic only",
        "candidates": summaries,
        "selected_by_type": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    for document_type, choice in selected.items():
        print(
            f"{document_type}: weight={choice['entity_loss_weight']}, "
            f"cross-fitted F1={choice['cross_fitted_threshold_f1']:.4f} "
            f"(raw OOF={choice['out_of_fold_f1']:.4f}), "
            f"median epoch={choice['median_best_epoch']}, "
            f"threshold={choice['median_validation_threshold']:+.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
