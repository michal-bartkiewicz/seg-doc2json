from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extract import ExtractionError
from .hybrid import HybridExtractor
from .pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract appeal-document JSON locally with regex and optional ModernBERT verification."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/documents")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--roberta-bundle",
        type=Path,
        help="optional local fine-tuned ModernBERT bundle; model files are never downloaded",
    )
    parser.add_argument(
        "--roberta-config",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/roberta/results/transfer-order-candidate-inference-config.json",
        help="local per-type ModernBERT token thresholds and window configuration",
    )
    parser.add_argument(
        "--roberta-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="device for local ModernBERT inference (default: cpu)",
    )
    parser.add_argument(
        "--roberta-policy",
        choices=["rank"],
        default="rank",
        help="rank principal and attorney candidates and fill missing fields from OCR spans",
    )
    args = parser.parse_args(argv)

    try:
        extractor = HybridExtractor(
            roberta_bundle=args.roberta_bundle,
            roberta_config=args.roberta_config if args.roberta_bundle else None,
            device=args.roberta_device,
            policy=args.roberta_policy,
        )
        report = run_pipeline(
            input_dir=args.input.resolve(),
            output_dir=args.output.resolve(),
            extractor=extractor,
        )
    except (ValueError, OSError, ExtractionError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    verification = "ModernBERT candidate verification enabled" if report["model_enabled"] else "regex-only (no ModernBERT bundle)"
    print(
        f"Saved {report['result_count']}/{report['document_count']} JSON files under "
        f"{args.output}; local OCR pages: {report['ocr_page_count']}; {verification}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
