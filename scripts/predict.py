#!/usr/bin/env python
"""CLI: score customers from processed CSV."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predict import predict_from_csv


def main():
    parser = argparse.ArgumentParser(description="Predict telco churn from processed CSV")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(ROOT / "data" / "processed" / "cleaned.csv"),
        help="Path to processed CSV (default: data/processed/cleaned.csv)",
    )
    parser.add_argument("-o", "--output", help="Optional output CSV path")
    args = parser.parse_args()
    out = predict_from_csv(args.input)
    print(out.head(10).to_string())
    print(f"\nScored {len(out)} rows. Mean P(churn)={out['churn_probability'].mean():.3f}")
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
