#!/usr/bin/env python
"""Train all churn models and write metrics + evaluation figures."""
import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make_dataset import preprocess_data
from src.models.train import train_all_models


def main():
    print("Preprocessing raw data...")
    preprocess_data()
    print("Training models...")
    out = train_all_models()
    champion = out["champion"]
    auc = out["results"][champion]["roc_auc"]
    f1 = out["results"][champion]["f1"]
    print(f"Champion: {champion} | Test ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
    print(f"Artifacts: models/, reports/figures/, reports/metrics/")


if __name__ == "__main__":
    main()
