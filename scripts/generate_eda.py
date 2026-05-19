#!/usr/bin/env python
"""Generate EDA figures from raw telco data."""
import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, FIGURES_DIR
from src.visualization.plots import (
    plot_categorical_churn_rates,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numerical_by_churn,
)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def main():
    df = load_raw()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_churn_distribution(df, "Churn", FIGURES_DIR / "eda_churn_distribution.png")
    plot_numerical_by_churn(
        df, ["tenure", "MonthlyCharges", "TotalCharges"], "Churn",
        FIGURES_DIR / "eda_numerical_boxplots.png",
    )

    # Processed-style correlation on encoded data
    from src.features.preprocess import load_processed_data
    proc = load_processed_data()
    plot_correlation_heatmap(proc, FIGURES_DIR / "eda_correlation_heatmap.png")

    cat_cols = [
        "Contract", "InternetService", "PaymentMethod",
        "PaperlessBilling", "SeniorCitizen", "Partner",
    ]
    plot_categorical_churn_rates(df, cat_cols, "Churn", FIGURES_DIR / "eda_categorical_churn_rates.png")
    print(f"EDA figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
