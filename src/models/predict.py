"""Inference utilities for trained churn models."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.features.preprocess import load_processed_data


def load_model(name: str = "champion_bundle"):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run scripts/train.py first.")
    return joblib.load(path)


def predict_proba(df: pd.DataFrame, bundle=None) -> np.ndarray:
    bundle = bundle or load_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    threshold = bundle.get("threshold", 0.5)
    X = df[feature_names] if "Churn" not in df.columns else df.drop(columns=["Churn"])[feature_names]
    proba = model.predict_proba(X)[:, 1]
    return proba


def predict(df: pd.DataFrame, bundle=None) -> np.ndarray:
    bundle = bundle or load_model()
    proba = predict_proba(df, bundle)
    return (proba >= bundle.get("threshold", 0.5)).astype(int)


def predict_from_csv(path: str | Path) -> pd.DataFrame:
    df = load_processed_data(path)
    if "Churn" in df.columns:
        y_true = df["Churn"]
        X_df = df
    else:
        y_true = None
        X_df = df
    proba = predict_proba(X_df)
    out = pd.DataFrame({"churn_probability": proba, "churn_predicted": predict(X_df)})
    if y_true is not None:
        out["actual_churn"] = y_true.values
    return out
