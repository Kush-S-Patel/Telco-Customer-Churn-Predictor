"""Model training utilities."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RF_PARAMS, RANDOM_STATE, XGB_PARAMS
from src.evaluation.metrics import compute_metrics, optimal_threshold_f1, save_metrics
from src.features.preprocess import load_processed_data, train_test_split_data
from src.visualization.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _scale_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / max(n_pos, 1)


def get_model_registry(y_train=None):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", solver="saga", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(**RF_PARAMS),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB and y_train is not None:
        params = {**XGB_PARAMS, "scale_pos_weight": _scale_pos_weight(y_train)}
        models["xgboost"] = xgb.XGBClassifier(**params)
    return models

def train_all_models(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_processed_data()

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    results = {}
    proba_test = {}
    fitted = {}

    for name, model in get_model_registry(y_train).items():
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        proba_test[name] = y_proba

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[name] = metrics

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Champion by ROC-AUC
    champion_name = max(results, key=lambda k: results[k]["roc_auc"])
    champion = fitted[champion_name]
    champion_proba = proba_test[champion_name]

    threshold = optimal_threshold_f1(y_test, champion_proba)
    y_pred_champion = (champion_proba >= threshold).astype(int)

    champion_metrics = compute_metrics(
        y_test,
        y_pred_champion,
        champion_proba
    )

    champion_metrics["optimal_threshold"] = threshold
    champion_metrics["model_name"] = champion_name

    joblib.dump(
        {
            "model": champion,
            "threshold": threshold,
            "feature_names": list(X_train.columns),
        },
        MODELS_DIR / "champion_bundle.joblib",
    )

    save_metrics(results, METRICS_DIR / "all_models_test.json")
    save_metrics(champion_metrics, METRICS_DIR / "champion_test.json")

    meta = {
        "champion": champion_name,
        "threshold": threshold,
        "n_features": X_train.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    with open(METRICS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _generate_eval_figures(
        X_train,
        y_train,
        y_test,
        proba_test,
        results,
        champion_name,
        champion_proba,
        y_pred_champion,
        fitted[champion_name],
        list(X_train.columns),
    )

    return {
        "results": results,
        "champion": champion_name,
        "models": fitted,
    }


def train_champion(df: pd.DataFrame | None = None):
    out = train_all_models(df)
    return out["champion"]


def _generate_eval_figures(
    X_train, y_train, y_test, proba_test, results, champion_name,
    champion_proba, y_pred_champion, champion_model, feature_names,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y_test, proba_test, FIGURES_DIR / "roc_curves.png")
    plot_pr_curve(y_test, proba_test, FIGURES_DIR / "pr_curves.png")
    plot_confusion_matrix(
        y_test, y_pred_champion, FIGURES_DIR / "confusion_matrix_champion.png",
        title=f"Champion Model ({champion_name})",
    )
    plot_calibration(y_test, champion_proba, FIGURES_DIR / "calibration_champion.png")

    metrics_df = pd.DataFrame(
        {k: {"roc_auc": v["roc_auc"], "f1": v["f1"], "recall": v["recall"]} for k, v in results.items()}
    ).T
    plot_model_comparison(metrics_df, FIGURES_DIR / "model_comparison.png")

    if hasattr(champion_model, "feature_importances_"):
        imp = pd.Series(champion_model.feature_importances_, index=feature_names)
    elif hasattr(champion_model, "coef_"):
        imp = pd.Series(np.abs(champion_model.coef_[0]), index=feature_names)
    else:
        imp = None
    if imp is not None:
        plot_feature_importance(imp, FIGURES_DIR / "feature_importance.png")

    # Learning curve for champion
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        champion_model.__class__(**champion_model.get_params()),
        X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=RANDOM_STATE,
    )
    plot_learning_curve(train_sizes, train_scores, val_scores, FIGURES_DIR / "learning_curve.png")

    # CV summary table
    cv_scores = {}
    for name, model in get_model_registry(y_train).items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = {"mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std())}
    save_metrics(cv_scores, METRICS_DIR / "cv_scores.json")
