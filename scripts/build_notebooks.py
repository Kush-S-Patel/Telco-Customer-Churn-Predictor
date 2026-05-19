#!/usr/bin/env python
"""Regenerate analysis notebooks with consistent structure."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write(name, cells):
    path = NB / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("Wrote", path)


write(
    "01-eda.ipynb",
    [
        md("# 01 — Exploratory Data Analysis\n\nTelco customer churn dataset: demographics, services, billing, and churn label."),
        code(
            """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, DATA_RAW, FIGURES_DIR
from src.visualization.plots import (
    plot_categorical_churn_rates,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_numerical_by_churn,
    setup_style,
)

setup_style()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)"""
        ),
        md("## Raw data overview"),
        code(
            """raw = pd.read_csv(DATA_RAW)
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
raw["Churn_num"] = raw["Churn"].map({"No": 0, "Yes": 1})

print("Shape:", raw.shape)
print("\\nMissing values:\\n", raw.isnull().sum()[raw.isnull().sum() > 0])
display(raw.head())
raw.describe(include="all").T.head(15)"""
        ),
        md("## Target distribution"),
        code(
            """plot_df = raw.copy()
plot_df["Churn"] = plot_df["Churn_num"]
plot_churn_distribution(plot_df, "Churn", FIGURES_DIR / "eda_churn_distribution.png")
raw["Churn_num"].value_counts(normalize=True).rename({0: "Stay", 1: "Churn"})"""
        ),
        md("## Numerical features vs churn"),
        code(
            """plot_numerical_by_churn(
    plot_df, ["tenure", "MonthlyCharges", "TotalCharges"], "Churn",
    FIGURES_DIR / "eda_numerical_boxplots.png",
)
plt.imshow(plt.imread(FIGURES_DIR / "eda_numerical_boxplots.png"))
plt.axis("off")"""
        ),
        md("## Categorical churn rates"),
        code(
            """cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "SeniorCitizen", "Partner"]
plot_categorical_churn_rates(plot_df, cat_cols, "Churn", FIGURES_DIR / "eda_categorical_churn_rates.png")
plt.imshow(plt.imread(FIGURES_DIR / "eda_categorical_churn_rates.png"))
plt.axis("off")"""
        ),
        md("## Processed feature correlations"),
        code(
            """proc = pd.read_csv(DATA_PROCESSED)
proc = proc.fillna(proc.median(numeric_only=True))
plot_correlation_heatmap(proc, FIGURES_DIR / "eda_correlation_heatmap.png")
plt.imshow(plt.imread(FIGURES_DIR / "eda_correlation_heatmap.png"))
plt.axis("off")"""
        ),
        md("### Key EDA insights\n- **Class imbalance:** ~26.5% churn rate — use stratified splits and class weights.\n- **Tenure & contract** strongly separate churners from loyal customers.\n- **Fiber + month-to-month + electronic check** segments show elevated churn.\n- **TotalCharges** missing for new customers; impute with `MonthlyCharges`."),
    ],
)

write(
    "02-feature_engineering.ipynb",
    [
        md("# 02 — Feature Engineering\n\nBuild model-ready matrix from raw CSV via one-hot encoding."),
        code(
            """import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, DATA_RAW
from src.data.make_dataset import preprocess_data
from src.features.preprocess import load_processed_data, prepare_xy"""
        ),
        md("## Rebuild processed dataset"),
        code(
            """df = preprocess_data(DATA_RAW, DATA_PROCESSED)
print(df.shape)
df.head()"""
        ),
        md("## Feature matrix"),
        code(
            """X, y = prepare_xy(df)
print(f"Features: {X.shape[1]} | Samples: {len(y)} | Churn rate: {y.mean():.3f}")
bool_cols = X.select_dtypes(include="bool").columns
if len(bool_cols):
    X[bool_cols] = X[bool_cols].astype(int)
X.describe().T.head(10)"""
        ),
        md("## Train / test split (stratified)"),
        code(
            """from src.features.preprocess import train_test_split_data

X_train, X_test, y_train, y_test = train_test_split_data(df)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train churn rate:", y_train.mean(), "| Test:", y_test.mean())"""
        ),
    ],
)

write(
    "03-modeling.ipynb",
    [
        md("# 03 — Model Training\n\nTrain logistic regression, random forest, gradient boosting, and XGBoost; persist champion bundle."),
        code(
            """import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.preprocess import load_processed_data
from src.models.train import train_all_models"""
        ),
        code(
            """df = load_processed_data()
results = train_all_models(df)
champion = results["champion"]
print("Champion model:", champion)
pd.DataFrame({k: {"roc_auc": v["roc_auc"], "f1": v["f1"], "recall": v["recall"]} for k, v in results["results"].items()}).T"""
        ),
        md("## Optional: quick hyperparameter search (XGBoost)"),
        code(
            """# Uncomment to run Optuna tuning (~2–5 min)
# import optuna
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# import xgboost as xgb
# from src.features.preprocess import train_test_split_data
# X_train, X_test, y_train, y_test = train_test_split_data(df)
# cv = StratifiedKFold(5, shuffle=True, random_state=42)
# def objective(trial):
#     params = dict(
#         n_estimators=trial.suggest_int("n_estimators", 200, 600),
#         max_depth=trial.suggest_int("max_depth", 3, 8),
#         learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
#         subsample=trial.suggest_float("subsample", 0.6, 1.0),
#         colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
#         eval_metric="logloss", random_state=42, n_jobs=-1,
#     )
#     model = xgb.XGBClassifier(**params)
#     return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30, show_progress_bar=True)
# print("Best CV ROC-AUC:", study.best_value)"""
        ),
    ],
)

write(
    "04-evaluation.ipynb",
    [
        md("# 04 — Model Evaluation\n\nReview test metrics, ROC/PR curves, confusion matrix, calibration, and feature importance."),
        code(
            """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, METRICS_DIR
from src.models.predict import load_model, predict_proba
from src.features.preprocess import load_processed_data, train_test_split_data"""
        ),
        code(
            """with open(METRICS_DIR / "all_models_test.json") as f:
    metrics = json.load(f)
pd.DataFrame({k: {m: metrics[k][m] for m in ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]} for k in metrics}).T"""
        ),
        code(
            """with open(METRICS_DIR / "champion_test.json") as f:
    champion = json.load(f)
print("Champion metrics:", {k: champion[k] for k in ["roc_auc", "f1", "recall", "optimal_threshold", "model_name"]})"""
        ),
        md("## Evaluation figures"),
        code(
            """figures = [
    "model_comparison.png", "roc_curves.png", "pr_curves.png",
    "confusion_matrix_champion.png", "calibration_champion.png",
    "feature_importance.png", "learning_curve.png",
]
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()
for ax, name in zip(axes, figures):
    path = FIGURES_DIR / name
    if path.exists():
        ax.imshow(plt.imread(path))
        ax.set_title(name.replace("_", " ").replace(".png", "").title(), fontsize=10)
    ax.axis("off")
for j in range(len(figures), len(axes)):
    axes[j].axis("off")
plt.tight_layout()"""
        ),
        md("## Sample predictions"),
        code(
            """df = load_processed_data()
_, X_test, _, y_test = train_test_split_data(df)
bundle = load_model()
proba = predict_proba(X_test, bundle)
preds = (proba >= bundle["threshold"]).astype(int)
out = pd.DataFrame({"actual": y_test.values, "probability": proba, "predicted": preds})
out.head(10)"""
        ),
    ],
)
