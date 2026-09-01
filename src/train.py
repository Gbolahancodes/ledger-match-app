"""
src/train.py
------------
Trains the pairwise match/no-match classifier.

Because generate_data.py keeps a ground-truth mapping, we can label every
candidate pair as a true match (1) or not (0) and get REAL precision/recall
on held-out data -- not a hand-wavy accuracy number.

Run:
    python -m src.train --scenario noisy --n 800
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.features import build_candidate_pairs, FEATURE_COLUMNS

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def label_pairs(pairs: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    true_pairs = set(
        zip(truth.dropna(subset=["external_id"])["internal_id"],
            truth.dropna(subset=["external_id"])["external_id"])
    )
    pairs = pairs.copy()
    pairs["label"] = pairs.apply(
        lambda r: int((r["internal_id"], r["external_id"]) in true_pairs), axis=1
    )
    return pairs


def train(scenario: str = "noisy", n: int = 800, seed: int = 42):
    internal = pd.read_csv(DATA_DIR / "internal_ledger.csv")
    external = pd.read_csv(DATA_DIR / "external_ledger.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth.csv")

    pairs, vectorizer = build_candidate_pairs(internal, external)
    pairs = label_pairs(pairs, truth)

    print(f"Candidate pairs: {len(pairs)}  |  positive rate: {pairs['label'].mean():.3f}")

    X = pairs[FEATURE_COLUMNS]
    y = pairs["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = dict(
        baseline_precision=precision_score(y_test, base_pred),
        baseline_recall=recall_score(y_test, base_pred),
        baseline_f1=f1_score(y_test, base_pred),
        gbm_precision=precision_score(y_test, pred),
        gbm_recall=recall_score(y_test, pred),
        gbm_f1=f1_score(y_test, pred),
        gbm_roc_auc=roc_auc_score(y_test, proba),
    )
    print(json.dumps(metrics, indent=2))

    # ---- SHAP: global feature importance for the writeup / demo ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(200, len(X_test)), random_state=seed))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(zip(FEATURE_COLUMNS, mean_abs_shap.tolist()),
                              key=lambda kv: -kv[1]))
    print("SHAP global feature importance:", json.dumps(importance, indent=2))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "matcher.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "shap_importance": importance}, f, indent=2)

    return model, vectorizer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="noisy")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.scenario, args.n, args.seed)
