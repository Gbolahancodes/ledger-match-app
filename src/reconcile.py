from pathlib import Path

import joblib
import pandas as pd

from src.features import build_candidate_pairs, FEATURE_COLUMNS

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"


def load_model():
    model = joblib.load(MODEL_DIR / "matcher.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    return model, vectorizer


def run_reconciliation(internal: pd.DataFrame, external: pd.DataFrame,
                        model, vectorizer, threshold: float = 0.5):
    pairs, _ = build_candidate_pairs(internal, external, vectorizer=vectorizer)
    if len(pairs) == 0:
        return pd.DataFrame(), internal.copy(), external.copy()

    pairs["match_proba"] = model.predict_proba(pairs[FEATURE_COLUMNS])[:, 1]

    # keep only each internal record's single best candidate above threshold
    pairs = pairs.sort_values("match_proba", ascending=False)
    best = pairs[pairs["match_proba"] >= threshold].drop_duplicates("internal_id")
    best = best.drop_duplicates("external_id")  # avoid double-claiming one external record

    matched_internal_ids = set(best["internal_id"])
    matched_external_ids = set(best["external_id"])

    unmatched_internal = internal[~internal["internal_id"].isin(matched_internal_ids)].copy()
    unmatched_external = external[~external["external_id"].isin(matched_external_ids)].copy()

    return best, unmatched_internal, unmatched_external
