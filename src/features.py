"""
src/features.py
----------------
Turns two raw ledgers into a set of CANDIDATE PAIRS with engineered
similarity features. This is the core "entity resolution" logic:

1. Blocking: don't compare every internal record to every external record
   (O(n*m) is wasteful and mostly irrelevant pairs) -- only compare records
   whose date and amount are already in the same rough neighbourhood.
2. Feature engineering: for each surviving candidate pair, compute numeric
   deltas (amount, time) and text-similarity signals (string similarity +
   lightweight character n-gram "embedding" cosine similarity via TF-IDF,
   which needs no external model download -- everything runs offline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AMOUNT_TOLERANCE_PCT = 0.03   # block on amounts within 3%
DATE_TOLERANCE_DAYS = 1        # block on same day +/- 1


def _fit_char_vectorizer(internal: pd.DataFrame, external: pd.DataFrame) -> TfidfVectorizer:
    """Character n-gram TF-IDF acts as a lightweight, fully-offline text
    embedding: similar-looking strings ('OKAFOR J.' vs 'John Okafor') end up
    with high cosine similarity even though there's no exact token overlap."""
    corpus = pd.concat([
        internal["name"] + " " + internal["reference"],
        external["name"] + " " + external["reference"],
    ]).astype(str).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vec.fit(corpus)
    return vec


def build_candidate_pairs(internal: pd.DataFrame, external: pd.DataFrame,
                           vectorizer: TfidfVectorizer | None = None) -> pd.DataFrame:
    internal = internal.copy()
    external = external.copy()
    internal["timestamp"] = pd.to_datetime(internal["timestamp"])
    external["timestamp"] = pd.to_datetime(external["timestamp"])
    internal["date"] = internal["timestamp"].dt.date
    external["date"] = external["timestamp"].dt.date

    if vectorizer is None:
        vectorizer = _fit_char_vectorizer(internal, external)

    int_vecs = vectorizer.transform(internal["name"] + " " + internal["reference"])
    ext_vecs = vectorizer.transform(external["name"] + " " + external["reference"])

    rows = []
    # ---- blocking: group external records by date to avoid full cross join ----
    ext_by_date: dict = {}
    for idx, d in enumerate(external["date"]):
        for offset in (-1, 0, 1):
            key = pd.Timestamp(d) + pd.Timedelta(days=offset)
            ext_by_date.setdefault(key.date(), []).append(idx)

    for i_idx, i_row in internal.iterrows():
        candidates = set(ext_by_date.get(i_row["date"], []))
        if not candidates:
            continue
        for e_idx in candidates:
            e_row = external.iloc[e_idx]
            amount_diff = abs(i_row["amount"] - e_row["amount"])
            amount_pct_diff = amount_diff / max(i_row["amount"], 1e-6)
            if amount_pct_diff > AMOUNT_TOLERANCE_PCT * 5:  # loose pre-filter
                continue

            time_diff_sec = abs((i_row["timestamp"] - e_row["timestamp"]).total_seconds())
            name_ratio = fuzz.token_sort_ratio(str(i_row["name"]), str(e_row["name"])) / 100.0
            ref_ratio = fuzz.ratio(str(i_row["reference"]), str(e_row["reference"])) / 100.0
            text_cos = float(cosine_similarity(int_vecs[i_idx], ext_vecs[e_idx])[0, 0])

            rows.append(dict(
                internal_id=i_row["internal_id"],
                external_id=e_row["external_id"],
                amount_diff=amount_diff,
                amount_pct_diff=amount_pct_diff,
                time_diff_sec=time_diff_sec,
                name_ratio=name_ratio,
                ref_ratio=ref_ratio,
                text_cos_sim=text_cos,
            ))

    return pd.DataFrame(rows), vectorizer


FEATURE_COLUMNS = [
    "amount_diff", "amount_pct_diff", "time_diff_sec",
    "name_ratio", "ref_ratio", "text_cos_sim",
]
