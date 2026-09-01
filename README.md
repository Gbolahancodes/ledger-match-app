# LedgerMatch — Automated Reconciliation & Break Explainer

Matches transaction records across two inconsistent ledgers (e.g. a bank's
internal ledger vs. an external switch/card-network feed) using entity
resolution, and explains — in plain English — why the leftovers ("breaks")
didn't match.

## Why this problem

Reconciliation is unglamorous but genuinely high-stakes: unmatched breaks
between an internal ledger and an external system tie up capital and
trigger audit findings, and are normally resolved by analysts manually
comparing spreadsheets. Exact-match rules fail constantly on real-world
messiness — timestamp jitter, rounding, truncated names, settlement
delays, duplicates — which is why record linkage / entity resolution is a
named ML subfield in its own right.

## Architecture

```
Raw ledgers (internal, external)
        │
        ▼
Blocking (src/features.py)          -- restrict comparisons to same-day ± amount tolerance
        │
        ▼
Feature engineering                  -- amount delta, time delta, string similarity
                                         (rapidfuzz) + character n-gram TF-IDF cosine
                                         similarity (a lightweight, fully offline text
                                         "embedding" — no external model download needed)
        │
        ▼
Pairwise classifier (src/train.py)   -- GradientBoostingClassifier vs. a
                                         LogisticRegression baseline
        │
        ▼
SHAP explainability                  -- global + per-pair feature importance
        │
        ▼
Break investigator (src/agent.py)    -- tool-calling agent: runs independent
                                         "tools" (near-amount check, timing-delay
                                         check, duplicate-reference check, fuzzy-name
                                         check) against every unmatched record and
                                         narrates a root-cause hypothesis. Uses a live
                                         Claude call if ANTHROPIC_API_KEY is set,
                                         otherwise a deterministic offline fallback.
        │
        ▼
FastAPI service (api/main.py)  <──┐
Streamlit demo (app/streamlit_app.py) ── both call the identical
                                         src/reconcile.py logic, so the
                                         demo and the "production" service
                                         are provably consistent.
```

## Real, honest evaluation

Because `generate_data.py` produces both ledgers from one true source, we
keep the ground-truth mapping (`data/ground_truth.csv`) and can report real
precision/recall on the matching task — not a hand-wavy accuracy number.
Example run (`noisy` scenario, n=800):

```
GBM   precision=0.934  recall=1.000  f1=0.966  roc_auc=0.999
LogReg precision=0.907  recall=1.000  f1=0.951   (baseline)
```

SHAP global feature importance on that run: `text_cos_sim` (0.46) >
`ref_ratio` (0.26) > `time_diff_sec` (0.18) > amount/name features — i.e.
the character-level text similarity carries most of the discriminative
signal, which matches intuition (amount and blocking already narrow
candidates heavily before the classifier sees them).

## Setup

```bash
pip install -r requirements.txt

# 1. Generate a synthetic dual-ledger dataset
python generate_data.py --scenario noisy --n 800

# 2. Train the pairwise matcher (prints precision/recall/F1/ROC-AUC + SHAP importance)
python -m src.train --scenario noisy --n 800

# 3a. Run the interactive demo (auto-bootstraps data/model on first run)
streamlit run app/streamlit_app.py

# 3b. OR run the FastAPI microservice
uvicorn api.main:app --reload --port 8000
```

### Scenarios

| Scenario    | What it simulates                                   |
|-------------|------------------------------------------------------|
| `clean`     | Low noise — a well-behaved day                       |
| `noisy`     | Realistic everyday noise (default)                   |
| `migration` | A system migration / format change — heavy noise, tests drift monitoring |

### Optional: live LLM narration

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
If set, `src/agent.py` calls Claude to write the break explanation instead
of the deterministic template. Entirely optional — the project runs free
and offline without it.

## Docker

```bash
docker build -t ledgermatch .
docker run -p 8000:8000 ledgermatch
```

## Deploying the demo for free

Push this repo to GitHub, then deploy `app/streamlit_app.py` on
[Streamlit Community Cloud](https://streamlit.io/cloud) — zero hosting
cost. The app bootstraps its own data and model on first run, so no
external database or pre-trained model upload is required.

## Project structure

```
ledgermatch/
├── README.md
├── requirements.txt
├── Dockerfile
├── generate_data.py       # synthetic dual-ledger generator (3 noise scenarios)
├── data/                  # generated at runtime: internal/external ledgers + ground truth
├── models/                # generated at runtime: trained matcher + vectorizer + metrics
├── src/
│   ├── __init__.py
│   ├── features.py        # blocking + pairwise feature engineering
│   ├── train.py            # trains + evaluates the matcher, computes SHAP importance
│   ├── reconcile.py        # shared matching logic used by API and Streamlit
│   └── agent.py             # tool-calling break-investigation agent
├── api/
│   └── main.py              # FastAPI microservice
└── app/
    └── streamlit_app.py     # interactive Streamlit demo
```
