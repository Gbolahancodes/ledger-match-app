from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.reconcile import load_model, run_reconciliation
from src.agent import investigate_break

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

app = FastAPI(title="LedgerMatch API", version="1.0")

_model, _vectorizer = None, None


@app.on_event("startup")
def _load():
    global _model, _vectorizer
    try:
        _model, _vectorizer = load_model()
    except FileNotFoundError:
        # model not trained yet -- endpoints will return a clear error
        _model, _vectorizer = None, None


class ReconcileRequest(BaseModel):
    threshold: float = 0.5


class InvestigateRequest(BaseModel):
    record_id: str
    side: Literal["internal", "external"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/reconcile")
def reconcile(req: ReconcileRequest):
    if _model is None:
        raise HTTPException(500, "Model not trained yet. Run `python -m src.train` first.")

    internal = pd.read_csv(DATA_DIR / "internal_ledger.csv")
    external = pd.read_csv(DATA_DIR / "external_ledger.csv")

    matches, unmatched_internal, unmatched_external = run_reconciliation(
        internal, external, _model, _vectorizer, threshold=req.threshold
    )

    return {
        "n_internal": len(internal),
        "n_external": len(external),
        "n_matched": len(matches),
        "n_unmatched_internal": len(unmatched_internal),
        "n_unmatched_external": len(unmatched_external),
        "match_rate": round(len(matches) / max(len(internal), 1), 4),
        "matches": matches.to_dict(orient="records"),
        "unmatched_internal": unmatched_internal.to_dict(orient="records"),
        "unmatched_external": unmatched_external.to_dict(orient="records"),
    }


@app.post("/investigate_break")
def investigate(req: InvestigateRequest):
    internal = pd.read_csv(DATA_DIR / "internal_ledger.csv")
    external = pd.read_csv(DATA_DIR / "external_ledger.csv")

    if req.side == "internal":
        record_df = internal[internal["internal_id"] == req.record_id]
        other_ledger, own_ledger = external, internal
    else:
        record_df = external[external["external_id"] == req.record_id]
        other_ledger, own_ledger = internal, external

    if record_df.empty:
        raise HTTPException(404, f"Record {req.record_id} not found on {req.side} ledger")

    result = investigate_break(record_df.iloc[0], other_ledger, own_ledger)
    return result
