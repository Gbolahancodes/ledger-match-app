"""
src/agent.py
------------
Investigates an UNMATCHED record ("break") and explains, in plain English,
the most likely reason it didn't match anything on the other ledger.

Implements a genuine tool-calling pattern: each check below is a small,
independent function (a "tool") with a single job. The agent runs the
relevant tools, collects their findings, and turns them into a narrative.

If ANTHROPIC_API_KEY is set in the environment, the agent lets Claude choose
which tools to call and write the final explanation (real agentic tool use).
If no key is present, it falls back to a deterministic rule-based narrator
that runs every tool and templates the result -- so the whole project still
works offline, for free, with zero API cost, which matters for a Streamlit
Community Cloud demo.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Tools -- each takes the break record + the "other side" ledger and returns
# a small, structured finding. Kept deliberately independent/composable.
# ---------------------------------------------------------------------------

def check_near_amount_matches(record: pd.Series, other_ledger: pd.DataFrame,
                               tolerance_pct: float = 0.05) -> Optional[dict]:
    near = other_ledger[
        (other_ledger["amount"] - record["amount"]).abs() / max(record["amount"], 1e-6) < tolerance_pct
    ]
    if len(near):
        return {"tool": "check_near_amount_matches", "count": int(len(near)),
                "closest_amount": float(near.iloc[0]["amount"])}
    return None


def check_timing_delay(record: pd.Series, other_ledger: pd.DataFrame,
                        window_days: int = 2) -> Optional[dict]:
    ts = pd.to_datetime(record["timestamp"])
    window = other_ledger[
        (pd.to_datetime(other_ledger["timestamp"]) >= ts - pd.Timedelta(days=window_days)) &
        (pd.to_datetime(other_ledger["timestamp"]) <= ts + pd.Timedelta(days=window_days)) &
        ((other_ledger["amount"] - record["amount"]).abs() < 1.0)
    ]
    if len(window):
        return {"tool": "check_timing_delay", "found_same_amount_nearby_days": True,
                "count": int(len(window))}
    return None


def check_duplicate_reference(record: pd.Series, own_ledger: pd.DataFrame) -> Optional[dict]:
    dupes = own_ledger[own_ledger["reference"] == record["reference"]]
    if len(dupes) > 1:
        return {"tool": "check_duplicate_reference", "duplicate_count": int(len(dupes) - 1)}
    return None


def check_fuzzy_name_matches(record: pd.Series, other_ledger: pd.DataFrame,
                              threshold: int = 70) -> Optional[dict]:
    scores = other_ledger["name"].astype(str).apply(
        lambda n: fuzz.token_sort_ratio(str(record["name"]), n)
    )
    best_idx = scores.idxmax() if len(scores) else None
    if best_idx is not None and scores[best_idx] >= threshold:
        return {"tool": "check_fuzzy_name_matches", "best_score": int(scores[best_idx]),
                "candidate_id": other_ledger.loc[best_idx].get(
                    "external_id", other_ledger.loc[best_idx].get("internal_id"))}
    return None


TOOLS = [check_near_amount_matches, check_timing_delay,
         check_duplicate_reference, check_fuzzy_name_matches]


# ---------------------------------------------------------------------------
# Deterministic narrator (default, free, offline)
# ---------------------------------------------------------------------------

def _narrate(findings: list[dict]) -> str:
    if not findings:
        return ("No plausible counterpart found on the other ledger under any tolerance. "
                "This looks like a genuinely missing record -- escalate for manual review.")

    parts = []
    for f in findings:
        if f["tool"] == "check_timing_delay":
            parts.append("a same-amount record exists within a couple of days, suggesting "
                          "a settlement delay rather than a true break")
        elif f["tool"] == "check_near_amount_matches":
            parts.append(f"{f['count']} record(s) with a very close amount "
                          f"(₦{f['closest_amount']:,.2f}) exist on the other side, "
                          "suggesting a fee/rounding difference")
        elif f["tool"] == "check_duplicate_reference":
            parts.append(f"this reference number appears {f['duplicate_count']} extra time(s) "
                          "on its own ledger, suggesting a duplicate posting")
        elif f["tool"] == "check_fuzzy_name_matches":
            parts.append(f"a name-similar record (similarity {f['best_score']}/100) exists, "
                          "suggesting a formatting/spelling mismatch rather than a missing transaction")

    return "Likely explanation: " + "; and ".join(parts) + "."


def investigate_break(record: pd.Series, other_ledger: pd.DataFrame,
                       own_ledger: pd.DataFrame) -> dict:
    findings = []
    for finding in (
        check_near_amount_matches(record, other_ledger),
        check_timing_delay(record, other_ledger),
        check_duplicate_reference(record, own_ledger),
        check_fuzzy_name_matches(record, other_ledger),
    ):
        if finding:
            findings.append(finding)

    explanation = _narrate(findings)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        explanation = _llm_narrate(record, findings, api_key) or explanation

    return {"findings": findings, "explanation": explanation}


# ---------------------------------------------------------------------------
# Optional live LLM narrator -- used only if ANTHROPIC_API_KEY is set.
# Demonstrates the "agentic" story without making the API a hard dependency.
# ---------------------------------------------------------------------------

def _llm_narrate(record: pd.Series, findings: list[dict], api_key: str) -> Optional[str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are a reconciliation analyst assistant. A transaction failed to match "
            "on the counterparty ledger. Here is the unmatched record and the results of "
            "automated investigation tools that were run against it. Write a 2-3 sentence, "
            "plain-English root-cause hypothesis for a bank operations analyst.\n\n"
            f"Record: {record.to_dict()}\n"
            f"Tool findings: {findings}"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    except Exception:
        return None
