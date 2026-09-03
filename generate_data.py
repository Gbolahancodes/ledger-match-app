"""
generate_data.py
-----------------
Creates a synthetic "internal ledger" and "external ledger" that represent
the same underlying set of transactions as seen by two different systems
(e.g. a bank's core ledger vs. a card network / switch).

Because we generate both sides from one true source, we KEEP the ground
truth mapping separately (ground_truth.csv) so the matching model can be
trained and evaluated with real precision/recall -- and so the Streamlit
demo can show "breaks" that are genuinely unmatched, not just noise.

Usage:
    python generate_data.py --scenario clean --n 800
    python generate_data.py --scenario noisy --n 800
    python generate_data.py --scenario migration --n 800
"""

import argparse
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker('en_NG')

DATA_DIR = Path(__file__).parent / "data"

SCENARIOS = {
    
    "clean":     dict(time_jitter=15,  amount_noise=0.03, name_corrupt=0.05,
                       ref_corrupt=0.02, missing=0.01, duplicate=0.01, extra=0.02),
    "noisy":     dict(time_jitter=180, amount_noise=0.15, name_corrupt=0.30,
                       ref_corrupt=0.15, missing=0.06, duplicate=0.05, extra=0.08),
    "migration": dict(time_jitter=600, amount_noise=0.25, name_corrupt=0.55,
                       ref_corrupt=0.35, missing=0.10, duplicate=0.08, extra=0.12),
}


def random_reference():
    return "TXN" + "".join(random.choices(string.digits, k=8))


def corrupt_name(name: str) -> str:
    """Simulate how a name gets mangled crossing systems."""
    choice = random.random()
    parts = name.split()
    if choice < 0.35 and len(parts) > 1:
        return f"{parts[-1].upper()} {parts[0][0]}."          
    elif choice < 0.6:
        return name.upper().replace(" ", "")                   
    elif choice < 0.8:
        return name[: max(4, len(name) - 3)]                   
    else:
        return name.replace("o", "0").replace("i", "1")         


def corrupt_reference(ref: str) -> str:
    ref = list(ref)
    idx = random.randint(3, len(ref) - 1)
    ref[idx] = random.choice(string.digits)
    return "".join(ref)


def generate(n: int, scenario: str, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    cfg = SCENARIOS[scenario]

    base_date = pd.Timestamp("2026-08-01")
    truth_rows, internal_rows, external_rows = [], [], []

    for i in range(n):
        txn_id = f"T{i:06d}"
        name = fake.name()
        amount = float(np.round(np.random.lognormal(mean=9.5, sigma=1.0), 2))
        amount = min(amount, 500_000.0)
        ts = base_date + pd.Timedelta(
            days=random.randint(0, 6),
            hours=random.randint(7, 21),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        reference = random_reference()

        internal_rows.append(
            dict(internal_id=f"INT-{i:06d}", name=name, amount=amount,
                 timestamp=ts, reference=reference)
        )

        if random.random() < cfg["missing"]:
            truth_rows.append(dict(internal_id=f"INT-{i:06d}", external_id=None, txn_id=txn_id))
            continue  

        ext_amount = amount
        if random.random() < cfg["amount_noise"]:
            fee_pct = random.uniform(0.0, 0.015)
            ext_amount = round(amount * (1 - fee_pct), 2)

        ext_ts = ts + pd.Timedelta(seconds=random.randint(-cfg["time_jitter"], cfg["time_jitter"]))

        ext_name = corrupt_name(name) if random.random() < cfg["name_corrupt"] else name
        ext_ref = corrupt_reference(reference) if random.random() < cfg["ref_corrupt"] else reference

        ext_id = f"EXT-{i:06d}"
        external_rows.append(
            dict(external_id=ext_id, name=ext_name, amount=ext_amount,
                 timestamp=ext_ts, reference=ext_ref)
        )
        truth_rows.append(dict(internal_id=f"INT-{i:06d}", external_id=ext_id, txn_id=txn_id))

        # ---- occasional duplicate posting on the external side ----
        if random.random() < cfg["duplicate"]:
            dup_id = f"EXT-{i:06d}-DUP"
            external_rows.append(
                dict(external_id=dup_id, name=ext_name, amount=ext_amount,
                     timestamp=ext_ts + pd.Timedelta(seconds=random.randint(1, 30)),
                     reference=ext_ref)
            )

    # ---- extra external records with no internal counterpart at all ----
    n_extra = int(n * cfg["extra"])
    for j in range(n_extra):
        ext_id = f"EXT-EXTRA-{j:04d}"
        external_rows.append(
            dict(external_id=ext_id, name=fake.name(),
                 amount=float(np.round(np.random.lognormal(9.3, 1.0), 2)),
                 timestamp=base_date + pd.Timedelta(
                     days=random.randint(0, 6), hours=random.randint(7, 21),
                     minutes=random.randint(0, 59)),
                 reference=random_reference())
        )

    internal_df = pd.DataFrame(internal_rows)
    external_df = pd.DataFrame(external_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows)

    DATA_DIR.mkdir(exist_ok=True)
    internal_df.to_csv(DATA_DIR / "internal_ledger.csv", index=False)
    external_df.to_csv(DATA_DIR / "external_ledger.csv", index=False)
    truth_df.to_csv(DATA_DIR / "ground_truth.csv", index=False)

    print(f"[{scenario}] internal={len(internal_df)} external={len(external_df)} "
          f"true_matches={truth_df['external_id'].notna().sum()} "
          f"genuine_missing={truth_df['external_id'].isna().sum()}")
    return internal_df, external_df, truth_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="noisy")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.n, args.scenario, args.seed)
