import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from generate_data import generate, SCENARIOS
from src.train import train
from src.reconcile import load_model, run_reconciliation
from src.agent import investigate_break

st.set_page_config(page_title="LedgerMatch", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* Hide Streamlit menu and deploy button, but KEEP the header so sidebar toggle works */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Clean up metric typography */
    [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 600; color: #1E3A8A; }
    [data-testid="stMetricLabel"] { font-weight: 500; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Polish the tabs to look like a modern SaaS navigation */
    .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 2px solid #F1F5F9; }
    .stTabs [data-baseweb="tab"] { padding: 12px 16px; border-radius: 6px 6px 0 0; transition: background-color 0.2s; }
    .stTabs [aria-selected="true"] { background-color: rgba(30, 58, 138, 0.05); border-bottom-color: transparent; }
    </style>
""", unsafe_allow_html=True)

st.title("LedgerMatch: Automated Reconciliation")
st.caption("Matches records across two inconsistent ledgers and explains, in plain English, "
           "why the leftovers don't match. All data below is synthetically generated.")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


st.sidebar.header("Scenario Configuration")
scenario = st.sidebar.radio(
    "Test Environment Data",
    options=list(SCENARIOS.keys()),
    format_func=lambda s: {"clean": "Clean Day (Low Noise)",
                            "noisy": "Regular Noisy Day (Standard)",
                            "migration": "System Migration (High Noise)"}[s],
    index=1,
)

st.sidebar.divider()

n_records = st.sidebar.slider("Transaction Volume", 200, 2000, 800, step=100)

if st.sidebar.button("Regenerate Data & Retrain", type="primary", use_container_width=True):
    with st.spinner("Synthesizing ledgers and training model pipeline..."):
        generate(n_records, scenario)
        train(scenario, n_records)

st.sidebar.divider()

threshold = st.sidebar.slider(
    "Confidence Threshold", 0.05, 0.95, 0.5, 0.05,
    help="Higher = fewer, more confident matches (higher precision). Lower = catches more edge cases (higher recall)."
)


if not (DATA_DIR / "internal_ledger.csv").exists():
    with st.spinner("System Bootstrap: Generating synthetic ledgers..."):
        generate(n_records, scenario)

if not (MODEL_DIR / "matcher.joblib").exists():
    with st.spinner("System Bootstrap: Training entity resolution matcher..."):
        train(scenario, n_records)

internal = pd.read_csv(DATA_DIR / "internal_ledger.csv")
external = pd.read_csv(DATA_DIR / "external_ledger.csv")
model, vectorizer = load_model()

matches, unmatched_internal, unmatched_external = run_reconciliation(
    internal, external, model, vectorizer, threshold=threshold
)


with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Internal Records", len(internal))
    c2.metric("Matched Records", len(matches), f"{len(matches) / max(len(internal),1):.0%} Match Rate")
    c3.metric("Unmatched (Internal)", len(unmatched_internal))
    c4.metric("Unmatched (External)", len(unmatched_external))

st.write("") 

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Reconciliation Results")
    tab1, tab2, tab3 = st.tabs(["Matches", "Breaks: Internal", "Breaks: External"])

    with tab1:
        if len(matches):
            show = matches[["internal_id", "external_id", "amount_diff", "time_diff_sec",
                             "name_ratio", "ref_ratio", "match_proba"]].sort_values(
                "match_proba", ascending=False)
            st.dataframe(
                show,
                use_container_width=True, 
                height=400,
                hide_index=True,
                column_config={
                    "match_proba": st.column_config.ProgressColumn(
                        "Confidence", help="Match probability based on ML model", format="%.2f", min_value=0, max_value=1
                    ),
                    "amount_diff": st.column_config.NumberColumn(
                        "Amount Delta", format="₦%.2f"
                    ),
                    "time_diff_sec": st.column_config.NumberColumn(
                        "Time Delta (s)"
                    ),
                    "name_ratio": st.column_config.NumberColumn(
                        "Name Match %", format="%.2f"
                    ),
                    "ref_ratio": st.column_config.NumberColumn(
                        "Ref Match %", format="%.2f"
                    )
                }
            )
        else:
            st.info("No records met the selected confidence threshold.")

    with tab2:
        st.dataframe(unmatched_internal, use_container_width=True, height=400, hide_index=True)
        if len(unmatched_internal):
            pick = st.selectbox("Select Break ID to Investigate (Internal):",
                                 unmatched_internal["internal_id"].tolist(), key="int_pick")
            if st.button("Run Diagnostics", key="investigate_internal"):
                record = internal[internal["internal_id"] == pick].iloc[0]
                result = investigate_break(record, external, internal)
                st.session_state["last_investigation"] = result

    with tab3:
        st.dataframe(unmatched_external, use_container_width=True, height=400, hide_index=True)
        if len(unmatched_external):
            pick_e = st.selectbox("Select Break ID to Investigate (External):",
                                   unmatched_external["external_id"].tolist(), key="ext_pick")
            if st.button("Run Diagnostics", key="investigate_external"):
                record = external[external["external_id"] == pick_e].iloc[0]
                result = investigate_break(record, internal, external)
                st.session_state["last_investigation"] = result

with right:
    st.subheader("Root-Cause Agent")
    with st.container(border=True):
        result = st.session_state.get("last_investigation")
        if result:
            st.success("Diagnostics Complete")
            st.write(f"**Analysis:** {result['explanation']}")
            
            st.write("")
            st.caption("TELEMETRY LOGS")
            if result["findings"]:
                for f in result["findings"]:
                    st.code(f, language="json")
            else:
                st.code('{"status": "no_signals_detected"}', language="json")
        else:
            st.info("Select a break record from the tables on the left and click **Run Diagnostics** to execute root-cause analysis.")

    st.write("")
    st.subheader("Match Discrepancy Distribution")
    with st.container(border=True):
        if len(matches):
            diag = pd.cut(matches["amount_diff"], bins=[-0.01, 0.5, 50, 1e9],
                           labels=["Exact Amount", "Minor Rounding/Fee", "Large Discrepancy"])
            fig = px.pie(diag.value_counts().reset_index(), names="amount_diff", values="count",
                         hole=0.4)
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for visualization.")