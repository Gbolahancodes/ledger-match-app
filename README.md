
# LedgerMatch: Automated Reconciliation & Break Explainer
LedgerMatch is a machine learning system designed to reconcile transaction records across inconsistent ledgers—such as a company's internal database and an external bank feed. Traditional exact-match reconciliation software fails frequently on messy, real-world data like timestamp jitter, OCR errors, or truncated names. LedgerMatch replaces these rigid rules with a probabilistic entity-resolution pipeline to link corresponding records despite the noise.

When transactions genuinely have no counterpart, they are flagged as unmatched "breaks". The system routes these breaks to an automated, tool-calling AI agent that runs diagnostic checks (e.g., looking for duplicate references, fee deductions, or settlement delays) and generates a plain-English explanation of the root cause. This eliminates the need for operations analysts to manually cross-reference spreadsheets.


## Core Features
* **Probabilistic Matching:** Uses a Gradient Boosting Classifier and lightweight character n-gram TF-IDF embeddings to link records based on amount deltas, time delays, and text similarity.
* **AI Break Investigator:** An agent that runs independent diagnostic tools to analyze unmatched records[cite: 1]. It functions entirely offline via deterministic rules, or optionally through a live Anthropic Claude AI integration.
* **Synthetic Data Generator:** Ships with a built-in simulator to generate clean, noisy, or system-migration data scenarios without requiring private financial data.
* **Interactive Dashboard:** A fully functional Streamlit frontend for visual reconciliation, confidence-threshold adjustment, and break investigation.
* **Production-Ready API:** A FastAPI microservice exposing the exact same matching logic for real-world system integration.

## Getting Started

### Prerequisites
Ensure you have Python 3.11+ installed.

### Installation
1. Clone the repository and navigate to the project folder:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ledgermatch.git](https://github.com/YOUR_USERNAME/ledgermatch.git)
   cd ledgermatch



2. Create a virtual environment and install dependencies:
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt

```



### Running the Application

Launch the interactive web dashboard to explore the data and run investigations. The app will automatically generate the test data and train the AI model on its first run:

```bash
streamlit run app/streamlit_app.py

```

*(Optional)* Start the backend FastAPI microservice to access the system programmatically:

```bash
uvicorn api.main:app --reload --port 8000

```

## Repository Structure

* `app/` - Streamlit frontend dashboard.


* `api/` - FastAPI microservice endpoints.


* `src/` - Core entity resolution, matching pipeline, and AI agent logic.


* `generate_data.py` - Synthetic dual-ledger data simulator.


