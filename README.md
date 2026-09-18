# INFINITY-IQ — SIH26170 Burn-In Screening Platform

AI-driven engineering component screening using burn-in measurement data.
Detects latent defects, predicts 168-hour degradation and produces an
explainable PASS / MONITOR / REVIEW / REJECT decision for every component.

**Team / App:** INFINITY IQ · **Problem Statement:** SIH26170 — AI-Driven
Anomaly Detection in Component Burn-In & Screening

## Live & Links

- **Live application:** https://infinity-iq.streamlit.app
- **Project source code:** https://github.com/rkvvenu-lgtm/INFINITY-IQ-
- **Demo video:** https://github.com/rkvvenu-lgtm/INFINITY-IQ-/blob/main/docs/demo/INFINITY-IQ_Demo.mp4
## Problem → Solution Mapping (SIH26170)

| Problem | Our Solution |
| --- | --- |
| Early-life failures are hard to detect from raw burn-in data | Statistical + AI anomaly detection on every parameter, globally and per lot |
| Multiple parameters and phases make manual screening inconsistent | Automatic feature discovery + a fixed, documented decision table |
| One dataset template should not limit the tool | Domain-agnostic "Universal Screening" accepts any CSV/Excel dataset |
| Results must be explainable | Every record carries an AI explanation (culprit parameter, drift %, spec violations) |
| Accuracy must be provable | Built-in Algorithm Self-Test reports recall and false-positive rate on controlled data |

## Features

- **Electronics burn-in screening** — Phase-1 pipeline (anomaly detection, 168h drift prediction, specification evaluation, risk fusion).
- **Universal Generic Screening** — upload ANY dataset; auto-discovers numeric parameters, lot/group columns, time/phase columns and spec-limit columns; screens it with the same scoring, thresholds and decision table.
- **Algorithm Self-Test** — injects 6σ outliers and validates on clean data. Verified: recall **100%**, false-positive rate **1.3%**.
- **AI vs statistical-only baseline** — shows how many anomalies the AI pipeline catches over a plain |z| ≥ 3.5 threshold screen.
- **Reports & CSV export** of full screening results.

## Quick Start (local)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
streamlit run user_app.py
```

Open http://localhost:8501.

### Try the demo in 2 clicks
1. Open **⚙️ Screening** → click **⚡ Load Sample Burn-In Dataset** → **🚀 Start AI Screening**.
2. Or open **🌐 Universal Screening** → **⚡ Load Sample Generic Dataset** → **🚀 Run Universal Screening**.

## Data Format

- CSV / XLSX / XLS.
- Electronics flow expects burn-in measurement columns
  (`Temperature_0h/24h/96h…`, `Press_*`, `Iddq_*`, `Leakage_*`, `Delay_*`, `P_final` or similar).
- Universal flow accepts any numeric dataset; columns are discovered automatically.

Built-in samples:
- `data/burn_in_measurements.csv` — 2000 electronics burn-in records.
- `data/sample_generic_data.csv` — 750 any-domain records with injected defects.

## Decision Table (documented)

| Cumulative risk | Decision | Action |
| --- | --- | --- |
| < 15% | PASS | Accept component |
| 15–35% | MONITOR | Watchlist / retest |
| 35–65% | REVIEW | Engineering review required |
| ≥ 65% | REJECT | Failure-prone, exclude |

## Architecture

```
modules/
├── electronics_pipeline.py       # Phase-1 orchestration
├── anomaly_detector.py           # IsolationForest + robust z-score + lot neighbours
├── drift_predictor.py            # 168h drift / prediction
├── specification_engine.py       # Spec-limit evaluation
├── risk_fusion.py                # Weighted risk scoring
├── explainability_engine.py      # Per-record explanations
├── data_quality_intelligence.py  # Readiness checks
└── generic_screening/            # Domain-agnostic subsystem
    ├── generic_feature_discovery.py
    ├── generic_anomaly.py
    ├── generic_drift.py
    ├── generic_rules.py
    ├── generic_explainability.py
    ├── generic_engine.py
    └── self_test.py              # Algorithm self-test
```

## Verification

Run the built-in self-test from the **🌐 Universal Screening** page
(**🧪 Algorithm Self-Test** button):

```
Test 1  Clean data            false positives 13 / 1000  → FPR 1.3%
Test 2  Injected 6σ outliers  detected 31 / 31           → Recall 100%
Self-test result: PASS
```

Electronics regression (2000 burn-in records): PASS 1537 / MONITOR 463,
no exceptions across all UI flows.

## Tech Stack

Python 3.13 · Streamlit 1.63 · scikit-learn 1.9 · pandas 3.0 · numpy 2.5 ·
plotly 7.0 · joblib 1.6
