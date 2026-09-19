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
| Future 168h behaviour must be predicted from early readings | Drift models take Value_0h + Value_24h and forecast Value_168h |
| A drift-rate safety limit must trigger early rejection | Safety-slope check flags components drifting faster than the limit allows |
| Results must be explainable | Every record carries an AI explanation (culprit parameter, drift %, spec violations) |
| Accuracy must be provable | Built-in self-tests report recall and false-positive rate; Training/Evaluation mode shows live MAE / RMSE / R² against 168h ground truth |

## Features

- **Electronics burn-in screening** — Phase-1 pipeline (anomaly detection, 168h drift prediction, specification evaluation, risk fusion).
- **Universal Generic Screening** — upload ANY dataset; auto-discovers numeric parameters, lot/group columns, time/phase columns and spec-limit columns; screens it with the same scoring, thresholds and decision table.
- **Drift Model Evaluation (Training/Evaluation mode)** — per-parameter table with held-out and live MAE / RMSE / R² against the measured 168h ground truth, plus a predicted-vs-actual 168h scatter chart.
- **Safety-Slope Check (PS Module B)** — computes the maximum hourly drift that still keeps each component inside its engineering limit at 168h, and flags faster-drifting components for early rejection.
- **Algorithm Self-Test** — injects 6σ outliers and validates on clean data. Verified: recall **100%**, false-positive rate **1.3%**.
- **Electronics Anomaly Spot-Check** — one-click 6σ injection test on burn-in data. Measured: recall **100%**, clean-data FPR **5.1%**.
- **AI vs statistical-only baseline** — shows how many anomalies the AI pipeline catches over a plain |z| ≥ 3.5 threshold screen.
- **Engineer review workflow** — reviewer name, action and comment per component, persisted to an audit trail.
- **Branded experience** — INFINITY IQ logo (sidebar + browser tab) and a 5-second animated splash intro on first load (`?splash=0` skips it).
- **Reports & CSV export** of full screening results.

## Quick Start (local)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
python -m streamlit run user_app.py
```

Open http://localhost:8501.

### Try the demo in 2 clicks
1. Open **⚙️ Screening** → click **⚡ Load Sample Burn-In Dataset** → **🚀 Start AI Screening**.
2. Or open **🌐 Universal Screening** → **⚡ Load Sample Generic Dataset** → **🚀 Run Universal Screening**.

### Operating modes (⚙️ Screening page sidebar)
- **Live Future Screening** — predicts unseen 168h behaviour from early measurements using the pre-trained drift models.
- **Training / Evaluation** — evaluates predictions against 168h ground truth (MAE / RMSE / R² panel, safety-slope check, spot-check). Model retraining is strictly opt-in via the retrain checkbox, since refitting all drift models is slow and overwrites the shipped `models/` files.

## Data Format

- CSV / XLSX / XLS.
- Electronics flow expects burn-in columns such as `Component_ID`, `Lot_ID`,
  `Component_Type`, `Temperature_C`, `Iddq_0h/24h/96h/168h_uA`,
  `Leakage_0h/24h/96h/168h_uA`, `Delay_0h/24h/96h/168h_ns`, plus per-parameter
  `*_Max_Limit_*` spec columns.
- Universal flow accepts any numeric dataset; columns are discovered automatically.

Built-in samples:

- `data/burn_in_measurements.csv` — 2000 electronics burn-in records.
- `data/sample_generic_data.csv` — 750 any-domain records with injected defects.

## Decision Table (documented and code-enforced)

| Cumulative risk | Decision | Action |
| --- | --- | --- |
| < 15% | PASS | Accept component |
| 15–35% | MONITOR | Watchlist / retest |
| 35–65% | REVIEW | Engineering review required |
| ≥ 65% | REJECT | Failure-prone, exclude |

Thresholds are enforced in `modules/risk_fusion.py` (spec FAIL always escalates
to REJECT / REVIEW regardless of score).

## Architecture

```
modules/
├── anomaly_detector.py           # IsolationForest + robust z-score + lot-relative risk
├── drift_predictor.py            # 168h drift training / prediction (early 0h+24h features)
├── specification_engine.py       # Spec-limit evaluation
├── risk_fusion.py                # Weighted risk scoring + decision table
├── explainability_engine.py      # Per-record quantitative WHY
├── data_quality_intelligence.py  # Readiness checks
├── universal_screening_engine.py # Electronics / universal routing
├── component_intelligence.py     # Component-level views
├── engineering_review.py         # Review records + audit trail
├── final_ui_integration.py       # Review + report integration
├── report_engine.py              # Report generation
└── generic_screening/            # Domain-agnostic subsystem
    ├── generic_feature_discovery.py
    ├── generic_anomaly.py
    ├── generic_drift.py
    ├── generic_rules.py
    ├── generic_explainability.py
    ├── generic_engine.py
    └── self_test.py              # Algorithm self-test
models/                           # Registry + pre-trained drift models (Iddq / Leakage / Delay)
assets/                           # INFINITY IQ logo + splash artwork
```

## Verification

Run the built-in self-test from the **🌐 Universal Screening** page
(**Run Algorithm Self-Test** button, seed 42):

```
Test 1  Clean data (300 rows)        flagged 4 / 300   → FPR 1.33% (limit 10%)
Test 2  Injected 6σ outliers (10)    detected 10 / 10  → Recall 100% (required ≥ 90%)
Self-test result: PASS
```

Electronics spot-check from the **📊 Results** page in Training/Evaluation mode
(**Run electronics spot-check** button, seed 42, 2000 burn-in records):

```
Injected 6σ outliers (20)            detected 20 / 20  → Recall 100%
Clean-data false-positive rate                             → FPR 5.1%
```

Electronics regression on the sample dataset (2000 burn-in records,
Training/Evaluation mode, no retraining):

```
PASS 833 / MONITOR 998 / REVIEW 169 / REJECT 0 — no exceptions.
```

## Tech Stack

Python 3.13 · Streamlit 1.63 · scikit-learn 1.9 · pandas 3.0 · numpy 2.5 ·
plotly 7.0 · joblib 1.6
