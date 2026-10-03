**Hybrid Quantum Machine Learning Platform for Early Disease Detection**
Team: Egreen Quanta · [Add Team ID]

This is the complete, working submission package: all five Delivery Table items, each independently documented, tested, and — where testing found a real problem — fixed and written up honestly rather than hidden.

## Start here

- **Try it live**: the published dashboard (link given in chat) runs the actual trained model in your browser — check some symptoms and see a real prediction.
- **The pitch deck**: `QuantaDx_SIH2026.pptx`, built to match last year's winning SIH template structure.
- **Everything below** is the engineering behind both of those.

## The five deliverables

| # | Folder | What it is | Status |
|---|---|---|---|
| 1 | `aikya-hqml-deliverable1-preprocessing/` | Two-tier dataset: 50 curated common/lung diseases (the real training data) + a cleaned HPO rare-disease layer (secondary), feature-selected down to a qubit-feasible 34 features | Built, tested |
| 2 | `aikya-hqml-deliverable2-architecture/` | Classical front-end → amplitude encoding (6 qubits) → trainable ansatz → measurement → classical readout. Backend-agnostic (simulator today, one-line swap to real quantum hardware later) | Built, verified end-to-end on real data |
| 3 | `aikya-hqml-deliverable3-models/` | The hybrid quantum model, trained, benchmarked head-to-head against classical baselines on identical data splits | Built, trained, **honestly reported** (classical currently ahead — see that folder's README for why, and for the concrete next steps to close the gap) |
| 4 | `aikya-hqml-deliverable4-decision-support/` | Disease probability scores → risk stratification (4 urgency tiers) → sensitivity/specificity-tuned per-disease thresholds | Built; a real false-emergency-alarm bug was caught by testing and fixed (see that folder's README) |
| 5 | `aikya-hqml-deliverable5-platform/` | FastAPI backend (`api.py`) serving the full hybrid ensemble + the published client-side dashboard | Built, both pieces tested; a path-resolution bug found while wiring them together was fixed |

## How the pieces connect

```
Deliverable 1 (data)
       │  tier1_qubit_ready_34feat.csv
       ▼
Deliverable 2 (architecture)
       │  frontend.py, quantum_encoding.py, hybrid_circuit.py
       ▼
Deliverable 3 (trained models)
       │  hybrid_model_weights.npz, benchmark_report.json
       ▼
Deliverable 4 (decision support)
       │  predict.py, risk_thresholds.py, decision_support.py
       ▼
Deliverable 5 (platform)
          api.py  (full ensemble, for your real AIKYA backend)
          quantadx_dashboard.html  (classical model, runs standalone)
```

Each arrow is a real import, not just a conceptual link — `predict.py` loads Deliverable 3's actual saved weights, `decision_support.py` loads Deliverable 4's own calibrated thresholds, `api.py` imports `decision_support.py` directly. Running any later stage requires the earlier ones to have been run first, in order (each folder's README says what to run).

## What's still on you

1. **Clinical review** of `data_preprocessing/curated_common_disease_symptoms.py` — the 50-disease symptom list was built from general medical knowledge, not a clinical cohort, flagged since Deliverable #1.
2. **Decide whether to invest more compute** improving the quantum model's accuracy before presenting — Deliverable #3's README has the concrete next steps (more epochs, deeper ansatz, parameter-matched comparison).
3. **Wire `api.py` into your actual Antigravity-built AIKYA frontend** — one `fetch` call from the existing `ChatService`, documented in Deliverable #5's README.
4. **Fill in the remaining PPT placeholders** — official Problem Statement ID and Team ID, flagged on the title slide.

## Why the honesty matters for judging

Several things in here are not flattering on the surface — the quantum model losing to classical baselines, a dataset that turned out mostly useless for the actual target diseases, bugs caught mid-build. Every one of them is documented rather than smoothed over, because *"we built a real benchmarking harness and can explain exactly where our hybrid approach currently falls short and why"* is a stronger answer to a judge's question than a suspiciously perfect demo with no visible engineering process behind it.
