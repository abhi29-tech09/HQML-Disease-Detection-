# QuantaDx — Deliverable #5: Software Platform / Prototype

SIH 2026, Problem Statement 3 — *Hybrid Quantum Machine Learning Platform for Early Disease Detection*

Maps to row 5 of the Delivery Table: *"End-to-end usable system — user interface or API, dataset upload, model training & evaluation dashboard, result visualization."*

## Two pieces, honestly scoped differently

| Piece | What it is | Why it's built this way |
|---|---|---|
| **`api.py`** | A real FastAPI backend serving the full Deliverable #4 decision-support engine (hybrid quantum + classical ensemble) | This is server code — it needs Python, PennyLane, and your trained model files to run. You run it yourself (locally or deployed); it's not something that can live inside a published static page. |
| **Published dashboard** (link above) | A self-contained, interactive symptom checker + benchmark viewer, running entirely in the browser | A published page can't reach an arbitrary backend running in my sandbox (no public URL, and published pages can't make outbound network calls anyway) — so this runs the **real trained classical model** (the logistic-regression weights, ported to a ~10-line JS function) client-side instead of faking it. It's genuinely the trained model, just the classical half of the ensemble rather than the full hybrid one. |

## What the dashboard actually does

- Embeds the real exported model weights, thresholds, urgency tiers, and benchmark numbers from Deliverables #3–#4 (`export_dashboard_data.py` produces this)
- Lets you check symptoms and get back the same structured output `decision_support.py` produces: flagged candidates, risk tier, recommended action, disclaimer — computed with `softmax(coef_ @ x + intercept_)`, the actual trained logistic regression
- Shows the honest benchmark comparison from Deliverable #3 (logistic regression and random forest both ahead of the 6-qubit hybrid quantum model)
- Is explicit in its own footer about what it is and isn't running, rather than implying it's the full quantum system

## Wiring `api.py` into your AIKYA website

```bash
cd platform
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --port 8000
```

Then from your Next.js `ChatService` (the same place that calls `/api/extract-symptoms` from the earlier AIKYA UI work), add a call to `http://localhost:8000/predict` with the extracted symptom dict after extraction finishes. The response shape is exactly what `decision_support.py`'s `assess()` returns — ready to drive the Summary Card / risk-tier UI described in the original UI spec.

Endpoints:
- `POST /predict` — `{symptoms: {...}, method: "ensemble"|"quantum"|"classical"}` → full decision-support result
- `POST /predict-batch` — upload a CSV (one row per case, columns matching `/features`) → results for every row. This is the **dataset upload** component of this deliverable.
- `GET /diseases` — every disease the model knows, with urgency tier
- `GET /features` — the exact symptom vocabulary the model understands (align your chat extraction's `canonical_name` values to this list)
- `GET /health` — basic healthcheck

## A bug this deliverable caught

Wiring `api.py` to import `decision_support.py` from a different directory immediately broke — both `predict.py` and `decision_support.py` were using paths relative to the *working directory* (`"outputs/risk_thresholds.json"`), which only happened to work because every earlier test was run from inside their own folder. Fixed by resolving every path relative to each file's own location (`Path(__file__).resolve().parent`) instead — the kind of bug that's invisible until something actually tries to integrate the pieces, which is exactly what this deliverable is for.

## Known limitations

- The published dashboard's client-side model is the classical baseline only — the hybrid quantum ensemble needs `api.py` running, by necessity (a quantum circuit simulation isn't something to port to browser JS for a hackathon demo).
- `api.py`'s CORS is wide open (`allow_origins=["*"]`) for prototype convenience — lock this to your actual frontend's domain before any real deployment.
- `/predict-batch` has no row-count limit or async job queue — fine for a demo CSV, not for a production-scale upload.

## All five deliverables, end to end

1. **Data Pre-processing & Feature Engineering** — two-tier dataset (curated common/lung diseases + cleaned HPO rare-disease layer), feature-selected and qubit-ready
2. **Hybrid Quantum-Classical Architecture** — amplitude encoding, backend-agnostic device abstraction, verified end-to-end
3. **Quantum ML Models** — trained and honestly benchmarked against classical baselines
4. **Prediction & Decision Support** — probability scores, risk stratification, tuned thresholds (with a real false-alarm bug caught and fixed)
5. **Software Platform** — API backend + interactive published dashboard (this deliverable)

Every deliverable's README documents what was built, what was tested, what broke during testing, and what's still a known limitation — that trail of honest engineering decisions is itself something worth presenting to judges, not just the final numbers.
