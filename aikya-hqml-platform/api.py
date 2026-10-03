"""
QuantaDx -- Backend API
SIH 2026 PS3, Deliverable #5 (Software Platform / Prototype)

Exposes the full Deliverable #4 decision-support engine (hybrid quantum +
classical ensemble) over HTTP, for the AIKYA chat UI's ChatService to call
after symptom extraction -- exactly the "/api/extract-symptoms" pattern
established earlier in this project, but for the prediction step.

Run with:  uvicorn api:app --reload --port 8000
Requires:  pip install fastapi uvicorn python-multipart
"""

import sys
import csv
import io
sys.path.insert(0, "../decision_support")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from decision_support import DecisionSupportEngine

app = FastAPI(title="QuantaDx API", version="0.1.0")

# Allow the AIKYA frontend (any origin, for prototype purposes -- lock this
# down to your actual frontend's domain before any real deployment) to call
# this API directly from the browser.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

engine = DecisionSupportEngine()


class PredictRequest(BaseModel):
    symptoms: dict[str, float]  # {canonical_name: confidence_0_to_1}
    method: str = "ensemble"    # "ensemble" | "quantum" | "classical"


@app.get("/health")
def health():
    return {"status": "ok", "n_diseases": engine.predictor.n_classes}


@app.get("/diseases")
def list_diseases():
    """All diseases the model knows, with urgency tier -- for the UI to
    show, e.g., what the model does and doesn't cover."""
    from disease_urgency import URGENCY_TIER
    return [{"disease": d, "tier": URGENCY_TIER[d]} for d in engine.predictor.diseases]


@app.get("/features")
def list_features():
    """The exact symptom vocabulary this model understands -- the AIKYA
    chat-extraction module's canonical_name values should be drawn from
    (or mapped to) this list."""
    return engine.predictor.feature_order


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return engine.assess(req.symptoms, method=req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Dataset upload for batch evaluation: a CSV with one row per case, one
    column per symptom (matching /features), values 0-1. Returns the
    decision-support result for every row -- useful for evaluating the
    model against a larger held-out set, or demoing it on many cases
    without typing each one into a form.
    """
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    results = []
    for i, row in enumerate(reader):
        symptom_dict = {k: float(v) for k, v in row.items() if v not in (None, "", "0")}
        try:
            result = engine.assess(symptom_dict)
            results.append({"row": i, **result})
        except ValueError as e:
            results.append({"row": i, "error": str(e)})

    return {"n_rows": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
