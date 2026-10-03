"""
QuantaDx -- Export model weights + metadata for the client-side dashboard.
SIH 2026 PS3, Deliverable #5 (Software Platform / Prototype)

The published dashboard runs entirely in the browser (no backend reachable
from a published page), so it uses the trained CLASSICAL model (which
Deliverable #3 showed is currently the stronger of the two) ported to
plain JS: softmax(coef_ @ x + intercept_). This is the real trained model,
not a fake -- just the classical half of the ensemble, client-side. The
full hybrid quantum ensemble is what api.py (the real backend) serves.
"""

import sys
import json
sys.path.insert(0, "../decision_support")
sys.path.insert(0, "../architecture")

from predict import QuantaDxPredictor  # noqa: E402
from disease_urgency import URGENCY_TIER, RECOMMENDED_ACTION  # noqa: E402

predictor = QuantaDxPredictor()

with open("../model_training/outputs/benchmark_report.json") as f:
    benchmark = json.load(f)

with open("../decision_support/outputs/risk_thresholds.json") as f:
    thresholds = json.load(f)

export = {
    "feature_order": predictor.feature_order,
    "diseases": predictor.diseases,
    "coef": predictor.classical_model.coef_.tolist(),
    "intercept": predictor.classical_model.intercept_.tolist(),
    "thresholds": {d: v["threshold"] for d, v in thresholds.items()},
    "urgency_tier": URGENCY_TIER,
    "recommended_action": RECOMMENDED_ACTION,
    "benchmark": {
        "random_forest": {"accuracy": benchmark["random_forest"]["accuracy"],
                            "macro_f1": benchmark["random_forest"]["macro_f1"]},
        "logistic_regression": {"accuracy": benchmark["logistic_regression"]["accuracy"],
                                  "macro_f1": benchmark["logistic_regression"]["macro_f1"]},
        "hybrid_quantum": {"accuracy": benchmark["hybrid_quantum"]["accuracy"],
                             "macro_f1": benchmark["hybrid_quantum"]["macro_f1"]},
    },
}

with open("outputs/dashboard_data.json", "w") as f:
    json.dump(export, f)

print(f"Exported {len(export['diseases'])} diseases, "
      f"{len(export['feature_order'])} features, "
      f"coef shape {len(export['coef'])}x{len(export['coef'][0])}")
