"""
QuantaDx -- Decision Support Module
SIH 2026 PS3, Deliverable #4 (Prediction & Decision Support Module)

Ties together:
  predict.py          -- disease probability scores (Delivery Table metric 1)
  disease_urgency.py   -- urgency tiers for risk stratification (metric 2)
  risk_thresholds.py   -- tier-aware sensitivity/specificity thresholds (metric 3)

into one function a chat UI (or the earlier AIKYA extraction pipeline) can
call directly.

SAFETY: this module never claims to diagnose. Every response includes an
explicit disclaimer, and the overall risk level is driven by the single
most urgent FLAGGED candidate, not just the single most probable one --
a low-probability emergency-tier disease that clears its (deliberately
low) threshold still escalates the whole response, because in a screening
context a missed emergency is worse than an unnecessary caution flag.
"""

import json
from pathlib import Path
from predict import QuantaDxPredictor

_THIS_DIR = Path(__file__).resolve().parent

TIER_RANK = {"emergency": 3, "urgent": 2, "routine": 1, "self_care": 0}

DISCLAIMER = (
    "This is not a diagnosis. QuantaDx is a research prototype intended to "
    "help you decide how soon to seek care, not to replace a clinician's "
    "assessment. If symptoms are severe, worsening, or you are otherwise "
    "concerned, consult a qualified healthcare professional."
)


class DecisionSupportEngine:
    def __init__(self):
        self.predictor = QuantaDxPredictor()
        with open(_THIS_DIR / "outputs/risk_thresholds.json") as f:
            self.thresholds = json.load(f)

    def assess(self, symptom_dict, method="ensemble", max_candidates=5):
        """
        symptom_dict: {canonical_name: confidence_0_to_1}
        Returns a structured decision-support object. See bottom of this
        file for the exact shape.
        """
        proba = self.predictor.predict_proba(symptom_dict, method=method)
        ranked = sorted(proba.items(), key=lambda x: -x[1])

        flagged = []
        for disease, p in ranked:
            threshold = self.thresholds[disease]["threshold"]
            if p >= threshold:
                flagged.append({
                    "disease": disease,
                    "probability": round(p, 4),
                    "tier": self.thresholds[disease]["tier"],
                    "threshold_used": threshold,
                })

        # Always show at least the top candidate even if nothing cleared
        # its threshold, so the response is never empty -- but mark it
        # clearly as below-threshold / low-confidence.
        if not flagged:
            top_disease, top_p = ranked[0]
            flagged = [{
                "disease": top_disease,
                "probability": round(top_p, 4),
                "tier": self.thresholds[top_disease]["tier"],
                "threshold_used": self.thresholds[top_disease]["threshold"],
                "note": "below this disease's calibrated threshold -- low-confidence signal only",
            }]

        flagged = flagged[:max_candidates]

        # Overall risk = the single most urgent tier among FLAGGED
        # candidates, not just the top-probability one.
        most_urgent = max(flagged, key=lambda c: TIER_RANK[c["tier"]])
        overall_tier = most_urgent["tier"]

        from disease_urgency import RECOMMENDED_ACTION
        return {
            "top_candidates": flagged,
            "overall_risk_tier": overall_tier,
            "recommended_action": RECOMMENDED_ACTION[overall_tier],
            "driven_by": most_urgent["disease"],
            "model_method": method,
            "disclaimer": DISCLAIMER,
        }


if __name__ == "__main__":
    engine = DecisionSupportEngine()

    scenarios = {
        "Likely pneumonia": {
            "fever": 1.0, "productive_cough": 0.9, "chest_pain": 0.6,
            "shortness_of_breath": 0.7, "fatigue": 0.8,
        },
        "Mostly a cold, but mentions one red-flag symptom": {
            "runny_nose": 0.9, "sneezing": 0.8, "sore_throat": 0.6,
            "sudden_shortness_of_breath": 0.3,  # low-confidence PE/pneumothorax signal
        },
        "Classic migraine": {
            "severe_headache": 1.0, "sensitivity_to_light": 0.9,
            "sensitivity_to_sound": 0.7, "nausea": 0.5,
        },
    }

    for name, symptoms in scenarios.items():
        print(f"\n{'='*60}\nScenario: {name}\n{'='*60}")
        result = engine.assess(symptoms)
        print(f"Overall risk tier: {result['overall_risk_tier'].upper()} "
              f"(driven by: {result['driven_by']})")
        print(f"Recommended action: {result['recommended_action']}")
        print("Top candidates:")
        for c in result["top_candidates"]:
            flag = f" [{c['note']}]" if "note" in c else ""
            print(f"  {c['disease']:28s} p={c['probability']:.3f}  "
                  f"tier={c['tier']:10s} threshold={c['threshold_used']:.3f}{flag}")
