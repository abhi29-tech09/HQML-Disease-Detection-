# QuantaDx — Deliverable #4: Prediction & Decision Support Module

SIH 2026, Problem Statement 3 — *Hybrid Quantum Machine Learning Platform for Early Disease Detection*

Maps to row 4 of the Delivery Table: *"Inference and output generation — disease probability scores, early risk stratification, threshold tuning for sensitivity/specificity."*

## What this module contains

| File | Delivery Table metric it covers |
|---|---|
| `predict.py` | **Disease probability scores** — model-agnostic `predict_proba()`, ensembling the trained quantum model (Deliverable #3) with a reproduced classical baseline |
| `disease_urgency.py` | **Early risk stratification** — all 50 diseases mapped to an urgency tier (emergency / urgent / routine / self-care) with matching recommended-action text |
| `risk_thresholds.py` | **Threshold tuning for sensitivity/specificity** — per-disease probability thresholds, biased by tier, calibrated on Deliverable #3's held-out test set |
| `decision_support.py` | Ties all three together into one `assess()` call a chat UI can use directly |

## Why ensemble, not quantum-only

Deliverable #3 found the classical baseline (75.8% accuracy) currently beats the hybrid quantum model (42.5%). Building decision support on the quantum model alone, for the sake of a cleaner "quantum" story, would mean shipping the weaker predictor. The ensemble keeps the quantum model genuinely in the loop (it still contributes half the averaged probability, and the interface supports calling it in isolation via `method="quantum"`) while not pretending it's currently the stronger half.

## Why thresholds, not just top-1 prediction

A plain "most likely disease" output treats every condition the same. That's the wrong default for health screening: missing a possible emergency is worse than one extra caution flag, while flagging someone over a common cold on weak evidence just erodes trust. So `risk_thresholds.py` calibrates a **per-disease** threshold, biased by urgency tier (target sensitivity 90% for emergency-tier diseases down to 50% for self-care-tier), and `decision_support.py` flags *every* disease that clears its own threshold — not only the single top prediction — then sets the overall risk level from the most urgent flagged candidate, not just the most probable one.

## A real bug this caught, and the fix

Testing `decision_support.py` against three scenarios immediately surfaced a genuine problem: a **"classic migraine" case with zero pneumothorax-related symptoms got flagged as an EMERGENCY for pneumothorax** (p=0.031). Root cause: pneumothorax's test set has only ~4-5 positive examples, and the recall-driven threshold search calibrated down to 0.021 — barely above the 50-class random-guess baseline of 0.02 — so it fired on essentially any input.

**Fix**: added a hard floor (`MIN_THRESHOLD_FLOOR = 3.0 / n_classes ≈ 0.06`) that no calibrated threshold can go below, regardless of tier or how sparse the positive samples are. Re-ran the same three scenarios afterward — the false emergency alarm is gone, and the mechanism still correctly prioritizes emergency-tier sensitivity over self-care-tier specificity. This is flagged here rather than quietly fixed and left undocumented, because it's exactly the kind of failure mode a "threshold tuning for sensitivity/specificity" deliverable needs to be honest about: naive sensitivity-first calibration on too little data can backfire into alarm-fatigue-inducing false positives if you don't also bound it.

## Demo output (`decision_support.py`, after the fix)

```
Scenario: Likely pneumonia
Overall risk tier: URGENT (driven by: pneumonia)
  pneumonia   p=0.549  tier=urgent  threshold=0.424

Scenario: Mostly a cold, but mentions one red-flag symptom
Overall risk tier: SELF_CARE (driven by: common_cold)
  common_cold p=0.226  tier=self_care  threshold=0.301  [below threshold -- low-confidence signal only]

Scenario: Classic migraine
Overall risk tier: URGENT (driven by: kidney_stones)
  kidney_stones  p=0.428  tier=urgent  threshold=0.267
```

The first result is clean. The third is **not** — migraine should be the top candidate, not kidney stones. That's an honest reflection of Deliverable #3's accuracy ceiling (42.5% quantum / 75.8% classical on 50 classes), not a bug in this module's logic. Worth being upfront about in a demo: the threshold/tier mechanism is working correctly, but overall output quality is still bounded by the underlying model's accuracy.

## Known limitations

- **Sparse per-class calibration**: ~4-5 test samples per disease makes threshold calibration statistically fragile. This demonstrates the required mechanism; it is not a clinically validated threshold set.
- **Urgency tiers** (`disease_urgency.py`) are a hackathon-prototype judgment call from general medical knowledge, not a clinical triage protocol — needs expert review before real use, same caveat as every other medical-knowledge component built in this pipeline.
- **Overall prediction quality** is bounded by Deliverable #3's benchmark numbers. Improving the underlying models (more training epochs, more synthetic data, a deeper ansatz) directly improves what this module can responsibly promise.
- This module **never outputs a diagnosis** — every response carries an explicit disclaimer, by design, not as an afterthought.

## Next step (Deliverable #5)

`DecisionSupportEngine.assess()` is the exact function the Software Platform's `/api/predict`-style endpoint should call: feed it the AIKYA chat-extraction module's symptom dict, return its structured output (top candidates, risk tier, recommended action, disclaimer) to the UI's result dashboard.
