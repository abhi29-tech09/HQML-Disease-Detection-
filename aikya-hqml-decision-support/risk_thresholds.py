"""
QuantaDx -- Per-disease threshold tuning for sensitivity/specificity
SIH 2026 PS3, Deliverable #4 (Prediction & Decision Support Module)

A plain argmax prediction treats every disease the same way. In a health
screening context that's the wrong default: missing a possible emergency
(a false negative) is far worse than an unnecessary caution flag (a false
positive), while for a self-care-tier condition the opposite trade-off is
reasonable -- don't alarm someone over a common cold on shaky evidence.

This module computes a per-disease probability THRESHOLD, calibrated on
the Deliverable #3 held-out test set, biased by urgency tier:
  - emergency/urgent diseases: lower threshold (favor sensitivity/recall --
    flag it even on weaker evidence, so fewer true cases are missed)
  - routine/self_care diseases: higher threshold (favor specificity/
    precision -- require stronger evidence before flagging)

HONEST LIMITATION: the test set has only ~4-5 samples per disease (240
samples / 50 classes). Per-class threshold calibration on that few points
is statistically fragile -- this demonstrates the MECHANISM the Delivery
Table asks for ("threshold tuning for sensitivity/specificity"), not a
clinically validated threshold set. More synthetic (or eventually real)
samples per class would make this calibration meaningfully more stable.
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, "../architecture")
sys.path.insert(0, "../model_training")
sys.path.insert(0, "../data_preprocessing")

from frontend import preprocess_for_encoding  # noqa: E402
from train_hybrid import build_trainable_circuit, forward  # noqa: E402
from synthetic_patients import generate_synthetic_patients  # noqa: E402
from disease_urgency import URGENCY_TIER, TIER_SENSITIVITY_TARGET  # noqa: E402
from predict import QuantaDxPredictor, TIER1_PATH, N_PER_CLASS, SEED  # noqa: E402


def compute_thresholds():
    predictor = QuantaDxPredictor()
    diseases = predictor.diseases
    label_map = predictor.label_map
    n_classes = len(diseases)

    # Safety floor: with ~4-5 positive test samples per class, the recall-
    # driven search below can select a threshold barely above the random-
    # guess baseline (1/n_classes) -- which then fires on almost any input,
    # unrelated symptoms included (verified: this happened for pneumothorax
    # at threshold=0.021, which flagged a pure migraine case as an
    # emergency). No threshold is allowed below this floor, regardless of
    # what the sparse PR curve suggests, even for emergency-tier diseases.
    MIN_THRESHOLD_FLOOR = 3.0 / n_classes  # ~0.06 for 50 classes

    # Reproduce Deliverable #3's exact split so this calibrates on the same
    # held-out test set the benchmark report used -- no data leakage into
    # the thresholds from samples the model was trained on.
    df = pd.read_csv(TIER1_PATH, index_col=0)
    X_raw, y_labels = generate_synthetic_patients(df, n_per_class=N_PER_CLASS, seed=SEED)
    y_idx = np.array([label_map[d] for d in y_labels])
    _, X_test, _, y_test = train_test_split(
        X_raw, y_idx, test_size=0.2, stratify=y_idx, random_state=SEED
    )

    # Ensemble probability for every test sample, every class.
    proba_matrix = np.array([
        [predictor.predict_proba(
            {f: v for f, v in zip(predictor.feature_order, row) if v > 0}
        ).get(d, 0.0) for d in diseases]
        for row in X_test
    ])

    thresholds = {}
    for i, disease in enumerate(diseases):
        tier = URGENCY_TIER[disease]
        target_sensitivity = TIER_SENSITIVITY_TARGET[tier]
        y_true_binary = (y_test == i).astype(int)
        scores = proba_matrix[:, i]

        if y_true_binary.sum() == 0:
            # No positive examples for this class in the test split -- fall
            # back to the tier's target used directly as the threshold.
            fallback = max(1 - target_sensitivity, MIN_THRESHOLD_FLOOR)
            thresholds[disease] = {
                "tier": tier, "threshold": round(fallback, 4),
                "note": "no positive test examples; using tier default"
            }
            continue

        precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, scores)
        # Find the highest threshold that still achieves >= target recall.
        achievable = [(t, r) for t, r in zip(pr_thresholds, recall[:-1]) if r >= target_sensitivity]
        if achievable:
            chosen_threshold = max(achievable, key=lambda x: x[0])[0]
        else:
            # Target sensitivity unreachable with this few samples -- use
            # the lowest observed threshold so no positive case is missed.
            chosen_threshold = float(pr_thresholds.min()) if len(pr_thresholds) else MIN_THRESHOLD_FLOOR

        chosen_threshold = max(float(chosen_threshold), MIN_THRESHOLD_FLOOR)

        thresholds[disease] = {
            "tier": tier,
            "target_sensitivity": target_sensitivity,
            "threshold": round(float(chosen_threshold), 4),
            "n_positive_test_samples": int(y_true_binary.sum()),
        }

    return thresholds


if __name__ == "__main__":
    thresholds = compute_thresholds()
    with open("outputs/risk_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    for tier in ["emergency", "urgent", "routine", "self_care"]:
        print(f"\n--- {tier} ---")
        for d, info in thresholds.items():
            if info["tier"] == tier:
                print(f"  {d:28s} threshold={info['threshold']:.3f}")
