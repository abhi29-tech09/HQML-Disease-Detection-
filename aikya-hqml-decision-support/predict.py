"""
QuantaDx -- Prediction module
SIH 2026 PS3, Deliverable #4 (Prediction & Decision Support Module)

Loads the trained hybrid quantum model (Deliverable #3) and retrains the
classical baseline (fast, <1s, same seed/data as Deliverable #3 for
reproducibility) to provide a model-agnostic predict_proba() interface.

Default output is an ENSEMBLE (simple average) of both models' calibrated
probabilities. This is a deliberate, documented choice: Deliverable #3's
benchmark showed the classical model currently outperforms the quantum one
on this data, so an ensemble that leans on both -- rather than quantum
alone -- is the more honest choice for anything resembling real decision
support, while still keeping the quantum model genuinely in the loop
rather than sidelined.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Resolve every path relative to THIS FILE's location, not the caller's
# working directory -- predict.py gets imported from other directories
# (e.g. the platform/api.py backend), and a cwd-relative path silently
# breaks the moment that happens instead of failing loudly where it's
# obvious. Caught exactly this bug while wiring up Deliverable #5.
_THIS_DIR = Path(__file__).resolve().parent
ARCH_DIR = _THIS_DIR / "../architecture"
TRAIN_DIR = _THIS_DIR / "../model_training"
TIER1_PATH = _THIS_DIR / "../data_preprocessing/outputs/tier1_qubit_ready_34feat.csv"

sys.path.insert(0, str(ARCH_DIR))
sys.path.insert(0, str(_THIS_DIR / "../data_preprocessing"))
sys.path.insert(0, str(TRAIN_DIR))

from frontend import align_to_feature_order, preprocess_for_encoding, load_feature_order  # noqa: E402
from train_hybrid import build_trainable_circuit, forward  # noqa: E402
from synthetic_patients import generate_synthetic_patients  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

N_QUBITS = 6
N_LAYERS = 2
SEED = 42
N_PER_CLASS = 24


class QuantaDxPredictor:
    def __init__(self):
        self.feature_order = load_feature_order(str(ARCH_DIR / "feature_order.json"))
        with open(TRAIN_DIR / "outputs/label_map.json") as f:
            self.label_map = json.load(f)
        self.diseases = sorted(self.label_map, key=lambda d: self.label_map[d])
        self.n_classes = len(self.diseases)

        weights = np.load(TRAIN_DIR / "outputs/hybrid_model_weights.npz")
        self.ansatz_weights = weights["ansatz_weights"]
        self.W = weights["W"]
        self.b = weights["b"]
        self.circuit = build_trainable_circuit(N_QUBITS, N_LAYERS)

        self.classical_model = self._retrain_classical_baseline()

    def _retrain_classical_baseline(self):
        """
        Deliverable #3 didn't persist the sklearn model object (only its
        metrics), and it trains in under a second -- so we reproduce it
        deterministically here (identical seed + data generation) rather
        than adding a second serialization path to keep in sync.
        """
        df = pd.read_csv(TIER1_PATH, index_col=0)
        X_raw, y_labels = generate_synthetic_patients(df, n_per_class=N_PER_CLASS, seed=SEED)
        y_idx = np.array([self.label_map[d] for d in y_labels])
        model = LogisticRegression(max_iter=2000, random_state=SEED)
        model.fit(X_raw, y_idx)
        return model

    def _quantum_proba(self, feature_vector):
        padded, _ = preprocess_for_encoding(feature_vector, n_qubits=N_QUBITS)
        logits = forward(self.circuit, padded, self.ansatz_weights, self.W, self.b)
        logits = np.array(logits)
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def _classical_proba(self, feature_vector):
        return self.classical_model.predict_proba(feature_vector.reshape(1, -1))[0]

    def predict_proba(self, symptom_dict, method="ensemble"):
        """
        symptom_dict: {canonical_name: confidence_0_to_1}, e.g. straight from
        the AIKYA chat-extraction module's SymptomRecord data.
        method: "ensemble" (default), "quantum", or "classical".
        Returns: dict {disease_name: probability}, sums to 1.0.
        """
        vec = align_to_feature_order(symptom_dict, self.feature_order)
        if vec.sum() == 0:
            raise ValueError("No recognized symptoms in input -- cannot predict on an empty vector.")

        q_proba = self._quantum_proba(vec)
        c_proba = self._classical_proba(vec)

        if method == "quantum":
            proba = q_proba
        elif method == "classical":
            proba = c_proba
        elif method == "ensemble":
            proba = (q_proba + c_proba) / 2
        else:
            raise ValueError(f"Unknown method: {method!r}")

        return {self.diseases[i]: float(proba[i]) for i in range(self.n_classes)}


if __name__ == "__main__":
    predictor = QuantaDxPredictor()
    demo_symptoms = {
        "fever": 1.0, "productive_cough": 0.9, "chest_pain": 0.6,
        "shortness_of_breath": 0.7, "fatigue": 0.8,
    }
    for method in ["ensemble", "quantum", "classical"]:
        proba = predictor.predict_proba(demo_symptoms, method=method)
        top5 = sorted(proba.items(), key=lambda x: -x[1])[:5]
        print(f"\n[{method}] top 5:")
        for disease, p in top5:
            print(f"  {disease:30s} {p:.4f}")
