"""
QuantaDx -- Classical Front-End
SIH 2026 PS3, Deliverable #2 (Hybrid Quantum-Classical Architecture)

Takes a raw symptom feature vector (from either the Tier-1 training matrix
or a live AIKYA chat-extraction result) and prepares it for quantum
amplitude encoding:
  1. Validate & align to the locked feature order from training
  2. Handle missing/unknown values
  3. L2-normalize to a unit vector (amplitude encoding requires this --
     the sum of squared amplitudes of a quantum state must equal 1)
  4. Zero-pad to the next power of 2 (2^n_qubits basis states)
"""

import json
import numpy as np

FEATURE_ORDER_PATH = "feature_order.json"


def save_feature_order(columns, path=FEATURE_ORDER_PATH):
    """Lock the exact feature order used at training time. Inference MUST
    use this same order, or the quantum state encodes the wrong meaning
    for each amplitude."""
    with open(path, "w") as f:
        json.dump(list(columns), f, indent=2)


def load_feature_order(path=FEATURE_ORDER_PATH):
    with open(path) as f:
        return json.load(f)


def align_to_feature_order(raw_symptom_dict, feature_order):
    """
    raw_symptom_dict: {canonical_name: confidence_0_to_1} -- e.g. straight
    from the AIKYA chat-extraction module's SymptomRecord.confidence, or a
    plain 0/1 presence flag from the training matrix.
    Missing symptoms (not mentioned at all) default to 0 -- absence of
    evidence, not evidence of absence, but 0 is the correct encoding input
    since we have nothing to place non-zero amplitude on.
    """
    return np.array([float(raw_symptom_dict.get(f, 0.0)) for f in feature_order], dtype=float)


def next_power_of_2(n):
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


def preprocess_for_encoding(vector, n_qubits=None):
    """
    Returns (normalized_padded_vector, n_qubits_used).

    Raises ValueError on an all-zero input rather than silently dividing
    by zero -- an all-zero symptom vector has no information to encode,
    and the caller (the chat UI) should ask for at least one symptom
    before submitting, not receive a meaningless uniform-superposition
    prediction.
    """
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(
            "All-zero symptom vector -- nothing to encode. "
            "The caller should collect at least one symptom before calling the model."
        )

    normalized = vector / norm

    target_len = next_power_of_2(len(normalized)) if n_qubits is None else 2 ** n_qubits
    if target_len < len(normalized):
        raise ValueError(
            f"n_qubits={n_qubits} gives only {target_len} amplitudes, "
            f"but the feature vector has {len(normalized)} entries."
        )

    padded = np.zeros(target_len, dtype=float)
    padded[: len(normalized)] = normalized
    # Padding zeros don't need re-normalizing -- they contribute 0 to the
    # sum of squares, so the vector is still unit-norm.

    n_qubits_used = int(np.log2(target_len))
    return padded, n_qubits_used


if __name__ == "__main__":
    # Smoke test with a toy vector.
    demo = {"fever": 1.0, "dry_cough": 0.8, "fatigue": 0.6}
    order = ["fever", "dry_cough", "fatigue", "rash", "joint_pain"]
    vec = align_to_feature_order(demo, order)
    padded, nq = preprocess_for_encoding(vec)
    print("aligned vector:", vec)
    print("padded (len", len(padded), "):", padded)
    print("norm check (should be 1.0):", np.linalg.norm(padded))
    print("qubits needed:", nq)
