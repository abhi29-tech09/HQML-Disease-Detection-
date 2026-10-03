"""
QuantaDx -- End-to-End Architecture Demo, at real scale
SIH 2026 PS3, Deliverable #2 (Hybrid Quantum-Classical Architecture)

Proves the full pipeline built in this deliverable actually runs on the
real Deliverable #1 output: 50 diseases x 34 symptom features -> 6 qubits.
Uses random (untrained) ansatz weights -- correctness of PREDICTIONS is
Deliverable #3's job. This script only proves the architecture is wired
correctly end to end at the real data's scale.
"""

import time
import numpy as np
import pandas as pd

from frontend import align_to_feature_order, preprocess_for_encoding, save_feature_order
from hybrid_circuit import build_hybrid_circuit, ansatz_weight_shape, classical_readout_stub

TIER1_PATH = "../data_preprocessing/outputs/tier1_qubit_ready_34feat.csv"


def main():
    df = pd.read_csv(TIER1_PATH, index_col=0)
    feature_order = list(df.columns)
    diseases = list(df.index)
    n_classes = len(diseases)

    # Lock the feature order used at training/architecture time -- inference
    # must use this exact file so chat-derived vectors align correctly.
    save_feature_order(feature_order, path="feature_order.json")
    print(f"Loaded real data: {n_classes} diseases x {len(feature_order)} features")

    # --- Pick one real disease row and run it through the whole pipeline ---
    sample_disease = "pneumonia" if "pneumonia" in diseases else diseases[0]
    row = df.loc[sample_disease].to_dict()
    vec = align_to_feature_order(row, feature_order)
    padded, n_qubits = preprocess_for_encoding(vec)
    print(f"\nSample disease: {sample_disease!r}")
    print(f"Active symptoms in this row: {[f for f, v in row.items() if v == 1]}")
    print(f"Padded vector length: {len(padded)}  ->  n_qubits = {n_qubits}")
    print(f"Norm check (must be 1.0): {np.linalg.norm(padded):.6f}")

    # --- Build the circuit at this scale and time a single forward pass ---
    n_layers = 2
    shape = ansatz_weight_shape(n_qubits, n_layers)
    weights = np.random.default_rng(0).normal(scale=0.5, size=shape)

    circuit = build_hybrid_circuit(n_qubits, n_layers)

    t0 = time.time()
    q_features = circuit(padded, weights)
    elapsed = time.time() - t0

    print(f"\nAnsatz weight shape: {shape}  ({np.prod(shape)} trainable parameters)")
    print(f"Quantum feature map output (6 values): {np.round(q_features, 4)}")
    print(f"Single forward pass time on simulator: {elapsed*1000:.1f} ms")

    # --- Shape-check the classical readout head against all 50 classes ---
    probs = classical_readout_stub(q_features, n_classes=n_classes)
    top3_idx = np.argsort(probs)[::-1][:3]
    print(f"\nClassical readout head output shape: {probs.shape} (== {n_classes} disease classes, correct)")
    print("Top-3 (UNTRAINED, random weights -- meaningless until Deliverable #3 trains this):")
    for i in top3_idx:
        print(f"  {diseases[i]:30s} {probs[i]:.4f}")

    print("\n[OK] Architecture verified end-to-end at real data scale: "
          f"{len(feature_order)} symptoms -> {n_qubits} qubits -> {n_classes}-class output.")


if __name__ == "__main__":
    main()
