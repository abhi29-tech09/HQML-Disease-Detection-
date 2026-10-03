"""
QuantaDx -- Full Hybrid Quantum-Classical Circuit (untrained skeleton)
SIH 2026 PS3, Deliverable #2 (Hybrid Quantum-Classical Architecture)

encoding (fixed, no parameters) -> variational ansatz (trainable) -> measurement

This is the architecture. Deliverable #3 trains the ansatz weights on
tier1_qubit_ready_34feat.csv; this module just proves the shapes and the
circuit run correctly with randomly-initialized weights, and gives
Deliverable #3 the exact function signature to import and train against.
"""

import numpy as np
import pennylane as qml

from quantum_encoding import get_device


def ansatz_weight_shape(n_qubits, n_layers=2):
    """StronglyEntanglingLayers weight shape: (n_layers, n_qubits, 3)."""
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


def build_hybrid_circuit(n_qubits, n_layers=2, backend="simulator", device=None):
    """
    Returns a QNode: hybrid_circuit(padded_vector, weights) -> list of
    per-qubit <Z> expectation values.

    padded_vector: classical front-end output (unit-norm, length 2**n_qubits).
    weights: trainable ansatz parameters, shape = ansatz_weight_shape(n_qubits, n_layers).
    """
    dev = device or get_device(n_qubits, backend=backend)

    @qml.qnode(dev)
    def circuit(padded_vector, weights):
        # --- Encoding: classical data -> quantum state (fixed, no training here) ---
        qml.AmplitudeEmbedding(features=padded_vector, wires=range(n_qubits), normalize=False)
        # --- Variational ansatz: this is what Deliverable #3 trains ---
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # --- Measurement: quantum feature map output for the classical readout head ---
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def classical_readout_stub(quantum_features, n_classes, rng=None):
    """
    Placeholder classical readout head: quantum expectation values -> class
    logits. Deliverable #3 replaces this random-weight stub with a properly
    trained linear layer (or small MLP) as part of the end-to-end training
    loop -- this function exists here only to prove the full shape chain
    (qubits -> classical head -> class probabilities) is wired correctly.
    """
    rng = rng or np.random.default_rng(42)
    n_qubits = len(quantum_features)
    W = rng.normal(scale=0.5, size=(n_classes, n_qubits))
    b = rng.normal(scale=0.1, size=n_classes)
    logits = W @ np.array(quantum_features) + b
    probs = np.exp(logits) / np.exp(logits).sum()
    return probs


if __name__ == "__main__":
    from frontend import align_to_feature_order, preprocess_for_encoding

    demo = {"fever": 1.0, "dry_cough": 0.8, "fatigue": 0.6}
    fake_order = ["fever", "dry_cough", "fatigue", "rash", "joint_pain"]
    vec = align_to_feature_order(demo, fake_order)
    padded, n_qubits = preprocess_for_encoding(vec)

    n_layers = 2
    shape = ansatz_weight_shape(n_qubits, n_layers)
    weights = np.random.default_rng(0).normal(size=shape)

    circuit = build_hybrid_circuit(n_qubits, n_layers)
    q_features = circuit(padded, weights)
    print(f"n_qubits={n_qubits}, ansatz weight shape={shape}")
    print("quantum feature map output:", np.round(q_features, 4))

    probs = classical_readout_stub(q_features, n_classes=5)
    print("stub class probabilities (untrained, for shape-check only):", np.round(probs, 4))
