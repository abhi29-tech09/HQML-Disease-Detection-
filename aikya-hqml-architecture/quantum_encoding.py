"""
QuantaDx -- Quantum Encoding Layer + Backend Abstraction
SIH 2026 PS3, Deliverable #2 (Hybrid Quantum-Classical Architecture)

Maps a classical, L2-normalized, power-of-2-length vector onto a quantum
state via amplitude encoding, and provides a single place to swap
simulator <-> real quantum hardware without touching the circuit code
above it (the "backend-agnostic" design point from the pitch deck).
"""

import pennylane as qml


def get_device(n_qubits, backend="simulator", shots=None):
    """
    Single point of backend control.

    backend="simulator" (default): PennyLane's default.qubit simulator --
        free, fast, unlimited use, what this whole prototype runs on.
    backend="ibmq": routes to IBM Quantum hardware/cloud simulators via
        PennyLane-Qiskit, IF that plugin and IBM Quantum credentials are
        configured in this environment. Not wired up by default here --
        see README for what's needed to actually enable it.

    Everything downstream (the encoding + ansatz + measurement in
    hybrid_circuit.py) is written against this device object and does not
    know or care which backend it is.
    """
    if backend == "simulator":
        return qml.device("default.qubit", wires=n_qubits, shots=shots)

    if backend == "ibmq":
        raise NotImplementedError(
            "IBM Quantum hardware backend requires the pennylane-qiskit plugin "
            "and IBM Quantum API credentials, neither of which are configured "
            "in this environment. Install with `pip install pennylane-qiskit`, "
            "set IBMQ credentials, then swap this branch to: "
            "qml.device('qiskit.ibmq', wires=n_qubits, backend='ibmq_qasm_simulator', ...)."
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def encoding_circuit(padded_vector, n_qubits, device=None, backend="simulator"):
    """
    Returns a PennyLane QNode that amplitude-encodes padded_vector onto
    n_qubits and returns the per-qubit Z expectation values -- the
    "quantum feature map" output that Deliverable #3's classical readout
    head consumes.

    This circuit has NO trainable parameters yet -- it's pure encoding.
    Deliverable #3 inserts a variational ansatz between the embedding and
    the measurement (see hybrid_circuit.py).
    """
    dev = device or get_device(n_qubits, backend=backend)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(features=padded_vector, wires=range(n_qubits), normalize=False)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


if __name__ == "__main__":
    import numpy as np
    from frontend import align_to_feature_order, preprocess_for_encoding

    demo = {"fever": 1.0, "dry_cough": 0.8, "fatigue": 0.6}
    order = ["fever", "dry_cough", "fatigue", "rash", "joint_pain"]
    vec = align_to_feature_order(demo, order)
    padded, n_qubits = preprocess_for_encoding(vec)

    circuit = encoding_circuit(padded, n_qubits)
    result = circuit()
    print(f"Encoded {len(vec)} symptoms into {n_qubits} qubits.")
    print("Per-qubit <Z> expectation values:", np.round(result, 4))
    print()
    print("Circuit diagram:")
    print(qml.draw(circuit)())
