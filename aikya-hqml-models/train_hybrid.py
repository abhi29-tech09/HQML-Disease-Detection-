"""
QuantaDx -- Hybrid Quantum Model Training
SIH 2026 PS3, Deliverable #3 (Quantum ML Models)

Trains the ansatz weights + classical readout head from Deliverable #2's
hybrid_circuit.py jointly, using PennyLane's native autograd (pennylane.numpy)
with backprop differentiation through the simulator. No PyTorch dependency
-- this environment ran out of disk space installing it, and PennyLane's
own autograd interface is a completely standard, lighter-weight choice for
a circuit this small (6 qubits, 36 ansatz parameters).
"""

import sys
import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, "../architecture")
from quantum_encoding import get_device  # noqa: E402


def softmax_cross_entropy(logits, label_idx):
    # Numerically stable log-softmax, autograd-compatible (pnp ops only).
    shifted = logits - pnp.max(logits)
    log_probs = shifted - pnp.log(pnp.sum(pnp.exp(shifted)))
    return -log_probs[label_idx]


def build_trainable_circuit(n_qubits, n_layers):
    dev = get_device(n_qubits, backend="simulator")

    @qml.qnode(dev, diff_method="backprop")
    def circuit(padded_vector, ansatz_weights):
        qml.AmplitudeEmbedding(features=padded_vector, wires=range(n_qubits), normalize=False)
        qml.StronglyEntanglingLayers(ansatz_weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def forward(circuit, padded_vector, ansatz_weights, W, b):
    q_features = pnp.stack(circuit(padded_vector, ansatz_weights))
    logits = W @ q_features + b
    return logits


def train_hybrid_model(X_train_padded, y_train_idx, n_qubits, n_classes,
                        n_layers=2, epochs=15, batch_size=16, lr=0.05, seed=0,
                        X_val_padded=None, y_val_idx=None, verbose=True):
    """
    X_train_padded: (n_samples, 2**n_qubits) unit-norm padded vectors.
    y_train_idx: integer class labels, 0..n_classes-1.
    Returns (ansatz_weights, W, b, history).
    """
    rng = np.random.default_rng(seed)
    circuit = build_trainable_circuit(n_qubits, n_layers)

    ansatz_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    ansatz_weights = pnp.array(rng.normal(scale=0.1, size=ansatz_shape), requires_grad=True)
    W = pnp.array(rng.normal(scale=0.3, size=(n_classes, n_qubits)), requires_grad=True)
    b = pnp.array(np.zeros(n_classes), requires_grad=True)

    def batch_cost(ansatz_weights, W, b, X_batch, y_batch):
        total = 0.0
        for xi, yi in zip(X_batch, y_batch):
            logits = forward(circuit, xi, ansatz_weights, W, b)
            total = total + softmax_cross_entropy(logits, yi)
        return total / len(X_batch)

    opt = qml.AdamOptimizer(stepsize=lr)
    n_train = len(X_train_padded)
    history = {"epoch": [], "train_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            X_batch = X_train_padded[idx]
            y_batch = y_train_idx[idx]

            cost_fn = lambda aw, w, bb: batch_cost(aw, w, bb, X_batch, y_batch)  # noqa: E731
            (ansatz_weights, W, b), loss_val = opt.step_and_cost(cost_fn, ansatz_weights, W, b)
            epoch_loss += float(loss_val)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        val_acc = None
        if X_val_padded is not None:
            preds = predict_batch(circuit, X_val_padded, ansatz_weights, W, b)
            val_acc = float(np.mean(preds == y_val_idx))

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            msg = f"epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}"
            if val_acc is not None:
                msg += f"  val_acc={val_acc:.3f}"
            print(msg)

    total_time = time.time() - t0
    return ansatz_weights, W, b, history, circuit, total_time


def predict_batch(circuit, X_padded, ansatz_weights, W, b):
    preds = []
    for xi in X_padded:
        logits = forward(circuit, xi, ansatz_weights, W, b)
        preds.append(int(np.argmax(logits)))
    return np.array(preds)


if __name__ == "__main__":
    # Tiny smoke test: 3 classes, 8 qubits' worth of padding not needed --
    # use n_qubits=2 (4-dim) for a fast sanity check before the real run.
    rng = np.random.default_rng(0)
    n_qubits = 2
    n_classes = 3
    X = rng.normal(size=(30, 2 ** n_qubits))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.integers(0, n_classes, size=30)

    aw, W, b, hist, circuit, t = train_hybrid_model(
        X, y, n_qubits=n_qubits, n_classes=n_classes, n_layers=1,
        epochs=3, batch_size=8, verbose=True
    )
    print(f"\nSmoke test completed in {t:.2f}s -- training loop is wired correctly.")
