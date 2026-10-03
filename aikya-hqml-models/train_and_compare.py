"""
QuantaDx -- Deliverable #3: Quantum ML Models + Classical Benchmark
SIH 2026 PS3.

Trains:
  1. Classical baselines (Random Forest, Logistic Regression) on the raw
     34-dim symptom features.
  2. The hybrid quantum model (amplitude encoding + trainable variational
     ansatz + classical readout head) from Deliverable #2.

Both on IDENTICAL train/test splits of the same synthetic-patient dataset,
for a fair benchmark -- matching the Delivery Table's explicit requirement:
"Benchmark the hybrid approach against classical models in terms of
accuracy, computational efficiency, and generalization performance."
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, "../architecture")
from frontend import preprocess_for_encoding  # noqa: E402
from synthetic_patients import generate_synthetic_patients  # noqa: E402
from train_hybrid import train_hybrid_model, predict_batch  # noqa: E402

SEED = 42
TIER1_PATH = "../data_preprocessing/outputs/tier1_qubit_ready_34feat.csv"
N_PER_CLASS = 24
N_QUBITS = 6
N_LAYERS = 2
EPOCHS = 18
BATCH_SIZE = 24
LR = 0.08


def main():
    print("=== Loading Tier 1 disease templates ===")
    df = pd.read_csv(TIER1_PATH, index_col=0)
    diseases = sorted(df.index)
    n_classes = len(diseases)
    label_map = {d: i for i, d in enumerate(diseases)}

    print(f"=== Generating synthetic patients ({N_PER_CLASS} per disease) ===")
    X_raw, y_labels = generate_synthetic_patients(df, n_per_class=N_PER_CLASS, seed=SEED)
    y_idx = np.array([label_map[d] for d in y_labels])
    print(f"Total synthetic patients: {X_raw.shape[0]}, features: {X_raw.shape[1]}, classes: {n_classes}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_idx, test_size=0.2, stratify=y_idx, random_state=SEED
    )
    print(f"Train: {X_train_raw.shape[0]}, Test: {X_test_raw.shape[0]} (stratified by disease)")

    results = {}

    # ---------------------------------------------------------- Classical baselines
    print("\n=== Classical baseline: Random Forest ===")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    rf.fit(X_train_raw, y_train)
    rf_train_time = time.time() - t0
    t0 = time.time()
    rf_preds = rf.predict(X_test_raw)
    rf_inference_time = (time.time() - t0) / len(X_test_raw)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    print(f"Random Forest: acc={rf_acc:.4f}  macro-F1={rf_f1:.4f}  "
          f"train_time={rf_train_time:.3f}s  inference={rf_inference_time*1000:.3f}ms/sample")
    results["random_forest"] = {
        "accuracy": rf_acc, "macro_f1": rf_f1,
        "train_time_sec": rf_train_time, "inference_ms_per_sample": rf_inference_time * 1000,
        "n_parameters": sum(t.tree_.node_count for t in rf.estimators_),
    }

    print("\n=== Classical baseline: Logistic Regression ===")
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=2000, random_state=SEED)
    lr_model.fit(X_train_raw, y_train)
    lr_train_time = time.time() - t0
    t0 = time.time()
    lr_preds = lr_model.predict(X_test_raw)
    lr_inference_time = (time.time() - t0) / len(X_test_raw)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro")
    print(f"Logistic Regression: acc={lr_acc:.4f}  macro-F1={lr_f1:.4f}  "
          f"train_time={lr_train_time:.3f}s  inference={lr_inference_time*1000:.3f}ms/sample")
    results["logistic_regression"] = {
        "accuracy": lr_acc, "macro_f1": lr_f1,
        "train_time_sec": lr_train_time, "inference_ms_per_sample": lr_inference_time * 1000,
        "n_parameters": lr_model.coef_.size + lr_model.intercept_.size,
    }

    # ---------------------------------------------------------- Hybrid quantum model
    print(f"\n=== Hybrid quantum model: {N_QUBITS} qubits, {N_LAYERS} ansatz layers, "
          f"{EPOCHS} epochs ===")
    X_train_padded = np.array([preprocess_for_encoding(x, n_qubits=N_QUBITS)[0] for x in X_train_raw])
    X_test_padded = np.array([preprocess_for_encoding(x, n_qubits=N_QUBITS)[0] for x in X_test_raw])

    ansatz_w, W, b, history, circuit, hybrid_train_time = train_hybrid_model(
        X_train_padded, y_train, n_qubits=N_QUBITS, n_classes=n_classes,
        n_layers=N_LAYERS, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED,
        X_val_padded=X_test_padded, y_val_idx=y_test, verbose=True,
    )

    t0 = time.time()
    hybrid_preds = predict_batch(circuit, X_test_padded, ansatz_w, W, b)
    hybrid_inference_time = (time.time() - t0) / len(X_test_padded)
    hybrid_acc = accuracy_score(y_test, hybrid_preds)
    hybrid_f1 = f1_score(y_test, hybrid_preds, average="macro")
    n_quantum_params = int(np.prod(ansatz_w.shape)) + W.size + b.size

    print(f"\nHybrid quantum model: acc={hybrid_acc:.4f}  macro-F1={hybrid_f1:.4f}  "
          f"train_time={hybrid_train_time:.1f}s  inference={hybrid_inference_time*1000:.3f}ms/sample")
    results["hybrid_quantum"] = {
        "accuracy": hybrid_acc, "macro_f1": hybrid_f1,
        "train_time_sec": hybrid_train_time, "inference_ms_per_sample": hybrid_inference_time * 1000,
        "n_parameters": n_quantum_params,
        "n_qubits": N_QUBITS, "n_ansatz_layers": N_LAYERS,
        "training_history": history,
    }

    # ---------------------------------------------------------- Save everything
    with open("outputs/benchmark_report.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez("outputs/hybrid_model_weights.npz",
             ansatz_weights=np.array(ansatz_w), W=np.array(W), b=np.array(b))

    with open("outputs/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    report_text = classification_report(y_test, hybrid_preds,
                                          target_names=diseases, zero_division=0)
    with open("outputs/hybrid_classification_report.txt", "w") as f:
        f.write(report_text)

    print("\n=== DONE ===")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "training_history"}
                       for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
