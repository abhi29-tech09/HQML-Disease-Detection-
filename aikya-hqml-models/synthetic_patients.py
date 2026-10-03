"""
QuantaDx -- Synthetic patient-symptom augmentation
SIH 2026 PS3, Deliverable #3 (Quantum ML Models)

WHY THIS EXISTS: tier1_qubit_ready_34feat.csv has exactly 50 rows -- one
DISEASE-LEVEL TEMPLATE per disease, not patient-level observations. A real
patient rarely reports every canonical symptom of their disease (someone
with pneumonia might only mention fever and cough, not all 7 template
symptoms), and training on exactly 1 sample per class would let a model
trivially "memorize" 50 points without learning anything about
partial/noisy real-world symptom reports -- which is exactly what the
chat-extraction module hands this model at inference time.

This module turns each disease's template into a distribution: for every
synthetic patient, each of that disease's true symptoms is INCLUDED
independently with probability keep_prob (simulating partial reporting),
and a small number of symptoms NOT in the template are added with
probability noise_prob (simulating incidental/comorbid mentions or minor
extraction noise). This is a synthetic-data design choice, not real patient
data -- documented here, not hidden, same as every other limitation in this
pipeline.
"""

import numpy as np
import pandas as pd


def generate_synthetic_patients(template_df, n_per_class=30, keep_prob=0.75,
                                  noise_prob=0.03, seed=42):
    """
    template_df: disease x symptom binary matrix (50 x 34), index = disease name.
    Returns (X, y): X is (n_per_class * n_diseases, n_features) binary array,
    y is matching disease-label array.
    """
    rng = np.random.default_rng(seed)
    features = template_df.columns.to_numpy()
    n_features = len(features)

    X_rows = []
    y_rows = []

    for disease, template_row in template_df.iterrows():
        template = template_row.to_numpy().astype(float)
        true_idx = np.where(template == 1)[0]
        false_idx = np.where(template == 0)[0]

        for _ in range(n_per_class):
            sample = np.zeros(n_features, dtype=float)

            # Each true symptom independently reported with prob keep_prob.
            keep_mask = rng.random(len(true_idx)) < keep_prob
            sample[true_idx[keep_mask]] = 1.0

            # A few off-template symptoms slip in with small probability.
            noise_mask = rng.random(len(false_idx)) < noise_prob
            sample[false_idx[noise_mask]] = 1.0

            # Guard against an all-zero sample (possible if keep_prob rolls
            # badly on a disease with few symptoms) -- force back the single
            # most-common true symptom rather than emit an empty case.
            if sample.sum() == 0 and len(true_idx) > 0:
                sample[true_idx[0]] = 1.0

            X_rows.append(sample)
            y_rows.append(disease)

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y


if __name__ == "__main__":
    df = pd.read_csv("../data_preprocessing/outputs/tier1_qubit_ready_34feat.csv", index_col=0)
    X, y = generate_synthetic_patients(df, n_per_class=5, seed=0)
    print(f"Generated {X.shape[0]} synthetic patients x {X.shape[1]} features "
          f"from {df.shape[0]} disease templates")
    print("All-zero rows (should be 0):", (X.sum(axis=1) == 0).sum())
    print("Example -- pneumonia template vs 3 synthetic patients:")
    print("  template:", df.loc["pneumonia"].to_numpy())
    mask = y == "pneumonia"
    print("  patient1:", X[mask][0].astype(int))
    print("  patient2:", X[mask][1].astype(int))
    print("  patient3:", X[mask][2].astype(int))
