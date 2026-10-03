"""
QuantaDx -- Data Pre-processing & Feature Engineering Module
SIH 2026, Problem Statement 3 (Hybrid Quantum ML for Early Disease Detection)
Deliverable #1: "Pipeline for handling biomedical data -- data cleaning,
normalization, dimensionality reduction, feature selection, handling of
missing/noisy data."

Two-tier design (see accompanying README.md for the reasoning):
  Tier 1 -- curated_common_disease_symptoms.py
            50 common/lung diseases x their real clinical symptoms.
            This is the actual training target for the classifier, since
            the HPO matrix (Tier 2) turned out to have near-zero coverage
            of these exact diseases (see README).
  Tier 2 -- the uploaded aikya_symptom_disease_matrix.xlsx (HPO-derived,
            15,496 diseases x 683 features). Kept as a secondary rare-
            disease phenotype layer: cleaned and compressed into a compact
            embedding, not used as the primary classifier's training data.

Output: two qubit-feasible, cleaned, documented feature sets ready for the
quantum encoding step (Deliverable #2/#3).
"""

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from curated_common_disease_symptoms import DISEASE_SYMPTOMS

RNG_SEED = 42
HPO_XLSX_PATH = "/mnt/user-data/uploads/1790766816754_aikya_symptom_disease_matrix.xlsx"
OUT_DIR = "outputs"

np.random.seed(RNG_SEED)


# ============================================================== TIER 1
def build_tier1_matrix():
    """Curated common/lung-disease symptom matrix -> clean wide binary table."""
    diseases = sorted(DISEASE_SYMPTOMS.keys())
    all_symptoms = sorted(set().union(*DISEASE_SYMPTOMS.values()))

    df = pd.DataFrame(0, index=diseases, columns=all_symptoms, dtype=int)
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        df.loc[disease, sorted(symptoms)] = 1
    df.index.name = "disease"

    # --- Cleaning / handling of noisy data ---
    # No missing values possible by construction (binary presence/absence),
    # but guard anyway in case this pipeline is later fed a messier source.
    assert df.isna().sum().sum() == 0, "unexpected missing values in Tier 1 matrix"
    # Drop any disease row that ended up with zero symptoms (data-entry guard).
    empty_rows = df.index[df.sum(axis=1) == 0].tolist()
    if empty_rows:
        print(f"[tier1] dropping {len(empty_rows)} empty disease rows: {empty_rows}")
        df = df.drop(index=empty_rows)

    return df


def select_tier1_features(df, max_disease_fraction=0.8):
    """
    Feature selection for the classifier's primary feature set.

    Only drop near-universal symptoms (present in > max_disease_fraction of
    diseases) -- these add no discriminative signal.

    IMPORTANT: earlier versions of this function also dropped
    "disease-unique" symptoms (present in only 1 disease) on the theory
    that they don't generalize. That was backwards for disease-level
    template data like this: a symptom that appears in exactly one disease
    (e.g. slurred_speech -> stroke, sudden_joint_pain -> gout) is often the
    single strongest identifier for that disease, not noise. Keeping them
    and using mutual information for the final ranking (below) handles
    this correctly instead of discarding them by a blunt frequency rule.
    """
    n_diseases = len(df)
    counts = df.sum(axis=0)
    keep = counts[counts <= max_disease_fraction * n_diseases].index
    dropped = sorted(set(df.columns) - set(keep))
    print(f"[tier1] feature selection: {len(df.columns)} -> {len(keep)} symptoms "
          f"(dropped {len(dropped)} near-universal, non-discriminative symptoms)")
    return df[sorted(keep)], dropped


def qubit_ready_tier1(df_selected, top_k=32):
    """
    Rank remaining symptoms by mutual information with the disease label
    and keep the top_k -- a statistically grounded discriminative-power
    ranking rather than an ad hoc frequency heuristic. With amplitude
    encoding this maps to ceil(log2(top_k)) qubits instead of one qubit
    per feature.

    Safety net: a pure top-k-by-MI cut can still leave two diseases with
    identical rows in the reduced feature set (this happened with
    gout/osteoarthritis during development -- both lost their
    disambiguating joint-symptom features). After the initial cut, greedily
    add back whichever remaining feature best splits each duplicate group,
    so no two diseases are left indistinguishable to the classifier.
    """
    from sklearn.feature_selection import mutual_info_classif

    X = df_selected.values
    y = df_selected.index.values  # one class per disease row
    mi = mutual_info_classif(X, y, discrete_features=True, random_state=RNG_SEED)
    mi_series = pd.Series(mi, index=df_selected.columns).sort_values(ascending=False)

    selected = list(mi_series.index[:top_k])
    remaining = list(mi_series.index[top_k:])

    def duplicate_groups(cols):
        sub = df_selected[cols]
        dup_mask = sub.duplicated(keep=False)
        if not dup_mask.any():
            return []
        groups = {}
        for idx, row in sub[dup_mask].iterrows():
            key = tuple(row.values)
            groups.setdefault(key, []).append(idx)
        return list(groups.values())

    guard = 0
    while True:
        groups = duplicate_groups(selected)
        if not groups or not remaining or guard > 50:
            break
        for group in groups:
            # Find the remaining feature (by MI rank) that best splits this group.
            for cand in remaining:
                vals = df_selected.loc[group, cand]
                if vals.nunique() > 1:
                    selected.append(cand)
                    remaining.remove(cand)
                    break
        guard += 1

    if guard > 0:
        print(f"[tier1] de-duplication safety net added {len(selected) - top_k} extra "
              f"feature(s) to separate diseases that would otherwise collide: "
              f"{selected[top_k:]}")

    reduced = df_selected[sorted(selected)]
    n_qubits_amplitude = int(np.ceil(np.log2(len(selected))))
    print(f"[tier1] final feature set: {len(selected)} symptoms "
          f"-> {n_qubits_amplitude} qubits via amplitude encoding")

    mi_report = sorted(zip(df_selected.columns, mi), key=lambda x: -x[1])
    return reduced, n_qubits_amplitude, mi_report


# ============================================================== TIER 2
def load_and_clean_tier2(path):
    df = pd.read_excel(path, sheet_name="Matrix")
    df = df.set_index("disease")

    # --- Cleaning / handling of noisy or sparse rows ---
    empty_rows = df.index[df.sum(axis=1) == 0]
    print(f"[tier2] loaded {df.shape[0]} diseases x {df.shape[1]} features; "
          f"dropping {len(empty_rows)} all-zero disease rows")
    df = df.loc[df.sum(axis=1) > 0]

    empty_cols = df.columns[df.sum(axis=0) == 0]
    if len(empty_cols):
        print(f"[tier2] dropping {len(empty_cols)} all-zero feature columns")
        df = df.drop(columns=empty_cols)

    return df


def reduce_tier2(df, n_components=20):
    """
    TruncatedSVD (works directly on sparse/binary data, unlike PCA which
    assumes centered continuous data) to compress the 683-feature HPO
    matrix into a compact embedding for use as a secondary rare-disease
    context signal -- not as the primary classifier's input.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=RNG_SEED)
    embedding = svd.fit_transform(df.values)
    explained = svd.explained_variance_ratio_.sum()
    print(f"[tier2] TruncatedSVD to {n_components} components: "
          f"{explained:.1%} variance retained")

    emb_df = pd.DataFrame(
        embedding,
        index=df.index,
        columns=[f"svd_{i+1}" for i in range(n_components)]
    )
    return emb_df, explained, svd.explained_variance_ratio_


# ============================================================== MAIN
def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Tier 1: curated common/lung disease matrix ===")
    t1_raw = build_tier1_matrix()
    t1_selected, dropped = select_tier1_features(t1_raw)
    t1_reduced, n_qubits, mi_report = qubit_ready_tier1(t1_selected, top_k=32)

    t1_raw.to_csv(f"{OUT_DIR}/tier1_full_matrix.csv")
    t1_selected.to_csv(f"{OUT_DIR}/tier1_selected_features.csv")
    t1_reduced.to_csv(f"{OUT_DIR}/tier1_qubit_ready_{t1_reduced.shape[1]}feat.csv")
    with open(f"{OUT_DIR}/tier1_mutual_information_ranking.json", "w") as f:
        json.dump([{"symptom": s, "mutual_information": round(float(v), 5)} for s, v in mi_report], f, indent=2)

    print("\n=== Tier 2: HPO matrix (secondary, rare-disease layer) ===")
    t2_clean = load_and_clean_tier2(HPO_XLSX_PATH)
    t2_embedding, explained, ratios = reduce_tier2(t2_clean, n_components=20)
    t2_embedding.to_csv(f"{OUT_DIR}/tier2_svd_embedding_20d.csv")

    report = {
        "tier1": {
            "diseases": t1_raw.shape[0],
            "raw_symptom_features": t1_raw.shape[1],
            "after_feature_selection": t1_selected.shape[1],
            "dropped_features": dropped,
            "qubit_ready_feature_count": t1_reduced.shape[1],
            "qubits_needed_amplitude_encoding": n_qubits,
            "qubits_needed_angle_encoding_1_per_feature": t1_reduced.shape[1],
        },
        "tier2": {
            "diseases_after_cleaning": t2_clean.shape[0],
            "features_after_cleaning": t2_clean.shape[1],
            "svd_components": t2_embedding.shape[1],
            "variance_retained": round(float(explained), 4),
            "per_component_variance": [round(float(r), 4) for r in ratios],
        },
    }
    with open(f"{OUT_DIR}/preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Done. Outputs written to", OUT_DIR, "===")
    print(json.dumps(report["tier1"], indent=2))
    print(json.dumps({k: v for k, v in report["tier2"].items() if k != "per_component_variance"}, indent=2))


if __name__ == "__main__":
    main()
