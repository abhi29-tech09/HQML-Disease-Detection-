# QuantaDx — Deliverable #1: Data Pre-processing & Feature Engineering Module

SIH 2026, Problem Statement 3 — *Hybrid Quantum Machine Learning Platform for Early Disease Detection*

Maps directly to row 1 of the Delivery Table: *"Pipeline for handling biomedical data — data cleaning, normalization, dimensionality reduction, feature selection, handling of missing/noisy data."*

## Why two tiers

The dataset built earlier (`aikya_symptom_disease_matrix.xlsx`, HPO-derived) turned out, on inspection, to have **near-zero coverage of the actual common/lung diseases** this problem statement targets — plain "asthma" and "COPD" had 0 features in the main matrix and only 1–2 in the full uncapped association set, because HPO is built for rare/genetic disease diagnosis, not everyday clinical presentation. Full reasoning and the specific numbers are in the earlier conversation; this module acts on that finding rather than ignoring it.

- **Tier 1 — `curated_common_disease_symptoms.py`**: 50 common/lung diseases mapped to their real clinical symptoms, built from general medical knowledge. This is the actual training target for the classifier.
- **Tier 2 — the uploaded HPO matrix**: kept as a secondary rare-disease phenotype layer, cleaned and compressed, not used as the primary classifier's training data.

## Pipeline (`preprocess_pipeline.py`)

### Tier 1 — cleaning → feature selection → qubit-ready reduction
1. **Build**: 50 diseases × 110 canonical symptoms, wide binary matrix. Symptom names use the same `canonical_name` vocabulary as the AIKYA chat-extraction schema, so this dataset and live chat-derived symptom vectors are compatible end-to-end.
2. **Clean**: assert no missing values (binary by construction); guard against any disease row that ends up empty.
3. **Feature selection**: drop near-universal symptoms (present in >80% of diseases — no discriminative value). *Earlier draft also dropped disease-unique symptoms on the theory they "don't generalize" — that was wrong and got corrected: a symptom that appears in only one disease (e.g. `slurred_speech` → stroke) is often the single strongest identifier for it, not noise.*
4. **Rank by mutual information** (not an ad hoc frequency heuristic) between each symptom and the disease label; keep the top candidates.
5. **De-duplication safety net**: after the top-K cut, check whether any two diseases collapsed to an identical row (this happened during development — gout and osteoarthritis became indistinguishable once their disambiguating joint-symptom features were cut). Greedily add back whichever excluded feature best splits each colliding pair, repeat until every disease has a unique fingerprint.
6. **Result**: 34 features, 0 duplicate rows, 0 empty rows, 6 qubits needed under amplitude encoding (`⌈log₂(34)⌉`).

### Tier 2 — cleaning → dimensionality reduction
1. **Clean**: drop the 1,426 all-zero disease rows (diseases with no features at all in this matrix) — 15,496 → 14,070 usable rows.
2. **Reduce**: TruncatedSVD (works directly on sparse binary data, unlike PCA) to 20 components.
3. **Honest result**: only **25% of variance retained** at 20 components — checked up to 100 components, which only reaches 50.6%. This data is genuinely high-dimensional and diffuse (15,496 near-unique disease "classes"), which is exactly why it's positioned as a secondary signal, not the primary classifier input.

## Output files (`outputs/`)

| File | Contents |
|---|---|
| `tier1_full_matrix.csv` | 50 × 110 raw curated matrix, before feature selection |
| `tier1_selected_features.csv` | 50 × 110 after near-universal-symptom removal (0 dropped this run) |
| `tier1_qubit_ready_34feat.csv` | **Final training input** — 50 × 34, ready for quantum encoding |
| `tier1_mutual_information_ranking.json` | Every symptom's MI score, for transparency/audit |
| `tier2_svd_embedding_20d.csv` | 14,070 × 20 compressed HPO embedding (secondary layer) |
| `preprocessing_report.json` | Machine-readable summary of every step above |

## Known limitations (documented, not hidden)

- Tier 1 is hand-curated from general medical knowledge, not a clinical cohort — **needs a qualified clinician's review before any real-world use**, same caveat as the earlier chat-extraction module.
- 50 diseases is a small class count for a "detect any disease" story — it's honest scope for a hackathon prototype, framed as v1 of a system meant to expand.
- Tier 2's 25% variance retention at qubit-feasible component counts is a real limitation, not a rounding error — it's presented in the pitch as a rare-disease *context* signal, not a claim of full rare-disease coverage.

## Next steps (Deliverables #2–#3)

`tier1_qubit_ready_34feat.csv` is the direct input to the Hybrid Quantum-Classical Architecture: 34 binary features → 6 qubits via amplitude encoding (or use `tier1_selected_features.csv` directly with angle encoding, 1 qubit per feature, if the circuit design calls for that instead).
