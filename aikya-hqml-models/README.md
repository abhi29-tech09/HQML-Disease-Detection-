# QuantaDx — Deliverable #3: Quantum ML Models + Classical Benchmark

SIH 2026, Problem Statement 3 — *Hybrid Quantum Machine Learning Platform for Early Disease Detection*

Maps to row 3 of the Delivery Table (*Quantum Machine Learning Models*) and directly fulfils the Expected Solution's explicit requirement to *"benchmark the hybrid approach against classical models in terms of accuracy, computational efficiency, and generalization performance."*

## Honest headline result

**The classical baselines beat the hybrid quantum model on this run, on every metric except parameter count.** That's the real number, not a rounded-up one — presenting it straight is more defensible to judges than a suspiciously perfect result would be, and it's also a well-documented, expected finding in current quantum ML research (see Discussion below), not a sign the pipeline is broken.

| Model | Accuracy | Macro-F1 | Train time | Inference/sample | Parameters |
|---|---|---|---|---|---|
| Random Forest | 69.6% | 67.2% | 0.39s | 0.09ms | 148,176 |
| Logistic Regression | **75.8%** | **74.0%** | **0.03s** | **0.001ms** | 1,750 |
| Hybrid Quantum (6 qubits) | 42.5% | 40.1% | 705.9s (~11.8 min) | 8.6ms | **386** |

## Methodology

1. **Synthetic patient generation** (`synthetic_patients.py`): the 50 disease *templates* from Deliverable #1 aren't patient data — one row per disease can't train a 50-class classifier meaningfully (a nearest-neighbor lookup would "solve" it trivially without learning anything generalizable). Each disease's true symptoms are independently included per synthetic patient with 75% probability (simulating partial real-world reporting), plus a 3% chance per off-template symptom (simulating incidental/noisy mentions) — 1,200 synthetic patients total (24 per disease), stratified 80/20 into 960 train / 240 test.
2. **Identical split** for all three models — the only fair way to compare them.
3. **Classical baselines**: scikit-learn Random Forest (200 trees) and Logistic Regression, trained directly on the 34 raw features.
4. **Hybrid quantum model**: Deliverable #2's architecture (amplitude encoding → `StronglyEntanglingLayers` ansatz, 2 layers → classical linear readout), trained jointly via PennyLane's native autograd with backprop differentiation (no PyTorch — this environment ran out of disk space installing it; PennyLane's own autograd is a standard, lighter-weight choice for a circuit this small). Adam optimizer, 18 epochs, batch size 24, lr 0.08, 386 total trainable parameters (36 ansatz + 350 readout).

## Discussion — why the quantum model underperforms here

This isn't a bug; it's the expected outcome for this kind of problem on today's tools, and worth explaining rather than hiding:

- **No inherent quantum advantage for generic tabular classification.** Quantum ML's theoretical edge comes from data with quantum-native structure (certain kernel/feature-space problems) or genuinely exponential classical-simulation cost. Binary symptom presence/absence data has neither — classical models are not handicapped here the way they would be on, say, quantum chemistry data.
- **386 parameters vs. 1,750–148,176.** The quantum model is working with roughly 5–400x fewer trainable parameters. A fairer classical comparison would be a logistic regression or small MLP capped at ~400 parameters, which would likely also score lower than the full models above — parameter-matched comparison is a natural next experiment.
- **18 epochs is shallow.** Per-sample Python-loop training (no batched circuit evaluation in this PennyLane version's `AmplitudeEmbedding`) makes each epoch expensive (~39s/epoch on 960 samples), which capped how long training could run in this environment. The loss was still decreasing, just slowly and noisily (val accuracy bounced between 30–45% rather than climbing smoothly) — more epochs would likely help.
- **Simple ansatz.** 2 layers of `StronglyEntanglingLayers` on 6 qubits is a modest-capacity circuit. Deeper ansätze, different entangling patterns, or data re-uploading (repeating the encoding between ansatz layers) are standard ways to add expressivity — at the cost of more parameters and, on real hardware, more noise.
- **Per-class pattern supports this read** (`hybrid_classification_report.txt`): diseases with *distinctive* symptom combinations classified well — allergic rhinitis (F1 0.89), ARDS (0.80), gout (0.71), dengue (0.67) — while diseases whose symptoms overlap heavily with many others did poorly — common cold, hypertension, and tuberculosis all scored F1 0.00. That's a data-separability problem more than a quantum-vs-classical problem; the classical models show the same underlying difficulty, just partly compensated by having far more parameters to fit decision boundaries with.

## What this means for the pitch

Presented honestly, this is a **stronger** hackathon story than a faked win: *"we built a real benchmarking harness, ran a fair head-to-head comparison, and found the expected NISQ-era result — quantum models today trade accuracy for a large reduction in parameter count, and the gap is a known open research question, not a flaw in this implementation."* Judges evaluating a "hybrid quantum ML" problem statement should find a team that understands and reports this tradeoff more credible than one claiming an implausible quantum win on tabular data with a 6-qubit simulator.

## Output files (`outputs/`)

| File | Contents |
|---|---|
| `benchmark_report.json` | Full metrics for all three models, machine-readable |
| `hybrid_model_weights.npz` | Trained ansatz weights + readout `W`, `b` |
| `hybrid_classification_report.txt` | Per-disease precision/recall/F1 |
| `label_map.json` | Disease name → class index mapping used by the trained model |

## Concrete next steps to close the gap

1. **More epochs** — loss was still decreasing at epoch 18; this just needs more wall-clock time than this environment's single run comfortably allowed.
2. **Parameter-matched classical baseline** — cap logistic regression at ~400 parameters for a fairer comparison.
3. **Deeper ansatz / data re-uploading** — more expressivity, testable cheaply on the simulator before ever touching real hardware.
4. **Batched circuit evaluation** if a PennyLane version/device supports it here — would directly cut the ~39s/epoch bottleneck and allow far more epochs in the same wall-clock budget.
