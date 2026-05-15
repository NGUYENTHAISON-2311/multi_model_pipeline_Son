# Pipeline Explanation — Multi-Model Ensemble

This document explains the full processing flow of the pipeline, from data → features → training → ensemble → benchmark.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Loading & Feature Extraction](#2-data-loading--feature-extraction)
3. [Data Sampling & Train-Test Split](#3-data-sampling--train-test-split)
4. [Monte Carlo Training Loop](#4-monte-carlo-training-loop)
5. [Hyperparameter Optimization](#5-hyperparameter-optimization)
6. [Model Selection & Ensemble](#6-model-selection--ensemble)
7. [Benchmark — Padded Sliding Window](#7-benchmark--padded-sliding-window)
8. [Benchmark — Adaptive Multi-Scale](#8-benchmark--adaptive-multi-scale)
9. [Metrics](#9-metrics)
10. [Seed Management & Reproducibility](#10-seed-management--reproducibility)
11. [Modifications & Extensions](#11-modifications--extensions)

---

## 1. Overview

```
Input data (JSON/FASTA)
    │
    ▼
Feature extraction (540 dimensions)
    │
    ▼
┌─────────────────────────────────────────────┐
│ For each algorithm (6 algorithms):          │
│                                             │
│   ┌─────────────────────────────────────┐   │
│   │ Monte Carlo loop (30 iterations):   │   │
│   │   1. Random train/test split (70/30)│   │
│   │   2. Balance classes (downsample)   │   │
│   │   3. Train classifier               │   │
│   │   4. Evaluate on test set           │   │
│   │   5. Record metrics                 │   │
│   └─────────────────────────────────────┘   │
│                                             │
│   → Select model with highest F1            │
│   → Compute mean ± std across 30 runs       │
└─────────────────────────────────────────────┘
    │
    ▼
Ensemble: combine 6 best models
    │
    ▼
Benchmark: evaluate on independent test set
```

**6 default algorithms:**
AdaBoost, Random Forest, Extra Trees, Gradient Boosting, SVM, Decision Tree.

---

## 2. Data Loading & Feature Extraction

### 2.1. Loading

The pipeline accepts 2 types of input:
- **JSON**: file containing a list of records, each with a `"Sequence"` field
- **FASTA**: standard bioinformatics format (`>ID` + sequence lines)

Two datasets are required:
- **Positive**: sequences containing amyloid core regions
- **Negative**: disordered regions (non-amyloid)

### 2.2. 540 Features

Each sequence (or window) is converted into a 540-dimensional vector:

| # | Feature Group | Count | Description |
|---|---|---|---|
| 1 | Sequence length | 1 | Length of the amino acid sequence |
| 2 | AA frequency | 20 | Frequency of each of the 20 standard amino acids (count / length) |
| 3 | Dipeptide transitions | 400 | 20×20 transition matrix between amino acid pairs, row-normalized |
| 4 | Group frequency | 17 | Frequency across 3 group classification systems (4 + 6 + 7 groups) |
| 5 | Intra-class transitions | 101 | Transitions within the same classification system (4² + 6² + 7² = 16 + 36 + 49) |
| 6 | IUPred disorder score | 1 | Average disorder score from IUPred3 |

**Total: 1 + 20 + 400 + 17 + 101 + 1 = 540**

### 2.3. Amino Acid Frequency (20 features)

Count occurrences of each amino acid, divided by the sequence length:

```
freq(A) = count(A) / length
freq(R) = count(R) / length
...
freq(V) = count(V) / length
```

20 standard amino acids: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V.

### 2.4. Dipeptide Transitions (400 features)

A 20×20 matrix where each cell (i, j) = the number of times amino acid `i` is followed by amino acid `j`.

Row-normalized: each row is divided by its sum → each row sums to 1.0.

Example: if Alanine (A) appears 10 times and is followed by Glycine (G) 3 times → `dipeptide[A][G] = 3/10 = 0.3`.

The matrix is flattened into a 400-element vector.

### 2.5. Group Classification (17 + 101 features)

Amino acids are assigned to groups according to 3 classification systems:

**Classification 1 (4 groups):**
| Group | Amino Acids | Property |
|---|---|---|
| A | R, K, E, D, Q, N | Hydrophilic / Charged |
| B | G, A, S, T, P, H, Y | Polar / Small |
| C | C, V, L, I, M, F, W | Hydrophobic |
| X | X | Unknown |

**Classification 2 (6 groups):**
| Group | Amino Acids | Property |
|---|---|---|
| A | H, N, T, Q, C, S | Polar uncharged |
| B | K, R, E, D | Charged |
| C | I, L, M, V, W, Y, F, A | Hydrophobic |
| G | G | Glycine |
| P | P | Proline |
| X | X | Unknown |

**Classification 3 (7 groups):**
| Group | Amino Acids | Property |
|---|---|---|
| A | H, T, C, S | Small polar |
| B | K, R, E, D | Charged |
| C | I, L, M, V, W, Y, F, A | Hydrophobic |
| D | Q, N | Amide |
| G | G | Glycine |
| P | P | Proline |
| X | X | Unknown |

**Group frequency (17 features):** Frequency of each group within each classification system.

**Intra-class transitions (101 features):** Count transitions (group_i → group_j) within the same classification system, divided by (length - 1).
- Classification 1: 4² = 16 transitions
- Classification 2: 6² = 36 transitions
- Classification 3: 7² = 49 transitions

### 2.6. IUPred Score (1 feature)

IUPred3 (external tool) is run to compute disorder scores for the full protein containing the sequence.
The average disorder score of the residues within the sequence is taken → 1 float value.

---

## 3. Data Sampling & Train-Test Split

### 3.1. Sampling Process

Each Monte Carlo iteration performs:

```
Positive features (N_pos vectors, 540-dim each)
Negative features (N_neg vectors, 540-dim each)
         │
         ▼
    ┌─────────────────────────────────┐
    │ Sampling strategy = "min":      │
    │                                 │
    │ n = min(N_pos, N_neg)           │
    │ Randomly sample n from each     │
    │ class → Balanced dataset: n + n │
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │ Shuffle & split:                │
    │                                 │
    │ 70% → Train set                 │
    │ 30% → Test set                  │
    └─────────────────────────────────┘
```

### 3.2. Two Balancing Strategies

**`"min"` (default):**
- Downsample both classes to the size of the smaller class
- Example: 500 positive + 800 negative → sample 500 from each class → 1000 samples
- Train: 700, Test: 300

**`"ratio"`:**
- Keep all data
- Compute sample weights to balance:
  - `w_pos = n / (2 × n_pos)`
  - `w_neg = n / (2 × n_neg)`
- Each class contributes a total weight of 0.5

### 3.3. Why Random Splits?

Each iteration uses a different random seed → **different data split**. 30 iterations = 30 different train/test versions. This provides:
- An assessment of **model stability** (low std = stable model)
- Protection against **overfitting** to a single split
- A **mean** metric that reflects real-world performance better than a single split

---

## 4. Monte Carlo Training Loop

### 4.1. Detailed Flow

```
For each algorithm (e.g., Random Forest):
    best_f1 = 0
    best_model = None
    all_scores = []

    For i = 1 to 30 (MC iterations):
        1. run_seed = generate_from(base_seed, i)

        2. Create classifier:
           clf = RandomForestClassifier(
               n_estimators=100, max_depth=None, ...,
               random_state=run_seed
           )

        3. Sampling:
           - Downsample both classes → balanced dataset
           - Shuffle with run_seed
           - Split 70/30

        4. Train:
           clf.fit(X_train, y_train)

        5. Evaluate:
           y_pred = clf.predict(X_test)
           f1 = f1_score(y_test, y_pred)
           acc = accuracy_score(y_test, y_pred)
           pre = precision_score(y_test, y_pred)
           rec = recall_score(y_test, y_pred)
           mcc = matthews_corrcoef(y_test, y_pred)

        6. Track best:
           if f1 > best_f1:
               best_f1 = f1
               best_model = clf

        7. Record:
           all_scores.append({f1, acc, pre, rec, mcc, run_seed})

    Summary:
        mean_F1 ± std_F1
        mean_Accuracy ± std_Accuracy
        mean_Precision ± std_Precision
        mean_Recall ± std_Recall
        mean_MCC ± std_MCC
        mean_Average = mean(mean_F1, mean_Acc, mean_Pre, mean_Rec, mean_MCC)
```

### 4.2. Suppressed Warnings

Sklearn may raise warnings when precision/recall/F1 are undefined (division by zero). The pipeline suppresses warnings and uses `zero_division=0.0` to avoid display errors.

### 4.3. Parallel Training

When `--workers > 1`:
- Each algorithm runs in a separate process
- `ProcessPoolExecutor` manages the worker pool
- Each process has its own progress bar (stacked tqdm bars)
- Shared counters (via `multiprocessing.Manager`) update progress

---

## 5. Hyperparameter Optimization

### 5.1. Random Search

Instead of using fixed params from the config, the pipeline **tries many random parameter combinations** and selects the best one.

```
┌───────────────────────────────────────────────────┐
│ For algorithm = Random Forest:                    │
│                                                   │
│   Sample 10 random param combos:                  │
│     combo_1: {n_est=2500, max_depth=30, ...}      │
│     combo_2: {n_est=800, max_depth=None, ...}     │
│     combo_3: {n_est=4200, max_depth=75, ...}      │
│     ...                                           │
│                                                   │
│   For each combo:                                 │
│     Run 30 MC iterations (SAME data splits)       │
│     → mean_F1, mean_Acc, ...                      │
│                                                   │
│   Select combo with highest mean_F1               │
│   → best_params = combo_3.params                  │
│   → best_model = best model from combo_3          │
└───────────────────────────────────────────────────┘
```

Reference: Bergstra & Bengio (2012) — 60 random samples are sufficient to cover 95% of the top-5% parameter space.

### 5.2. Fair Comparison — Shared Seeds

**Problem:** If each combo uses different data splits, fair comparison is impossible.

**Solution:** All combos use **the same data splits** (shared seeds):

```
base_seed
    │
    ├→ Combo sampling RNG (selects random params)
    │
    └→ split_seed = base_seed XOR 0x5DEECE66D
         │
         └→ shared_seeds = [seed_1, seed_2, ..., seed_30]
              │
              ├→ combo_1: train 30 runs with shared_seeds
              ├→ combo_2: train 30 runs with shared_seeds (SAME splits)
              └→ combo_3: train 30 runs with shared_seeds (SAME splits)
```

XOR with the constant `0x5DEECE66D` (Java LCG constant) ensures split seeds are **decorrelated** from combo sampling seeds.

### 5.3. Search Spaces

Each algorithm has its own search space (per Table S3, Gonay et al. 2025):

| Algorithm | Searched Parameters |
|---|---|
| Random Forest | n_estimators (500–5000), max_depth (5–110 / null), max_features, min_samples_split, min_samples_leaf, bootstrap |
| Extra Trees | Same as Random Forest |
| Gradient Boosting | Same as above + min_impurity_decrease (0–10) |
| AdaBoost | n_estimators (1–500) |
| SVM | C (0.01–5.0), kernel (linear/poly/rbf/sigmoid) |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf, max_features, criterion, splitter |
| CatBoost | n_estimators, max_depth (0–16), min_child_samples, bootstrap_type |

### 5.4. Deduplication

When sampling random combos, if a combo has already been tried → skip and resample. Maximum `20 × n_combos` attempts. Uses a `set` for O(1) duplicate checking.

---

## 6. Model Selection & Ensemble

### 6.1. Per-Algorithm Selection

Each algorithm retains **1 best model** — the model with the **highest F1** across 30 iterations.

```
Algorithm       Best F1    Best Model
─────────────────────────────────────
AdaBoost        0.82       clf_ada_17
Random Forest   0.87       clf_rf_23
Extra Trees     0.89       clf_et_05
Gradient Boost  0.86       clf_gb_11
SVM             0.84       clf_svm_29
Decision Tree   0.78       clf_dt_02
```

### 6.2. Ensemble Creation

The 6 best models are combined into a single `EnsembleClassifier`:

```python
ensemble = EnsembleClassifier(
    models=[clf_ada, clf_rf, clf_et, clf_gb, clf_svm, clf_dt],
    weights=[0.82, 0.87, 0.89, 0.86, 0.84, 0.78],  # F1 scores
    mode="soft_voting",  # or "weighted_voting", "best_model"
)
```

### 6.3. Three Ensemble Modes

**Soft Voting (default):**

Average probability across all models:

$$P(y=1|x) = \frac{1}{K}\sum_{k=1}^{K} P_k(y=1|x)$$

Where $K = 6$ models. Each model contributes equally regardless of performance.

**Weighted Voting:**

Weighted average, with weights = normalized F1 scores:

$$P(y=1|x) = \sum_{k=1}^{K} w_k \cdot P_k(y=1|x), \quad \sum w_k = 1$$

Models with higher F1 → greater influence on the final decision.

**Best Model:**

Only uses the model with the highest F1. All other models are ignored.

### 6.4. Prediction

```
Input: feature vector x (540-dim)
    │
    ▼
Each model computes probability: P_k(y=1|x)
    │
    ▼
Aggregate by mode → P_ensemble(y=1|x)
    │
    ▼
If P_ensemble(y=1|x) > 0.5 → predict 1 (amyloid)
If P_ensemble(y=1|x) ≤ 0.5 → predict 0 (non-amyloid)
```

---

## 7. Benchmark — Padded Sliding Window

### 7.1. Concept

Given a protein sequence, determine which residues belong to the amyloid core.

**Sliding window:** Slide a window of size $T$ (default 17) across the sequence, each position → 1 prediction score.

**Problem with standard sliding window:** Residues at the start/end of the sequence are covered by fewer windows than residues in the middle → bias.

**Solution: Padding.** Add special sequences to both ends so that every original residue is covered by exactly $T$ windows.

### 7.2. Padding

```
Original sequence:    ───────── S₁S₂S₃...Sₙ ─────────
                               ↑              ↑
                         first residue   last residue

Padded sequence:      [LEFT_PAD]S₁S₂S₃...Sₙ[RIGHT_PAD]
                      ←─ T-1 ──→              ←─ T-1 ──→
```

**Left padding:** A sequence ending with the same residue as $S_1$ (first amino acid).
**Right padding:** A sequence starting with the same residue as $S_n$ (last amino acid).

Example: if $S_1$ = A and $T$ = 17 → left pad = 16 amino acids, ending with A.

Padding sequences are pre-selected for each amino acid (lookup table), ensuring biological compatibility.

### 7.3. Scoring

```
Padded sequence: [pad]ABCDEFGH[pad]
Window size T = 5

Window 1: [pad₁ pad₂ pad₃ pad₄ A]    → score₁
Window 2: [pad₂ pad₃ pad₄ A B]        → score₂
Window 3: [pad₃ pad₄ A B C]           → score₃
Window 4: [pad₄ A B C D]              → score₄
Window 5: [A B C D E]                 → score₅
Window 6: [B C D E F]                 → score₆
...

For residue A (original position 1):
    Covered by windows 1, 2, 3, 4, 5 → exactly T=5 windows
    avg_score(A) = (score₁ + score₂ + score₃ + score₄ + score₅) / 5

For residue D (original position 4):
    Covered by windows 4, 5, 6, 7, 8 → exactly T=5 windows
    avg_score(D) = (score₄ + score₅ + score₆ + score₇ + score₈) / 5
```

### 7.4. Thresholding

$$\text{pred}(i) = \begin{cases} 1 & \text{if } \text{avg\_score}(i) > k \\ 0 & \text{otherwise} \end{cases}$$

Where $k = 0.5$ (default). Compare `pred` vs `true_labels` → compute metrics.

### 7.5. Full Benchmark Flow

```
For each sequence in the benchmark set:
    │
    ├─ 1. Pad sequence (add T-1 residues on each side)
    │
    ├─ 2. Slide window across padded sequence
    │     For each window position:
    │         Extract 540 features
    │         score = ensemble.predict_proba(features)[class=1]
    │
    ├─ 3. Compute average score for each original residue
    │     avg_score[i] = mean(T window scores covering residue i)
    │
    ├─ 4. Threshold: pred[i] = 1 if avg_score[i] > k else 0
    │
    ├─ 5. Compare with true labels
    │     (from annotated core regions)
    │
    └─ 6. Compute metrics: TP, TN, FP, FN → Precision, Recall, F1, Accuracy, MCC, SOV

Global metrics = average of per-sequence metrics
```

---

## 8. Benchmark — Adaptive Multi-Scale

### 8.1. Motivation

The padded benchmark (section 7) uses a **fixed** window size $T$ for all
positions.  However, training data contains sequences of **variable length**
(typically 11–25 residues).  A fixed window of 18 always produces feature
vectors computed from exactly 18 residues — even when the true amyloid core
is shorter or longer.

The adaptive benchmark solves this by trying multiple window lengths when the
model is **uncertain**, while keeping the start position fixed and only
changing the **end** position (extending or shortening the window).

### 8.2. Algorithm

```
For each start position (sliding +1, same as padded benchmark):
    │
    ├─ 1. Score with default window size (e.g. 18):
    │     score_default = predict_proba(window[start : start + 18])
    │
    ├─ 2. If |score_default − 0.5| ≥ confidence_margin:
    │     → Model is CONFIDENT → use score_default, skip exploration
    │
    └─ 3. If |score_default − 0.5| < confidence_margin:
          → Model is UNCERTAIN → keep start fixed, try other end positions:
            For each size S ∈ {11, 12, ..., 25} \ {18}:
                score_S = predict_proba(window[start : start + S])
            │
            → Select the size where |score − 0.5| is LARGEST
            → Use that score (may be confident positive OR negative)

    Accumulate chosen score for all residues covered by the chosen window.

Final: score[i] = sum(overlapping scores) / count(overlapping windows)
Threshold: pred[i] = 1 if score[i] > k else 0
```

### 8.3. End-Only Extension

Only the **end position** of the window changes.  The start position is the
same as in the standard padded benchmark (slides by +1 each step).

```
Start fixed at position p:

Default:    [p ─────────── p+17]        size 18
Shortened:  [p ────── p+10]             size 11
Extended:   [p ────────────────── p+24] size 25

  ← start fixed          end varies →
```

This design reflects the biological question: "from this starting point,
how far does the amyloid core extend?"  The start position provides the
spatial sampling; the end position provides the scale sampling.

### 8.4. Two-Directional Fairness

The selection criterion is **maximum confidence** (|score − 0.5|), not
maximum score.  This means:

- A score of 0.92 (confident positive) beats a score of 0.55 (weak positive)
- A score of 0.08 (confident negative) also beats a score of 0.55
- A score of 0.51 (uncertain) loses to both

This prevents the one-directional bias that would occur if we simply took
the highest score (which would always favour positive predictions).

### 8.5. IUPred Precomputation

IUPred is the main computational bottleneck (external call per window).
The adaptive benchmark precomputes IUPred scores **once** for the entire
maximally-padded sequence, then uses a lookup table for each window:

```
1. Pad sequence with max_window−1 residues on each side
2. iupred_scores = IUPred(full_padded_sequence)      ← 1 call
3. For any window [start : start+S]:
     iupred_mean = mean(iupred_scores[start : start+S])  ← O(S) lookup
```

The other 539 features (AA freq, dipeptides, etc.) are pure computation
and fast.  As a result, the adaptive benchmark adds **no extra IUPred calls**
regardless of how many window sizes are tried.

### 8.6. Padding

```
Padded sequence:  [LEFT_PAD] S₁S₂...Sₙ [RIGHT_PAD]
                  ←─ T-1 ──→            ←─ Tmax-1 ─→

Left pad  = default_window − 1  (same start positions as padded benchmark)
Right pad = max_window − 1      (room for extended windows)
```

### 8.7. Confidence Margin

The `confidence_margin` parameter (default 0.15) controls the trade-off:

| Margin | Uncertain band | Behaviour |
|---|---|---|
| 0.05 | [0.45, 0.55] | Rarely explores — almost same as fixed-size |
| 0.15 | [0.35, 0.65] | Moderate exploration (default) |
| 0.30 | [0.20, 0.80] | Aggressive — explores most positions |
| 0.50 | [0.00, 1.00] | Always explores all sizes (full multi-scale) |

### 8.8. Comparison: Padded vs Adaptive

| Property | Padded (section 7) | Adaptive (this section) |
|---|---|---|
| Window size per position | Fixed $T$ | $T$ default, $[T_{min}, T_{max}]$ when uncertain |
| Windows per residue | Exactly $T$ | Variable (depends on chosen sizes) |
| IUPred calls per sequence | $N + T - 1$ | **1** (precomputed) |
| Scale sensitivity | Single scale only | Multi-scale where needed |
| Computational cost | Baseline | Baseline + extra features for uncertain windows |
| Score bias | None | None (two-directional confidence) |

---

## 9. Metrics

### 9.1. Training Metrics

| Metric | Formula | Meaning |
|---|---|---|
| F1 Score | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of Precision and Recall |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Proportion of correct predictions |
| Precision | $\frac{TP}{TP + FP}$ | Of all predicted positives, how many are correct |
| Recall | $\frac{TP}{TP + FN}$ | Of all actual positives, how many are found |
| MCC | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Balanced metric, range [-1, 1] |

**Average metric:** the mean of all 5 metrics above (used when `--metric Average`).

### 9.2. Benchmark Metrics

In addition to the 5 metrics above, the benchmark also computes **SOV** (Segment Overlap):

$$SOV = \frac{1}{N_{true}} \sum_{\text{true segments}} \frac{\text{overlap} + \delta}{\text{union}} \times \text{length}_{obs}$$

Where $\delta$ rewards near-correct predictions (partial overlap), bounded by segment size. SOV evaluates prediction quality at the **region** level rather than per-residue.

---

## 10. Seed Management & Reproducibility

### 10.1. Seed Tree

```
config.training.random_seed (base_seed)
    │
    ├── data_rng = Random(base_seed)
    │   ├── per_run_seed[1] = data_rng.randint(0, 2³¹-1)
    │   ├── per_run_seed[2] = data_rng.randint(0, 2³¹-1)
    │   └── ...per_run_seed[30]
    │
    └── [Optimize mode only]
        ├── combo_rng = Random(base_seed)
        │   └── sample N random param combos
        │
        └── split_seed = base_seed XOR 0x5DEECE66D
            └── split_rng = Random(split_seed)
                ├── shared_seed[1]
                ├── shared_seed[2]
                └── ...shared_seed[30]
                    (shared across ALL combos)
```

### 10.2. Why XOR?

The combo sampling RNG and split RNG use the same `base_seed` → they may be correlated. XOR with the large constant `0x5DEECE66D` produces a **decorrelated** seed, ensuring that the combo sampling order does not affect data splits.

### 10.3. Traceability

| Information | Stored In |
|---|---|
| Base seed | `metadata.json` |
| Per-run seed | `scores.csv` (column `run_seed`) |
| Best model seed | `summary.json` (field `best_seed`) |
| Combo params & scores | `summary.json` (field `combos_tried`) |

When `random_seed = null` in config: each run uses a random seed → different results each time.
When `random_seed = 42`: results are identical across runs (reproducible).

---

---

## 11. Modifications & Extensions

This section documents all changes made on top of the original pipeline.

---

### 11.1. ESM2 Embeddings as a Feature Token

**Files:** `scripts/train_ensemble_esm2.py`, `esm2_embedding.py`, `feature_builder.py`

ESM2 (Evolutionary Scale Modeling 2, Meta AI) is a protein language model that produces
per-residue contextual embeddings. A new training script `train_ensemble_esm2.py` exposes
`esm2` as a first-class token in the `--features` argument, on equal footing with `builtin`
and `aaindex`.

#### Feature token system

```
--features builtin              → 540-dim handcrafted vector only
--features aaindex              → 304-dim AAindex means only
--features esm2                 → ESM2 embeddings only (default 64-dim after PCA)
--features builtin aaindex esm2 → all three concatenated (540 + 304 + 64 = 908-dim)
```

Any JSON feature spec path can also be mixed in.

#### ESM2 processing pipeline

```
sequence (variable length)
    → ESM2 model (facebook/esm2_t12_35M_UR50D by default)
    → per-residue hidden states: (seq_len, hidden_size=480)
    → pooling: (hidden_size,)
         mean  — average over all non-padding token positions (default)
         max   — element-wise maximum over non-padding positions
         cls   — [CLS] token representation
    → PCA: (hidden_size,) → (esm2_dim,)   [default: 64; set --esm2-dim 0 to skip]
    → concatenate with other features → classifier
```

#### PCA reducer

PCA is fitted on the **combined positive + negative** training set to avoid data leakage
from fitting on a single class. The fitted reducer is saved to
`<run_dir>/esm2_pca_reducer.pkl` alongside the model PKLs so it can be reused at
inference time.

#### Pre-computed embeddings

ESM2 inference is the most time-consuming step. Embeddings can be pre-computed once
and reused across multiple training runs:

```bash
python esm2_embedding.py -i pos.json -o pos_emb.pt
python esm2_embedding.py -i neg.json -o neg_emb.pt
python scripts/train_ensemble_esm2.py \
    --features builtin esm2 \
    --embeddings-pos pos_emb.pt \
    --embeddings-neg neg_emb.pt
```

#### Feature caching

To avoid recomputing all features after a crash, the built feature matrices can be
saved to disk with `--cache-dir`:

```bash
# First run: computes and saves
python scripts/train_ensemble_esm2.py --features builtin esm2 \
    --cache-dir .feature_cache/run1 ...

# Retry after crash: loads instantly, skips ESM2 / IUPred / PCA
python scripts/train_ensemble_esm2.py --features builtin esm2 \
    --cache-dir .feature_cache/run1 ...
```

Saved files: `pos_matrix.npy`, `neg_matrix.npy`, `pca_reducer.pkl`, `feature_info.json`.

---

### 11.2. Cython-Accelerated Feature Extraction

**Files:** `src/_feature_fast.pyx`, `setup_cython.py`

The three most-called inner loops in `feature_pipeline.py` were rewritten as Cython
extensions. The key property is that the hot C loops run **with the GIL released**
(`with nogil:`), allowing multiple Python threads to execute them truly in parallel
when `ThreadPoolExecutor` trains several algorithms concurrently.

#### Functions

| Cython function | Replaces | Speedup |
|---|---|---|
| `aa_frequencies(sequence, amino_acids)` | `compute_amino_acid_frequencies()` | ~3× |
| `dipeptide_transitions(sequence, amino_acids)` | `compute_dipeptide_aa_transitions()` | ~8× |
| `group_transitions(encoded, groups)` | `compute_cross_classification_transitions()` (intra-class) | ~6× |

All three use ASCII-indexed C arrays and `malloc`/`free` for zero-overhead memory
allocation. The GIL is re-acquired only when building the Python return list.

#### Build

```bash
pip install cython numpy
python setup_cython.py build_ext --inplace
# → produces src/_feature_fast.cpython-*.so
```

#### Transparent fallback

`feature_pipeline.py` imports the extension with a try/except:

```python
try:
    from ._feature_fast import aa_frequencies as _fast_aa_freq, ...
    _CYTHON = True
except ImportError:
    _CYTHON = False
```

If the `.so` is absent (extension not built), all functions silently fall back to the
original pure-Python implementations with no change in behaviour.

---

### 11.3. Training Pipeline Optimizations

**File:** `src/training_pipeline.py`

#### 11.3.1. ThreadPoolExecutor replaces ProcessPoolExecutor

The original parallel execution used `ProcessPoolExecutor`, which **pickles and
duplicates** the entire feature matrices (`pos_features`, `neg_features`) into each
worker process. With 6 algorithms × large matrices, this caused out-of-memory errors
and segmentation faults.

`ThreadPoolExecutor` replaces it. Threads **share the same memory**, so the feature
matrices exist only once regardless of how many algorithms run concurrently. sklearn's
C extensions release the GIL during `clf.fit()`, so true parallelism is preserved for
the computationally expensive parts.

```
Before (ProcessPoolExecutor):
    main process: pos_features (200 MB)
    worker 1:     pos_features (200 MB)   ← copy
    worker 2:     pos_features (200 MB)   ← copy
    ...
    Total RAM: 200 MB × (1 + N_algos)

After (ThreadPoolExecutor):
    shared memory: pos_features (200 MB)
    all threads read the same object
    Total RAM: 200 MB × 1
```

#### 11.3.2. n_jobs forwarded to tree classifiers

`create_classifier()` now accepts an `n_jobs` parameter forwarded to
`RandomForestClassifier` and `ExtraTreesClassifier`. These algorithms use OpenMP
internally and scale linearly with the number of cores.

```
Sequential mode (--workers 1):  n_jobs = all CPUs → single algo uses all cores
Parallel mode   (--workers N):  n_jobs = CPUs // N → total usage ≈ all CPUs
```

This provides a free 4-8× speedup on tree-based algorithms with no code change needed
in the calling scripts.

#### 11.3.3. Incremental saving (crash recovery)

Previously, the run directory was created **after** all algorithms finished. A crash
midway through training meant zero files saved.

Now the run directory is created **before** the training loop starts. After each
algorithm completes, its results are immediately written to disk:

```
outputs/training/run_YYYYMMDD_HHMMSS/
└── per_algorithm/
    ├── adaboost/          ← written as soon as adaboost finishes
    │   ├── best_model.pkl
    │   ├── scores.csv
    │   └── summary.json
    ├── random_forest/     ← written when RF finishes
    ...
```

If training crashes after 4 out of 6 algorithms, the first 4 are already saved on disk.

---

### 11.4. TensorBoard Logging

**File:** `scripts/train_ensemble_esm2.py`

After training completes, `_write_tb_logs()` reads the saved per-algorithm CSVs and
metadata and emits a full TensorBoard event log to `<run_dir>/tensorboard/`.

#### Logged quantities

| Tag | What it shows |
|---|---|
| `features/total_dim` | Total feature vector dimension |
| `features/pipeline_dim` | Dimension from builtin/aaindex features |
| `features/esm2_raw_dim` | ESM2 hidden size before PCA |
| `features/esm2_pca_explained_variance` | Fraction of variance retained by PCA |
| `features/config` (text) | ESM2 model name, pool strategy, PCA components |
| `run/title` (text) | Human-readable run label (from `--title`) |
| `<algo>/fold/<metric>` | Per-fold F1 / Accuracy / Precision / Recall / MCC (step = fold index) |
| `<algo>/summary/mean_<metric>` | Cross-fold mean for each metric |
| `<algo>/summary/std_<metric>` | Cross-fold standard deviation |
| `<algo>/combo/<metric>` | Hyperparameter search score per combo (step = combo index) |

#### Usage

```bash
python scripts/train_ensemble_esm2.py \
    --features builtin esm2 --title "builtin+esm2_pca64" ...

tensorboard --logdir outputs/training/run_YYYYMMDD_HHMMSS/tensorboard
```

Pass `--no-tensorboard` to disable logging.

---

### 11.5. Run Labelling

**File:** `scripts/train_ensemble_esm2.py`

The `--title` argument adds a human-readable label to a run. It is displayed as a
banner at the start of training and again in the completion summary, and recorded as
a text tag in TensorBoard.

```
════════════════════════════════════════════════════
  Run: builtin+esm2_seed20_pca64
════════════════════════════════════════════════════

...training...

════════════════════════════════════════════════════
  Training completed — 3 ensemble variant(s)
  Title  : builtin+esm2_seed20_pca64
  Run ID : run_20260513_142301
════════════════════════════════════════════════════
```

---

### 11.6. Summary of New Files

| File | Purpose |
|---|---|
| `scripts/train_ensemble_esm2.py` | ESM2-aware training script (replaces `train_ensemble.py` for ESM2 runs) |
| `esm2_embedding.py` | Standalone ESM2 embedding tool; saves `.pt` checkpoints |
| `feature_builder.py` | Combines src/ pipeline features + ESM2 into a single `.npy` matrix |
| `src/_feature_fast.pyx` | Cython inner loops for feature extraction (GIL-free) |
| `setup_cython.py` | Build script for the Cython extension |
| `classify_benchmark_sets.py` | Sequence-level classification for benchmark sets 1–10 |

### 11.7. Summary of Modified Files

| File | Change |
|---|---|
| `src/training_pipeline.py` | ThreadPoolExecutor, n_jobs, incremental saving, early run_dir creation |
| `src/feature_pipeline.py` | Cython import with pure-Python fallback; Cython used in hot paths |

---

## References

- Gonay V, et al. Developing machine-learning-based amyloidogenicity predictors
  with Cross-Beta DB. Alzheimer's Dement. 2025; 1-7.
  https://doi.org/10.1002/alz.14510
- Bergstra J, Bengio Y. Random Search for Hyper-Parameter Optimization.
  JMLR. 2012; 13:281-305.
