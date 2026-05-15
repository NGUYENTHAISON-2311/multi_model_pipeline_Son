# Hyperparameters vs Learned Parameters

For each algorithm, **hyperparameters** are configured before training (in
`default_config.json` or via random search), while **learned parameters** are
discovered automatically by `.fit(X_train, y_train)`.

---

## AdaBoost

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `n_estimators` — number of weak learners | Weight $\alpha_t$ of each weak learner |
| `algorithm` — boosting algorithm (`SAMME`) | Each weak learner's decision stump (feature index, threshold) |
| `learning_rate` — shrinkage per iteration | Sample weights $w_i$ at each boosting stage |

**How `.fit()` works:** Iteratively trains weak classifiers (decision stumps).
After each stump, misclassified samples get higher weight, forcing the next
stump to focus on hard examples. The final prediction is a weighted vote of
all stumps.

---

## Random Forest

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `n_estimators` — number of trees | Each tree's full structure (nodes, splits, leaves) |
| `max_depth` — maximum tree depth | Split feature and threshold at each node |
| `max_features` — features considered per split | Leaf values (class distribution) |
| `min_samples_split` — min samples to split a node | Bootstrap sample selection per tree |
| `min_samples_leaf` — min samples in a leaf | |
| `bootstrap` — whether to bootstrap samples | |

**How `.fit()` works:** Builds `n_estimators` independent decision trees, each
trained on a random bootstrap sample of the data. At each node, only
`max_features` random features are evaluated for the best split. Prediction =
majority vote (or average probability) across all trees.

---

## Extra Trees

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `n_estimators` — number of trees | Each tree's full structure |
| `max_depth` — maximum tree depth | Split feature and **random** threshold at each node |
| `max_features` — features considered per split | Leaf values (class distribution) |
| `min_samples_split` — min samples to split a node | |
| `min_samples_leaf` — min samples in a leaf | |
| `bootstrap` — whether to bootstrap samples | |

**How `.fit()` works:** Like Random Forest, but with an extra source of
randomness: instead of finding the *best* threshold for each candidate feature,
it picks a **random** threshold. This makes splits faster and adds more
diversity between trees, often reducing variance.

---

## Gradient Boosting

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `n_estimators` — number of boosting stages | Regression tree structure at each stage |
| `learning_rate` — shrinkage factor | Split feature and threshold at each node |
| `max_depth` — tree depth per stage | Leaf output values (fitted to residuals) |
| `min_samples_split` — min samples to split | Initial prediction $F_0$ (log-odds of class prior) |
| `min_samples_leaf` — min samples in a leaf | |
| `max_features` — features per split | |
| `min_impurity_decrease` — min impurity gain to split | |

**How `.fit()` works:** Starts with a constant prediction (class prior), then
iteratively adds small trees. Each tree is fitted to the **negative gradient**
(pseudo-residuals) of the loss function. The final prediction is the sum of
all trees scaled by `learning_rate`:

$$F(x) = F_0 + \eta \sum_{t=1}^{T} h_t(x)$$

where $\eta$ = learning_rate, $h_t$ = tree at stage $t$.

---

## SVM (Support Vector Machine)

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `C` — regularization (trade-off margin vs errors) | Support vectors (subset of training samples) |
| `kernel` — kernel function (`linear`, `poly`, `rbf`, `sigmoid`) | Dual coefficients $\alpha_i$ for each support vector |
| | Bias term $b$ (intercept) |

**How `.fit()` works:** Solves a quadratic optimization problem to find the
hyperplane that maximizes the margin between classes. Only a subset of training
samples (support vectors) define the decision boundary:

$$f(x) = \text{sign}\left(\sum_{i \in SV} \alpha_i \, y_i \, K(x_i, x) + b\right)$$

With `kernel="rbf"`, the data is implicitly mapped to a high-dimensional space
where a linear separator exists.

---

## Decision Tree

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `max_depth` — maximum tree depth | Tree structure (which feature splits where) |
| `min_samples_split` — min samples to split a node | Threshold at each internal node |
| `min_samples_leaf` — min samples in a leaf | Class distribution at each leaf |
| `max_features` — features considered per split | |
| `criterion` — split quality metric (`gini`, `entropy`, `log_loss`) | |
| `splitter` — split strategy (`best`, `random`) | |

**How `.fit()` works:** Recursively partitions the feature space by finding the
feature + threshold that maximizes information gain (or minimizes Gini
impurity). Stops splitting when `max_depth`, `min_samples_split`, or
`min_samples_leaf` constraints are reached.

---

## CatBoost

| Hyperparameters (set) | Learned Parameters (trained) |
|---|---|
| `n_estimators` — number of boosting iterations | Oblivious decision tree at each stage |
| `max_depth` — tree depth (oblivious = all nodes at same level use same split) | Split feature and threshold per level |
| `min_child_samples` — min samples in a leaf | Leaf output values |
| `bootstrap_type` — sampling method (`Bayesian`, `Bernoulli`, `MVS`, `No`) | Ordered boosting permutation |

**How `.fit()` works:** Similar to Gradient Boosting but uses **oblivious
decision trees** (symmetric: all nodes at the same depth split on the same
feature) and **ordered boosting** (each sample's residual is computed using a
model trained only on preceding samples) to reduce target leakage/overfitting.

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Before training                          │
│                                                             │
│  Hyperparameters:  n_estimators, max_depth, C, kernel, ...  │
│  → Set by user (config) or random search (--optimize)       │
│  → Define the MODEL STRUCTURE                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     .fit(X, y)                              │
│                                                             │
│  Learned parameters:  split thresholds, tree structures,    │
│                       support vectors, weights, biases      │
│  → Discovered from DATA                                     │
│  → Define the DECISION BOUNDARY                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    After training                           │
│                                                             │
│  Same hyperparameters + different data split                │
│  → Different learned parameters                             │
│  → Different model → different predictions                  │
│                                                             │
│  That's why 30 MC iterations matter:                        │
│  mean ± std measures how STABLE the learned parameters are  │
│  across different training data                             │
└─────────────────────────────────────────────────────────────┘
```
