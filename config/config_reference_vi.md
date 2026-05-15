# Config Reference — `default_config.json`

Tài liệu tham khảo cho tất cả key trong file config.

## `training`

### Paths & Input

| Key | Type | Mặc định | Mô tả |
|---|---|---|---|
| `positive_json` | string | `"train_test_dataset/Chloe_clustered_cores.json"` | JSON file chứa positive sequences |
| `negative_json` | string | `"train_test_dataset/Chloe_disordered_regions_clustered.json"` | JSON file chứa negative sequences |
| `positive_fasta` | string | `"train_test_dataset/Chloe_clustered_cores.fasta"` | FASTA file positive (cho IUPred) |
| `negative_fasta` | string | `"train_test_dataset/Chloe_disordered_regions_clustered.fasta"` | FASTA file negative (cho IUPred) |
| `iupred_script` | string | `"scripts/iupred3/iupred3.py"` | Path tới IUPred3 script |
| `iupred_input_type` | string | `"long"` | IUPred prediction type (`"long"` hoặc `"short"`) |

### Training Settings

| Key | Type | Mặc định | Mô tả |
|---|---|---|---|
| `test_ratio` | float | `0.3` | Tỷ lệ test set (0–1) |
| `sampling_strategy` | string | `"min"` | Cân bằng class: `"min"` = downsample về class nhỏ nhất |
| `random_seed` | int \| null | `null` | Seed cố định cho reproducibility. `null` = random |
| `output_dir` | string | `"outputs/training"` | Thư mục output cho kết quả training |

### `algorithms` (mảng)

Mỗi phần tử trong mảng `algorithms` là 1 thuật toán cần train:

| Key | Type | Mô tả |
|---|---|---|
| `type` | string | Tên thuật toán (xem bảng dưới) |
| `runs` | int | Số Monte Carlo iterations |
| `params` | object | Hyperparameters truyền thẳng vào sklearn |
| `n_combos` | int | Số random param combos (mode optimize) |

`runs` được dùng cho **cả 2 mode**:

- **Fixed params** (không có `--optimize`): train `runs` MC iterations với `params` cố định
- **Optimize** (có `--optimize`): mỗi combo train `runs` MC iterations, so sánh trên **cùng data splits**

CLI flag `--runs N` có thể override per-algorithm `runs` cho tất cả thuật toán.

### Metrics

Mỗi MC iteration tính 5 metrics: **F1_score**, **Accuracy**, **Precision**, **Recall**, **MCC**.

CLI flag `--metric` chọn metric để so sánh các combos (khi `--optimize`):

| Giá trị | Mô tả |
|---|---|
| `F1_score` | F1 score (mặc định) |
| `Accuracy` | Accuracy |
| `Precision` | Precision |
| `Recall` | Recall |
| `MCC` | Matthews Correlation Coefficient (range [-1, 1]) |
| `Average` | Trung bình cả 5 metrics trên |

---

## Algorithm Params

### AdaBoost (`adaboost`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `n_estimators` | int | `50` | 1 – 500, step 1 |
| `algorithm` | string | `"SAMME"` | — |
| `learning_rate` | float | `1.0` | — |

### Random Forest (`random_forest`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `n_estimators` | int | `100` | 500 – 5000, 500 values |
| `max_features` | string | `"sqrt"` | `"sqrt"`, `"log2"` |
| `max_depth` | int \| null | `null` | 5 – 110 (30 values) + `null` |
| `min_samples_split` | int | `2` | 2, 5, 10 |
| `min_samples_leaf` | int | `1` | 1, 2, 4, 8 |
| `bootstrap` | bool | `true` | `true`, `false` |

### Extra Trees (`extra_trees`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `n_estimators` | int | `100` | 500 – 5000, 500 values |
| `max_features` | string | `"sqrt"` | `"sqrt"`, `"log2"` |
| `max_depth` | int \| null | `null` | 5 – 110 (30 values) + `null` |
| `min_samples_split` | int | `2` | 2, 5, 10 |
| `min_samples_leaf` | int | `1` | 1, 2, 4, 8 |
| `bootstrap` | bool | `false` | `true`, `false` |

### Gradient Boosting (`gradient_boosting`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `n_estimators` | int | `100` | 500 – 5000, 500 values |
| `learning_rate` | float | `0.1` | — |
| `max_depth` | int | `3` | 5 – 110 (30 values) + `null` |
| `min_samples_split` | int | `2` | 2, 5, 10 |
| `min_samples_leaf` | int | `1` | 1, 2, 4, 8 |
| `max_features` | string \| null | `null` | `"sqrt"`, `"log2"` |
| `min_impurity_decrease` | float | `0.0` | 0.0 – 10.0, 100 values |

### SVM (`svm`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `C` | float | `1.0` | 0.01 – 5.0, 100 values |
| `kernel` | string | `"rbf"` | `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"` |

### Decision Tree (`decision_tree`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `max_depth` | int \| null | `null` | 5 – 110 (30 values) + `null` |
| `min_samples_split` | int | `2` | 2, 5, 10 |
| `min_samples_leaf` | int | `1` | 1, 2, 4, 8 |
| `max_features` | string \| null | `null` | `"sqrt"`, `"log2"` |
| `criterion` | string | `"gini"` | `"gini"`, `"entropy"`, `"log_loss"` |
| `splitter` | string | `"best"` | `"best"`, `"random"` |

### CatBoost (`catboost`)

| Param | Type | Mặc định | Search space |
|---|---|---|---|
| `n_estimators` | int | `100` | 500 – 5000, 500 values |
| `max_depth` | int | `6` | 0 – 16, step 1 |
| `min_child_samples` | int | `5` | 2, 5, 10 |
| `bootstrap_type` | string | `"MVS"` | `"Bayesian"`, `"Bernoulli"`, `"MVS"`, `"No"` |

> Search spaces dựa theo Table S3 — Gonay et al. (2025).

---

## `benchmark`

| Key | Type | Mặc định | Mô tả |
|---|---|---|---|
| `input_json` | string | `"benchmark_dataset/benchmark_set_test1.json"` | JSON benchmark set |
| `window_size` | int | `17` | Kích thước sliding window |
| `threshold` | float | `0.51` | Ngưỡng probability cho 1 window |
| `threshold_avg` | float | `0.5` | Ngưỡng trung bình probability |
| `aggregation_method` | string | `"max"` | Cách tổng hợp predictions: `"max"` (=1 nếu bất kỳ window nào ≥ threshold) \| `"mean"` (avg score ≥ threshold) \| `"vote"` (fraction windows ≥ vote_fraction) |
| `vote_fraction` | float | `0.5` | Tỷ lệ model phải đồng ý (ensemble voting) |
| `classifier_positive_label` | string | `"AMYLOID"` | Label dùng cho **classifier mode** (so sánh case-insensitive với trường `LABEL` trong input JSON). Override bằng `--positive-label` trên CLI. |
| `output_dir` | string | `"outputs/benchmark"` | Thư mục output benchmark |

CLI flags `--output` / `--benchmark-output` nhận **thư mục** (không phải file stem).
Files output: `benchmark_new_results.csv`, `.json`, `_windows.json` trong thư mục đó.

CLI flag `--window-size` override `window_size` trong config.

---

## Benchmark Modes

Cả 3 benchmark programs (`run_benchmark.py`, `run_benchmark_new.py`, `run_benchmark_adaptive.py`) và 2 full-pipeline scripts hỗ trợ 3 modes, tự động phát hiện hoặc chọn qua CLI flag:

### 1. Position benchmark (mặc định)

So sánh per-residue predictions với ground truth từ trường `matched_core_regions` trong input JSON.
Output: `<name>.csv`, `<name>.json`, `<name>_scores.csv`, `<name>_windows.json` với các metrics TP/TN/FP/FN, Precision, Recall, F1, Accuracy, MCC, SOV cho từng sequence và row GLOBAL.

### 2. Classifier mode (`--classifier`)

So sánh sequence-level prediction với ground truth từ trường `LABEL` trong input JSON.

- Một sequence được predict **positive** nếu ít nhất 1 residue được classifier là core.
- Ground truth: `LABEL` so sánh case-insensitive với `classifier_positive_label` (mặc định `"AMYLOID"`).
- Override positive label bằng `--positive-label <label>` trên CLI.
- Output: `<name>.csv`, `<name>.json`, `<name>_scores.csv`, `<name>_windows.json` với TP/TN/FP/FN, Precision, Recall, F1, Accuracy, MCC (không có SOV).

### 3. Prediction-only mode (tự động)

Kích hoạt tự động khi **tất cả** records trong file input đều không có `LABEL` (rỗng/thiếu) **và** không có `matched_core_regions` (rỗng/thiếu) — tức là file chỉ chứa `ID` và `Sequence`.

- Không tính bất kỳ benchmark metric nào.
- Output:
  - `<name>_scores.csv` — 3 cột: `ID`, `Sequence`, `Score_residues` (list float độ dài = sequence length, score mỗi residue tính theo `aggregation_method` như bình thường)
  - `<name>_windows.json` — window records chi tiết
  - **Không có** `<name>.csv` / `<name>.json` (results/metrics)

| Mode | Điều kiện | Output files |
|---|---|---|
| Position | `matched_core_regions` có data | `<name>.csv` (metrics) + `<name>.json` + `<name>_scores.csv` + `<name>_windows.json` |
| Classifier | `--classifier` flag | `<name>.csv` (metrics) + `<name>.json` + `<name>_scores.csv` + `<name>_windows.json` |
| Prediction-only | Tất cả records thiếu cả `LABEL` lẫn `matched_core_regions` | `<name>_scores.csv` + `<name>_windows.json` |
