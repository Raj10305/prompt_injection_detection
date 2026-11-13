# Model Training & Evaluation Results

- **Generated At**: 2026-11-13 12:38:54
- **Dataset**: `combined_dataset.jsonl`
- **Total Samples Evaluated**: 15,009 (3,002 held-out test split)
- **Optimal Soft-Voting Weight w***: `0.35` (XGBoost weight vs Sequence Classifier)

## Overall Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.60%** |
| **Precision** | **98.73%** |
| **Recall** | **97.74%** |
| **F1-Score** | **98.23%** |
| **AUC-ROC** | **0.9967** |
| **Average Latency** | `2.46 ms / prompt` |

## Per-Attack-Type Detection Breakdown (Test Split)

| Attack Category | Detection Rate / Recall |
| :--- | :--- |
| `code_execution` | 25/25 (100.0%) |
| `data_leakage` | 1/1 (100.0%) |
| `jailbreaking` | 442/469 (94.2%) |
| `obfuscation` | 13/13 (100.0%) |
| `role_playing` | 685/685 (100.0%) |

## Confusion Matrix

```
[[TN=1794, FP=15],
 [FN=27, TP=1166]]
```
