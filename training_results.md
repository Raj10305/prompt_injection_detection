# Model Training & Evaluation Results

- **Generated At**: 2026-11-14 19:24:19
- **Dataset**: `guychuk_dataset + Prompt_INJECTION_And_Benign_DATASET`
- **Total Samples Evaluated**: 35,029 (1,000 held-out test split)
- **Ensemble Architecture**: Primary BERT/DeBERTa (`75%`) + XGBoost (`25%`)

## Overall Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.90%** |
| **Precision** | **100.00%** |
| **Recall** | **99.81%** |
| **F1-Score** | **99.90%** |
| **AUC-ROC** | **0.9999** |
| **Average Latency** | `1.85 ms / prompt` |

## Per-Attack-Type Detection Breakdown (Test Split)

| Attack Category | Detection Rate / Recall |
| :--- | :--- |
| **jailbreaking** | 466/467 (99.8%) |
| **obfuscation** | 4/4 (100.0%) |
| **role_playing** | 42/42 (100.0%) |
