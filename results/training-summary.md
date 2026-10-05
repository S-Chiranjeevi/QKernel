# ULB/Worldline benchmark summary

Dataset: `creditcard.csv` (SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`)
Protocol: four ideal statevector benchmark runs; 10 prescribed seeds per run; 200 legitimate + k fraud training rows; 400 legitimate + 100 fraud test rows per seed; six qubits, one circuit repeat. Scaling, PCA, angle scaling, CV and thresholds are fit on training data only.
The CSV contains 1,081 exact duplicate rows; they were retained in accordance with the loaded-source protocol. Test sets are balanced and do not reflect natural fraud prevalence. The quantum circuit was simulated in software; no quantum hardware was used.

| Fraud labels k | Quantum PR-AUC mean ± SD | Best classical PR-AUC mean ± SD | Paired delta (Q − C) | 95% t interval | Verdict |
|---:|---:|---:|---:|---:|---|
| 6 | 0.667 ± 0.168 | 0.889 ± 0.036 | -0.222 | [-0.351, -0.092] | Classical preferred |
| 10 | 0.829 ± 0.077 | 0.913 ± 0.024 | -0.084 | [-0.144, -0.024] | Classical preferred |
| 20 | 0.853 ± 0.049 | 0.927 ± 0.015 | -0.074 | [-0.108, -0.040] | Classical preferred |
| 50 | 0.879 ± 0.069 | 0.929 ± 0.016 | -0.050 | [-0.098, -0.003] | Classical preferred |

Saved artifacts are in the four run folders alongside this summary: `run.json`, `metrics.csv`, `predictions.csv`, and `kernel_train.npz`. The fitted estimators themselves are not serialized; this project currently saves evaluation scores and kernels, not a deployable inference model.

Data source and license: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), made available under [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The raw CSV is excluded from this repository; its SHA-256 is recorded above.
