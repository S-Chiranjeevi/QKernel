# Published experiment artifacts

These four ideal statevector runs use the canonical ULB/Worldline credit-card
fraud dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
The dataset page attributes it to the Machine Learning Group at ULB and lists
Open Database License (ODbL) for database contents. This report and the derived
experiment files contain information from that database and are made available
under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).

The raw 150.8 MB CSV is not stored in this repository. Place it locally at
`datasets/raw/creditcard.csv`; its SHA-256 and expected class counts are recorded
in `../datasets/README.md`. The results include aggregated metrics, sampled
per-row test predictions, and fraud-first training-kernel matrices. They do not
include a serialized estimator for arbitrary-row inference.

See [training-summary.md](training-summary.md) for the protocol and aggregate
PR-AUC comparison. Each run directory contains `run.json`, `metrics.csv`,
`predictions.csv`, and `kernel_train.npz`.
