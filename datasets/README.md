# Local datasets

The raw ULB/Worldline credit-card fraud data is stored at
`datasets/raw/creditcard.csv`; the CSV and other files under this directory are
ignored by Git. It was downloaded from the [original Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which identifies
the source as a Worldline and ULB collaboration and lists the Open Database
License for database contents.

Verified file: 284,807 rows, 31 columns, 492 frauds, 284,315 legitimate rows,
1,081 exact duplicate rows, no missing values. SHA-256:
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

The benchmark uses the CSV as provided; it reports but does not remove duplicate
rows. Check the source license and terms before sharing derived row-level output.
