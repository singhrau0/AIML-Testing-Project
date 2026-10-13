# Loan approval MLOps pipeline

`runmain.py` runs all 13 stages using `loan_train.csv`. Every process has its own module in `loan_pipeline/` and produces a numbered artifact folder with `report.json`. The original Iris script and dependencies remain intact.

## Run

Use Python 3.11 (the machine's default Python 3.15 is incompatible with these dependencies):

```bash
source .venv/bin/activate
python runmain.py
```

For a fresh environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python runmain.py
python -m pytest -q
```

Paths default to the project directory even when invoked from another working directory. Optional arguments: `--data`, `--artifacts`, `--seed`, `--cv-folds`, `--min-roc-auc`, `--min-balanced-accuracy`.

## Processes and artifacts

| Stage | Module | Outputs |
|---|---|---|
| 01 | tracking.py | Environment versions, configuration, source snapshot, Git commit if available |
| 02 | ingestion.py | Immutable source CSV snapshot, SHA-256, dimensions, missing counts |
| 03 | validation.py | Column/domain/type checks, duplicate ID checks, invalid-label rejection |
| 04 | cleaning.py / cleaning_analysis.py | Before/after CSV snapshots, comparison CSV, standalone cleaning HTML report, detailed before/after analysis in report.json |
| 05 | profiling.py | Descriptive statistics, missingness, class distribution |
| 06 | splitting.py | Stratified train/test CSVs, disjoint ID verification |
| 07 | preprocessing.py | Feature contract and transform strategy |
| 08 | training.py | Logistic regression/random forest grid results, fitted pipelines, CV selection |
| 09 | evaluation.py | Holdout metrics, baseline comparison, bootstrap AUC interval, predictions, permutation importance |
| 10 | fairness.py | Gender, marital status and property-area subgroup metrics |
| 11 | monitoring.py | Training reference distributions, holdout drift diagnostic |
| 12 | registration.py | Model bundle, quality gate decision, metadata, monitoring reference, model card |
| 13 | reporting.py | Standalone HTML report combining process reports |

Each execution writes `artifacts/<UTC timestamp>_<unique id>/`, including `stages.json`, `result.json` and a SHA-256 manifest. `artifacts/latest.json` points to the most recent completed run, including rejected model runs; it is not a production model pointer. Failed runs retain their completed stage artifacts and traceback. Reports are local and contain dataset-derived information; access should match dataset permissions.

The cleaning report compares row/column counts, missing cells, rows with missing values, exact duplicates, duplicate IDs, whitespace, blank strings, column types, unique values, numeric distributions, descriptive IQR outliers and target counts. Open `04_cleaning/cleaning_report.html` or the cleaning section of the combined report. Missing values remain until training-only imputation; outliers are flagged and retained. Unchanged before/after values accurately indicate that no cleaning change was needed for those measures.

## Modeling strategy

The target is `Loan_Status`: N=0, Y=1. It describes historical approval, **not loan default or creditworthiness**. Loan IDs, Gender and Married are excluded as predictive features. Exclusion does not remove all proxy bias.

Reserve 20% as a stratified holdout. Fit median numeric imputation, missing indicators, scaling, categorical imputation and one-hot encoding inside each training CV fold. Select hyperparameters and model using five-fold training ROC AUC. Compare a prior-probability dummy baseline. Evaluate the selected model once on the holdout with a fixed 0.5 threshold. Do not tune gates, features or threshold in response to holdout results; use a new evaluation set if iterating.

Registration requires ROC AUC >= 0.70, balanced accuracy >= 0.60, and AUC above baseline. These are configurable engineering defaults, not validated business acceptance criteria. A passing model becomes a local candidate; failed candidates are recorded as rejected. No model is automatically authorized for production lending.

## Batch inference and monitoring

```bash
python predict.py --model-dir artifacts/<run-id>/12_registration --data new_loans.csv
```

Input uses the same columns; Loan_Status is optional. Prediction writes a new unique artifact run containing predictions, schema checks and drift metrics. The bundled preprocessing ensures training/serving consistency. Only load trusted joblib files. Rejected models and checksum mismatches are blocked. Unknown domain categories are rejected by the data contract; missing values are imputed.

Monitor PSI >= 0.20 and missingness changes >= 0.10 as investigation triggers. The holdout drift report is a demonstration against training data, not a live production signal. With delayed labels, periodically evaluate AUC, balanced accuracy, calibration and subgroup error rates. Retraining should follow reviewed drift/performance degradation or new representative labeled data, run the same pipeline, and compare against the approved champion on a fresh holdout. Keep previous run bundles for rollback. Never automatically retrain or promote based on unlabeled drift alone.

## Operational strategy and limits

Local artifact tracking is functional without an MLflow server or DVC remote. SHA-256 dataset/source snapshots and environment versions support provenance. Existing Iris DVC configuration is not reused for the unrelated loan data. To scale, configure controlled artifact storage, a governed dataset version store and an experiment registry; provision infrastructure separately.

CI runs validation, leakage boundary, drift, failure logging, end-to-end training and inference checks. No hosted service or scheduler is deployed by this project. For production, add approved serving infrastructure, access controls, retention, monitoring with labels, temporal/external validation, reviewed thresholds reflecting error costs, and human oversight. Source date and sampling/label provenance are unknown, so this dataset alone cannot demonstrate real-world lending suitability.
