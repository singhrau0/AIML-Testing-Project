# Loan model: random_forest

Status: candidate. Production approved: no.

Predicts historical Loan_Status (Y/N), not default risk or repayment probability. Training provenance and representativeness are unknown. Random stratified evaluation does not establish future or out-of-population performance. Gender and Married are excluded from prediction but proxies may remain. Imputation and encoding are bundled with the model.

Holdout ROC AUC: 0.8046; balanced accuracy: 0.7850.

Before production: review label definition, consent and permitted use, temporal/external validation, fairness, calibration and operational error costs. Set an approved threshold using training validation data, never the test set. Use human review, shadow deployment, monitored rollout and rollback.
