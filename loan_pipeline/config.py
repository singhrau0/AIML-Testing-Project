from dataclasses import dataclass, asdict

NUMERIC = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
CATEGORICAL = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
# Sensitive attributes are retained for auditing, excluded from prediction by default.
FEATURES = NUMERIC + [c for c in CATEGORICAL if c not in ('Gender', 'Married')]
TARGET = 'Loan_Status'
DOMAINS = {'Gender': ['Male', 'Female'], 'Married': ['Yes', 'No'],
           'Dependents': ['0', '1', '2', '3+'], 'Education': ['Graduate', 'Not Graduate'],
           'Self_Employed': ['Yes', 'No'], 'Property_Area': ['Urban', 'Rural', 'Semiurban']}

@dataclass
class Config:
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    min_roc_auc: float = 0.70
    min_balanced_accuracy: float = 0.60
    threshold: float = 0.5

    def to_dict(self):
        return asdict(self)
