from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import NUMERIC, FEATURES


def build_preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         [c for c in FEATURES if c not in NUMERIC])])


def describe_preprocessing(folder, report):
    report.update(features=FEATURES, excluded=['Loan_ID', 'Loan_Status', 'Gender', 'Married'],
                  numeric='Median imputation, missing indicators, standard scaling',
                  categorical='Most frequent imputation and one-hot encoding',
                  leakage_control='A fresh preprocessor is fitted within every CV training fold.')
