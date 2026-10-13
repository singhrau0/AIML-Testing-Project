import numpy as np
import pandas as pd
from .config import NUMERIC, CATEGORICAL, DOMAINS, TARGET


def validate(data, folder, report, training=True):
    required = NUMERIC + CATEGORICAL + ['Loan_ID'] + ([TARGET] if training else [])
    errors = []
    missing = sorted(set(required) - set(data.columns))
    extras = sorted(set(data.columns) - set(required) - {TARGET})
    if missing or extras:
        errors.append(f'Schema mismatch: missing={missing}, extra={extras}')
    if data.empty:
        errors.append('Dataset is empty')
    if not missing:
        if data.Loan_ID.isna().any() or data.Loan_ID.astype(str).str.strip().eq('').any():
            errors.append('Loan_ID must be populated')
        # Identical records can be deduplicated; conflicting IDs cannot.
        if data.drop_duplicates().Loan_ID.duplicated().any():
            errors.append('Conflicting duplicate Loan_ID values')
        for col in NUMERIC:
            numeric = pd.to_numeric(data[col], errors='coerce')
            if (data[col].notna() & numeric.isna()).any() or np.isinf(numeric).any():
                errors.append(f'{col}: invalid numeric values')
            if (numeric.dropna() < 0).any():
                errors.append(f'{col}: negative values')
            if col in ('LoanAmount', 'Loan_Amount_Term') and numeric.eq(0).any():
                errors.append(f'{col}: zero values')
            if col == 'Credit_History' and not numeric.dropna().isin([0, 1]).all():
                errors.append('Credit_History must be 0 or 1')
        for col in CATEGORICAL:
            values = data[col].dropna().astype(str).str.strip()
            if not values.isin(DOMAINS[col]).all():
                errors.append(f'{col}: unknown categories {sorted(set(values) - set(DOMAINS[col]))}')
        if training:
            labels = data[TARGET].astype('string').str.strip()
            if labels.isna().any() or not labels.isin(['Y', 'N']).all():
                errors.append('Loan_Status must contain only nonmissing Y/N labels')
            if labels.nunique() != 2:
                errors.append('Both target classes are required')
        all_missing = [c for c in NUMERIC + CATEGORICAL if data[c].isna().all()]
        if training and all_missing:
            errors.append(f'Entirely missing columns: {all_missing}')
    report.update(errors=errors, identical_duplicates=int(data.duplicated().sum()), valid=not errors)
    if errors:
        raise ValueError('; '.join(errors))
    return data
