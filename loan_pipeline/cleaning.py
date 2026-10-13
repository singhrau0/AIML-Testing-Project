import pandas as pd
from .config import NUMERIC
from .cleaning_analysis import analyze, comparison, render_analysis


def clean(data, folder, report):
    before = analyze(data)
    data.to_csv(folder / 'before_cleaning.csv', index=False)
    result = data.copy()
    for col in result.select_dtypes(include=['object', 'string']):
        result[col] = result[col].str.strip()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors='raise')
    result = result.drop_duplicates().reset_index(drop=True)
    if result.Loan_ID.duplicated().any():
        raise ValueError('Loan IDs conflict after whitespace normalization')
    result.to_csv(folder / 'cleaned.csv', index=False)
    report.update(input_rows=len(data), output_rows=len(result), removed_duplicates=len(data)-len(result),
                  strategy='Trim strings; remove exact duplicates; preserve outliers; defer imputation until training.')
    after = analyze(result)
    report.update(before=before, after=after, analysis_note=(
        'Before means the loaded CSV immediately before cleaning; after means immediately after '
        'trimming strings, numeric conversion and exact deduplication. Missing values and outliers '
        'are retained at this stage. Imputation happens later inside training folds to prevent leakage. '
        'IQR outliers are descriptive flags using each snapshot’s own 1.5-IQR fences, not invalid records.'))
    comparison(before, after).to_csv(folder / 'before_after_comparison.csv', index=False)
    (folder / 'cleaning_report.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Cleaning analysis</title>'
        '<style>body{font:16px system-ui;margin:32px;color:#182434}table{border-collapse:collapse;margin:20px 0}'
        'td,th{padding:8px 12px;border:1px solid #d7dfe8;text-align:left}th{background:#edf2f7}</style>'
        '</head><body>' + render_analysis(report) + '</body></html>')
    return result
