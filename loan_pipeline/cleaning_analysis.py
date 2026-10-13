"""Before/after cleaning diagnostics; no learned preprocessing is fitted here."""
import html
import pandas as pd
from .config import NUMERIC, TARGET


def analyze(data):
    columns = {}
    for name in data.columns:
        series = data[name]
        strings = series.dropna().map(str)
        item = {
            'dtype': str(series.dtype),
            'missing_count': int(series.isna().sum()),
            'missing_percent': round(float(series.isna().mean() * 100), 4) if len(data) else 0.0,
            'unique_nonmissing': int(series.nunique()),
            'blank_strings': int(strings.str.strip().eq('').sum()),
            'surrounding_whitespace': int(strings.ne(strings.str.strip()).sum()),
        }
        if name in NUMERIC:
            numeric = pd.to_numeric(series, errors='coerce')
            stats = numeric.describe()
            item['numeric_statistics'] = {str(k): float(v) if pd.notna(v) else None for k, v in stats.items()}
            q1, q3 = numeric.quantile([.25, .75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            item['iqr_outlier_count'] = int(((numeric < low) | (numeric > high)).sum())
        columns[name] = item
    return {
        'rows': len(data), 'columns': len(data.columns),
        'missing_cells': int(data.isna().sum().sum()),
        'rows_with_missing': int(data.isna().any(axis=1).sum()),
        'exact_duplicate_rows': int(data.duplicated().sum()),
        'duplicate_loan_ids': int(data.Loan_ID.duplicated().sum()),
        'whitespace_cells': sum(v['surrounding_whitespace'] for v in columns.values()),
        'blank_string_cells': sum(v['blank_strings'] for v in columns.values()),
        'target_counts': {str(k): int(v) for k, v in data[TARGET].value_counts(dropna=False).items()} if TARGET in data else {},
        'column_analysis': columns,
    }


def comparison(before, after):
    rows = []
    for name, old in before['column_analysis'].items():
        new = after['column_analysis'][name]
        row = {'column': name}
        for metric in ['dtype', 'missing_count', 'missing_percent', 'unique_nonmissing',
                       'blank_strings', 'surrounding_whitespace', 'iqr_outlier_count']:
            if metric in old:
                row[f'{metric}_before'] = old[metric]
                row[f'{metric}_after'] = new[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def render_analysis(report):
    before, after = report['before'], report['after']
    metrics = ['rows', 'columns', 'missing_cells', 'rows_with_missing', 'exact_duplicate_rows',
               'duplicate_loan_ids', 'whitespace_cells', 'blank_string_cells']
    summary = pd.DataFrame([{'Metric': key.replace('_', ' ').title(), 'Before': before[key],
                             'After': after[key], 'Change': after[key] - before[key]} for key in metrics])
    column_rows = []
    for name, old in before['column_analysis'].items():
        new = after['column_analysis'][name]
        column_rows.append({'Column': name, 'Dtype before → after': f'{old["dtype"]} → {new["dtype"]}',
                            'Missing before → after': f'{old["missing_count"]} → {new["missing_count"]}',
                            'Unique before → after': f'{old["unique_nonmissing"]} → {new["unique_nonmissing"]}',
                            'Whitespace before → after': f'{old["surrounding_whitespace"]} → {new["surrounding_whitespace"]}'})
    numeric_rows = []
    for name, old in before['column_analysis'].items():
        if 'numeric_statistics' not in old:
            continue
        new = after['column_analysis'][name]
        for stat, value in old['numeric_statistics'].items():
            numeric_rows.append({'Column': name, 'Statistic': stat, 'Before': value,
                                 'After': new['numeric_statistics'][stat]})
        numeric_rows.append({'Column': name, 'Statistic': 'IQR outliers (retained)',
                             'Before': old['iqr_outlier_count'], 'After': new['iqr_outlier_count']})
    targets = sorted(set(before['target_counts']) | set(after['target_counts']))
    target_table = pd.DataFrame([{'Label': key, 'Before': before['target_counts'].get(key, 0),
                                 'After': after['target_counts'].get(key, 0)} for key in targets])
    return ('<h2>Cleaning: before and after</h2><p>' + html.escape(report['analysis_note']) + '</p>'
            + summary.to_html(index=False, border=0) + '<h3>Column-level comparison</h3>'
            + pd.DataFrame(column_rows).to_html(index=False, border=0)
            + '<h3>Numeric distributions</h3>' + pd.DataFrame(numeric_rows).to_html(index=False, border=0)
            + '<h3>Target distribution</h3>' + target_table.to_html(index=False, border=0))
