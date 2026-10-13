import numpy as np
import pandas as pd
from .config import FEATURES, NUMERIC


def build_reference(train, folder, report):
    reference = {}
    for col in FEATURES:
        series = train[col]
        item = {'missing_fraction': float(series.isna().mean())}
        if col in NUMERIC:
            values = series.dropna().to_numpy()
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
            # Open end bins catch future values outside the training range.
            internal = edges[1:-1].tolist()
            counts = np.histogram(values, bins=[-np.inf] + internal + [np.inf])[0]
            item.update(kind='numeric', internal_edges=internal, proportions=(counts / counts.sum()).tolist())
        else:
            item.update(kind='categorical', proportions=series.fillna('__MISSING__').value_counts(normalize=True).to_dict())
        reference[col] = item
    from .artifacts import write_json
    write_json(folder / 'reference.json', reference)
    report.update(reference_population='training split only', features=list(reference),
                  alert_policy={'psi': 0.2, 'absolute_missing_fraction_change': 0.1},
                  limitation='Drift is a diagnostic signal, not evidence of performance loss; labels are needed for outcome monitoring.')
    return reference


def monitor(data, reference):
    results = {}
    for col, ref in reference.items():
        series = data[col]
        missing_change = abs(float(series.isna().mean()) - ref['missing_fraction'])
        if ref['kind'] == 'numeric':
            counts = np.histogram(pd.to_numeric(series).dropna(), bins=[-np.inf] + ref['internal_edges'] + [np.inf])[0]
            observed = counts / max(counts.sum(), 1)
            expected = np.array(ref['proportions'])
        else:
            current = series.fillna('__MISSING__').value_counts(normalize=True).to_dict()
            keys = sorted(set(current) | set(ref['proportions']))
            observed = np.array([current.get(k, 0) for k in keys])
            expected = np.array([ref['proportions'].get(k, 0) for k in keys])
        observed, expected = np.clip(observed, 1e-6, None), np.clip(expected, 1e-6, None)
        psi = float(np.sum((observed-expected) * np.log(observed/expected)))
        results[col] = {'psi': psi, 'missing_fraction_change': missing_change,
                        'alert': psi >= 0.2 or missing_change >= 0.1}
    return results
