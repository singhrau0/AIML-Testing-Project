import pandas as pd
from .config import FEATURES, TARGET


def audit(model, test, config, folder, report):
    y = test[TARGET].map({'N': 0, 'Y': 1})
    predicted = pd.Series(model.predict_proba(test[FEATURES])[:, 1] >= config.threshold, index=test.index)
    rows = []
    for attribute in ['Gender', 'Married', 'Property_Area']:
        for group, frame in test.groupby(test[attribute].fillna('Missing')):
            idx = frame.index
            actual, pred = y.loc[idx], predicted.loc[idx]
            positives, negatives = actual.eq(1), actual.eq(0)
            rows.append({'attribute': attribute, 'group': group, 'rows': len(idx),
                         'predicted_Y_rate': float(pred.mean()), 'accuracy': float((actual == pred).mean()),
                         'true_positive_rate': float(pred[positives].mean()) if positives.any() else None,
                         'false_positive_rate': float(pred[negatives].mean()) if negatives.any() else None,
                         'small_sample': len(idx) < 30})
    pd.DataFrame(rows).to_csv(folder / 'group_metrics.csv', index=False)
    report.update(groups=rows, conclusion='Descriptive audit only; small groups and historical approval labels cannot establish fairness.',
                  deployment_requirement='Domain, fairness and policy review required before lending use.')
