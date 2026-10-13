import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, log_loss, brier_score_loss, classification_report, confusion_matrix)
from sklearn.inspection import permutation_importance
from .config import FEATURES, TARGET


def metrics(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'roc_auc': float(roc_auc_score(y, probabilities)),
            'average_precision': float(average_precision_score(y, probabilities)),
            'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
            'brier_score': float(brier_score_loss(y, probabilities)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            'classification_report': classification_report(y, predicted, labels=[0, 1],
                target_names=['N', 'Y'], output_dict=True, zero_division=0)}


def evaluate(model, baseline, test, config, folder, report):
    X, y = test[FEATURES], test[TARGET].map({'N': 0, 'Y': 1}).to_numpy()
    p = model.predict_proba(X)[:, 1]
    result = metrics(y, p, config.threshold)
    base = metrics(y, baseline.predict_proba(X)[:, 1], config.threshold)
    rng = np.random.default_rng(config.seed)
    aucs = []
    for _ in range(500):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    result['roc_auc_bootstrap_95_interval'] = np.quantile(aucs, [0.025, 0.975]).tolist()
    pd.DataFrame({'Loan_ID': test.Loan_ID, 'actual': y, 'probability_Y': p,
                  'predicted': (p >= config.threshold).astype(int)}).to_csv(folder / 'predictions.csv', index=False)
    importance = permutation_importance(model, X, y, scoring='roc_auc', n_repeats=10,
                                         random_state=config.seed, n_jobs=1)
    pd.DataFrame({'feature': FEATURES, 'mean_auc_decrease': importance.importances_mean,
                  'std': importance.importances_std}).sort_values('mean_auc_decrease', ascending=False).to_csv(
                      folder / 'permutation_importance.csv', index=False)
    report.update(model=result, baseline=base, positive_label='Y', threshold=config.threshold,
                  confusion_matrix_order=['N', 'Y'], test_rows=len(test))
    return result, base
