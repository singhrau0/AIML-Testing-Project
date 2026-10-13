import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from .config import FEATURES, TARGET
from .preprocessing import build_preprocessor


def train_models(train, config, folder, report):
    X, y = train[FEATURES], train[TARGET].map({'N': 0, 'Y': 1})
    cv = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.seed)
    candidates = {
        'logistic_regression': (LogisticRegression(max_iter=2000, random_state=config.seed),
                                {'model__C': [0.1, 1, 10], 'model__class_weight': [None, 'balanced']}),
        'random_forest': (RandomForestClassifier(n_estimators=150, random_state=config.seed, n_jobs=1),
                          {'model__max_depth': [3, 6, None], 'model__min_samples_leaf': [3, 10]})}
    scores, models = {}, {}
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])
        search = GridSearchCV(pipeline, grid, scoring={'auc': 'roc_auc', 'balanced_accuracy': 'balanced_accuracy'},
                              refit='auc', cv=cv, n_jobs=1, error_score='raise')
        search.fit(X, y)
        pd.DataFrame(search.cv_results_).to_csv(folder / f'{name}_cv.csv', index=False)
        models[name] = search.best_estimator_
        joblib.dump(models[name], folder / f'{name}.joblib')
        scores[name] = {'cv_roc_auc': float(search.best_score_), 'best_params': search.best_params_}
    baseline = Pipeline([('preprocess', build_preprocessor()), ('model', DummyClassifier(strategy='prior'))])
    baseline_cv = cross_validate(baseline, X, y, cv=cv, scoring='roc_auc', error_score='raise')
    baseline.fit(X, y)
    winner = max(scores, key=lambda name: scores[name]['cv_roc_auc'])
    report.update(candidates=scores, selected=winner, selection_metric='training CV ROC AUC',
                  baseline_cv_roc_auc=float(baseline_cv['test_score'].mean()), folds=config.cv_folds,
                  uncertainty='Best CV score is selection-biased; holdout evaluation is independent.')
    return winner, models[winner], baseline
