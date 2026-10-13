import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from loan_pipeline.validation import validate
from loan_pipeline.config import Config, FEATURES
from loan_pipeline.splitting import split
from loan_pipeline.preprocessing import build_preprocessor
from loan_pipeline.monitoring import build_reference, monitor
from loan_pipeline.artifacts import Run
from runmain import run_pipeline, ROOT
from predict import predict


@pytest.fixture
def data():
    return pd.read_csv(ROOT / 'loan_train.csv')


def test_validation_rejects_bad_schema_and_values(data, tmp_path):
    for column, value in [('Loan_Status', None), ('Credit_History', 4), ('ApplicantIncome', np.inf)]:
        bad = data.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            validate(bad, tmp_path, {})
    with pytest.raises(ValueError):
        validate(data.drop(columns=['Education']), tmp_path, {})
    bad = pd.concat([data, data.iloc[[0]].assign(ApplicantIncome=1)])
    with pytest.raises(ValueError, match='duplicate'):
        validate(bad, tmp_path, {})


def test_split_and_training_only_imputation(data, tmp_path):
    train, test = split(data, Config(), tmp_path, {})
    assert set(train.Loan_ID).isdisjoint(test.Loan_ID)
    processor = build_preprocessor().fit(train[FEATURES])
    medians = processor.named_transformers_['numeric'].named_steps['imputer'].statistics_.copy()
    altered = test[FEATURES].copy()
    altered['ApplicantIncome'] = 1e12
    altered['Property_Area'] = 'NeverSeen'
    assert np.isfinite(processor.transform(altered)).all()
    np.testing.assert_array_equal(medians, processor.named_transformers_['numeric'].named_steps['imputer'].statistics_)
    assert medians[0] == train.ApplicantIncome.median()


def test_drift(data, tmp_path):
    ref = build_reference(data, tmp_path, {})
    assert all(abs(v['psi']) < 1e-10 for v in monitor(data, ref).values())
    changed = data.copy()
    changed['ApplicantIncome'] = 1e10
    assert monitor(changed, ref)['ApplicantIncome']['alert']


def test_failure_artifacts(tmp_path):
    run = Run(tmp_path)
    with pytest.raises(ValueError):
        with run.stage('failure'):
            raise ValueError('expected')
    assert json.loads((run.path / 'failure/report.json').read_text())['status'] == 'failed'


def test_end_to_end_and_inference(tmp_path):
    run = run_pipeline(artifacts=tmp_path / 'runs', config=Config(cv_folds=2))
    stages = json.loads((run / 'stages.json').read_text())
    assert len(stages) == 13 and all(s['status'] == 'passed' for s in stages)
    assert (run / '13_reporting/index.html').exists()
    model_dir = run / '12_registration'
    metadata = json.loads((model_dir / 'metadata.json').read_text())
    assert metadata['quality_gate_passed']
    batch = pd.read_csv(ROOT / 'loan_train.csv').head(10).drop(columns=['Loan_Status'])
    batch.to_csv(tmp_path / 'batch.csv', index=False)
    output = predict(model_dir, tmp_path / 'batch.csv', tmp_path / 'predictions')
    predictions = pd.read_csv(output / '01_inference/predictions.csv')
    assert len(predictions) == 10
    np.testing.assert_allclose(predictions.probability_Y, joblib.load(model_dir / 'model.joblib').predict_proba(batch[FEATURES])[:, 1])
    metadata['quality_gate_passed'] = False
    (model_dir / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='quality gates'):
        predict(model_dir, tmp_path / 'batch.csv', tmp_path / 'rejected')


def test_registration_rejects_poor_model(tmp_path):
    from loan_pipeline.registration import register
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier().fit([[1], [2]], [0, 1])
    metrics = {'roc_auc': 0.5, 'balanced_accuracy': 0.5}
    report = {}
    metadata = register(model, 'dummy', metrics, metrics, {}, Config(), tmp_path, report)
    assert metadata['status'] == 'rejected'
    assert not metadata['production_approved']


def test_cleaning_before_after_analysis(data, tmp_path):
    from loan_pipeline.cleaning import clean
    sample = data.head(3).copy()
    sample.loc[0, 'Gender'] = ' Male '
    sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    original = sample.copy(deep=True)
    report = {}
    cleaned = clean(sample, tmp_path, report)
    pd.testing.assert_frame_equal(sample, original)
    assert report['before']['rows'] == 4
    assert report['after']['rows'] == len(cleaned) == 3
    assert report['before']['exact_duplicate_rows'] == 1
    assert report['after']['exact_duplicate_rows'] == 0
    assert report['before']['whitespace_cells'] == 2
    assert report['after']['whitespace_cells'] == 0
    assert report['before']['column_analysis']['LoanAmount']['missing_count'] == 2
    assert report['after']['column_analysis']['LoanAmount']['missing_count'] == 1
    assert cleaned.LoanAmount.isna().sum() == 1
    assert report['after']['target_counts'] == {'Y': 2, 'N': 1}
    assert (tmp_path / 'before_after_comparison.csv').exists()
    assert 'Cleaning: before and after' in (tmp_path / 'cleaning_report.html').read_text()
    json.dumps(report, allow_nan=False)
