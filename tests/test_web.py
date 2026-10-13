"""Web tests use the unchanged bundled model; no fitting or retraining."""
import hashlib
import json
import shutil
from pathlib import Path
import pytest
from app import create_app, ROOT
from webapp.model_service import ModelService

PAYLOAD = {'ApplicantIncome': 5000, 'CoapplicantIncome': 0, 'LoanAmount': 120,
           'Loan_Amount_Term': 360, 'Credit_History': 1, 'Dependents': '0',
           'Education': 'Graduate', 'Self_Employed': 'No', 'Property_Area': 'Urban'}

@pytest.fixture(scope='module')
def app():
    application = create_app({'TESTING': True})
    assert application.extensions['loan_model'] is not None
    return application

@pytest.fixture
def client(app):
    return app.test_client()


def test_page_and_static_assets(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Build your loan profile' in response.data
    assert b'Gender" name=' not in response.data
    for path in ['/static/css/style.css', '/static/js/script.js', '/static/assets/favicon.svg']:
        assert client.get(path).status_code == 200
    assert client.get('/health').json == {'ok': True, 'status': 'ready'}
    assert "default-src 'self'" in response.headers['Content-Security-Policy']


def test_real_model_both_outcomes_and_parity(client, app):
    service = app.extensions['loan_model']
    for flag, outcome in [(1, 'approved'), (0, 'rejected')]:
        payload = {**PAYLOAD, 'Credit_History': flag}
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        result = response.json['prediction']
        assert result['decision'] == outcome
        assert result == service.predict(payload)
        assert 0 <= result['approval_probability'] <= 1
    assert app.extensions['loan_model'] is service


@pytest.mark.parametrize('patch', [
    {'ApplicantIncome': -1}, {'ApplicantIncome': '5000'}, {'ApplicantIncome': True},
    {'ApplicantIncome': None}, {'ApplicantIncome': float('inf')}, {'LoanAmount': 0},
    {'Loan_Amount_Term': 1.5}, {'Credit_History': 2}, {'Dependents': '4'},
    {'Education': '<script>alert(1)</script>'}, {'Property_Area': []}, {'Gender': 'Male'},
])
def test_invalid_input(client, patch):
    response = client.post('/predict', json={**PAYLOAD, **patch})
    assert response.status_code == 422
    assert response.json['ok'] is False
    assert response.json['error']['fields']


def test_missing_malformed_and_http_errors(client):
    for payload in [{}, [], None]:
        response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
        assert response.status_code == 422
    assert client.post('/predict', data='{', content_type='application/json').status_code == 400
    assert client.post('/predict', data='hello').status_code == 415
    assert client.get('/predict').status_code == 405
    assert client.get('/not-here').status_code == 404
    assert client.post('/predict', data='x' * 20000, content_type='application/json').status_code == 413


def test_backend_exception_is_sanitized(client, app, monkeypatch):
    def fail(_):
        raise RuntimeError('PRIVATE_STACK_TRACE')
    monkeypatch.setattr(app.extensions['loan_model'], 'predict', fail)
    response = client.post('/predict', json=PAYLOAD)
    assert response.status_code == 500
    assert b'PRIVATE_STACK_TRACE' not in response.data
    assert response.json['error']['code'] == 'prediction_failed'


def test_missing_model_is_unavailable(tmp_path):
    app = create_app({'TESTING': True, 'MODEL_DIR': tmp_path})
    client = app.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/health').status_code == 503
    assert client.post('/predict', json=PAYLOAD).status_code == 503


def test_model_integrity_and_version(tmp_path):
    directory = tmp_path / 'model'
    shutil.copytree(ROOT / 'model', directory)
    path = directory / 'model.joblib'
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(RuntimeError, match='integrity'):
        ModelService(directory)
    assert hashlib.sha256((ROOT / 'model/model.joblib').read_bytes()).hexdigest() == json.loads((ROOT / 'model/metadata.json').read_text())['model_sha256']
    runtime = json.loads((directory / 'runtime.json').read_text())
    runtime['packages']['scikit-learn'] = '0.0.0'
    (directory / 'runtime.json').write_text(json.dumps(runtime))
    with pytest.raises(RuntimeError, match='runtime mismatch'):
        ModelService(directory)
