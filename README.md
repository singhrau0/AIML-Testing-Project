# Clario · Loan Intelligence

A local Flask application with a dark FinTech interface, a guided three-step loan profile, and real predictions from the project's existing Random Forest pipeline. The original training pipeline and its before/after cleaning reports remain available; see [pipeline documentation](docs/PIPELINE.md).

The app estimates historical loan approval labels. It is an exploratory tool, not an official lending decision or a repayment-risk assessment. No model has been retrained for this application.

## Features and stack

- Responsive HTML, CSS and vanilla JavaScript; no frontend framework, CDN, external font, or analytics dependency.
- Dark navy surfaces, mint accents, translucent cards, animated progress, accessible native controls, and reduced-motion support.
- Three steps: finances, profile, credit history. Only the nine verified model features are collected.
- Asynchronous prediction, inline validation, duplicate-submission prevention, timeout handling, and clear API errors.
- Genuine model probabilities and distinct likely/unlikely outcomes. Probabilities have not been independently calibrated.
- Flask application factory, strict server-side validation, local model checksum/version verification, and JSON error handling.
- No database, authentication, model uploads, or stored application history.

## Project structure

```text
app.py                       Flask factory and local entry point
webapp/
  model_service.py           Trusted model loader and inference
  validation.py              Nine-feature JSON validation
model/                       REQUIRED for deployment; not ignored
  model.joblib               Exact copy of existing saved pipeline
  metadata.json              Features, label mapping, threshold, checksum
  runtime.json               Training package versions and source run
  model_card.md              Model scope and limitations
templates/index.html         Dashboard, guided form, result panel, dialogs
static/css/style.css         Responsive visual design
static/js/script.js          Validation, navigation, fetch, result animation
static/assets/favicon.svg    Local icon
requirements.txt             Web inference dependencies
requirements-web-dev.txt     Web dependencies plus pytest
requirements-iris.txt         Preserved original Iris dependencies
requirements-loan.txt        Original newer training environment
requirements-dev.txt         Training environment and tests
docs/PIPELINE.md              Existing MLOps documentation
docs/example-request.json    API smoke-test payload
runmain.py                   Original 13-stage training entry point
loan_pipeline/               Original modular MLOps processes
tests/test_web.py             Web/API/model tests; never retrains
tests/test_pipeline.py        Training pipeline tests (fits models)
```

## Prerequisites

Use **Python 3.11**. The bundled model was saved with scikit-learn **1.3.2**; `requirements.txt` pins matching inference dependencies. The older `.venv` contains scikit-learn 1.7.2 for training and must not be used for this web model. The app checks the version and fails closed instead of silently serving incompatible predictions. The machine's default Python 3.15 is not supported by these pinned packages.

The model was copied byte-for-byte from run `20261013T212330Z_2f0c0aed`. Its source file is preserved. The app always uses `model/`, never a changing `artifacts/latest.json` pointer. It does not load a model supplied by a request.

## macOS setup

Run from this project directory:

```bash
python3.11 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r requirements-web-dev.txt
python app.py
```

On this machine, `.venv-web` is already prepared. You can simply run:

```bash
.venv-web/bin/python app.py
```

## Windows setup

PowerShell:

```powershell
py -3.11 -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements-web-dev.txt
python app.py
```

If script activation is restricted, use the interpreter directly without changing security settings:

```powershell
.\.venv-web\Scripts\python.exe -m pip install -r requirements-web-dev.txt
.\.venv-web\Scripts\python.exe app.py
```

Command Prompt activation alternative: `.venv-web\Scripts\activate.bat`.

Open **http://127.0.0.1:5000**. Stop the server with Ctrl+C. Debug mode is off and the server binds only to localhost. No `.env` configuration or secrets are required.

## Model contract and prediction workflow

The saved pipeline performs numeric median imputation, missing indicators, scaling, categorical imputation and one-hot encoding, then Random Forest classification. The app supplies one DataFrame in the model's recorded feature order; it does not encode or scale a second time.

| Feature | Accepted value |
|---|---|
| ApplicantIncome | Nonnegative JSON number |
| CoapplicantIncome | Nonnegative JSON number; use 0 when absent |
| LoanAmount | Positive JSON number |
| Loan_Amount_Term | Positive whole number of months |
| Credit_History | Numeric 0 or 1; the source binary flag, not a credit score |
| Dependents | String `0`, `1`, `2`, or `3+` |
| Education | `Graduate` or `Not Graduate` |
| Self_Employed | `Yes` or `No` |
| Property_Area | `Urban`, `Semiurban`, or `Rural` |

All fields are required by the web form. Numeric values must be finite and <= 1 trillion as an input guard, not a lending limit. No gender, marital status, Loan_ID, or target input is collected. Unknown fields and categories are rejected. The source CSV does not document the currency, income period, or LoanAmount scaling; enter values on its original scale. No conversion is performed. Missing-data handling remains bundled in the original pipeline, but the interactive form requires complete inputs.

Browser validation → POST JSON → server validation → saved preprocessing/model pipeline → probability for actual class 1 (Y) → recorded 0.50 threshold → animated result. Labels map 0 to N and 1 to Y. Approval probability is returned only if the loaded model implements `predict_proba`.

## API

### `GET /health`

Returns `200 {"ok": true, "status": "ready"}` if model loading succeeded, otherwise HTTP 503 with `status: unavailable`.

### `POST /predict`

Content type: `application/json`. See [example request](docs/example-request.json).

macOS / Linux:

```bash
curl -s http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @docs/example-request.json
```

Windows PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType 'application/json' -Body (Get-Content docs/example-request.json -Raw)
```

Successful responses contain `ok: true` and `prediction` with `label`, `decision`, `title`, `approval_probability` (0–1 or null), `threshold`, `model`, and `disclaimer`. The executed smoke-test response is saved in `docs/example-response.json`; this file is documentation and is never used to produce application predictions.

Errors use a consistent envelope:

```json
{
  "ok": false,
  "error": {
    "code": "validation_error",
    "message": "Please check the highlighted fields.",
    "fields": {"ApplicantIncome": "This field is required."}
  }
}
```

HTTP statuses: 400 malformed JSON; 413 oversized request; 415 non-JSON body; 422 invalid fields; 503 unavailable model; 500 prediction failure. Stack traces are logged only on the server, never returned in API responses. Request bodies are not logged by this application. Security headers prohibit third-party resources and framing.

## Testing

```bash
# Web tests use the actual bundled model and do NOT retrain it.
.venv-web/bin/python -m pytest tests/test_web.py -q
node --check static/js/script.js
```

Windows: replace `.venv-web/bin/python` with `.\.venv-web\Scripts\python.exe`.

The API suite checks real approved and rejected outcomes, direct-model parity, invalid/missing fields, malformed and oversized requests, HTTP error handling, missing models, checksum/version failures, and sanitized backend exceptions. Browser verification results and limitations are recorded in [verification notes](docs/WEB_VERIFICATION.md).

For manual end-to-end testing, enter the example request values in the form, complete all three steps, and submit. Repeat with Credit_History=0 to exercise the actual model's unlikely result. Check the New Estimate button, blank/negative fields, mobile layout, keyboard navigation and dialogs. No canned success response is used.

Training tests remain separate and require local `loan_train.csv`: `.venv/bin/python -m pytest tests/test_pipeline.py -q`. They retrain temporary test models, so they are not part of web verification. Do not run `runmain.py` to start the website.

## Troubleshooting

- **Model unavailable / version mismatch:** use `.venv-web`, install `requirements.txt`, confirm all four files in `model/` exist, and restart the server. Check its logs. Do not bypass version or checksum verification.
- **Port 5000 occupied:** stop the service you own on that port, or for testing use `python -m flask --app 'app:create_app' run --port 5001`. On macOS, AirPlay Receiver may occupy port 5000. This project does not alter system settings.
- **Invalid values:** use exact category spellings and JSON numbers, not numeric strings. Unknown input keys are rejected.
- **Different result than expected:** an estimate reflects this model's historical patterns, not a lender's decision. Verify dataset units first; do not change the model to force an outcome.
- **A missing dependency on Windows:** confirm `py -3.11` selects 64-bit Python 3.11 and use the dedicated web environment.
- **Training dataset missing after cloning:** it is intentionally excluded. Restore it from your authorized dataset store for training. Web inference does not require it.

## Git and future deployment

`.gitignore` excludes local environments, secrets, datasets, caches, training outputs and browser test output. The necessary `model/` bundle, application code, templates, assets, requirements and documentation are intentionally included. `requirements-iris.txt` preserves the original dependencies. `.env.example` contains no secrets.

No cloud deployment is configured or performed. A future deployment must include the trusted model bundle and matching dependencies, use a production WSGI server with HTTPS, configure limits and access as appropriate, and review training provenance, input units, fairness, calibration and operational suitability. The Flask development server is for local testing. Applicant data should not be placed in analytics or access-log URLs. Model probabilities and historical approval labels must not be represented as lending authorization.
