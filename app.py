"""Local loan prediction web application. Run: python app.py."""
import logging
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from webapp.model_service import ModelService
from webapp.validation import validate_payload

ROOT = Path(__file__).resolve().parent


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16 * 1024, MODEL_DIR=ROOT / 'model')
    if test_config:
        app.config.update(test_config)
    try:
        app.extensions['loan_model'] = ModelService(app.config['MODEL_DIR'])
    except Exception:
        app.logger.exception('Unable to load the loan model')
        app.extensions['loan_model'] = None

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/health')
    def health():
        ready = app.extensions['loan_model'] is not None
        return jsonify(ok=ready, status='ready' if ready else 'unavailable'), 200 if ready else 503

    @app.post('/predict')
    def predict():
        if not request.is_json:
            return jsonify(ok=False, error={'code': 'unsupported_media_type', 'message': 'Use application/json.'}), 415
        payload = request.get_json()
        values, errors = validate_payload(payload)
        if errors:
            return jsonify(ok=False, error={'code': 'validation_error', 'message': 'Please check the highlighted fields.', 'fields': errors}), 422
        model = app.extensions['loan_model']
        if model is None:
            return jsonify(ok=False, error={'code': 'model_unavailable', 'message': 'The model is unavailable. Check the server setup and try again.'}), 503
        try:
            return jsonify(ok=True, prediction=model.predict(values))
        except Exception:
            app.logger.exception('Prediction failed')
            return jsonify(ok=False, error={'code': 'prediction_failed', 'message': 'We could not complete the estimate. Please try again.'}), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {400: 'The request contains invalid JSON.', 404: 'This resource was not found.',
                    405: 'This request method is not supported.', 413: 'The request is too large.'}
        return jsonify(ok=False, error={'code': f'http_{error.code}', 'message': messages.get(error.code, 'The request could not be processed.')}), error.code

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_app().run(host='127.0.0.1', port=5000, debug=False)
