"""Load the trusted, versioned local model once, with fail-closed checks."""
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from loan_pipeline.config import FEATURES


class ModelService:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        runtime = json.loads((directory / 'runtime.json').read_text())
        if importlib.metadata.version('scikit-learn') != runtime['packages']['scikit-learn']:
            raise RuntimeError('Model runtime mismatch. Install requirements.txt in the web environment.')
        path = directory / 'model.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata['model_sha256']:
            raise RuntimeError('Model integrity check failed.')
        if not self.metadata['quality_gate_passed']:
            raise RuntimeError('Model has not passed its candidate quality gates.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            self.model = joblib.load(path)
        if not isinstance(self.model, Pipeline):
            raise RuntimeError('Expected the saved preprocessing and estimator pipeline.')
        self.features = list(self.model.feature_names_in_)
        if self.features != FEATURES or self.features != self.metadata['features']:
            raise RuntimeError('Model feature contract mismatch.')
        self.classes = [int(v) for v in self.model.classes_]
        if set(self.classes) != {0, 1} or self.metadata['labels'] != {'0': 'N', '1': 'Y'}:
            raise RuntimeError('Model label contract mismatch.')
        self.threshold = float(self.metadata['threshold'])
        if not 0 < self.threshold < 1:
            raise RuntimeError('Invalid model threshold.')
        self.supports_probability = callable(getattr(self.model, 'predict_proba', None))

    def predict(self, values):
        frame = pd.DataFrame([values], columns=self.features)
        probability = None
        if self.supports_probability:
            probability = float(self.model.predict_proba(frame)[0, self.classes.index(1)])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise RuntimeError('Invalid model probability.')
            label = int(probability >= self.threshold)
        else:
            label = int(self.model.predict(frame)[0])
        if label not in (0, 1):
            raise RuntimeError('Invalid prediction label.')
        return {
            'label': self.metadata['labels'][str(label)],
            'decision': 'approved' if label == 1 else 'rejected',
            'title': 'Loan Approval Likely' if label == 1 else 'Loan Approval Unlikely',
            'approval_probability': probability,
            'threshold': self.threshold,
            'model': 'Random Forest',
            'disclaimer': 'AI-generated estimate, not an official lending decision. Model probabilities are not a guarantee of approval.',
        }
