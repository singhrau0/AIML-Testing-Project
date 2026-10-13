"""Batch inference using a trusted local registered candidate."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from loan_pipeline.artifacts import Run, digest, write_json
from loan_pipeline.validation import validate
from loan_pipeline.cleaning import clean
from loan_pipeline.monitoring import monitor
from loan_pipeline.config import FEATURES


def predict(model_dir, data_path, output_root):
    model_dir = Path(model_dir)
    run = Run(output_root)
    try:
        with run.stage('01_inference') as (folder, report):
            metadata = json.loads((model_dir / 'metadata.json').read_text())
            if not metadata['quality_gate_passed']:
                raise ValueError('Model failed registration quality gates')
            if digest(model_dir / 'model.joblib') != metadata['model_sha256']:
                raise ValueError('Model checksum mismatch')
            data = pd.read_csv(data_path)
            validate(data, folder, report, training=False)
            data = clean(data, folder, report)
            # Only load trusted local joblib artifacts: pickle can execute code.
            model = joblib.load(model_dir / 'model.joblib')
            p = model.predict_proba(data[FEATURES])[:, 1]
            pd.DataFrame({'Loan_ID': data.Loan_ID, 'probability_Y': p,
                          'prediction': ['Y' if v >= metadata['threshold'] else 'N' for v in p]}).to_csv(
                              folder / 'predictions.csv', index=False)
            reference = json.loads((model_dir / 'monitoring_reference.json').read_text())
            write_json(folder / 'drift.json', monitor(data, reference))
            report.update(model_directory=str(model_dir.resolve()), model_sha256=metadata['model_sha256'],
                          source_sha256=digest(data_path), rows=len(data), usage='Offline candidate inference')
        write_json(run.path / 'result.json', {'status': 'completed'})
        return run.path
    except Exception as exc:
        write_json(run.path / 'result.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        run.manifest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'predictions')
    args = parser.parse_args()
    print(predict(args.model_dir, args.data, args.output))
