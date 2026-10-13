"""Run the complete loan training pipeline: python runmain.py."""
import argparse
import shutil
from pathlib import Path
from loan_pipeline.artifacts import Run, write_json
from loan_pipeline.config import Config
from loan_pipeline.ingestion import ingest
from loan_pipeline.validation import validate
from loan_pipeline.cleaning import clean
from loan_pipeline.profiling import profile
from loan_pipeline.splitting import split
from loan_pipeline.preprocessing import describe_preprocessing
from loan_pipeline.training import train_models
from loan_pipeline.evaluation import evaluate
from loan_pipeline.fairness import audit
from loan_pipeline.monitoring import build_reference, monitor
from loan_pipeline.registration import register
from loan_pipeline.reporting import summarize
from loan_pipeline.tracking import track

ROOT = Path(__file__).resolve().parent


def run_pipeline(data_path=ROOT / 'loan_train.csv', artifacts=ROOT / 'artifacts', config=None):
    config = config or Config()
    if not 0 < config.test_size < 1 or config.cv_folds < 2 or not 0 < config.threshold < 1:
        raise ValueError('Invalid split, folds, or threshold configuration')
    run = Run(artifacts)
    print(f'Artifacts: {run.path}', flush=True)
    try:
        with run.stage('01_tracking') as (folder, report):
            track(run, config, folder, report)
            source = folder / 'source'
            source.mkdir()
            shutil.copy2(Path(__file__), source / 'runmain.py')
            for filename in ['predict.py', 'requirements-loan.txt', 'requirements-dev.txt', 'README.md']:
                shutil.copy2(ROOT / filename, source / filename)
            shutil.copytree(ROOT / 'loan_pipeline', source / 'loan_pipeline', ignore=shutil.ignore_patterns('__pycache__'))
        with run.stage('02_ingestion') as (folder, report):
            data = ingest(data_path, folder, report)
        with run.stage('03_validation') as (folder, report):
            validate(data, folder, report)
        with run.stage('04_cleaning') as (folder, report):
            data = clean(data, folder, report)
        with run.stage('05_profiling') as (folder, report):
            profile(data, folder, report)
        with run.stage('06_splitting') as (folder, report):
            train, test = split(data, config, folder, report)
        with run.stage('07_preprocessing') as (folder, report):
            describe_preprocessing(folder, report)
        with run.stage('08_training') as (folder, report):
            name, model, baseline = train_models(train, config, folder, report)
        with run.stage('09_evaluation') as (folder, report):
            metrics, base = evaluate(model, baseline, test, config, folder, report)
        with run.stage('10_fairness') as (folder, report):
            audit(model, test, config, folder, report)
        with run.stage('11_monitoring') as (folder, report):
            reference = build_reference(train, folder, report)
            write_json(folder / 'holdout_drift.json', monitor(test, reference))
        with run.stage('12_registration') as (folder, report):
            metadata = register(model, name, metrics, base, reference, config, folder, report)
        with run.stage('13_reporting') as (folder, report):
            summarize(run, folder, report)
        write_json(run.path / 'result.json', {'status': 'completed', 'model': metadata})
        write_json(Path(artifacts) / 'latest.json', {'run_path': str(run.path), 'model_status': metadata['status']})
        return run.path
    except Exception as exc:
        write_json(run.path / 'result.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        run.manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'loan_train.csv')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--min-roc-auc', type=float, default=.70)
    parser.add_argument('--min-balanced-accuracy', type=float, default=.60)
    args = parser.parse_args()
    path = run_pipeline(args.data, args.artifacts, Config(seed=args.seed, cv_folds=args.cv_folds,
        min_roc_auc=args.min_roc_auc, min_balanced_accuracy=args.min_balanced_accuracy))
    print(f'Completed. Report: {path / "13_reporting/index.html"}')


if __name__ == '__main__':
    main()
