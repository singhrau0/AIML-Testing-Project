import joblib
from .artifacts import write_json, digest
from .config import FEATURES


def register(model, name, metrics, baseline, reference, config, folder, report):
    checks = {'roc_auc': metrics['roc_auc'] >= config.min_roc_auc,
              'balanced_accuracy': metrics['balanced_accuracy'] >= config.min_balanced_accuracy,
              'beats_baseline_auc': metrics['roc_auc'] > baseline['roc_auc']}
    eligible = all(checks.values())
    model_path = folder / 'model.joblib'
    joblib.dump(model, model_path)
    metadata = {'model_name': name, 'status': 'candidate' if eligible else 'rejected',
                'quality_gate_passed': eligible, 'checks': checks, 'features': FEATURES,
                'threshold': config.threshold, 'labels': {'0': 'N', '1': 'Y'},
                'model_sha256': digest(model_path), 'config': config.to_dict(),
                'production_approved': False}
    write_json(folder / 'metadata.json', metadata)
    write_json(folder / 'monitoring_reference.json', reference)
    report.update(metadata)
    (folder / 'model_card.md').write_text(
        f'# Loan model: {name}\n\nStatus: {metadata["status"]}. Production approved: no.\n\n'
        'Predicts historical Loan_Status (Y/N), not default risk or repayment probability. '
        'Training provenance and representativeness are unknown. Random stratified evaluation '
        'does not establish future or out-of-population performance. Gender and Married are excluded '
        'from prediction but proxies may remain. Imputation and encoding are bundled with the model.\n\n'
        f'Holdout ROC AUC: {metrics["roc_auc"]:.4f}; balanced accuracy: {metrics["balanced_accuracy"]:.4f}.\n\n'
        'Before production: review label definition, consent and permitted use, temporal/external validation, '
        'fairness, calibration and operational error costs. Set an approved threshold using training validation '
        'data, never the test set. Use human review, shadow deployment, monitored rollout and rollback.\n')
    return metadata
