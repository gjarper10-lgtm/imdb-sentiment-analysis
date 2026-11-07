"""Verify completed full-data experiments against their saved predictions and bundles."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C
from src.artifacts import load_bundle, inference_loader
from src.engine import compute_metrics, predict


def main():
    torch.set_num_threads(2)
    fingerprints, identities, records = set(), [], {}
    for name in C.MODELS:
        folder = ROOT / 'results' / name
        result = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        predictions = pd.read_csv(folder / 'predictions.csv')
        assert len(predictions) == 7500
        assert predictions.label.value_counts().to_dict() == {0: 3750, 1: 3750}
        assert result['split_sizes'] == {'train':35000, 'val':7500, 'test':7500}
        assert result['max_len'] == 512 and result['subset'] is None
        assert result['best_epoch'] == max(history, key=lambda h: h['val_f1'])['epoch']
        metrics = compute_metrics(predictions.label.to_numpy(), predictions.prob_positive.to_numpy())
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            assert abs(metrics[key] - result['test'][key]) < 1e-12
        assert metrics['confusion_matrix'] == result['test']['confusion_matrix']
        selection = result['selection']
        assert selection['test_used_for_selection'] is False
        assert len(selection['trials']) == 2
        assert selection['selected'] == max(selection['trials'], key=lambda t:t['validation_f1'])['run']
        fingerprints.add(result['split_fingerprint'])
        identities.append(predictions.review_id.tolist())
        model, spec, info = load_bundle(folder, 'cpu')
        probs, _, _ = predict(model, inference_loader(
            ['A moving and beautifully acted film.', 'A dull and disappointing movie.'], spec, info),
            torch.device('cpu'), amp=False)
        assert np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all()
        with (folder / 'model.pt').open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records[name] = {'test_metrics_recomputed': True, 'offline_bundle_prediction': True,
                         'selected_checkpoint_sha256': digest,
                         'example_positive_probabilities': [float(x) for x in probs],
                         'accuracy': result['test']['accuracy'], 'f1': result['test']['f1']}
        del model
    assert len(fingerprints) == 1
    assert all(order == identities[0] for order in identities)
    for name in ['textcnn', 'bilstm']:
        assert json.loads((ROOT/'results'/name/'metrics.json').read_text())['glove'] is True
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT/'src').glob('*.py')}
    verification = {'verified_utc': datetime.now(timezone.utc).isoformat(),
                    'full_experiment_complete': True, 'same_test_reviews_in_same_order': True,
                    'regression_test_command': 'python -m pytest -q',
                    'regression_test_log': 'docs/test_results.txt',
                    'models': records, 'source_sha256': source_hashes,
                    'remote_github_publication': 'See docs/publication.json for current publication state.'}
    (ROOT/'docs/verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
