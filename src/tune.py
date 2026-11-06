"""Two-rate validation-only search; evaluate the selected checkpoint once on test.

python -m src.tune --model mlp --epochs 3
python -m src.tune --model distilbert --epochs 3
"""
import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from . import config as C
from . import run
from .artifacts import load_bundle, inference_loader
from .data import load_splits, split_fingerprint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=C.MODELS, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--learning-rates', type=float, nargs='+')
    p.add_argument('--subset', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--max-len', type=int, default=C.MAX_LEN)
    p.add_argument('--glove-path')
    p.add_argument('--no-glove', action='store_true')
    p.add_argument('--output-root', type=Path, default=C.RESULTS_DIR)
    args = p.parse_args(argv)
    rates = args.learning_rates or ([2e-5, 5e-5] if args.model == 'distilbert' else [1e-3, 3e-4])
    if args.epochs < 1 or any(rate <= 0 for rate in rates):
        p.error('epochs and learning rates must be positive')
    trials_root = args.output_root / 'tuning'
    candidates = []
    for index, rate in enumerate(rates, 1):
        tag = f'trial{index}'
        directory = trials_root / f'{args.model}_{tag}'
        call = ['--model', args.model, '--epochs', str(args.epochs), '--lr', str(rate),
                '--seed', str(args.seed), '--max-len', str(args.max_len),
                '--tag', tag, '--validation-only', '--output-root', str(trials_root)]
        for flag in ('subset', 'batch_size', 'glove_path'):
            value = getattr(args, flag)
            if value is not None:
                call.extend(['--' + flag.replace('_', '-'), str(value)])
        if args.no_glove:
            call.append('--no-glove')
        if (directory / 'bundle.json').exists() and (directory / 'run.json').exists():
            result = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
            expected_glove = args.model in ('textcnn', 'bilstm') and not args.no_glove
            if (result['hyperparameters']['lr'] != rate or result['hyperparameters']['epochs'] != args.epochs
                    or result['seed'] != args.seed or result['subset'] != args.subset
                    or result['max_len'] != args.max_len or result['glove'] != expected_glove
                    or (args.batch_size is not None and result['hyperparameters']['batch_size'] != args.batch_size)):
                raise ValueError(f'Existing trial configuration differs: {directory}; use another output-root')
            print(f'Resuming completed trial: {directory}')
        else:
            result = run.main(call)
        candidates.append((directory, result))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected, result = max(candidates, key=lambda pair: pair[1]['validation_f1'])
    summary = {'selection_metric': 'validation_f1', 'test_used_for_selection': False,
               'selected': selected.name,
               'total_search_train_time_s': sum(r['train_time_s'] for _, r in candidates),
               'trials': [{'run': d.name, 'lr': r['hyperparameters']['lr'],
                           'validation_f1': r['validation_f1'], 'best_epoch': r['best_epoch'],
                           'train_time_s': r['train_time_s']} for d, r in candidates]}
    (trials_root / f'{args.model}_selection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    destination = args.output_root / args.model
    if (destination / 'metrics.json').exists():
        previous = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
        if previous.get('selection') != summary:
            raise ValueError(f'{destination} has a different result; use another output-root')
        print(f'Final test result already exists: {destination}')
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    for source in selected.iterdir():
        if source.is_dir():
            shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination / source.name)
    splits = load_splits(args.subset, args.seed)
    if split_fingerprint(splits) != result['split_fingerprint']:
        raise ValueError('Dataset or split changed since tuning')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, spec, info = load_bundle(destination, device)
    test = splits['test']
    loader = inference_loader(test[0], spec, info, result['hyperparameters']['batch_size'], test[1])
    result['selection'] = summary
    return run.evaluate(model, loader, test, result, destination, device)


if __name__ == '__main__':
    main()
