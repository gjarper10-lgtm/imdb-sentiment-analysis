"""Reproduce the synopsis: EDA, tuning, test evaluation, analysis, documents."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=3, help='maximum epochs per tuning candidate')
    p.add_argument('--subset', type=int, help='smoke run: separate results directory')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--glove-path')
    p.add_argument('--models', nargs='+', choices=['mlp', 'textcnn', 'bilstm', 'distilbert'],
                   default=['mlp', 'textcnn', 'bilstm', 'distilbert'])
    p.add_argument('--no-glove', action='store_true')
    p.add_argument('--output-root', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    output = (args.output_root or (root / ('results_smoke' if args.subset else 'results'))).resolve()
    if args.no_glove and output == root / 'results':
        p.error('--no-glove is an ablation: specify --output-root results_random')
    def execute(arguments):
        subprocess.run([sys.executable, *arguments], cwd=root, check=True)
    if not args.subset:
        execute(['-m', 'src.eda', '--output-root', str(output)])
    for model in args.models:
        call = ['-m', 'src.tune', '--model', model, '--epochs', str(args.epochs), '--output-root', str(output)]
        for name in ('subset', 'batch_size', 'glove_path'):
            value = getattr(args, name)
            if value is not None:
                call.extend(['--' + name.replace('_', '-'), str(value)])
        if args.no_glove:
            call.append('--no-glove')
        execute(call)
    from src import config as C, compare, analysis
    C.RESULTS_DIR = output
    compare.main()
    analysis.main()
    if output == root / 'results':
        execute(['docs/build_report.py'])
        execute(['docs/build_slides.py'])
        execute(['docs/export_pdf.py'])
    print(f'Experiment artifacts: {output}')


if __name__ == '__main__':
    main()
