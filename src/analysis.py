"""Measured error analysis, confusion matrices, ROC curves and cost tradeoffs."""
import json
import os
from . import config as C
os.environ.setdefault('MPLCONFIGDIR', str(C.DATA_DIR / 'matplotlib-cache'))
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from . import config as C
from .compare import load_results


def wilson(correct, total):
    z = 1.96
    p = correct / total
    centre = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [centre-radius, centre+radius]


def main():
    df = load_results()
    if df.empty:
        return
    out = C.RESULTS_DIR
    analyses = {}
    fig_cm, axes = plt.subplots(1, len(df), figsize=(4*len(df), 4), squeeze=False)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    for i, row in df.iterrows():
        directory = out / row['run']
        metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
        predictions = pd.read_csv(directory / 'predictions.csv')
        correct = predictions.label == predictions.prediction
        cm = np.asarray(metrics['test']['confusion_matrix'])
        ax = axes[0, i]
        ax.imshow(cm, cmap='Blues')
        for y in range(2):
            for x in range(2):
                ax.text(x, y, str(cm[y, x]), ha='center', va='center',
                        color='white' if cm[y, x] > cm.max()/2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'], xlabel='Predicted', ylabel='Actual', title=row['run'])
        fpr, tpr, _ = roc_curve(predictions.label, predictions.prob_positive)
        auc = roc_auc_score(predictions.label, predictions.prob_positive)
        ax_roc.plot(fpr, tpr, label=f"{row['run']} (AUC {auc:.3f})")
        length_groups = {}
        for name, mask in [('at_most_512_words', predictions.length_words <= 512),
                            ('over_512_words', predictions.length_words > 512)]:
            length_groups[name] = {'n': int(mask.sum()),
                                   'error_rate': float((~correct[mask]).mean()) if mask.any() else None}
        examples = pd.read_csv(directory / 'misclassified.csv').head(3).to_dict('records')
        analyses[row['run']] = {'accuracy_95ci_wilson': wilson(int(correct.sum()), len(correct)),
            'roc_auc': float(auc), 'false_positive': int(cm[0, 1]), 'false_negative': int(cm[1, 0]),
            'length_groups': length_groups, 'most_confident_errors': examples}
    fig_cm.tight_layout()
    fig_cm.savefig(out / 'confusion_matrices.png', dpi=150)
    plt.close(fig_cm)
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate', title='IMDB test ROC curves')
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(out / 'roc_curves.png', dpi=150)
    plt.close(fig_roc)
    pareto = {}
    for cost in ['params_M', 'ms_per_review_single', 'train_time_min']:
        pareto[cost] = []
        for _, candidate in df.iterrows():
            dominates = ((df.accuracy >= candidate.accuracy) & (df[cost] <= candidate[cost])
                         & ((df.accuracy > candidate.accuracy) | (df[cost] < candidate[cost])))
            if not dominates.any():
                pareto[cost].append(candidate['run'])
    (out / 'analysis.json').write_text(json.dumps({'models': analyses, 'pareto_frontiers': pareto,
        'confidence_interval_note': 'Test-sample uncertainty only; does not capture variation across training seeds.',
        'length_note': 'Word lengths are before truncation. DistilBERT uses subword tokens; the word-length comparison is descriptive, not a causal test.'}, indent=2), encoding='utf-8')
    print('Saved confusion matrices, ROC curves and measured error analysis')


if __name__ == '__main__':
    main()
