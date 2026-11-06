"""Train and evaluate one model under the common protocol.

    python -m src.run --model textcnn
    python -m src.run --model distilbert --subset 2000 --epochs 1     # quick check
"""
import argparse
import json
import platform
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import bert_loaders, load_splits, sequence_loaders, set_seed, tfidf_loaders, split_fingerprint
from .artifacts import save_bundle
from .engine import compute_metrics, measure_latency, predict, train
from .models import BiLSTMAttention, DistilBERTClassifier, MLPBoW, TextCNN, count_parameters

DISPLAY_NAMES = {"mlp": "MLP (TF-IDF BoW)", "textcnn": "TextCNN (GloVe-300)",
                 "bilstm": "BiLSTM + Attention (GloVe-300)", "distilbert": "DistilBERT (fine-tuned)"}


def build(model_name, splits, args):
    bs = args.batch_size
    if model_name == "mlp":
        loaders, info = tfidf_loaders(splits, bs, args.max_len)
        return MLPBoW(info["input_dim"]), loaders, info
    if model_name in ("textcnn", "bilstm"):
        loaders, info = sequence_loaders(splits, bs, args.max_len)
        vocab = info["vocab"]
        emb = None
        if not args.no_glove:
            from .glove import build_embedding_matrix
            emb = build_embedding_matrix(vocab, args.glove_path, seed=args.seed)
        cls = TextCNN if model_name == "textcnn" else BiLSTMAttention
        return cls(len(vocab), emb, freeze_emb=args.freeze_emb), loaders, info
    if model_name == "distilbert":
        loaders, info = bert_loaders(splits, bs, args.max_len)
        return DistilBERTClassifier(info["model_name"]), loaders, info
    raise ValueError(model_name)


def save_error_analysis(out_dir, texts, labels, probs, k=50):
    """The k most confidently wrong test reviews, for qualitative error analysis."""
    preds = (probs >= 0.5).astype(int)
    wrong = np.where(preds != labels)[0]
    confidence = np.abs(probs[wrong] - 0.5)
    top = wrong[np.argsort(-confidence)][:k]
    pd.DataFrame({"label": labels[top], "prob_positive": probs[top].round(4),
                  "review": [texts[i] for i in top]}).to_csv(out_dir / "misclassified.csv",
                                                              index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, choices=C.MODELS)
    p.add_argument("--subset", type=int, default=None, help="use N reviews total (quick runs)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--max-len", type=int, default=C.MAX_LEN)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--no-glove", action="store_true", help="random embeddings instead of GloVe")
    p.add_argument("--glove-path", help="existing glove.6B.300d.txt (skips the download)")
    p.add_argument("--freeze-emb", action="store_true", help="keep GloVe embeddings frozen")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision")
    p.add_argument("--save-model", action="store_true", help="also save the best weights")
    p.add_argument("--tag", default="", help="suffix for the results folder")
    p.add_argument('--output-root', type=Path, default=C.RESULTS_DIR)
    p.add_argument('--validation-only', action='store_true', help='tuning trial: never evaluate test labels')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args(argv)

    hp = dict(C.MODEL_DEFAULTS[args.model])
    for key in ("epochs", "batch_size", "lr"):
        if getattr(args, key) is not None:
            hp[key] = getattr(args, key)
    args.batch_size = hp["batch_size"]
    if min(hp['epochs'], hp['batch_size']) < 1 or hp['lr'] <= 0 or not 4 <= args.max_len <= 512:
        p.error('epochs, batch-size and lr must be positive; max-len must be 4..512')
    if args.tag and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        p.error('tag may only contain letters, digits, underscores and hyphens')
    out_dir = args.output_root / (args.model + (f"_{args.tag}" if args.tag else ""))
    if (out_dir / 'run.json').exists() and not args.overwrite:
        p.error(f'{out_dir} already exists; use a new --tag or explicitly --overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    splits = load_splits(args.subset, args.seed)
    print({s: len(v[1]) for s, v in splits.items()})
    model, loaders, info = build(args.model, splits, args)
    total, trainable = count_parameters(model)
    print(f"{DISPLAY_NAMES[args.model]}: {total:,} params ({trainable:,} trainable)")

    history, train_time = train(
        model, loaders, device, lr=hp["lr"], epochs=hp["epochs"], patience=hp["patience"],
        weight_decay=hp.get("weight_decay", 0.0), grad_accum=hp.get("grad_accum", 1),
        warmup_frac=hp.get("warmup_frac", 0.0), amp=not args.no_amp,
        progress_path=out_dir / 'history.json')
    result = {
        "model": args.model,
        "display_name": DISPLAY_NAMES[args.model].replace("GloVe-300", "random emb.")
        if args.no_glove else DISPLAY_NAMES[args.model],
        'validation_f1': max(h['val_f1'] for h in history),
        'best_epoch': max(history, key=lambda h: h['val_f1'])['epoch'],
        "params_total": total, "params_trainable": trainable,
        "train_time_s": train_time, "epochs_run": len(history),
        "hyperparameters": hp, "subset": args.subset, "max_len": args.max_len,
        "glove": args.model in ("textcnn", "bilstm") and not args.no_glove,
        "seed": args.seed, "device": torch.cuda.get_device_name() if device.type == "cuda" else
        platform.processor() or "cpu",
        "split_sizes": {s: len(v[1]) for s, v in splits.items()},
        'split_fingerprint': split_fingerprint(splits),
        'freeze_emb': args.freeze_emb, 'amp': not args.no_amp and device.type == 'cuda',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'versions': {name: version(name) for name in ['torch', 'transformers', 'datasets', 'scikit-learn', 'numpy']},
    }
    (out_dir / 'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding='utf-8')
    save_bundle(model, info, args, out_dir)
    if not args.validation_only:
        result = evaluate(model, loaders['test'], splits['test'], result, out_dir, device)
    print(f"saved to {out_dir}")
    return result


def evaluate(model, loader, split, result, out_dir, device):
    probs, labels, loss = predict(model, loader, device, result['amp'])
    result = dict(result)
    result['test'] = {**compute_metrics(labels, probs), 'loss': loss}
    result.update(measure_latency(model, loader, device, amp=result['amp']))
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    save_error_analysis(out_dir, split[0], labels, probs)
    pd.DataFrame({'review_id': [hashlib.sha256(t.encode()).hexdigest() for t in split[0]],
                  'label': labels, 'prob_positive': probs, 'prediction': (probs >= .5).astype(int),
                  'length_words': [len(t.split()) for t in split[0]]}).to_csv(out_dir / 'predictions.csv', index=False)
    print(f"TEST {result['model']}: accuracy={result['test']['accuracy']:.4f}, F1={result['test']['f1']:.4f}")
    return result


if __name__ == "__main__":
    main()
