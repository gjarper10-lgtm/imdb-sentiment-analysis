"""Train and evaluate one model under the common protocol.

    python -m src.run --model textcnn
    python -m src.run --model distilbert --subset 2000 --epochs 1     # quick check
"""
import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import bert_loaders, load_splits, sequence_loaders, set_seed, tfidf_loaders
from .engine import compute_metrics, measure_latency, predict, train
from .models import BiLSTMAttention, DistilBERTClassifier, MLPBoW, TextCNN, count_parameters

DISPLAY_NAMES = {"mlp": "MLP (TF-IDF BoW)", "textcnn": "TextCNN (GloVe-300)",
                 "bilstm": "BiLSTM + Attention (GloVe-300)", "distilbert": "DistilBERT (fine-tuned)"}


def build(model_name, splits, args):
    bs = args.batch_size
    if model_name == "mlp":
        loaders, info = tfidf_loaders(splits, bs)
        return MLPBoW(info["input_dim"]), loaders
    if model_name in ("textcnn", "bilstm"):
        loaders, info = sequence_loaders(splits, bs, args.max_len)
        vocab = info["vocab"]
        emb = None
        if not args.no_glove:
            from .glove import build_embedding_matrix
            emb = build_embedding_matrix(vocab, args.glove_path)
        cls = TextCNN if model_name == "textcnn" else BiLSTMAttention
        return cls(len(vocab), emb, freeze_emb=args.freeze_emb), loaders
    if model_name == "distilbert":
        loaders, info = bert_loaders(splits, bs, args.max_len)
        return DistilBERTClassifier(info["model_name"]), loaders
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
    args = p.parse_args(argv)

    hp = dict(C.MODEL_DEFAULTS[args.model])
    for key in ("epochs", "batch_size", "lr"):
        if getattr(args, key) is not None:
            hp[key] = getattr(args, key)
    args.batch_size = hp["batch_size"]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    splits = load_splits(args.subset, args.seed)
    print({s: len(v[1]) for s, v in splits.items()})
    model, loaders = build(args.model, splits, args)
    total, trainable = count_parameters(model)
    print(f"{DISPLAY_NAMES[args.model]}: {total:,} params ({trainable:,} trainable)")

    history, train_time = train(
        model, loaders, device, lr=hp["lr"], epochs=hp["epochs"], patience=hp["patience"],
        weight_decay=hp.get("weight_decay", 0.0), grad_accum=hp.get("grad_accum", 1),
        warmup_frac=hp.get("warmup_frac", 0.0), amp=not args.no_amp)

    probs, labels, test_loss = predict(model, loaders["test"], device, not args.no_amp)
    metrics = compute_metrics(labels, probs)
    latency = measure_latency(model, loaders["test"], device, amp=not args.no_amp)

    out_dir = C.RESULTS_DIR / (args.model + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "display_name": DISPLAY_NAMES[args.model].replace("GloVe-300", "random emb.")
        if args.no_glove else DISPLAY_NAMES[args.model],
        "test": {**metrics, "loss": test_loss}, **latency,
        "params_total": total, "params_trainable": trainable,
        "train_time_s": train_time, "epochs_run": len(history),
        "hyperparameters": hp, "subset": args.subset, "max_len": args.max_len,
        "glove": args.model in ("textcnn", "bilstm") and not args.no_glove,
        "seed": args.seed, "device": torch.cuda.get_device_name() if device.type == "cuda" else
        platform.processor() or "cpu",
        "split_sizes": {s: len(v[1]) for s, v in splits.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_error_analysis(out_dir, splits["test"][0], labels, probs)
    if args.save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")

    m = result["test"]
    print(f"\nTEST  acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  | train {train_time:.0f}s | "
          f"{latency['latency_ms_single_review']:.2f} ms/review (single)")
    print(f"saved to {out_dir}")
    return result


if __name__ == "__main__":
    main()
