"""Exploratory data analysis: review lengths, vocabulary size, class balance.

    python -m src.eda
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .data import load_splits, tokenize


def main():
    out = C.RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    texts = [t for s in splits.values() for t in s[0]]
    labels = np.concatenate([s[1] for s in splits.values()])
    lengths = np.array([len(tokenize(t)) for t in texts])
    train_counts = Counter(w for t in splits["train"][0] for w in tokenize(t))
    top10k = sum(c for _, c in train_counts.most_common(C.VOCAB_SIZE - 2))

    stats = {
        "n_reviews": len(texts),
        "split_sizes": {s: len(v[1]) for s, v in splits.items()},
        "positive_fraction": float(labels.mean()),
        "length_words": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                         "p90": float(np.percentile(lengths, 90)),
                         "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max())},
        "fraction_truncated_at_max_len": float((lengths > C.MAX_LEN).mean()),
        "train_vocabulary_size": len(train_counts),
        "train_token_coverage_of_10k_vocab": top10k / sum(train_counts.values()),
    }
    (out / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, name in ((1, "positive"), (0, "negative")):
        ax.hist(np.clip(lengths[labels == lab], 0, 1500), bins=60, alpha=0.6, label=name)
    ax.axvline(C.MAX_LEN, color="k", ls="--", lw=1, label=f"max_len = {C.MAX_LEN}")
    ax.set(xlabel="review length (words, clipped at 1500)", ylabel="reviews",
           title="IMDB review length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=150)

    # Most class-indicative words (log-odds ratio with add-one smoothing) on the train split.
    pos = Counter(w for t, y in zip(*splits["train"]) if y == 1 for w in set(tokenize(t)))
    neg = Counter(w for t, y in zip(*splits["train"]) if y == 0 for w in set(tokenize(t)))
    common = [w for w, c in train_counts.most_common(5000)]
    lor = {w: np.log((pos[w] + 1) / (neg[w] + 1)) for w in common}
    ranked = sorted(lor, key=lor.get)
    (out / "indicative_words.json").write_text(json.dumps(
        {"negative": ranked[:30], "positive": ranked[::-1][:30]}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
