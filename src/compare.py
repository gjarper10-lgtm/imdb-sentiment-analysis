"""Collect results/*/metrics.json into a comparison table and trade-off plots.

    python -m src.compare
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def load_results():
    rows = []
    order = lambda f: (C.MODELS.index(f.parent.name.split("_")[0])
                       if f.parent.name.split("_")[0] in C.MODELS else 99, f.parent.name)
    for f in sorted(C.RESULTS_DIR.glob("*/metrics.json"), key=order):
        r = json.loads(f.read_text())
        rows.append({
            "run": f.parent.name, "model": r["display_name"],
            "accuracy": r["test"]["accuracy"], "precision": r["test"]["precision"],
            "recall": r["test"]["recall"], "f1": r["test"]["f1"],
            "params_M": r["params_total"] / 1e6, "trainable_M": r["params_trainable"] / 1e6,
            "train_time_min": r["train_time_s"] / 60, "epochs": r["epochs_run"],
            "ms_per_review_single": r["latency_ms_single_review"],
            "ms_per_review_batched": r["latency_ms_per_review_batched"],
            "subset": r.get("subset"),
        })
    return pd.DataFrame(rows)


def plot_curves(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f in sorted(C.RESULTS_DIR.glob("*/history.json")):
        h = pd.DataFrame(json.loads(f.read_text()))
        axes[0].plot(h["epoch"], h["val_loss"], marker="o", label=f.parent.name)
        axes[1].plot(h["epoch"], h["val_f1"], marker="o", label=f.parent.name)
    axes[0].set(xlabel="epoch", ylabel="validation loss")
    axes[1].set(xlabel="epoch", ylabel="validation F1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)


def main():
    df = load_results()
    if df.empty:
        print("no results yet - run `python -m src.run --model <name>` first")
        return
    out = C.RESULTS_DIR
    df.to_csv(out / "comparison.csv", index=False)

    table = df.drop(columns=["run", "subset"]).copy()
    for col in ("accuracy", "precision", "recall", "f1"):
        table[col] = (table[col] * 100).map("{:.2f}".format)
    for col in ("params_M", "trainable_M", "train_time_min"):
        table[col] = table[col].map("{:.2f}".format)
    for col in ("ms_per_review_single", "ms_per_review_batched"):
        table[col] = table[col].map("{:.3f}".format)
    (out / "comparison.md").write_text(table.to_markdown(index=False))
    print(table.to_markdown(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, x, label in ((axes[0], "params_M", "parameters (millions, log)"),
                         (axes[1], "ms_per_review_single", "latency per review (ms, log)"),
                         (axes[2], "train_time_min", "training time (min, log)")):
        ax.scatter(df[x], df["accuracy"] * 100, s=70)
        for _, r in df.iterrows():
            ax.annotate(r["model"].split(" (")[0], (r[x], r["accuracy"] * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set(xscale="log", xlabel=label, ylabel="test accuracy (%)")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs. cost on IMDB (common protocol)")
    fig.tight_layout()
    fig.savefig(out / "tradeoffs.png", dpi=150)
    plot_curves(out)
    print(f"\nsaved comparison.md / comparison.csv / tradeoffs.png / learning_curves.png to {out}")


if __name__ == "__main__":
    main()
