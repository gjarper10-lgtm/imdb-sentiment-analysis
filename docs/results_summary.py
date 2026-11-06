"""Read the experiment outputs for the report and slides. Missing values -> None."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ORDER = ["mlp", "textcnn", "bilstm", "distilbert"]
SHORT = {"mlp": "MLP (TF-IDF BoW)", "textcnn": "TextCNN", "bilstm": "BiLSTM + Attention",
         "distilbert": "DistilBERT"}
TBD = "TBD"


def load():
    """{model: metrics dict or None}, eda stats or None, and whether runs were full-data."""
    runs = {}
    for m in ORDER:
        f = RESULTS / m / "metrics.json"
        runs[m] = json.loads(f.read_text()) if f.exists() else None
    eda_f = RESULTS / "eda" / "eda_stats.json"
    eda = json.loads(eda_f.read_text()) if eda_f.exists() else None
    complete = all(runs.values())
    full = complete and all(r.get("subset") is None for r in runs.values())
    return runs, eda, complete, full


def fmt_pct(x):
    return TBD if x is None else f"{x * 100:.2f}"


def rows(runs):
    """One row of display strings per model for the results table."""
    out = []
    for m in ORDER:
        r = runs[m]
        if r is None:
            out.append([SHORT[m]] + [TBD] * 7)
            continue
        t = r["test"]
        out.append([SHORT[m], fmt_pct(t["accuracy"]), fmt_pct(t["precision"]),
                    fmt_pct(t["recall"]), fmt_pct(t["f1"]),
                    f"{r['params_total'] / 1e6:.2f}M", f"{r['train_time_s'] / 60:.1f}",
                    f"{r['latency_ms_single_review']:.2f}"])
    return out


HEADER = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "Params",
          "Train time (min)", "Latency (ms/review)"]


def findings(runs):
    """Data-driven sentences for the discussion; empty until all four runs exist."""
    if not all(runs.values()):
        return []
    acc = {m: runs[m]["test"]["accuracy"] for m in ORDER}
    best = max(acc, key=acc.get)
    scratch_best = max(["mlp", "textcnn", "bilstm"], key=acc.get)
    gap = (acc["distilbert"] - acc[scratch_best]) * 100
    params = {m: runs[m]["params_total"] for m in ORDER}
    lat = {m: runs[m]["latency_ms_single_review"] for m in ORDER}
    ttime = {m: runs[m]["train_time_s"] for m in ORDER}
    cnn_share = acc["textcnn"] / acc["distilbert"] * 100
    return [
        f"{SHORT[best]} achieved the highest test accuracy ({acc[best] * 100:.2f}%).",
        f"The strongest model trained from scratch was {SHORT[scratch_best]} "
        f"({acc[scratch_best] * 100:.2f}%); the fine-tuned transformer "
        f"{'outperformed' if gap > 0 else 'trailed'} it by {abs(gap):.2f} percentage points.",
        f"TextCNN reached {cnn_share:.1f}% of DistilBERT's accuracy with "
        f"{params['textcnn'] / params['distilbert'] * 100:.1f}% of its parameters and "
        f"{lat['textcnn'] / lat['distilbert'] * 100:.1f}% of its single-review latency.",
        f"The MLP bag-of-words baseline reached {acc['mlp'] * 100:.2f}% while training in "
        f"{ttime['mlp'] / 60:.1f} min, versus {ttime['distilbert'] / 60:.1f} min for DistilBERT "
        f"fine-tuning.",
    ]


if __name__ == "__main__":
    r, e, c, f = load()
    print("complete:", c, "full-data:", f)
    for row in rows(r):
        print(row)
    print("\n".join(findings(r)), file=sys.stdout)
