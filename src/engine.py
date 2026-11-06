"""Training loop and evaluation harness shared by all four models."""
import copy
import time
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm


def _to(inputs, device):
    return {k: (v if k == 'lengths' else v.to(device, non_blocking=True)) for k, v in inputs.items()}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, preds)), "precision": float(p),
            "recall": float(r), "f1": float(f1),
            "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist()}


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for inputs, y in loader:
        inputs, y = _to(inputs, device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(**inputs)
        loss_sum += loss_fn(logits.float(), y).item()
        n += len(y)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(np.int64), loss_sum / max(n, 1)


def train(model, loaders, device, *, lr, epochs, patience, weight_decay=0.0,
          grad_accum=1, warmup_frac=0.0, amp=True, clip=1.0, progress_path=None):
    """Train with AdamW; keep the weights with the best validation F1 (early stopping)."""
    if min(epochs, patience, grad_accum) < 1:
        raise ValueError('epochs, patience and grad_accum must be positive')
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    total_steps = epochs * math.ceil(len(loaders["train"]) / grad_accum)
    warmup = int(warmup_frac * total_steps)
    sched = None
    if warmup_frac > 0:   # linear warmup then linear decay (transformer fine-tuning recipe)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
            else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, seen = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        accumulated_samples = 0
        bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}", leave=False)
        for step, (inputs, y) in enumerate(bar, 1):
            inputs, y = _to(inputs, device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**inputs).float(), y)
            # Sum losses then normalise by the actual number of samples, including
            # the final partial batch / accumulation group.
            scaler.scale(loss * len(y)).backward()
            accumulated_samples += len(y)
            if step % grad_accum == 0 or step == len(loaders["train"]):
                scaler.unscale_(opt)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accumulated_samples = 0
                if sched:
                    sched.step()
            running += loss.item() * len(y)
            seen += len(y)
            bar.set_postfix(loss=f"{running / seen:.4f}")
        _sync(device)
        epoch_time = time.perf_counter() - t0

        val_probs, val_labels, val_loss = predict(model, loaders["val"], device, amp)
        val = compute_metrics(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss,
                        "val_accuracy": val["accuracy"], "val_f1": val["f1"],
                        "epoch_time_s": epoch_time})
        print(f"epoch {epoch}: train_loss={running / seen:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f} ({epoch_time:.1f}s)")

        if val["f1"] > best_f1:
            best_f1, bad_epochs = val["f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping (no val F1 gain for {patience} epochs)")
                break

        if progress_path is not None:
            import json
            progress_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    train_time = time.perf_counter() - train_start
    model.load_state_dict(best_state)
    return history, train_time


@torch.no_grad()
def measure_latency(model, loader, device, n_single=200, amp=True):
    """Throughput over the full test set (batched) and mean latency for single reviews."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    ctx = lambda: torch.autocast(device.type, dtype=torch.float16, enabled=use_amp)

    for i, (inputs, _) in enumerate(loader):
        with ctx():
            model(**_to(inputs, device))
        if i >= 2:
            break
    _sync(device)
    t0, n = time.perf_counter(), 0
    for inputs, y in loader:
        with ctx():
            model(**_to(inputs, device))
        n += len(y)
    _sync(device)
    batched_ms = (time.perf_counter() - t0) * 1000 / n

    singles = []
    for inputs, _ in loader:
        for i in range(len(next(iter(inputs.values())))):
            one = {k: v[i:i + 1] for k, v in inputs.items()}
            if "lengths" in one:      # trim padding to this review's real length
                one["ids"] = one["ids"][:, :int(one["lengths"][0])]
            if "attention_mask" in one:
                L = int(one["attention_mask"].sum())
                one = {k: v[:, :L] for k, v in one.items()}
            singles.append(one)
            if len(singles) >= n_single + 10:
                break
        if len(singles) >= n_single + 10:
            break
    for one in singles[:10]:          # warm-up
        with ctx():
            model(**_to(one, device))
    _sync(device)
    measured = singles[10:] or singles
    t0 = time.perf_counter()
    for one in measured:
        with ctx():
            model(**_to(one, device))
        _sync(device)
    single_ms = (time.perf_counter() - t0) * 1000 / max(1, len(measured))
    return {"latency_ms_per_review_batched": batched_ms, "latency_ms_single_review": single_ms,
            "latency_single_samples": len(measured),
            "latency_scope": "pretokenized input, includes loader and device transfer; excludes tokenization"}
