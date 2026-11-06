"""Dataset loading, the shared preprocessing pipeline, and per-model DataLoaders."""
import random
import re
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config as C

_HTML = re.compile(r"<[^>]+>")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    """Lowercase, strip HTML line breaks and punctuation, collapse whitespace."""
    text = _HTML.sub(" ", text.lower())
    text = text.replace("'", "")            # don't -> dont, keeps negations as one token
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return text.split()


def load_splits(subset: int | None = None, seed: int = C.SEED):
    """Pool the 50k IMDB reviews and make a stratified 70/15/15 split.

    Returns {"train"|"val"|"test": (cleaned_texts, labels)}. `subset` keeps a
    stratified sample of that many reviews in total (useful for quick runs).
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    labels = np.array(list(ds["train"]["label"]) + list(ds["test"]["label"]), dtype=np.int64)

    if subset and subset < len(texts):
        idx, _ = train_test_split(np.arange(len(texts)), train_size=subset,
                                  stratify=labels, random_state=seed)
        texts = [texts[i] for i in idx]
        labels = labels[idx]

    texts = [clean_text(t) for t in texts]
    idx = np.arange(len(texts))
    train_idx, rest = train_test_split(idx, train_size=C.TRAIN_FRAC, stratify=labels,
                                       random_state=seed)
    val_share = C.VAL_FRAC / (C.VAL_FRAC + C.TEST_FRAC)
    val_idx, test_idx = train_test_split(rest, train_size=val_share, stratify=labels[rest],
                                         random_state=seed)
    pick = lambda ix: ([texts[i] for i in ix], labels[ix])
    return {"train": pick(train_idx), "val": pick(val_idx), "test": pick(test_idx)}


# ---------------------------------------------------------------- vocabulary

class Vocab:
    def __init__(self, train_texts, size: int = C.VOCAB_SIZE):
        counts = Counter(tok for t in train_texts for tok in tokenize(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(size - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, texts, max_len: int = C.MAX_LEN):
        """Map to ids, truncate/pad to max_len. Returns (ids[N, max_len], lengths[N])."""
        ids = np.full((len(texts), max_len), C.PAD_IDX, dtype=np.int64)
        lengths = np.zeros(len(texts), dtype=np.int64)
        for row, text in enumerate(texts):
            toks = [self.stoi.get(w, C.UNK_IDX) for w in tokenize(text)[:max_len]] or [C.UNK_IDX]
            ids[row, :len(toks)] = toks
            lengths[row] = len(toks)
        return ids, lengths


# ------------------------------------------------------------------- datasets

class _IndexDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _loader(n, collate, batch_size, shuffle):
    return DataLoader(_IndexDataset(n), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0)


def tfidf_loaders(splits, batch_size):
    """MLP baseline: sparse TF-IDF vectors (fit on train only), densified per batch."""
    vec = TfidfVectorizer(max_features=C.VOCAB_SIZE, tokenizer=tokenize,
                          token_pattern=None, lowercase=False, sublinear_tf=True,
                          dtype=np.float32)
    mats = {"train": vec.fit_transform(splits["train"][0])}
    for s in ("val", "test"):
        mats[s] = vec.transform(splits[s][0])

    def make(split, shuffle):
        X, y = mats[split], splits[split][1]
        collate = lambda ix: ({"x": torch.from_numpy(X[ix].toarray())},
                              torch.from_numpy(y[ix]).float())
        return _loader(len(y), collate, batch_size, shuffle)

    loaders = {s: make(s, s == "train") for s in splits}
    return loaders, {"input_dim": len(vec.vocabulary_)}


def sequence_loaders(splits, batch_size, max_len=C.MAX_LEN):
    """TextCNN / BiLSTM: padded token-id sequences over the shared 10k vocabulary."""
    vocab = Vocab(splits["train"][0])
    encoded = {s: vocab.encode(splits[s][0], max_len) for s in splits}

    def make(split, shuffle):
        (ids, lengths), y = encoded[split], splits[split][1]

        def collate(ix):
            ix = np.asarray(ix)
            L = int(lengths[ix].max())       # trim batch to its longest review
            return ({"ids": torch.from_numpy(ids[ix, :L]),
                     "lengths": torch.from_numpy(lengths[ix])},
                    torch.from_numpy(y[ix]).float())
        return _loader(len(y), collate, batch_size, shuffle)

    loaders = {s: make(s, s == "train") for s in splits}
    return loaders, {"vocab": vocab}


def bert_loaders(splits, batch_size, max_len=C.MAX_LEN,
                 model_name="distilbert-base-uncased"):
    """DistilBERT: WordPiece tokenisation of the same cleaned text, dynamic padding."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    enc = {s: tok(splits[s][0], truncation=True, max_length=max_len)["input_ids"]
           for s in splits}

    def make(split, shuffle):
        ids_list, y = enc[split], splits[split][1]

        def collate(ix):
            batch = tok.pad({"input_ids": [ids_list[i] for i in ix]}, return_tensors="pt")
            return ({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                    torch.from_numpy(y[np.asarray(ix)]).float())
        return _loader(len(y), collate, batch_size, shuffle)

    loaders = {s: make(s, s == "train") for s in splits}
    return loaders, {"tokenizer": tok, "model_name": model_name}
