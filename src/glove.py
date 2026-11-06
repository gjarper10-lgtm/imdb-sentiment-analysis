"""GloVe-300 (6B, Wikipedia + Gigaword) embedding matrix for the shared vocabulary."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import config as C

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"  # Stanford's download host
GLOVE_FILE = "glove.6B.300d.txt"


def _ensure_glove_file(path=None):
    if path:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    target = C.GLOVE_DIR / GLOVE_FILE
    if target.exists():
        return target
    C.GLOVE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = C.GLOVE_DIR / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading GloVe from {GLOVE_URL} (~822 MB, one time)...")
        partial = zip_path.with_suffix('.part')
        urllib.request.urlretrieve(GLOVE_URL, partial)
        partial.replace(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(GLOVE_FILE, C.GLOVE_DIR)
    return target


def build_embedding_matrix(vocab, glove_path=None, dim=C.EMBED_DIM, seed=C.SEED):
    """Rows for words found in GloVe are copied; OOV words get N(0, 0.6^2); <pad> is zero.

    The matrix is cached in data/glove keyed by a hash of the vocabulary.
    """
    source = Path(_ensure_glove_file(glove_path))
    provenance = f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{dim}:{seed}'
    key = hashlib.sha256((provenance + '\n' + "\n".join(vocab.itos)).encode()).hexdigest()[:16]
    cache = C.GLOVE_DIR / f"emb_{key}.npy"
    if cache.exists():
        return np.load(cache)

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.6, (len(vocab), dim)).astype(np.float32)
    matrix[C.PAD_IDX] = 0.0
    found = 0
    with open(source, encoding="utf8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            idx = vocab.stoi.get(word)
            if idx is not None and idx != C.PAD_IDX:
                vector = np.asarray(rest.split(), dtype=np.float32)
                if len(vector) != dim:
                    raise ValueError(f'Expected {dim} dimensions, got {len(vector)} for {word}')
                matrix[idx] = vector
                found += 1
    print(f"GloVe coverage: {found}/{len(vocab)} vocabulary words")
    C.GLOVE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    return matrix
