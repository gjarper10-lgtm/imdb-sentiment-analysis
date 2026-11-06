"""Shared configuration: one evaluation protocol for every model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GLOVE_DIR = DATA_DIR / "glove"

SEED = 42

# Common preprocessing (synopsis: 10k vocabulary, 512-token reviews)
VOCAB_SIZE = 10_000          # includes <pad> and <unk>
MAX_LEN = 512
PAD_IDX, UNK_IDX = 0, 1

# 70 / 15 / 15 stratified split over all 50,000 reviews
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

EMBED_DIM = 300              # GloVe-300

# Per-model training defaults; any of these can be overridden from the CLI.
MODEL_DEFAULTS = {
    "mlp": dict(lr=1e-3, batch_size=64, epochs=10, patience=3, weight_decay=1e-5),
    "textcnn": dict(lr=1e-3, batch_size=64, epochs=10, patience=3, weight_decay=1e-5),
    "bilstm": dict(lr=1e-3, batch_size=64, epochs=10, patience=3, weight_decay=1e-5),
    "distilbert": dict(lr=2e-5, batch_size=8, grad_accum=2, epochs=3, patience=2,
                       weight_decay=0.01, warmup_frac=0.1),
}

MODELS = list(MODEL_DEFAULTS)
