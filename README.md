# IMDB: comparative deep learning sentiment classification

Arnav Kumar · 230911308 · ICT-4442 · MIT Manipal

This project implements the supplied synopsis: TF-IDF MLP, GloVe-300 TextCNN,
stacked BiLSTM with attention, and fine-tuned DistilBERT under a shared evaluation
protocol. See `docs/SYNOPSIS_CHECKLIST.txt` for the audit of the original ZIP and
`docs/verification.json` for the actual run/test status. Missing results are never invented.

## Open these first

- `docs/Project_Report.docx`: methodology, measured results, tuning and error analysis.
- `docs/Project_Presentation.pptx`: 13 slides with results read from artifacts.
- `results/comparison.csv`: metrics, parameter counts, training/search time and latency.
- `results/analysis.json`: confusion counts, ROC AUC, accuracy intervals and Pareto frontiers.
- `IMDB_Experiments.ipynb`: local Jupyter/Google Colab reproduction notebook.

## Run on this computer

From this project folder, PowerShell can use the environment prepared alongside it:

```powershell
.\run_local.ps1 -Action test
.\run_local.ps1 -Action predict -Model mlp -Text "A wonderful film with excellent acting."
.\run_local.ps1 -Action report
.\run_local.ps1 -Action train
```

The launcher uses the project's `.venv`, or the prepared workspace environment,
or an installed `python`. It does not install software automatically.

## Set up on another computer

Python 3.12 is recommended. Create a virtual environment at a short Windows path.
The executed environment uses PyTorch 2.6.0 with CUDA 12.4; CPU training also works
but is considerably slower. Choose a compatible PyTorch build for your system.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m pytest -q
```

GloVe is downloaded from Stanford on the first sequence-model run. If needed,
download `glove.6B.zip`, extract `glove.6B.300d.txt`, and use `--glove-path PATH`.
The raw IMDB dataset and pretrained DistilBERT are cached by HuggingFace. Downloads
are a one-time setup cost. `requirements-lock.txt` records the executed ML stack.

## Reproduce the full experiment

```powershell
python run_all.py --epochs 3
```

For each architecture, two learning rates are compared using validation F1 only.
The best validation checkpoint of the winning trial is evaluated once on the test
set. Completed trials can be resumed by running the same command. This default is
a limited three-epoch search, not a claim that each model is fully converged.
Use a separate output directory to investigate a larger budget:

```powershell
python run_all.py --epochs 10 --output-root results_longer
python run_all.py --subset 200 --epochs 1
python -m src.tune --model mlp --epochs 3
python -m src.run --model textcnn --freeze-emb --tag frozen
python -m src.predict --model-dir results/mlp --text "The story was dull and predictable."
```

Smoke runs go to `results_smoke`, separate from submission results. Random-embedding
ablations require `--no-glove --output-root results_random`. Never compare scores
from different subsets, seeds or token limits as if they shared the same protocol.

## Protocol and interpretation

- Pool the 50,000 labelled IMDB reviews; stratified 35,000/7,500/7,500 split, seed 42.
  The unlabelled split is excluded. This follows the synopsis, but is different from
  Stanford's official train/test split; published leaderboard scores are not directly comparable.
- Decode HTML entities, strip HTML, lowercase, remove punctuation, collapse whitespace.
- Fit TF-IDF and word vocabulary on training texts only. Maximum 10,000 features/tokens;
  sequence vocabulary includes padding and unknown tokens. MLP/CNN/BiLSTM use at most 512 words.
- DistilBERT retains its pretrained WordPiece vocabulary; its 512-token limit includes
  special tokens. Token units differ from the word models. It receives the same cleaned text.
- CNN: GloVe-300, 100 filters per kernel (2,3,4), masked max pooling, dropout 0.5.
- BiLSTM: GloVe-300, two bidirectional layers of 128 units per direction, masked additive attention.
- AdamW, binary cross-entropy, gradient clipping 1.0; mixed precision on CUDA.
  Validation selects the checkpoint. Threshold 0.5 is fixed, not tuned using test labels.
- Search: learning rates 1e-3/3e-4 for non-transformers and 2e-5/5e-5 for DistilBERT.
- Positive is class 1. Precision/recall/F1 refer to the positive class.
- Training time includes validation for the selected trial. Total search training time
  is separate. Downloads, preprocessing and final testing are excluded.
- Warmed-up latency uses pretokenized reviews, including loader/device-transfer cost;
  tokenization is excluded. Batch sizes are recorded and affect throughput.
- GloVe is pretrained: CNN/BiLSTM are not entirely trained from scratch.
  The shared PyTorch loop replaces the synopsis's proposed HuggingFace Trainer.
- A fixed seed does not ensure bit-identical GPU results. Wilson intervals reflect
  test-sample uncertainty, not training-seed variation. Error causes are not inferred
  from confidence or review length alone.

## Artifacts

Each selected model directory contains `model.pt`, `bundle.json`, fitted vocabulary/
TF-IDF/tokenizer, `run.json`, `metrics.json`, `history.json`, `predictions.csv` and
`misclassified.csv`. Only load trusted bundles because TF-IDF joblib files use pickle.
`results/tuning/` contains trial histories and validation-selection records.
The distribution ZIP may omit duplicate trial checkpoints and downloaded datasets;
selected model bundles are sufficient for offline prediction.

Regenerate figures and documents after the canonical `results/` runs:

```powershell
python -m src.compare
python -m src.analysis
python docs/build_report.py
python docs/build_slides.py
```

In Word, update the table of contents using References → Update Table.

## GitHub submission

Publish source, tests, notebook, results and report to your own repository. Large
downloaded embeddings/caches/model weights are ignored by Git; use a release or
external artifact storage for weights. No remote repository or historical work
has been fabricated. Local history, if supplied, records only genuine snapshots.

## Primary references

1. [Maas et al., IMDB dataset (2011)](https://aclanthology.org/P11-1015/)
2. [Kim, TextCNN (2014)](https://aclanthology.org/D14-1181/)
3. [Devlin et al., BERT (2019)](https://aclanthology.org/N19-1423/)
4. [Howard and Ruder, ULMFiT (2018)](https://aclanthology.org/P18-1031/)
5. [Sanh et al., DistilBERT (2019)](https://arxiv.org/abs/1910.01108)
6. [Yang et al., Hierarchical Attention Networks (2016)](https://aclanthology.org/N16-1174/)
7. [Pennington et al., GloVe (2014)](https://aclanthology.org/D14-1162/)
8. [PyTorch mixed precision and gradient accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
9. [HuggingFace DistilBERT implementation](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
