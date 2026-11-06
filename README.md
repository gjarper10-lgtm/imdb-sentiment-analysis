# Comparative Evaluation of Deep Learning Models for Binary Sentiment Classification (IMDB)

Deep Learning Project, ICT-4442 · Arnav Kumar (230911308) · School of Computer Engineering, MIT Manipal

Four architectures with increasing inductive bias are trained and evaluated on the IMDB Large Movie
Review Dataset under one common protocol:

| Family | Model | Input | Key idea |
|---|---|---|---|
| Classical baseline | **MLP over TF-IDF BoW** | 10k-dim TF-IDF vector | No word order; sets the performance floor |
| CNN | **TextCNN** (Kim, 2014) | GloVe-300 embeddings | Kernels of 2/3/4 tokens detect sentiment n-grams; max-over-time pooling |
| RNN | **BiLSTM + attention** | GloVe-300 embeddings | 2-layer bidirectional LSTM; additive attention picks out sentiment words |
| Transformer | **DistilBERT** (fine-tuned) | WordPiece tokens | `distilbert-base-uncased`, pre-trained then fine-tuned end to end |

## Common protocol

- **Data:** all 50,000 reviews (HuggingFace `stanfordnlp/imdb`) pooled and split **70 / 15 / 15**
  (35,000 / 7,500 / 7,500), stratified, seed 42.
- **Preprocessing:** lowercase, HTML `<br />` and punctuation removed, whitespace tokenisation,
  vocabulary of the **10,000** most frequent training tokens (incl. `<pad>`, `<unk>`), reviews
  truncated/padded to **512** tokens. DistilBERT receives the same cleaned text, tokenised by its
  own WordPiece tokenizer, also capped at 512.
- **Training:** AdamW, binary cross-entropy on one logit, fp16 mixed precision on GPU, gradient
  clipping 1.0, early stopping on **validation F1** (best checkpoint restored before testing).
- **Metrics (test split):** accuracy, precision, recall, F1, confusion matrix, parameter count,
  training time, inference latency (single review and batched).

Default hyperparameters are in [src/config.py](src/config.py).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On Windows, keep the project (or at least the venv) at a short path: pip fails when paths exceed 260
characters.

GloVe (`glove.6B.zip`, ~822 MB) is downloaded automatically into `data/glove/` the first time TextCNN
or BiLSTM runs. If you already have `glove.6B.300d.txt`, pass `--glove-path <file>`.

## Running

```bash
python run_all.py --subset 2000 --epochs 1
```
Quick end-to-end smoke test (2,000 reviews, 1 epoch per model, a few minutes).

```bash
python run_all.py
```
Full experiment: EDA, all four models, comparison.

Individual steps:

```bash
python -m src.eda
```

```bash
python -m src.run --model mlp
```

```bash
python -m src.run --model textcnn --freeze-emb --tag frozen
```

```bash
python -m src.compare
```

`src.run` options: `--model {mlp,textcnn,bilstm,distilbert}`, `--epochs`, `--batch-size`, `--lr`,
`--subset N`, `--max-len`, `--seed`, `--no-glove`, `--freeze-emb`, `--no-amp`, `--save-model`, `--tag`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `eda/eda_stats.json`, `eda/length_distribution.png`, `eda/indicative_words.json` | Dataset statistics |
| `<model>/metrics.json` | Test metrics, params, training time, latency, hyperparameters |
| `<model>/history.json` | Per-epoch train/val loss, val accuracy/F1, epoch time |
| `<model>/misclassified.csv` | 50 most confidently wrong test reviews (error analysis) |
| `comparison.md`, `comparison.csv` | Side-by-side results table |
| `tradeoffs.png` | Accuracy vs. parameters / latency / training time (Pareto view) |
| `learning_curves.png` | Validation loss and F1 per epoch, all models |

## Report and presentation (`docs/`)

| File | Content |
|---|---|
| `Project_Report.docx` | Final report: abstract, literature review, dataset, methodology, results, discussion, references |
| `Project_Presentation.pptx` | 13-slide presentation with speaker notes |
| `build_report.py`, `build_slides.py` | Regenerate both from the latest `results/` |

Results tables, figures and key findings show **TBD** until the experiments have run. After
`python run_all.py`, regenerate both documents:

```bash
python docs/build_report.py
```

```bash
python docs/build_slides.py
```

Then open the report in Word and update the table of contents (right-click it, Update Field).

## Project layout

```
src/
  config.py    shared protocol and hyperparameters
  data.py      loading, cleaning, 70/15/15 split, vocabulary, TF-IDF, DataLoaders
  glove.py     GloVe-300 download and embedding matrix
  models.py    MLPBoW, TextCNN, BiLSTMAttention, DistilBERTClassifier
  engine.py    training loop, early stopping, metrics, latency
  run.py       train + evaluate one model (CLI)
  eda.py       exploratory data analysis
  compare.py   comparison table and plots
run_all.py     whole pipeline
docs/          report, presentation and their generators
```

## References

1. A. L. Maas et al., "Learning Word Vectors for Sentiment Analysis," ACL, 2011.
2. Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.
3. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.
4. J. Howard and S. Ruder, "Universal Language Model Fine-Tuning for Text Classification," ACL, 2018.
5. V. Sanh et al., "DistilBERT, a distilled version of BERT," arXiv:1910.01108, 2019.
6. Z. Yang et al., "Hierarchical Attention Networks for Document Classification," NAACL, 2016.
