"""Generate docs/Project_Report.docx. Re-run after experiments to pull in the latest results.

    python docs/build_report.py
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_summary import HEADER, RESULTS, TBD, findings, load, rows  # noqa: E402

OUT = Path(__file__).resolve().parent / "Project_Report.docx"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)

TITLE = ("Comparative Evaluation of Deep Learning Models for Binary Sentiment "
         "Classification on the IMDB Movie Review Dataset")


# ------------------------------------------------------------------ helpers

def setup_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.5
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        h = doc.styles[name]
        h.font.name = "Times New Roman"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = ACCENT
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        h.paragraph_format.space_after = Pt(6)
    for sec in doc.sections:
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = sec.right_margin = Cm(2.54)
        sec.top_margin = sec.bottom_margin = Cm(2.54)


def para(doc, text, bold=False, italic=False, size=None, align=None, after=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def body(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullets(doc, items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(2)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(doc, header, data, widths=None, font=10):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "1F3A5F")
    for k, row in enumerate(data):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
            if k % 2:
                shade(cells[i], "EEF2F7")
    for row in t.rows:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            c.paragraphs[0].paragraph_format.line_spacing = 1.0
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def caption(doc, text):
    p = para(doc, text, italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    return p


def figure(doc, path, text, width_cm=15):
    if path.exists():
        doc.add_picture(str(path), width=Cm(width_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        para(doc, f"[Figure pending: generated at {path.relative_to(RESULTS.parent)} "
                  f"after running the experiments]", italic=True, size=10,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    caption(doc, text)


def page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t"); txt.text = "Right-click and choose Update Field to build the table of contents."
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, txt, end):
        run._r.append(el)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ content

def title_page(doc):
    for _ in range(3):
        doc.add_paragraph()
    para(doc, TITLE, bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)
    para(doc, "Final Report on", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "Deep Learning Project [ICT-4442]", bold=True, size=14,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
    para(doc, "Submitted By", italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "ARNAV KUMAR  |  230911308", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
         after=60)
    for line in ("SCHOOL OF COMPUTER ENGINEERING", "MANIPAL INSTITUTE OF TECHNOLOGY",
                 "MANIPAL ACADEMY OF HIGHER EDUCATION"):
        para(doc, line, bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "October 2026", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    page_break(doc)


def build():
    runs, eda, complete, full = load()
    doc = Document()
    setup_styles(doc)
    page_numbers(doc.sections[0])
    title_page(doc)

    # Abstract
    doc.add_heading("Abstract", 1)
    acc = lambda m: TBD if runs[m] is None else f"{runs[m]['test']['accuracy'] * 100:.2f}%"
    body(doc,
         "Binary sentiment classification is a foundational task in natural language processing "
         "and underpins applications ranging from product-review mining to social-media "
         "monitoring. This project compares four deep learning architectures of increasing "
         "inductive bias on the IMDB Large Movie Review Dataset (50,000 reviews): a multilayer "
         "perceptron over TF-IDF bag-of-words features, a TextCNN over GloVe-300 embeddings, a "
         "bidirectional LSTM with an attention layer, and a fine-tuned DistilBERT transformer. All "
         "models share one preprocessing pipeline, a stratified 70/15/15 split and a single "
         "evaluation harness reporting accuracy, precision, recall, F1, parameter count, training "
         "time and inference latency. "
         f"Test accuracies were: MLP {acc('mlp')}, TextCNN {acc('textcnn')}, BiLSTM-Attention "
         f"{acc('bilstm')} and DistilBERT {acc('distilbert')}. The study quantifies the "
         "accuracy gap between models trained from scratch and a pre-trained transformer, and "
         "the compute cost at which that gap is bought.")
    para(doc, "Keywords: sentiment analysis, IMDB, TextCNN, BiLSTM, attention, DistilBERT, "
              "transfer learning, model comparison", italic=True, size=11)
    page_break(doc)

    doc.add_heading("Table of Contents", 1)
    toc(doc)
    page_break(doc)

    # 1 Introduction
    doc.add_heading("1. Introduction", 1)
    body(doc,
         "Automatically deciding whether a piece of text expresses positive or negative sentiment "
         "is one of the most studied problems in NLP. Despite its apparent simplicity, the task "
         "exercises most of what makes language hard: sentiment is carried by individual words "
         "(\"brilliant\", \"dull\"), by short phrases (\"not bad at all\"), by negation and "
         "contrast spanning many words (\"I expected to hate it, but...\"), and occasionally by "
         "sarcasm that only world knowledge resolves.")
    body(doc,
         "Each family of neural architecture covered in ICT-4442 encodes a different assumption "
         "about which of these signals matter. A bag-of-words MLP assumes word identity is enough. "
         "A convolutional network assumes local n-gram patterns matter. A recurrent network "
         "assumes word order and long-range context matter. A pre-trained transformer assumes "
         "that general language knowledge, learned from large unlabelled corpora, transfers to "
         "the target task. Comparing these models under identical conditions shows how much each "
         "additional assumption buys, and at what computational cost.")
    doc.add_heading("1.1 Objectives", 2)
    bullets(doc, [
        "To evaluate MLP bag-of-words, TextCNN, BiLSTM-Attention and DistilBERT architectures on "
        "binary sentiment classification accuracy, F1-score and inference latency on the IMDB "
        "dataset under a common evaluation protocol.",
        "To measure the performance gap between models trained from scratch (MLP, CNN, BiLSTM) "
        "and a fine-tuned pre-trained transformer (DistilBERT), and to characterise the "
        "cost-benefit trade-off in training time and parameter count.",
    ], style="List Number")
    doc.add_heading("1.2 Motivation", 2)
    body(doc,
         "Sentiment analysis of product and entertainment reviews supports consumer decision "
         "making at very large scale. IMDB is the standard benchmark for binary sentiment, so "
         "results are directly comparable with more than a decade of published work. The four "
         "architectures studied span the practical range of compute budgets, from a model that "
         "trains in seconds on a CPU to one that needs a GPU for fine-tuning. For deployment "
         "settings such as embedded review analysers or on-device NLP, knowing how close a small "
         "model gets to a transformer, and at what fraction of the parameters and latency, is "
         "directly useful.")

    # 2 Literature
    doc.add_heading("2. Literature Review", 1)
    lit = [
        ("Maas et al. (2011) [1]. ", "Introduced the IMDB Large Movie Review Dataset and learned "
         "word vectors that capture both semantic and sentiment information, establishing the "
         "benchmark used here."),
        ("Kim (2014) [2]. ", "Showed that a single-layer CNN with multiple filter widths over "
         "pre-trained word vectors is competitive with far more complex models on sentence "
         "classification, including sentiment. Our TextCNN follows this design."),
        ("Yang et al. (2016) [6]. ", "Hierarchical Attention Networks applied attention over "
         "recurrent states to weight the most informative words and sentences in a document, "
         "motivating the attention pooling used in our BiLSTM."),
        ("Howard and Ruder (2018) [4]. ", "ULMFiT demonstrated that pre-training a language model "
         "and fine-tuning it with discriminative learning rates substantially outperforms "
         "training from scratch on sentiment tasks, including IMDB."),
        ("Devlin et al. (2019) [3]. ", "BERT's masked-language-model pre-training and "
         "fine-tuning reached above 95% accuracy on IMDB, defining the modern transfer-learning "
         "baseline. RoBERTa later improved on BERT through better pre-training choices."),
        ("Sanh et al. (2019) [5]. ", "DistilBERT used knowledge distillation to produce a model "
         "40% smaller and 60% faster than BERT-base while retaining about 97% of its language "
         "understanding performance, making it the transformer used in this study."),
    ]
    bullets(doc, lit)

    # 3 Dataset
    doc.add_heading("3. Dataset and Exploratory Analysis", 1)
    body(doc,
         "The IMDB Large Movie Review Dataset (Stanford AI Lab) contains 50,000 English movie "
         "reviews labelled positive or negative, released as 25,000 training and 25,000 test "
         "reviews with 12,500 of each class in each half. There are no missing labels. The raw "
         "text version from the HuggingFace datasets library (stanfordnlp/imdb) was used so that "
         "a custom preprocessing pipeline could be applied identically to every model.")
    body(doc,
         "To obtain a validation set and a single protocol, all 50,000 reviews were pooled and "
         "split 70/15/15 with stratification on the label and a fixed seed (42).")
    if eda:
        L = eda["length_words"]
        sz = eda["split_sizes"]
        table(doc, ["Statistic", "Value"], [
            ["Reviews (train / val / test)", f"{sz['train']:,} / {sz['val']:,} / {sz['test']:,}"],
            ["Positive fraction", f"{eda['positive_fraction'] * 100:.1f}%"],
            ["Mean / median length (words)", f"{L['mean']:.0f} / {L['median']:.0f}"],
            ["90th / 95th percentile length", f"{L['p90']:.0f} / {L['p95']:.0f}"],
            ["Longest review (words)", f"{L['max']:,}"],
            ["Reviews longer than 512 words", f"{eda['fraction_truncated_at_max_len'] * 100:.1f}%"],
            ["Training vocabulary size", f"{eda['train_vocabulary_size']:,}"],
            ["Training tokens covered by 10k vocabulary",
             f"{eda['train_token_coverage_of_10k_vocab'] * 100:.1f}%"],
        ], widths=[9, 6])
    else:
        table(doc, ["Statistic", "Value"], [
            ["Reviews (train / val / test)", "35,000 / 7,500 / 7,500"],
            ["Positive fraction", "50.0%"],
            ["Mean review length", "~231 words"],
            ["Other statistics", f"{TBD}: run python -m src.eda"],
        ], widths=[9, 6])
    caption(doc, "Table 1: Dataset statistics after cleaning.")
    figure(doc, RESULTS / "eda" / "length_distribution.png",
           "Figure 1: Distribution of review lengths by class; the dashed line marks the "
           "512-token cut-off.")

    # 4 Methodology
    doc.add_heading("4. Methodology", 1)
    doc.add_heading("4.1 Shared preprocessing pipeline", 2)
    bullets(doc, [
        "Lowercasing, removal of HTML line breaks (<br />) and punctuation; apostrophes are "
        "dropped so that negations such as \"don't\" remain one token (\"dont\").",
        "Whitespace tokenisation and a vocabulary of the 10,000 most frequent training tokens, "
        "including <pad> and <unk>; the vocabulary is built from the training split only.",
        "Reviews truncated or padded to 512 tokens. Batches are trimmed to their longest review "
        "to avoid wasted computation.",
        "DistilBERT receives the same cleaned text, tokenised by its own WordPiece tokenizer and "
        "capped at 512 sub-word tokens.",
    ])
    doc.add_heading("4.2 Model 1: MLP over TF-IDF bag-of-words (classical baseline)", 2)
    body(doc,
         "Each review is represented by a 10,000-dimensional TF-IDF vector (sub-linear term "
         "frequency, fitted on the training split). A feed-forward network with hidden layers of "
         "512 and 128 units, ReLU activations and dropout 0.5 maps this vector to a single logit. "
         "Word order is discarded entirely, so this model sets the performance floor achievable "
         "from word identity alone.")
    doc.add_heading("4.3 Model 2: TextCNN", 2)
    body(doc,
         "Following Kim (2014), tokens are embedded with 300-dimensional GloVe vectors (glove.6B, "
         "fine-tuned during training; out-of-vocabulary words are initialised randomly). Three "
         "parallel 1D convolutions with kernel widths 2, 3 and 4 and 100 filters each act as "
         "learned bigram, trigram and four-gram detectors. Max-over-time pooling keeps the "
         "strongest response of each filter anywhere in the review; the 300 pooled features pass "
         "through dropout 0.5 to a sigmoid output.")
    doc.add_heading("4.4 Model 3: BiLSTM with attention", 2)
    body(doc,
         "GloVe-300 embeddings feed a two-layer bidirectional LSTM with 128 hidden units per "
         "direction. Padded positions are excluded using packed sequences. Instead of using only "
         "the final hidden state, an additive attention layer scores each time step, "
         "e_t = v^T tanh(W h_t), normalises the scores with a masked softmax and forms a weighted "
         "sum of the hidden states. This lets the model focus on the words that determine "
         "sentiment and handle negation and contrast that depend on word order.")
    doc.add_heading("4.5 Model 4: DistilBERT (fine-tuned)", 2)
    body(doc,
         "The pre-trained distilbert-base-uncased encoder (6 transformer layers, 66M parameters) "
         "is fine-tuned end to end with a single-logit classification head on the [CLS] "
         "representation. Self-attention over the whole review, together with knowledge acquired "
         "during pre-training, gives this model the strongest inductive bias of the four.")
    doc.add_heading("4.6 Training and evaluation protocol", 2)
    body(doc,
         "All models minimise binary cross-entropy on one logit with the AdamW optimiser, "
         "gradient clipping at norm 1.0 and fp16 mixed precision on the GPU. After every epoch "
         "the validation F1-score is computed; training stops early when it fails to improve, "
         "and the best checkpoint is restored before the single evaluation on the test split. "
         "The same evaluation harness computes accuracy, precision, recall, F1 and the confusion "
         "matrix, counts parameters, records wall-clock training time and measures inference "
         "latency both for single reviews and batched over the test set.")
    table(doc, ["Model", "Learning rate", "Batch size", "Max epochs", "Patience", "Other"], [
        ["MLP (TF-IDF)", "1e-3", "64", "10", "3", "weight decay 1e-5"],
        ["TextCNN", "1e-3", "64", "10", "3", "weight decay 1e-5"],
        ["BiLSTM + Attention", "1e-3", "64", "10", "3", "weight decay 1e-5"],
        ["DistilBERT", "2e-5", "8 x 2 (accum.)", "3", "2", "10% linear warm-up, wd 0.01"],
    ], widths=[3.6, 2.4, 2.6, 2.1, 1.8, 4.0])
    caption(doc, "Table 2: Training hyperparameters.")

    # 5 Experimental setup
    doc.add_heading("5. Experimental Setup", 1)
    device = next((r["device"] for r in runs.values() if r), "NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)")
    bullets(doc, [
        ("Hardware: ", f"{device}."),
        ("Software: ", "Python, PyTorch, HuggingFace Transformers and Datasets, scikit-learn."),
        ("Reproducibility: ", "fixed random seed 42 for the split, initialisation and data "
                              "order; all hyperparameters in src/config.py."),
        ("Code: ", "single entry point run_all.py runs EDA, all four models and the comparison."),
    ])

    # 6 Results
    doc.add_heading("6. Results", 1)
    if not complete:
        body(doc, "Note: some experiments have not been run yet; cells marked TBD are filled in "
                  "automatically when this report is regenerated after running run_all.py.")
    elif not full:
        body(doc, "Note: these numbers come from a reduced-data run (--subset) and are for "
                  "pipeline verification only; regenerate after the full run for final results.")
    table(doc, HEADER, rows(runs), widths=[3.6, 1.8, 1.8, 1.7, 1.5, 1.5, 1.8, 2.0], font=9)
    caption(doc, "Table 3: Test-set results under the common protocol (7,500 reviews).")
    figure(doc, RESULTS / "tradeoffs.png",
           "Figure 2: Test accuracy against parameter count, single-review latency and training "
           "time (log scales).", width_cm=16)
    figure(doc, RESULTS / "learning_curves.png",
           "Figure 3: Validation loss and F1 per epoch for each model.")

    # 7 Discussion
    doc.add_heading("7. Discussion", 1)
    found = findings(runs)
    if found:
        doc.add_heading("7.1 Key findings", 2)
        bullets(doc, found)
    else:
        body(doc, "[Key findings are generated from the results once all four models have been "
                  "run.]")
    doc.add_heading("7.2 Effect of inductive bias", 2)
    body(doc,
         "The bag-of-words baseline is expected to be strong on IMDB because long reviews "
         "contain many sentiment-bearing words, so word identity alone is highly informative. "
         "Its errors concentrate on reviews whose sentiment is reversed by negation or contrast, "
         "which TextCNN partly captures through short n-grams and the BiLSTM through sequential "
         "context. The pre-trained transformer additionally benefits from language knowledge "
         "that a model trained on 35,000 reviews cannot acquire on its own.")
    doc.add_heading("7.3 Error analysis", 2)
    body(doc,
         "For each model the 50 most confidently misclassified test reviews are saved to "
         "results/<model>/misclassified.csv. Recurring error categories to examine are: mixed "
         "reviews that praise some aspects and criticise others, sarcasm and irony, sentiment "
         "expressed only in the final sentences of reviews truncated at 512 tokens, and reviews "
         "that discuss the plot's negative events while recommending the film.")
    doc.add_heading("7.4 Cost-benefit trade-off", 2)
    body(doc,
         "Figure 2 places every model on an accuracy-versus-cost plane. Models on the Pareto "
         "front are the rational choices for some budget: the MLP when training and inference "
         "must be nearly free, TextCNN when a small, fast model must approach transformer "
         "accuracy, and DistilBERT when accuracy is paramount and a GPU is available.")
    doc.add_heading("7.5 Limitations", 2)
    bullets(doc, [
        "Results come from one seed; repeating runs with several seeds would give confidence "
        "intervals.",
        "Punctuation was removed for all models to keep preprocessing identical, which may cost "
        "DistilBERT some accuracy compared with feeding it raw text.",
        "Hyperparameters were not tuned exhaustively for each model.",
        "Truncation at 512 tokens discards the end of the longest reviews.",
    ])

    # 8 Conclusion
    doc.add_heading("8. Conclusion and Future Work", 1)
    body(doc,
         "This project implemented and compared four architectures from different families for "
         "binary sentiment classification on IMDB under a single preprocessing pipeline and "
         "evaluation harness. The comparison quantifies how much accuracy each additional "
         "inductive bias provides and what it costs in parameters, training time and latency, "
         "giving practical guidance for choosing a model under a given compute budget.")
    body(doc, "Future work includes:")
    bullets(doc, [
        "multi-seed runs with confidence intervals and significance tests;",
        "head-and-tail truncation or hierarchical models for reviews longer than 512 tokens;",
        "larger pre-trained models (RoBERTa, DeBERTa) and parameter-efficient fine-tuning (LoRA);",
        "quantisation and distillation of DistilBERT into TextCNN for on-device deployment;",
        "visualising BiLSTM attention weights to interpret its decisions.",
    ])

    # References
    doc.add_heading("References", 1)
    refs = [
        "A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, \"Learning Word "
        "Vectors for Sentiment Analysis,\" in Proc. ACL, pp. 142-150, 2011.",
        "Y. Kim, \"Convolutional Neural Networks for Sentence Classification,\" in Proc. EMNLP, "
        "pp. 1746-1751, 2014.",
        "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: Pre-training of Deep "
        "Bidirectional Transformers for Language Understanding,\" in Proc. NAACL, pp. 4171-4186, "
        "2019.",
        "J. Howard and S. Ruder, \"Universal Language Model Fine-Tuning for Text "
        "Classification,\" in Proc. ACL, pp. 328-339, 2018.",
        "V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version of BERT: "
        "smaller, faster, cheaper and lighter,\" arXiv:1910.01108, 2019.",
        "Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy, \"Hierarchical Attention "
        "Networks for Document Classification,\" in Proc. NAACL, pp. 1480-1489, 2016.",
        "J. Pennington, R. Socher, and C. D. Manning, \"GloVe: Global Vectors for Word "
        "Representation,\" in Proc. EMNLP, pp. 1532-1543, 2014.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {r}")
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.line_spacing = 1.15

    # Appendix
    doc.add_heading("Appendix A: Reproducing the Results", 1)
    for line in ("pip install torch --index-url https://download.pytorch.org/whl/cu128",
                 "pip install -r requirements.txt",
                 "python run_all.py",
                 "python docs/build_report.py",
                 "python docs/build_slides.py"):
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0

    doc.save(OUT)
    print(f"saved {OUT}  (results complete: {complete}, full data: {full})")


if __name__ == "__main__":
    build()
