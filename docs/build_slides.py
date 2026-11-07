"""Generate docs/Project_Presentation.pptx. Re-run after experiments to pull in results.

    python docs/build_slides.py
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_summary import HEADER, ORDER, RESULTS, SHORT, findings, load, rows, hyperparameter_rows  # noqa: E402

OUT = Path(__file__).resolve().parent / "Project_Presentation.pptx"

# Cinema palette: dark screening-room ink, marquee amber accent, review green/red.
INK = RGBColor(0x16, 0x1A, 0x2B)
INK_2 = RGBColor(0x24, 0x2A, 0x42)
AMBER = RGBColor(0xF2, 0xA5, 0x41)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
MIST = RGBColor(0xF1, 0xF2, 0xF6)
MUTED = RGBColor(0x5B, 0x61, 0x75)
POS = RGBColor(0x2E, 0x9E, 0x6B)
NEG = RGBColor(0xD6, 0x4F, 0x4F)
MODEL_COLORS = [RGBColor(0x8A, 0x93, 0xAD), RGBColor(0x3E, 0x7C, 0xB1),
                RGBColor(0x6A, 0x4C, 0x93), AMBER]
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# ------------------------------------------------------------------ helpers

def bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=None, name=None):
    """content: str, or list of str / (str, dict) paragraphs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, item in enumerate(paras):
        t, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        if spacing or opt.get("after"):
            p.space_after = Pt(opt.get("after", spacing))
        r = p.add_run()
        r.text = t
        f = r.font
        f.name = opt.get("font", font)
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.italic = opt.get("italic", italic)
        f.color.rgb = opt.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, after=8):
    tb = text(slide, x, y, w, h, [(i, {"after": after}) for i in items], size=size, color=color)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar",
                             {"char": "•"})
        pPr.append(bu)
    return tb


def box(slide, x, y, w, h, fill, radius=True, name=None, shadow=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 x, y, w, h)
    if radius:
        shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    if not shadow:
        shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def circle(slide, x, y, d, fill, label, size=18, color=INK):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name = Pt(size), True, HEAD
    r.font.color.rgb = color
    return c


def title(slide, t, dark=False):
    text(slide, MARGIN, Inches(0.45), W - 2 * MARGIN, Inches(0.9), t, size=34, bold=True,
         font=HEAD, color=PAPER if dark else INK, anchor=MSO_ANCHOR.MIDDLE, name="Title")


def footer(slide, n, dark=False):
    c = RGBColor(0x9A, 0xA0, 0xB5) if dark else MUTED
    text(slide, MARGIN, H - Inches(0.45), Inches(8), Inches(0.3),
         "IMDB Sentiment: Comparative Evaluation  |  ICT-4442  |  Arnav Kumar",
         size=10, color=c)
    text(slide, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n),
         size=10, color=c, align=PP_ALIGN.RIGHT)


def new_slide(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg(s, INK if dark else PAPER)
    return s


def picture_or_note(slide, path, x, y, w, h, note):
    if path.exists():
        pic = slide.shapes.add_picture(str(path), x, y, width=w)
        if pic.height > h:              # keep inside the frame
            ratio = h / pic.height
            pic.height, pic.width = h, int(pic.width * ratio)
        return pic
    box(slide, x, y, w, h, MIST)
    text(slide, x + Inches(0.4), y, w - Inches(0.8), h, note, size=14, color=MUTED,
         italic=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ------------------------------------------------------------------ slides

def build():
    runs, eda, complete, full = load()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1. Title
    s = new_slide(prs, dark=True)
    n += 1
    text(s, MARGIN, Inches(1.3), Inches(4), Inches(0.4), "DEEP LEARNING PROJECT  |  ICT-4442",
         size=14, bold=True, color=AMBER)
    text(s, MARGIN, Inches(1.85), Inches(10.5), Inches(2.4),
         "Comparative Evaluation of Deep Learning Models for Binary Sentiment Classification",
         size=40, bold=True, font=HEAD, color=PAPER)
    text(s, MARGIN, Inches(4.25), Inches(10), Inches(0.5),
         "MLP  vs  TextCNN  vs  BiLSTM-Attention  vs  DistilBERT  on the IMDB Movie Review Dataset",
         size=18, color=RGBColor(0xC9, 0xCD, 0xDB))
    text(s, MARGIN, Inches(5.6), Inches(8), Inches(1.0), [
        ("Arnav Kumar  |  230911308", {"size": 18, "bold": True, "color": PAPER}),
        ("School of Computer Engineering, Manipal Institute of Technology",
         {"size": 14, "color": RGBColor(0xC9, 0xCD, 0xDB)}),
    ])
    for i, (lab, col) in enumerate((("+", POS), ("–", NEG))):
        circle(s, W - Inches(2.6) + i * Inches(1.05), Inches(5.65), Inches(0.85), col, lab,
               size=30, color=PAPER)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the project: four architecture families compared on one sentiment benchmark "
        "under a single protocol.")

    # 2. Problem and motivation
    s = new_slide(prs)
    n += 1
    title(s, "Why compare architectures on sentiment?")
    bullets(s, MARGIN, Inches(1.7), Inches(6.4), Inches(4.8), [
        "Sentiment classification powers review mining, brand monitoring and recommendation.",
        "Sentiment lives in words, short phrases, negation (\"not bad at all\") and long-range "
        "contrast.",
        "Each architecture family assumes different signals matter: word identity, n-grams, "
        "order, or pre-trained language knowledge.",
        "Question: how much accuracy does each assumption buy, and at what compute cost?",
    ], size=18, after=14)
    stats = [("50,000", "labelled IMDB reviews"), ("4", "architecture families"),
             ("1", "shared protocol & harness")]
    for i, (big, small) in enumerate(stats):
        y = Inches(1.7) + i * Inches(1.6)
        box(s, Inches(7.6), y, Inches(5.1), Inches(1.35), MIST)
        text(s, Inches(7.9), y, Inches(2.2), Inches(1.35), big, size=40, bold=True, font=HEAD,
             color=AMBER if i == 0 else INK, anchor=MSO_ANCHOR.MIDDLE)
        text(s, Inches(10.1), y, Inches(2.5), Inches(1.35), small, size=16, color=MUTED,
             anchor=MSO_ANCHOR.MIDDLE)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Motivate the comparison: the same task, four different inductive biases.")

    # 3. Objectives
    s = new_slide(prs)
    n += 1
    title(s, "Objectives")
    objs = [
        ("Evaluate", "MLP bag-of-words, TextCNN, BiLSTM-Attention and DistilBERT on accuracy, "
                     "F1-score and inference latency on IMDB, under one common evaluation "
                     "protocol."),
        ("Measure", "the performance gap between non-transformer models and a fine-tuned "
                    "pre-trained transformer, and the cost-benefit trade-off in training time "
                    "and parameter count."),
    ]
    for i, (verb, rest) in enumerate(objs):
        x = MARGIN + i * Inches(6.15)
        box(s, x, Inches(1.8), Inches(5.85), Inches(4.4), MIST)
        circle(s, x + Inches(0.45), Inches(2.25), Inches(0.9), AMBER, str(i + 1), size=26)
        text(s, x + Inches(0.45), Inches(3.45), Inches(5.0), Inches(0.6), verb, size=26,
             bold=True, font=HEAD)
        text(s, x + Inches(0.45), Inches(4.1), Inches(5.0), Inches(2.0), rest, size=17,
             color=MUTED)
    footer(s, n)

    # 4. Dataset
    s = new_slide(prs)
    n += 1
    title(s, "Dataset: IMDB Large Movie Review Dataset")
    if eda:
        L, sz = eda["length_words"], eda["split_sizes"]
        facts = [(f"{sz['train']:,} / {sz['val']:,} / {sz['test']:,}", "train / val / test"),
                 (f"{L['mean']:.0f} words", f"mean length (median {L['median']:.0f})"),
                 (f"{eda['fraction_truncated_at_max_len'] * 100:.1f}%", "reviews > 512 tokens"),
                 (f"{eda['train_token_coverage_of_10k_vocab'] * 100:.1f}%",
                  "tokens covered by 10k vocab")]
    else:
        facts = [("35,000 / 7,500 / 7,500", "train / val / test"),
                 ("~231 words", "average review length"),
                 ("50 / 50", "positive / negative balance"),
                 ("0", "missing labels")]
    for i, (big, small) in enumerate(facts):
        y = Inches(1.65) + i * Inches(1.22)
        text(s, MARGIN, y, Inches(4.6), Inches(0.6), big, size=26, bold=True, font=HEAD,
             color=INK)
        text(s, MARGIN, y + Inches(0.55), Inches(4.6), Inches(0.4), small, size=14, color=MUTED)
    picture_or_note(s, RESULTS / "eda" / "length_distribution.png", Inches(5.6), Inches(1.6),
                    Inches(7.1), Inches(4.9),
                    "Review-length histogram appears here after running python -m src.eda")
    text(s, MARGIN, Inches(6.55), Inches(4.8), Inches(0.4),
         "Source: Stanford AI Lab, via HuggingFace stanfordnlp/imdb", size=11, color=MUTED,
         italic=True)
    footer(s, n)

    # 5. Common protocol
    s = new_slide(prs)
    n += 1
    title(s, "One protocol for every model")
    steps = [("Clean", "lowercase, strip HTML & punctuation"),
             ("Split", "70 / 15 / 15 stratified, seed 42"),
             ("Tokenise", "10k-word vocab from train split"),
             ("Pad", "truncate or pad to 512 tokens"),
             ("Train", "AdamW, fp16, early stop on val F1"),
             ("Evaluate", "acc, P, R, F1, latency, params")]
    cw, gap = Inches(1.85), Inches(0.17)
    for i, (head, sub) in enumerate(steps):
        x = MARGIN + i * (cw + gap)
        box(s, x, Inches(2.2), cw, Inches(2.6), INK if i in (4, 5) else MIST)
        circle(s, x + Inches(0.6), Inches(2.45), Inches(0.62), AMBER, str(i + 1), size=18)
        dark = i in (4, 5)
        text(s, x + Inches(0.18), Inches(3.25), cw - Inches(0.36), Inches(0.5), head, size=18,
             bold=True, font=HEAD, color=PAPER if dark else INK, align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.18), Inches(3.8), cw - Inches(0.36), Inches(0.95), sub, size=13,
             color=RGBColor(0xC9, 0xCD, 0xDB) if dark else MUTED, align=PP_ALIGN.CENTER)
    text(s, MARGIN, Inches(5.3), W - 2 * MARGIN, Inches(1.0),
         "DistilBERT receives the same cleaned text, tokenised by its own WordPiece tokenizer "
         "(max 512 sub-words). The best validation checkpoint is restored before a single "
         "evaluation on the held-out test split.", size=16, color=MUTED)
    footer(s, n)

    # 6. Four models
    s = new_slide(prs)
    n += 1
    title(s, "Four architectures for sentiment classification")
    models = [
        ("MLP + TF-IDF", "Classical baseline",
         "Dense 512-128 net over a 10k TF-IDF vector. Ignores word order: the performance floor."),
        ("TextCNN", "Convolutional",
         "GloVe-300 + 1D convolutions, kernels 2/3/4 (100 filters each), max-over-time pooling."),
        ("BiLSTM + Attention", "Recurrent",
         "2-layer BiLSTM (128/dir) over GloVe-300; additive attention weights sentiment words."),
        ("DistilBERT", "Pre-trained transformer",
         "distilbert-base-uncased, 6 layers, fine-tuned end to end with a single-logit head."),
    ]
    cw, ch = Inches(5.9), Inches(2.3)
    for i, (name, fam, desc) in enumerate(models):
        x = MARGIN + (i % 2) * (cw + Inches(0.33))
        y = Inches(1.6) + (i // 2) * (ch + Inches(0.3))
        box(s, x, y, cw, ch, MIST)
        circle(s, x + Inches(0.35), y + Inches(0.35), Inches(0.75), MODEL_COLORS[i], str(i + 1),
               size=20, color=PAPER if i in (1, 2) else INK)
        text(s, x + Inches(1.3), y + Inches(0.3), cw - Inches(1.6), Inches(0.45), name,
             size=22, bold=True, font=HEAD)
        text(s, x + Inches(1.3), y + Inches(0.78), cw - Inches(1.6), Inches(0.35), fam.upper(),
             size=11, bold=True, color=MUTED)
        text(s, x + Inches(1.3), y + Inches(1.18), cw - Inches(1.6), Inches(1.0), desc,
             size=14, color=INK)
    footer(s, n)

    # 7. Inside the sequence models
    s = new_slide(prs)
    n += 1
    title(s, "Inside the sequence models")
    cols = [
        ("TextCNN: local n-gram detectors", [
            "Embedding: GloVe-300, fine-tuned; random init for OOV words",
            "Conv1D widths 2, 3, 4 act as bigram, trigram, 4-gram detectors",
            "Max-over-time keeps each filter's strongest phrase anywhere in the review",
            "300 pooled features, dropout 0.5, sigmoid output",
        ]),
        ("BiLSTM + attention: order and context", [
            "Packed sequences so padding never reaches the LSTM",
            "Forward and backward states capture negation and contrast",
            "Attention: e_t = vᵀ tanh(W h_t), masked softmax over real tokens",
            "Context vector = attention-weighted sum of hidden states",
        ]),
    ]
    for i, (head, items) in enumerate(cols):
        x = MARGIN + i * Inches(6.2)
        text(s, x, Inches(1.7), Inches(5.8), Inches(0.5), head, size=21, bold=True, font=HEAD,
             color=MODEL_COLORS[i + 1])
        bullets(s, x, Inches(2.4), Inches(5.7), Inches(3.8), items, size=16, after=12)
    footer(s, n)

    # 8. Training setup
    s = new_slide(prs)
    n += 1
    title(s, "Training setup")
    data = [["Model", "LR", "Batch", "Epoch cap", "Patience", "Selected"]] + hyperparameter_rows(runs)
    tbl = s.shapes.add_table(len(data), len(data[0]), MARGIN, Inches(1.7), Inches(8.2),
                             Inches(2.9)).table
    widths = [2.1, 0.8, 1.4, 1.2, 1.0, 1.7]
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if i == 0 else (MIST if i % 2 == 0 else PAPER)
            cell.text_frame.paragraphs[0].text = ""
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = v
            r.font.size, r.font.name = Pt(13), BODY
            r.font.bold = i == 0
            r.font.color.rgb = PAPER if i == 0 else INK
    box(s, Inches(9.2), Inches(1.7), Inches(3.5), Inches(4.6), INK)
    text(s, Inches(9.5), Inches(1.95), Inches(3.0), Inches(4.2), [
        ("Shared", {"size": 20, "bold": True, "font": HEAD, "color": AMBER, "after": 10}),
        ("AdamW optimiser", {"after": 8}),
        ("Binary cross-entropy, one logit", {"after": 8}),
        ("fp16 mixed precision", {"after": 8}),
        ("Gradient clipping at 1.0", {"after": 8}),
        ("Early stopping on val F1", {"after": 8}),
        ("Seed 42 everywhere", {"after": 8}),
        ("RTX 4060 Laptop GPU, 8 GB", {}),
    ], size=15, color=PAPER)
    footer(s, n)

    # 9. Results table
    s = new_slide(prs)
    n += 1
    title(s, "Results on the test split")
    data = [HEADER] + rows(runs)
    tbl = s.shapes.add_table(len(data), len(data[0]), MARGIN, Inches(1.7), W - 2 * MARGIN,
                             Inches(3.0)).table
    widths = [2.8, 1.35, 1.4, 1.25, 1.1, 1.15, 1.45, 1.63]
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if i == 0 else (MIST if i % 2 == 0 else PAPER)
            cell.text_frame.paragraphs[0].text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = v
            r.font.size, r.font.name = Pt(14 if i else 12), BODY
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = PAPER if i == 0 else INK
    note = ("Cells marked TBD are filled automatically: run python run_all.py, then "
            "python docs/build_slides.py." if not complete else
            "Reduced-data run (--subset): for pipeline verification only." if not full else
            "7,500 test reviews; pretokenized latency. Custom split: not an official leaderboard score.")
    text(s, MARGIN, Inches(5.1), W - 2 * MARGIN, Inches(0.5), note, size=14, italic=True,
         color=MUTED)
    footer(s, n)

    # 10. Accuracy vs cost
    s = new_slide(prs)
    n += 1
    title(s, "Accuracy vs. cost")
    if complete:
        cd = CategoryChartData()
        cd.categories = [SHORT[m] for m in ORDER]
        cd.add_series("Test accuracy (%)", [round(runs[m]["test"]["accuracy"] * 100, 2)
                                            for m in ORDER])
        gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, MARGIN, Inches(1.6), Inches(5.8),
                                Inches(4.9), cd)
        ch = gf.chart
        ch.has_legend = False
        ch.has_title = True
        ch.chart_title.text_frame.text = "Test accuracy (%)"
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
        plot = ch.plots[0]
        plot.has_data_labels = True
        plot.data_labels.number_format = "0.00"
        plot.data_labels.number_format_is_linked = False
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        plot.data_labels.font.size = Pt(12)
        for i, pt in enumerate(plot.series[0].points):
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = MODEL_COLORS[i]
        va = ch.value_axis
        lo = min(runs[m]["test"]["accuracy"] for m in ORDER) * 100
        va.minimum_scale = max(0, int(lo // 5) * 5 - 5)
        va.maximum_scale = 100
        va.has_major_gridlines = False
        va.tick_labels.font.size = Pt(11)
        ch.category_axis.tick_labels.font.size = Pt(12)
        cost_rows = [['Model', 'Params', 'Train min', 'Single ms']]
        cost_rows += [[SHORT[m], f"{runs[m]['params_total']/1e6:.2f}M",
                       f"{runs[m]['train_time_s']/60:.1f}",
                       f"{runs[m]['latency_ms_single_review']:.2f}"] for m in ORDER]
        cost_table = s.shapes.add_table(5, 4, Inches(6.7), Inches(2.25), Inches(6.0), Inches(2.7)).table
        for j, width in enumerate([2.1, 1.2, 1.35, 1.35]):
            cost_table.columns[j].width = Inches(width)
        for i, values in enumerate(cost_rows):
            for j, value in enumerate(values):
                cell = cost_table.cell(i, j)
                cell.text = value
                cell.fill.solid()
                cell.fill.fore_color.rgb = INK if i == 0 else MIST
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size, run.font.name = Pt(13), BODY
                        run.font.color.rgb = PAPER if i == 0 else INK
        text(s, Inches(6.7), Inches(5.15), Inches(6.0), Inches(1.1),
             'Selected-trial training time. Single-review latency excludes tokenization. '
             'Full cost plots and total search times are in the report and results files.', size=14, color=MUTED)
    else:
        picture_or_note(s, RESULTS / "tradeoffs.png", MARGIN, Inches(1.6), W - 2 * MARGIN,
                        Inches(4.9),
                        "Accuracy vs. parameters, latency and training time appears here after "
                        "the full run (python run_all.py, then python docs/build_slides.py).")
    footer(s, n)

    # 11. Discussion
    s = new_slide(prs)
    n += 1
    title(s, "What the comparison shows")
    found = findings(runs) or [
        "Key findings are generated from the results once all four models have been run.",
    ]
    bullets(s, MARGIN, Inches(1.7), Inches(7.2), Inches(4.8), found, size=17, after=14)
    box(s, Inches(8.3), Inches(1.7), Inches(4.4), Inches(4.8), MIST)
    text(s, Inches(8.6), Inches(1.95), Inches(3.9), Inches(4.4), [
        ("Error analysis", {"size": 20, "bold": True, "font": HEAD, "after": 10}),
        ("50 most confident mistakes per model, saved to misclassified.csv", {"after": 12,
                                                                              "color": MUTED}),
        ("Candidate causes to inspect:", {"after": 6}),
        ("Mixed sentiment, sarcasm, irony", {"after": 6}),
        ("Verdict in the truncated tail", {"after": 6}),
        ("See measured errors in the report", {}),
    ], size=15)
    footer(s, n)

    # 12. Conclusion
    s = new_slide(prs, dark=True)
    n += 1
    title(s, "Conclusion and future work", dark=True)
    bullets(s, MARGIN, Inches(1.8), Inches(6.0), Inches(4.5), [
        (f"DistilBERT led this experiment at {runs['distilbert']['test']['accuracy']*100:.2f}% test accuracy."
         if complete else 'Results are pending completed experiments.'),
        'The MLP offers a low compute baseline; the BiLSTM is the strongest non-transformer in this run.',
        'One training seed and duplicate reviews across splits limit the conclusions. '
        'A grouped split and repeated seeds are needed to assess robustness.',
    ], size=17, color=PAPER, after=14)
    text(s, Inches(7.2), Inches(1.8), Inches(5.5), Inches(0.5), "Next steps", size=21,
         bold=True, font=HEAD, color=AMBER)
    bullets(s, Inches(7.2), Inches(2.45), Inches(5.5), Inches(4.0), [
        "Multi-seed runs with confidence intervals",
        "Head + tail truncation for long reviews",
        "RoBERTa / DeBERTa and LoRA fine-tuning",
        "Distil DistilBERT into TextCNN for on-device use",
        "Visualise BiLSTM attention weights",
    ], size=16, color=RGBColor(0xC9, 0xCD, 0xDB), after=10)
    footer(s, n, dark=True)

    # 13. References
    s = new_slide(prs)
    n += 1
    title(s, "References")
    refs = [
        "[1] A. L. Maas et al., \"Learning Word Vectors for Sentiment Analysis,\" ACL, 2011.",
        "[2] Y. Kim, \"Convolutional Neural Networks for Sentence Classification,\" EMNLP, 2014.",
        "[3] J. Devlin et al., \"BERT: Pre-training of Deep Bidirectional Transformers,\" "
        "NAACL, 2019.",
        "[4] J. Howard and S. Ruder, \"Universal Language Model Fine-Tuning,\" ACL, 2018.",
        "[5] V. Sanh et al., \"DistilBERT, a distilled version of BERT,\" arXiv:1910.01108, 2019.",
        "[6] Z. Yang et al., \"Hierarchical Attention Networks,\" NAACL, 2016.",
        "[7] J. Pennington et al., \"GloVe: Global Vectors for Word Representation,\" EMNLP, 2014.",
    ]
    text(s, MARGIN, Inches(1.7), W - 2 * MARGIN, Inches(4.8),
         [(r, {"after": 10}) for r in refs], size=15)
    footer(s, n)

    # OOXML axis IDs are unsigned integers; python-pptx's legacy defaults can be
    # negative, which causes strict importers to reject otherwise editable charts.
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_chart:
                for element in shape.chart._chartSpace.iter():
                    if element.tag.rsplit('}', 1)[-1] in ('axId', 'crossAx'):
                        value = element.get('val')
                        if value is not None and int(value) < 0:
                            element.set('val', str(int(value) + 2**32))
    prs.save(OUT)
    print(f"saved {OUT}  ({n} slides; results complete: {complete}, full data: {full})")


if __name__ == "__main__":
    build()
