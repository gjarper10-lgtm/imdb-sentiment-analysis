"""Keep six existing, editable slides for a four-minute classroom talk."""
from pathlib import Path
from pptx import Presentation

HERE = Path(__file__).resolve().parent
prs = Presentation(HERE / 'Project_Presentation.pptx')
keep = {0, 3, 5, 7, 8, 11}
for i in range(len(prs.slides) - 1, -1, -1):
    if i not in keep:
        item = prs.slides._sldIdLst[i]
        prs.part.drop_rel(item.rId)
        prs.slides._sldIdLst.remove(item)
notes = [
    '20 seconds. I compared four deep learning architectures for positive versus negative movie-review classification. The aim is to compare accuracy and compute cost under one evaluation protocol.',
    '40 seconds. The dataset has 50,000 labelled reviews. I pooled and stratified it into 35,000 training, 7,500 validation and 7,500 test reviews. Vocabulary and TF-IDF fitting use training data only. This is a custom split, and duplicate texts across splits are a limitation.',
    '60 seconds. MLP uses TF-IDF word features. TextCNN detects local phrases using GloVe embeddings. BiLSTM reads both directions and uses attention pooling. DistilBERT transfers pretrained contextual language knowledge and is fine-tuned on the reviews.',
    '35 seconds. I tried two learning rates per model with a three-epoch cap. I selected the learning rate and checkpoint by validation F1, then evaluated the selected models once on the test set. DistilBERT uses batch eight with two gradient accumulation steps.',
    '50 seconds. DistilBERT achieved 93.29 percent accuracy and 93.30 percent F1. BiLSTM achieved 91.52 percent accuracy, while MLP and TextCNN were around 89.6 percent. These are measured scores from saved predictions on 7,500 reviews.',
    '35 seconds. DistilBERT was best on this experiment, while MLP used much less compute. Limitations are one seed, limited tuning, truncation and duplicate texts across splits. Next I will show live predictions using the saved model weights.'
]
for n, slide in enumerate(prs.slides, 1):
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text.strip().isdigit() and shape.top > prs.slide_height * .9:
            shape.text_frame.paragraphs[0].runs[0].text = str(n)
    slide.notes_slide.notes_text_frame.text = notes[n-1]
prs.save(HERE / 'Presentation_10_Minutes.pptx')
print('Saved six slides with speaking notes.')
