"""Portable model bundles: weights and the exact fitted preprocessing objects."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Vocab, clean_text, tokenize, _IndexDataset
from .models import MLPBoW, TextCNN, BiLSTMAttention, DistilBERTClassifier


def save_bundle(model, info, args, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / 'model.pt')
    spec = {'model': args.model, 'max_len': args.max_len, 'format_version': 1}
    if args.model == 'mlp':
        joblib.dump(info['vectorizer'], directory / 'tfidf.joblib')
        spec['input_dim'] = info['input_dim']
    elif args.model in ('textcnn', 'bilstm'):
        (directory / 'vocab.json').write_text(json.dumps(info['vocab'].itos), encoding='utf-8')
        spec['vocab_size'] = len(info['vocab'])
        spec['freeze_emb'] = args.freeze_emb
    else:
        model.model.config.save_pretrained(directory / 'transformer')
        info['tokenizer'].save_pretrained(directory / 'transformer')
    (directory / 'bundle.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')


def load_bundle(directory, device='cpu'):
    """Only load trusted bundles: scikit-learn's joblib format uses pickle."""
    directory = Path(directory)
    spec = json.loads((directory / 'bundle.json').read_text(encoding='utf-8'))
    name = spec['model']
    info = {}
    if name == 'mlp':
        model = MLPBoW(spec['input_dim'])
        info['vectorizer'] = joblib.load(directory / 'tfidf.joblib')
    elif name in ('textcnn', 'bilstm'):
        vocab = Vocab([])
        vocab.itos = json.loads((directory / 'vocab.json').read_text(encoding='utf-8'))
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        info['vocab'] = vocab
        cls = TextCNN if name == 'textcnn' else BiLSTMAttention
        model = cls(spec['vocab_size'])
    else:
        from transformers import AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(directory / 'transformer', local_files_only=True)
        model = DistilBERTClassifier(config=cfg)
        info['tokenizer'] = AutoTokenizer.from_pretrained(directory / 'transformer', local_files_only=True)
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    model.to(device).eval()
    return model, spec, info


def inference_loader(texts, spec, info, batch_size=32, labels=None):
    if not texts:
        raise ValueError('At least one review is required')
    texts = [clean_text(t) for t in texts]
    labels = np.zeros(len(texts), dtype=np.float32) if labels is None else np.asarray(labels, dtype=np.float32)
    name, length = spec['model'], spec['max_len']
    if name == 'mlp':
        matrix = info['vectorizer'].transform([' '.join(tokenize(t)[:length]) for t in texts])
        def collate(ix):
            return {'x': torch.from_numpy(matrix[ix].toarray())}, torch.from_numpy(labels[ix])
    elif name in ('textcnn', 'bilstm'):
        ids, lengths = info['vocab'].encode(texts, length)
        def collate(ix):
            width = int(lengths[ix].max())
            return {'ids': torch.from_numpy(ids[ix, :width]), 'lengths': torch.from_numpy(lengths[ix])}, torch.from_numpy(labels[ix])
    else:
        tok = info['tokenizer']
        ids = tok(texts, truncation=True, max_length=length)['input_ids']
        def collate(ix):
            batch = tok.pad({'input_ids': [ids[i] for i in ix]}, return_tensors='pt')
            return dict(batch), torch.from_numpy(labels[ix])
    return DataLoader(_IndexDataset(len(texts)), batch_size=batch_size, collate_fn=collate)
