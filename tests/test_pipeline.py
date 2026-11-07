"""Offline regression checks for preprocessing, padding, training and saved models."""
from argparse import Namespace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import Vocab, clean_text, tfidf_loaders, split_fingerprint
from src.models import TextCNN, BiLSTMAttention, MLPBoW, DistilBERTClassifier
from src.engine import train, compute_metrics, measure_latency
from src.artifacts import save_bundle, load_bundle, inference_loader

torch.set_num_threads(2)


def fixture_splits():
    train = (['good lovely movie', 'bad boring film', 'good film', 'bad movie'], np.array([1, 0, 1, 0]))
    return {'train': train, 'val': (['uniquevalidation good', 'bad'], np.array([1, 0])),
            'test': (['uniquetest movie', 'lovely'], np.array([0, 1]))}


def test_cleaning_and_empty_input():
    assert clean_text("<br />I DON'T &amp; won't!!!") == 'i dont wont'
    vocab = Vocab(['good movie'])
    ids, lengths = vocab.encode(['', 'unseen good movie'], 2)
    assert ids.tolist() == [[1, 0], [1, vocab.stoi['good']]]
    assert lengths.tolist() == [1, 2]


def test_train_only_features_and_truncation():
    splits = fixture_splits()
    _, info = tfidf_loaders(splits, 2, max_len=1)
    assert set(info['vectorizer'].vocabulary_) == {'good', 'bad'}
    vocab = Vocab(splits['train'][0])
    assert 'uniquetest' not in vocab.stoi and 'uniquevalidation' not in vocab.stoi
    assert split_fingerprint(splits) == split_fingerprint(fixture_splits())


@pytest.mark.parametrize('cls', [TextCNN, BiLSTMAttention])
def test_padding_invariance_and_backward(cls):
    torch.manual_seed(13)
    model = cls(20).eval()
    short = torch.tensor([[2, 3]])
    padded = torch.tensor([[2, 3, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9]])
    a = model(short, torch.tensor([2]))
    b = model(padded, torch.tensor([2, 6]))[:1]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    b.sum().backward()
    assert torch.isfinite(model.embedding.weight.grad).all()
    if cls is BiLSTMAttention:
        assert model.last_attention[0, 2:].sum() == 0
        torch.testing.assert_close(model.last_attention.sum(1), torch.ones(2))


def test_metrics_known_confusion():
    result = compute_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .7, .9]))
    assert result['confusion_matrix'] == [[1, 1], [0, 2]]
    assert result['accuracy'] == .75
    assert result['f1'] == .8


@pytest.mark.parametrize('name', ['mlp', 'textcnn', 'bilstm'])
def test_saved_prediction_roundtrip(tmp_path, name):
    splits = fixture_splits()
    if name == 'mlp':
        _, info = tfidf_loaders(splits, 2)
        model = MLPBoW(info['input_dim'])
    else:
        info = {'vocab': Vocab(splits['train'][0])}
        model = (TextCNN if name == 'textcnn' else BiLSTMAttention)(len(info['vocab']))
    model.eval()
    args = Namespace(model=name, max_len=16, freeze_emb=False)
    save_bundle(model, info, args, tmp_path)
    restored, spec, fitted = load_bundle(tmp_path)
    inputs, _ = next(iter(inference_loader(['good movie'], spec, fitted)))
    torch.testing.assert_close(model(**inputs), restored(**inputs))


def test_distilbert_forward_offline():
    from transformers import DistilBertConfig
    cfg = DistilBertConfig(vocab_size=20, n_layers=1, dim=16, hidden_dim=32, n_heads=2, num_labels=1)
    model = DistilBERTClassifier(config=cfg)
    logits = model(torch.tensor([[2, 3, 4]]), torch.ones(1, 3, dtype=torch.long))
    assert logits.shape == (1,)
    logits.sum().backward()


class TinyDataset(Dataset):
    def __len__(self): return 5
    def __getitem__(self, i):
        return {'x': torch.tensor([float(i) / 5])}, torch.tensor(float(i % 2))


def test_accumulation_partial_group_matches_large_batches():
    torch.manual_seed(9)
    model_a = MLPBoW(1, hidden=(), dropout=0)
    model_b = MLPBoW(1, hidden=(), dropout=0)
    model_b.load_state_dict(model_a.state_dict())
    validation = DataLoader(TinyDataset(), batch_size=5)
    # Two groups of 4 and 1 examples: microbatches [2,2,1] versus batches [4,1].
    for model, batch, accum in [(model_a, 2, 2), (model_b, 4, 1)]:
        loaders = {'train': DataLoader(TinyDataset(), batch_size=batch), 'val': validation}
        train(model, loaders, torch.device('cpu'), lr=.01, epochs=1, patience=1, grad_accum=accum, amp=False)
    for a, b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_latency_small_dataset():
    loader = DataLoader(TinyDataset(), batch_size=2)
    result = measure_latency(MLPBoW(1, hidden=()), loader, torch.device('cpu'), amp=False)
    assert result['latency_single_samples'] == 5
    assert result['latency_ms_single_review'] > 0


def test_glove_cache_respects_seed_and_source(tmp_path, monkeypatch):
    from src import config as C
    from src.glove import build_embedding_matrix
    monkeypatch.setattr(C, 'GLOVE_DIR', tmp_path / 'cache')
    source = tmp_path / 'glove.txt'
    source.write_text('good 1 2 3\n', encoding='utf-8')
    vocab = Vocab(['good unseen'])
    first = build_embedding_matrix(vocab, source, dim=3, seed=1)
    second = build_embedding_matrix(vocab, source, dim=3, seed=2)
    np.testing.assert_equal(first[vocab.stoi['good']], [1, 2, 3])
    assert not np.array_equal(first[vocab.stoi['unseen']], second[vocab.stoi['unseen']])
    np.testing.assert_equal(first[0], [0, 0, 0])


def test_distilbert_bundle_roundtrip_offline(tmp_path):
    from transformers import DistilBertConfig, DistilBertTokenizerFast
    cfg = DistilBertConfig(vocab_size=8, n_layers=1, dim=16, hidden_dim=32, n_heads=2, num_labels=1)
    vocab_file = tmp_path / 'tokens.txt'
    vocab_file.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\n')
    tok = DistilBertTokenizerFast(vocab_file=str(vocab_file))
    model = DistilBERTClassifier(config=cfg).eval()
    destination = tmp_path / 'bundle'
    save_bundle(model, {'tokenizer': tok}, Namespace(model='distilbert', max_len=16), destination)
    restored, spec, info = load_bundle(destination)
    inputs, _ = next(iter(inference_loader(['good movie'], spec, info)))
    torch.testing.assert_close(model(**inputs), restored(**inputs))


def test_comparison_rejects_mixed_splits(tmp_path, monkeypatch):
    import json
    from src import config as C
    from src.compare import load_results
    monkeypatch.setattr(C, 'RESULTS_DIR', tmp_path)
    for name, fingerprint in [('mlp', 'split-a'), ('textcnn', 'split-b')]:
        folder = tmp_path / name
        folder.mkdir()
        record = {'display_name': name, 'split_fingerprint': fingerprint, 'max_len':512,
                  'test': {'accuracy':.8, 'precision':.8, 'recall':.8, 'f1':.8},
                  'params_total':100, 'params_trainable':100, 'train_time_s':1,
                  'epochs_run':1, 'latency_ms_single_review':1, 'latency_ms_per_review_batched':1,
                  'validation_f1':.8}
        (folder/'metrics.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Mixed data splits'):
        load_results()
