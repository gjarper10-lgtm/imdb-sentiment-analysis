"""The four architectures. Every model returns one logit per review (positive = 1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import config as C


def _embedding(vocab_size, embeddings, freeze):
    if embeddings is not None:
        return nn.Embedding.from_pretrained(torch.as_tensor(embeddings), freeze=freeze,
                                            padding_idx=C.PAD_IDX)
    return nn.Embedding(vocab_size, C.EMBED_DIM, padding_idx=C.PAD_IDX)


class MLPBoW(nn.Module):
    """Classical baseline: dense network over a TF-IDF bag-of-words vector (no word order)."""

    def __init__(self, input_dim, hidden=(512, 128), dropout=0.5):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TextCNN(nn.Module):
    """Kim (2014): parallel 1D convolutions (n-gram detectors) + max-over-time pooling."""

    def __init__(self, vocab_size, embeddings=None, freeze_emb=False,
                 kernel_sizes=(2, 3, 4), n_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = _embedding(vocab_size, embeddings, freeze_emb)
        self.convs = nn.ModuleList(nn.Conv1d(C.EMBED_DIM, n_filters, k) for k in kernel_sizes)
        self.max_k = max(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, ids, lengths=None):
        if ids.size(1) < self.max_k:                       # very short reviews
            ids = F.pad(ids, (0, self.max_k - ids.size(1)), value=C.PAD_IDX)
        x = self.embedding(ids).transpose(1, 2)            # [B, E, T]
        pooled = [F.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(-1)


class BiLSTMAttention(nn.Module):
    """Stacked BiLSTM with additive (Bahdanau-style) attention pooling over time steps."""

    def __init__(self, vocab_size, embeddings=None, freeze_emb=False,
                 hidden=128, num_layers=2, dropout=0.3, attn_dim=128):
        super().__init__()
        self.embedding = _embedding(vocab_size, embeddings, freeze_emb)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(C.EMBED_DIM, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_proj = nn.Linear(2 * hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)
        self.last_attention = None                          # [B, T], kept for analysis

    def forward(self, ids, lengths):
        x = self.emb_dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))  # [B,T,2H]

        scores = self.attn_v(torch.tanh(self.attn_proj(out))).squeeze(-1)  # [B, T]
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        self.last_attention = weights.detach()
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)          # [B, 2H]
        return self.fc(self.dropout(context)).squeeze(-1)


class DistilBERTClassifier(nn.Module):
    """Pre-trained DistilBERT (base, uncased) fine-tuned end to end with a single-logit head."""

    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
