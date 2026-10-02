"""Transformer viết tay - không dùng nn.Transformer / nn.MultiheadAttention.

Chỉ dùng các khối cơ bản (Linear, Embedding, LayerNorm, Dropout) để tự cài:
scaled dot-product attention, multi-head, positional encoding, encoder/decoder layer.

Hai model:
- TaggerTransformer : encoder-only, mỗi ký tự -> 1 trong 30 nhãn (shape, tone)
- Seq2SeqTransformer: encoder-decoder, sinh câu có dấu token-by-token (BPE)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout: nn.Module | None = None):
    """q,k,v: (B, H, T, d). mask: True = ĐƯỢC nhìn, broadcast được về (B, H, Tq, Tk).

    Chia sqrt(d) để tích vô hướng không lớn dần theo d -> softmax không bị bão hoà
    (gradient gần 0) khi d lớn.
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None  # giữ lại để vẽ attention map
        self.keep_attn = False

    def _split(self, x):  # (B, T, D) -> (B, H, T, d)
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        q, k, v = self._split(self.q_proj(x_q)), self._split(self.k_proj(x_kv)), self._split(self.v_proj(x_kv))
        y, attn = scaled_dot_product_attention(q, k, v, mask, self.drop)
        if self.keep_attn:
            self.last_attn = attn.detach()
        b, _, t, _ = y.shape
        return self.out(y.transpose(1, 2).reshape(b, t, self.h * self.d))


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...).

    Chọn sinusoidal thay vì learned: không giới hạn cứng độ dài lúc suy luận và
    không tốn tham số. Ký tự-level nên câu dài 200+ ký tự là chuyện thường.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN: LayerNorm trước sub-layer. Ổn định hơn post-LN (paper gốc), train được
    không cần warmup quá dài - quan trọng khi chỉ có vài giờ GPU Kaggle."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, self_mask, cross_mask):
        h = self.ln1(y)
        y = y + self.drop(self.self_attn(h, h, self_mask))
        y = y + self.drop(self.cross_attn(self.ln2(y), memory, cross_mask))
        return y + self.drop(self.ff(self.ln3(y)))


def padding_mask(ids, pad_id: int = 0):
    """(B, T) -> (B, 1, 1, T): True ở vị trí không phải padding."""
    return (ids != pad_id)[:, None, None, :]


def causal_mask(t: int, device=None):
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))[None, None]


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len=2048, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, ids, mask):
        x = self.drop(self.pos(self.emb(ids) * self.scale))
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)


class TaggerTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int, label_mask: torch.Tensor, d_model=256, n_heads=8,
                 n_layers=6, d_ff=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout, pad_id=pad_id)
        self.head = nn.Linear(d_model, n_labels)
        # (V, L) - nhãn nào hợp lệ với ký tự nào; không train
        self.register_buffer("label_mask", label_mask.bool(), persistent=True)

    def forward(self, ids):
        h = self.encoder(ids, padding_mask(ids, self.pad_id))
        logits = self.head(h)
        allowed = self.label_mask[ids]  # (B, T, L)
        return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)

    def loss(self, ids, labels):
        logits = self(ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)

    def set_keep_attn(self, flag: bool):
        for layer in self.encoder.layers:
            layer.attn.keep_attn = flag


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=8, n_enc=4, n_dec=4, d_ff=1024, dropout=0.1,
                 pad_id=0, tie_embeddings=True):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(src_vocab, d_model, n_heads, n_enc, d_ff, dropout, pad_id=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec))
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_embeddings:
            self.out.weight = self.tgt_emb.weight
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def decode(self, tgt_in, memory, src_mask):
        t = tgt_in.size(1)
        self_mask = padding_mask(tgt_in, self.pad_id) & causal_mask(t, tgt_in.device)
        y = self.drop(self.pos(self.tgt_emb(tgt_in) * self.scale))
        for layer in self.layers:
            y = layer(y, memory, self_mask, src_mask)
        return self.out(self.ln(y))

    def forward(self, src, tgt_in):
        src_mask = padding_mask(src, self.pad_id)
        return self.decode(tgt_in, self.encoder(src, src_mask), src_mask)

    def loss(self, src, tgt_in, tgt_out, label_smoothing=0.1):
        logits = self(src, tgt_in)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt_out.reshape(-1),
                               ignore_index=self.pad_id, label_smoothing=label_smoothing)

    @torch.no_grad()
    def greedy(self, src, bos_id, eos_id, max_len=256):
        src_mask = padding_mask(src, self.pad_id)
        memory = self.encoder(src, src_mask)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            # không dùng KV-cache cho gọn; đủ nhanh với câu ngắn
            nxt = self.decode(ys, memory, src_mask)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos_id
            if done.all():
                break
        return ys[:, 1:]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
