"""Dataset + batching theo độ dài cho 2 hướng tagger / seq2seq."""

import random

import torch
from torch.utils.data import Dataset, Sampler

from .bpe import BOS_ID, EOS_ID, PAD_ID
from .labels import encode_target
from .vocab import CharVocab


def read_lines(path, limit: int | None = None) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(line)
                if limit and len(out) >= limit:
                    break
    return out


class TaggerDataset(Dataset):
    def __init__(self, lines: list[str], vocab: CharVocab, max_len: int = 256):
        self.lines = lines
        self.vocab = vocab
        self.max_len = max_len
        self.mask = vocab.label_mask()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        folded, labels = encode_target(self.lines[i][: self.max_len])
        ids = self.vocab.encode(folded)
        # nhãn không hợp lệ với ký tự (ký tự lạ bị map sang UNK) -> bỏ qua khi tính loss
        labels = [lab if self.mask[c][lab] else -100 for c, lab in zip(ids, labels, strict=True)]
        return ids, labels

    def length(self, i):
        return min(len(self.lines[i]), self.max_len)


class Seq2SeqDataset(Dataset):
    def __init__(self, lines: list[str], src_tok, tgt_tok, max_len: int = 128):
        from .labels import strip

        self.lines = lines
        self.src_tok, self.tgt_tok = src_tok, tgt_tok
        self.max_len = max_len
        self.strip = strip

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        tgt_text = self.lines[i]
        src = self.src_tok.encode(self.strip(tgt_text)).ids[: self.max_len]
        tgt = self.tgt_tok.encode(tgt_text).ids[: self.max_len - 2]
        return src, [BOS_ID] + tgt + [EOS_ID]

    def length(self, i):
        return len(self.lines[i]) // 3  # ước lượng số token, đủ để gom batch


def collate_tagger(batch):
    t = max(len(ids) for ids, _ in batch)
    ids = torch.zeros(len(batch), t, dtype=torch.long)
    labels = torch.full((len(batch), t), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
    return ids, labels


def collate_seq2seq(batch):
    ts = max(len(s) for s, _ in batch)
    tt = max(len(t) for _, t in batch)
    src = torch.full((len(batch), ts), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), tt), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt[:, :-1], tgt[:, 1:]


class LengthGroupedSampler(Sampler):
    """Gom câu có độ dài gần nhau vào cùng batch -> ít padding, nhanh hơn ~2x.
    Vẫn xáo trộn ở mức batch để không học theo thứ tự độ dài."""

    def __init__(self, dataset, batch_size: int, mega: int = 50, seed: int = 0):
        self.ds = dataset
        self.bs = batch_size
        self.mega = mega
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.ds)))
        rng.shuffle(idx)
        batches = []
        step = self.bs * self.mega
        for s in range(0, len(idx), step):
            chunk = sorted(idx[s : s + step], key=self.ds.length)
            batches += [chunk[j : j + self.bs] for j in range(0, len(chunk), self.bs)]
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return (len(self.ds) + self.bs - 1) // self.bs
