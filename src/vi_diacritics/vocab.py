"""Bảng ký tự đầu vào (đã bỏ dấu) + mask nhãn hợp lệ cho từng ký tự."""

import json
from collections import Counter
from pathlib import Path

from .labels import N_LABELS, allowed_labels

PAD, UNK = "<pad>", "<unk>"


class CharVocab:
    def __init__(self, chars: list[str]):
        self.itos = [PAD, UNK] + [c for c in chars if c not in (PAD, UNK)]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, folded_texts, min_freq: int = 20, max_size: int = 300) -> "CharVocab":
        cnt = Counter()
        for t in folded_texts:
            cnt.update(t)
        chars = [c for c, n in cnt.most_common(max_size) if n >= min_freq]
        # luôn có đủ chữ cái a-z, A-Z dù corpus thiếu
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ":
            if c not in chars:
                chars.append(c)
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(c, unk) for c in text]

    def label_mask(self) -> list[list[bool]]:
        """mask[char_id][label] = True nếu nhãn hợp lệ với ký tự đó."""
        rows = []
        for c in self.itos:
            ok = set(allowed_labels(c)) if c not in (PAD, UNK) else {0}
            rows.append([i in ok for i in range(N_LABELS)])
        return rows

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        itos = json.loads(Path(path).read_text("utf-8"))["itos"]
        v = cls([])
        v.itos = itos
        v.stoi = {c: i for i, c in enumerate(itos)}
        return v
