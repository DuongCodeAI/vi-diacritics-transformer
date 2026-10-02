"""Đánh giá thêm dấu.

- char_acc: chỉ tính trên ký tự "có lựa chọn" (nguyên âm, d). Tính trên mọi ký tự
  thì phụ âm/dấu cách luôn đúng làm số đẹp giả tạo.
- word_acc: âm tiết đúng hoàn toàn. Đây là số người dùng cảm nhận được.
- sent_acc: cả câu đúng.
- hallucination: bỏ dấu output mà KHÁC input -> model đã đổi chữ/thêm bớt từ.
  Với cách gán nhãn từng ký tự thì luôn = 0 (by design); seq2seq thì không.
"""

import re
from collections import Counter

from .labels import is_ambiguous, strip

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def evaluate(preds: list[str], golds: list[str]) -> dict:
    amb_ok = amb_n = w_ok = w_n = s_ok = halluc = 0
    for p, g in zip(preds, golds, strict=True):
        s_ok += p == g
        if strip(p) != strip(g):
            halluc += 1
        if len(p) == len(g):
            for pc, gc in zip(p, g, strict=True):
                b = strip(gc)
                if is_ambiguous(b):
                    amb_n += 1
                    amb_ok += pc == gc
        else:
            amb_n += sum(is_ambiguous(strip(c)) for c in g)
        pw, gw = _TOKEN_RE.findall(p), _TOKEN_RE.findall(g)
        w_n += len(gw)
        if len(pw) == len(gw):
            w_ok += sum(a == b for a, b in zip(pw, gw, strict=True))
    n = max(len(golds), 1)
    return {
        "char_acc": amb_ok / max(amb_n, 1),
        "word_acc": w_ok / max(w_n, 1),
        "sent_acc": s_ok / n,
        "hallucination_rate": halluc / n,
        "n": len(golds),
    }


def confusions(preds: list[str], golds: list[str], top: int = 30) -> list[tuple[str, str, int]]:
    """Cặp (đúng -> đoán sai) hay gặp nhất, ở mức âm tiết, không phân biệt hoa thường."""
    cnt: Counter = Counter()
    for p, g in zip(preds, golds, strict=True):
        pw, gw = _TOKEN_RE.findall(p.lower()), _TOKEN_RE.findall(g.lower())
        if len(pw) != len(gw):
            continue
        for a, b in zip(pw, gw, strict=True):
            if a != b:
                cnt[(b, a)] += 1
    return [(g, p, c) for (g, p), c in cnt.most_common(top)]


def errors_by_syllable(preds: list[str], golds: list[str], top: int = 30) -> list[tuple[str, int, int, float]]:
    """Âm tiết không dấu nào khó nhất: (âm tiết, số lỗi, số lần xuất hiện, tỉ lệ lỗi)."""
    err: Counter = Counter()
    tot: Counter = Counter()
    for p, g in zip(preds, golds, strict=True):
        pw, gw = _TOKEN_RE.findall(p.lower()), _TOKEN_RE.findall(g.lower())
        if len(pw) != len(gw):
            continue
        for a, b in zip(pw, gw, strict=True):
            f = strip(b)
            tot[f] += 1
            err[f] += a != b
    rows = [(f, err[f], tot[f], err[f] / tot[f]) for f in tot if tot[f] >= 20]
    rows.sort(key=lambda r: -r[1])
    return rows[:top]
