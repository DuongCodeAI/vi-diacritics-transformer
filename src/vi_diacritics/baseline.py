"""Baseline thống kê: chọn dạng có dấu phổ biến nhất cho mỗi âm tiết, có xét từ đứng trước.

Phải có baseline trước khi làm deep learning, không thì không biết Transformer
hơn được bao nhiêu. Bigram đơn giản này đã đúng ~85-90% âm tiết - phần còn lại
(ma/mà/má/mã/mạ phụ thuộc ngữ cảnh xa) mới là chỗ Transformer phải chứng minh.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .labels import strip

_TOKEN_RE = re.compile(r"(\w+)", re.UNICODE)


def _restore_case(src: str, word: str) -> str:
    """Chép kiểu hoa/thường của từng ký tự gốc sang từ đã thêm dấu (bỏ dấu không đổi độ dài).

    Bản đầu chỉ xử lý "viết hoa hết" / "hoa chữ đầu" -> "iPhone", "McDonald" bị đổi chữ, chạy trên
    Colab ra hallucination_rate 1.6% cho baseline (lẽ ra phải 0).
    """
    if len(src) != len(word):
        return src
    return "".join(c.upper() if s.isupper() else c for s, c in zip(src, word, strict=True))


class NgramBaseline:
    def __init__(self):
        self.uni: dict[str, Counter] = defaultdict(Counter)
        self.bi: dict[tuple[str, str], Counter] = defaultdict(Counter)

    def fit(self, sentences) -> "NgramBaseline":
        for s in sentences:
            words = [w.lower() for w in _TOKEN_RE.findall(s)]
            prev = "<s>"
            for w in words:
                f = strip(w)
                self.uni[f][w] += 1
                self.bi[(prev, f)][w] += 1
                prev = w
        return self

    def prune(self, min_count: int = 2) -> None:
        self.bi = defaultdict(Counter, {k: v for k, v in self.bi.items() if sum(v.values()) >= min_count})

    def restore(self, text: str) -> str:
        parts = _TOKEN_RE.split(text)
        prev = "<s>"
        out = []
        for i, p in enumerate(parts):
            if i % 2 == 0:  # phần không phải từ (khoảng trắng, dấu câu)
                out.append(p)
                continue
            f = strip(p).lower()
            cand = self.bi.get((prev, f)) or self.uni.get(f)
            w = cand.most_common(1)[0][0] if cand else p.lower()  # từ lạ: giữ nguyên, không đoán
            out.append(_restore_case(p, w))
            prev = w
        return "".join(out)

    def save(self, path: str | Path) -> None:
        data = {"uni": {k: dict(v) for k, v in self.uni.items()},
                "bi": {f"{a}\t{b}": dict(v) for (a, b), v in self.bi.items()}}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NgramBaseline":
        data = json.loads(Path(path).read_text("utf-8"))
        m = cls()
        m.uni = defaultdict(Counter, {k: Counter(v) for k, v in data["uni"].items()})
        m.bi = defaultdict(Counter, {tuple(k.split("\t")): Counter(v) for k, v in data["bi"].items()})
        return m
