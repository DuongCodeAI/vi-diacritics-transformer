"""Chuẩn bị dữ liệu: Wikipedia tiếng Việt -> câu sạch -> train/val/test.

Dữ liệu "miễn phí": lấy câu có dấu, bỏ dấu đi là có ngay cặp (input, target).
Hai cái bẫy:
1. Câu không có dấu sẵn (tên riêng nước ngoài, code, tiếng Anh) -> model học
   "không thêm dấu". Lọc câu có quá ít âm tiết mang dấu.
2. Cùng một từ có 2 kiểu bỏ dấu (hoà/hòa, thuỷ/thủy) -> nhãn nhiễu. Chuẩn hoá về
   một kiểu trước khi tạo nhãn.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

from .labels import strip

_TONES = {"̀", "́", "̃", "̉", "̣"}
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ\"“(])")
_BAD = re.compile(r"[{}<>|\\=_#@^~`]|https?://")


def normalize_tone_placement(syl: str) -> str:
    """hoà/thuý/khoẻ -> hòa/thúy/khỏe (đặt dấu ở nguyên âm đầu của oa, oe, uy khi âm tiết mở)."""
    nfd = unicodedata.normalize("NFD", syl)
    tone = next((c for c in nfd if c in _TONES), None)
    if tone is None:
        return syl
    bare = unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _TONES))
    low = bare.lower()
    if not low.endswith(("oa", "oe", "uy")) or (low.endswith("uy") and low[:-2].endswith("q")):
        return syl
    pos = len(bare) - 2
    return bare[:pos] + unicodedata.normalize("NFC", bare[pos] + tone) + bare[pos + 1:]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _WORD_RE.sub(lambda m: normalize_tone_placement(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


def accent_ratio(text: str) -> float:
    words = _WORD_RE.findall(text)
    words = [w for w in words if not w.isdigit()]
    if not words:
        return 0.0
    return sum(1 for w in words if strip(w) != w) / len(words)


def is_good_sentence(s: str, min_len: int = 15, max_len: int = 256, min_accent: float = 0.3) -> bool:
    if not (min_len <= len(s) <= max_len) or _BAD.search(s):
        return False
    letters = sum(c.isalpha() for c in s)
    if letters < 0.6 * len(s):
        return False
    return accent_ratio(s) >= min_accent


def split_sentences(paragraph: str) -> list[str]:
    return [p.strip() for p in _SENT_SPLIT.split(paragraph) if p.strip()]


def sentences_from_docs(docs: Iterable[str]) -> Iterator[str]:
    for doc in docs:
        for para in doc.split("\n"):
            para = para.strip()
            if len(para) < 20 or para.startswith(("Thể loại:", "Tham khảo", "Liên kết ngoài")):
                continue
            for s in split_sentences(para):
                s = normalize_text(s)
                if is_good_sentence(s):
                    yield s


def split_of(sentence: str, val_pct: float = 0.5, test_pct: float = 0.5) -> str:
    """Chia theo hash của câu -> câu trùng luôn rơi vào cùng một tập, không bị rò rỉ train/test."""
    h = int(hashlib.md5(sentence.encode("utf-8")).hexdigest()[:8], 16) % 10000 / 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def build_splits(docs: Iterable[str], out_dir: str | Path, max_train: int = 2_000_000) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (out_dir / f"{k}.txt").open("w", encoding="utf-8") for k in ("train", "val", "test")}
    counts = dict.fromkeys(files, 0)
    seen: set[int] = set()
    try:
        for s in sentences_from_docs(docs):
            key = hash(s)
            if key in seen:
                continue
            seen.add(key)
            sp = split_of(s)
            files[sp].write(s + "\n")
            counts[sp] += 1
            if counts["train"] >= max_train:
                break
    finally:
        for f in files.values():
            f.close()
    return counts


def iter_wikipedia(config: str = "20231101.vi") -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    for row in ds:
        yield row["text"]
