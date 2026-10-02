"""API suy luận dùng cho các dự án khác (RAG luật, viet-copilot). Chỉ cần numpy + onnxruntime.

    from vi_diacritics import Restorer
    r = Restorer.load("export/")
    r.restore("xe may vuot den do phat bao nhieu")       # 'xe máy vượt đèn đỏ phạt bao nhiêu'
    r.restore_detail("ma toi khong biet")               # kèm độ tự tin từng từ

- Giữ nguyên dấu người dùng đã gõ (gõ thiếu dấu một vài chữ là chuyện thường).
- Từ nào model không chắc (< min_conf) thì để nguyên không dấu thay vì đoán bừa:
  với RAG, chữ không dấu vẫn khớp được index bỏ dấu, còn đoán sai dấu thì hỏng.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import decode, encode_target, is_ambiguous
from .vocab import CharVocab

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class WordInfo:
    text: str
    conf: float
    changed: bool


class Restorer:
    def __init__(self, session, vocab: CharVocab, min_conf: float = 0.0, max_len: int = 256):
        self.session = session
        self.vocab = vocab
        self.mask = np.array(vocab.label_mask(), dtype=bool)
        self.min_conf = min_conf
        self.max_len = max_len

    @classmethod
    def load(cls, folder: str | Path, int8: bool = True, min_conf: float = 0.0, threads: int = 2) -> "Restorer":
        import onnxruntime as ort

        folder = Path(folder)
        f = folder / ("tagger.int8.onnx" if int8 and (folder / "tagger.int8.onnx").exists() else "tagger.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(f), opts, providers=["CPUExecutionProvider"])
        return cls(sess, CharVocab.load(folder / "vocab.json"), min_conf)

    def _probs(self, folded: str) -> np.ndarray:
        ids = np.array([self.vocab.encode(folded)], dtype=np.int64)
        return self.session.run(None, {"ids": ids})[0][0]  # (T, L)

    def restore_detail(self, text: str) -> tuple[str, list[WordInfo]]:
        if not text.strip():
            return text, []
        pieces = [text[i : i + self.max_len] for i in range(0, len(text), self.max_len)]
        out_chars, out_conf = [], []
        for piece in pieces:
            folded, typed = encode_target(piece)
            probs = self._probs(folded)
            # ép các ký tự đã có dấu giữ nguyên nhãn người dùng gõ
            for i, lab in enumerate(typed):
                if lab != 0:
                    probs[i] = 0.0
                    probs[i, lab] = 1.0
            labels = probs.argmax(-1)
            conf = probs.max(-1)
            out_chars.append(decode(folded, labels.tolist()))
            out_conf.append(np.where([is_ambiguous(c) for c in folded], conf, 1.0))
        restored = "".join(out_chars)
        conf = np.concatenate(out_conf)

        words = []
        result = list(restored)
        for m in _WORD_RE.finditer(restored):
            c = float(conf[m.start() : m.end()].min())
            src = text[m.start() : m.end()]
            if c < self.min_conf:
                result[m.start() : m.end()] = list(src)
                words.append(WordInfo(src, c, False))
            else:
                words.append(WordInfo(m.group(0), c, m.group(0) != src))
        return "".join(result), words

    def restore(self, text: str) -> str:
        return self.restore_detail(text)[0]

