"""Biến bài toán thêm dấu thành gán nhãn từng ký tự.

Mỗi ký tự tiếng Việt = chữ gốc + dấu mũ/móc (shape) + dấu thanh (tone):
    'ệ' = 'e' + mũ (^) + nặng
Bỏ dấu không làm đổi độ dài chuỗi, nên chỉ cần dự đoán cho MỖI ký tự đầu vào
một nhãn (shape, tone). 5 shape x 6 tone = 30 lớp. Với mỗi chữ gốc chỉ một số
lớp là hợp lệ ('b' chỉ có lớp 0, 'i' không có mũ...) -> dùng mask khi giải mã,
nên model KHÔNG BAO GIỜ sinh ra ký tự không tồn tại hay đổi chữ.
"""

import unicodedata

SHAPES = ["", "̂", "̆", "̛", "stroke"]  # không, mũ (â ê ô), trăng (ă), móc (ơ ư), gạch (đ)
TONES = ["", "̀", "́", "̉", "̃", "̣"]  # ngang, huyền, sắc, hỏi, ngã, nặng
N_LABELS = len(SHAPES) * len(TONES)

_SHAPE_IDX = {m: i for i, m in enumerate(SHAPES) if m and m != "stroke"}
_TONE_IDX = {m: i for i, m in enumerate(TONES) if m}

# chữ gốc (thường) -> các shape hợp lệ
_ALLOWED_SHAPES = {
    "a": [0, 1, 2],
    "e": [0, 1],
    "o": [0, 1, 3],
    "u": [0, 3],
    "i": [0],
    "y": [0],
    "d": [0, 4],
}


def label_id(shape: int, tone: int) -> int:
    return shape * len(TONES) + tone


def split_label(lab: int) -> tuple[int, int]:
    return divmod(lab, len(TONES))


def decompose(ch: str) -> tuple[str, int, int]:
    """'Ệ' -> ('E', 1, 5); 'đ' -> ('d', 4, 0); 'x' -> ('x', 0, 0)."""
    if ch in "đĐ":
        return ("d" if ch == "đ" else "D"), 4, 0
    nfd = unicodedata.normalize("NFD", ch)
    base, marks = nfd[0], nfd[1:]
    shape = tone = 0
    rest = []
    for m in marks:
        if m in _SHAPE_IDX:
            shape = _SHAPE_IDX[m]
        elif m in _TONE_IDX:
            tone = _TONE_IDX[m]
        else:
            rest.append(m)
    if rest or (shape == 0 and tone == 0 and marks):
        # ký tự có dấu lạ (ç, ñ...) -> coi như ký tự thường, không gán dấu
        return ch, 0, 0
    if base.lower() not in _ALLOWED_SHAPES and (shape or tone):
        return ch, 0, 0
    return base, shape, tone


def compose(base: str, shape: int, tone: int) -> str:
    if shape == 4:
        return "đ" if base == "d" else "Đ" if base == "D" else base
    s = base + (SHAPES[shape] if shape else "") + (TONES[tone] if tone else "")  # mũ/móc phải đứng trước thanh
    return unicodedata.normalize("NFC", s)


def allowed_labels(base: str) -> list[int]:
    shapes = _ALLOWED_SHAPES.get(base.lower())
    if shapes is None:
        return [0]
    if base.lower() == "d":
        return [label_id(0, 0), label_id(4, 0)]
    return [label_id(s, t) for s in shapes for t in range(len(TONES))]


def encode_target(text: str) -> tuple[str, list[int]]:
    """Câu có dấu -> (câu không dấu, nhãn từng ký tự)."""
    text = unicodedata.normalize("NFC", text)
    chars, labels = [], []
    for ch in text:
        b, s, t = decompose(ch)
        chars.append(b)
        labels.append(label_id(s, t))
    return "".join(chars), labels


def strip(text: str) -> str:
    return encode_target(text)[0]


def decode(folded: str, labels: list[int]) -> str:
    return "".join(compose(c, *split_label(lab)) for c, lab in zip(folded, labels, strict=True))


def is_ambiguous(base: str) -> bool:
    """Ký tự có hơn 1 lựa chọn -> chỉ tính accuracy trên những ký tự này mới có ý nghĩa."""
    return len(allowed_labels(base)) > 1
