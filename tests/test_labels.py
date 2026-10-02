import unicodedata

from vi_diacritics.labels import N_LABELS, allowed_labels, decode, decompose, encode_target, is_ambiguous
from vi_diacritics.vocab import CharVocab

SENTS = [
    "Tôi đang đi học ở Hà Nội.",
    "Người Việt Nam yêu hòa bình, khỏe mạnh và thủy chung.",
    "ĐƯỜNG CAO TỐC BẮC – NAM, xe ô tô chạy 120km/h",
    "Ngữ nghĩa, nghiêng ngả, ưu tiên, quyển sách, giặt giũ",
]


def test_roundtrip_all_sentences():
    for s in SENTS:
        folded, labels = encode_target(s)
        assert len(folded) == len(s) == len(labels)
        assert decode(folded, labels) == unicodedata.normalize("NFC", s)


def test_folded_has_no_vietnamese_marks():
    folded, _ = encode_target("Đường phố Sài Gòn")
    assert folded == "Duong pho Sai Gon"


def test_decompose():
    assert decompose("ệ") == ("e", 1, 5)
    assert decompose("Ữ") == ("U", 3, 4)
    assert decompose("đ") == ("d", 4, 0)
    assert decompose("ç") == ("ç", 0, 0)


def test_every_target_label_is_allowed():
    for s in SENTS:
        folded, labels = encode_target(s)
        for c, lab in zip(folded, labels, strict=True):
            assert lab in allowed_labels(c), (c, lab)


def test_label_counts():
    assert N_LABELS == 30
    assert len(allowed_labels("a")) == 18
    assert len(allowed_labels("d")) == 2
    assert allowed_labels("b") == [0]
    assert is_ambiguous("o") and not is_ambiguous("k")


def test_vocab_mask():
    v = CharVocab.build([encode_target(s)[0] for s in SENTS], min_freq=1)
    mask = v.label_mask()
    assert sum(mask[v.stoi["d"]]) == 2
    assert sum(mask[v.stoi["k"]]) == 1
