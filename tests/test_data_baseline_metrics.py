from vi_diacritics.baseline import NgramBaseline
from vi_diacritics.data import is_good_sentence, normalize_text, split_of, split_sentences
from vi_diacritics.labels import strip
from vi_diacritics.metrics import confusions, evaluate


def test_normalize_tone_style():
    assert normalize_text("hoà bình, khoẻ, thuỷ, quý") == "hòa bình, khỏe, thủy, quý"


def test_sentence_filter():
    assert is_good_sentence("Hà Nội là thủ đô của nước Cộng hòa xã hội chủ nghĩa Việt Nam.")
    assert not is_good_sentence("The quick brown fox jumps over the lazy dog again.")
    assert not is_good_sentence("Xem https://example.com để biết thêm chi tiết nhé bạn.")


def test_split_sentences():
    assert split_sentences("Tôi đi học. Trời mưa to! Sao vậy?") == ["Tôi đi học.", "Trời mưa to!", "Sao vậy?"]


def test_split_deterministic():
    s = "Câu này luôn rơi vào cùng một tập."
    assert split_of(s) == split_of(s)


def test_baseline_uses_context():
    train = ["con ma đáng sợ", "con ma ám ảnh", "mà tôi không biết", "mà sao vậy", "mà thôi"]
    m = NgramBaseline().fit(train)
    assert m.restore("con ma") == "con ma"
    assert m.restore("Ma toi khong biet") == "Mà tôi không biết"


def test_evaluate_and_confusions():
    golds = ["Tôi đi học", "Mà thôi"]
    preds = ["Tôi đi học", "Má thôi"]
    r = evaluate(preds, golds)
    assert r["sent_acc"] == 0.5
    assert r["hallucination_rate"] == 0.0
    assert 0.7 < r["word_acc"] < 0.9
    assert confusions(preds, golds) == [("mà", "má", 1)]
    # seq2seq bịa thêm chữ
    r2 = evaluate(["Tôi đi học bài"], ["Tôi đi học"])
    assert r2["hallucination_rate"] == 1.0
    assert strip("Tôi") == "Toi"
