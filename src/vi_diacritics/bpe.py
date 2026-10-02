"""BPE cho hướng seq2seq: train 2 tokenizer riêng (nguồn không dấu, đích có dấu).

Dùng thư viện `tokenizers` để train cho nhanh (Rust), còn thuật toán BPE thì
nhắc lại cho nhớ: bắt đầu từ ký tự, lặp lại việc gộp cặp token kề nhau xuất hiện
nhiều nhất thành token mới cho tới khi đủ vocab_size.
"""

from collections.abc import Iterable
from pathlib import Path

SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_bpe(texts: Iterable[str], vocab_size: int, out_path: str | Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Metaspace giữ thông tin khoảng trắng -> decode lại đúng câu gốc
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=5)
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(out_path))
    return tok


def load(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
