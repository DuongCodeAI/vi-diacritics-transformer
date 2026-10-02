"""So sánh baseline / tagger fp32 / tagger int8 / seq2seq trên nhiều bộ test.

python -m vi_diacritics.evaluate --export export/ --baseline runs/baseline.json \
    --test wiki=data/splits/test.txt --test nrl=data/nrl_eval.txt --test real=data/real_typing.txt

Mỗi file test: 1 câu có dấu đúng / dòng. Model chỉ thấy bản bỏ dấu.
"""

import argparse
import json
import time
from pathlib import Path

from .labels import strip
from .metrics import confusions, errors_by_syllable, evaluate


def _read(p, limit=None):
    lines = [x.strip() for x in Path(p).read_text("utf-8").splitlines() if x.strip()]
    return lines[:limit] if limit else lines


def _timed(fn, inputs):
    t0 = time.perf_counter()
    preds = [fn(x) for x in inputs]
    ms = 1000 * (time.perf_counter() - t0) / max(len(inputs), 1)
    return preds, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="export", help="thư mục ONNX (tagger.onnx, tagger.int8.onnx, vocab.json)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--s2s-run", default=None, help="thư mục run seq2seq (cần torch)")
    ap.add_argument("--test", action="append", required=True, help="tên=đường_dẫn")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    systems = {}
    if args.baseline:
        from .baseline import NgramBaseline

        systems["bigram baseline"] = NgramBaseline.load(args.baseline).restore
    from .restorer import Restorer

    exp = Path(args.export)
    systems["tagger fp32"] = Restorer.load(exp, int8=False, threads=1).restore
    if (exp / "tagger.int8.onnx").exists():
        systems["tagger int8"] = Restorer.load(exp, int8=True, threads=1).restore
    if args.s2s_run:
        systems["seq2seq"] = _seq2seq_fn(Path(args.s2s_run))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    md = ["| test set | system | char acc | word acc | sent acc | hallucination | ms/câu (1 luồng CPU) |",
          "|---|---|---|---|---|---|---|"]
    for spec in args.test:
        name, path = spec.split("=", 1)
        golds = _read(path, args.limit)
        inputs = [strip(g) for g in golds]
        for sys_name, fn in systems.items():
            preds, ms = _timed(fn, inputs)
            m = evaluate(preds, golds)
            m["ms_per_sent"] = round(ms, 2)
            report[f"{name}/{sys_name}"] = m
            md.append(f"| {name} | {sys_name} | {m['char_acc']:.4f} | {m['word_acc']:.4f} | {m['sent_acc']:.4f} | "
                      f"{m['hallucination_rate']:.2%} | {ms:.1f} |")
            if sys_name.startswith("tagger int8") or sys_name == "seq2seq":
                (out / f"errors_{name}_{sys_name.replace(' ', '_')}.json").write_text(json.dumps({
                    "confusions": confusions(preds, golds),
                    "hard_syllables": errors_by_syllable(preds, golds),
                    "examples": [{"input": i, "pred": p, "gold": g}
                                 for i, p, g in zip(inputs, preds, golds, strict=True) if p != g][:100],
                }, ensure_ascii=False, indent=1), "utf-8")
            print(md[-1])
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    (out / "eval.md").write_text("\n".join(md) + "\n", "utf-8")


def _seq2seq_fn(run: Path):
    import torch

    from .bpe import load as load_bpe
    from .model import Seq2SeqTransformer
    from .train import predict_seq2seq

    cfg = json.loads((run / "config.json").read_text("utf-8"))
    src_tok, tgt_tok = load_bpe(run / "src_bpe.json"), load_bpe(run / "tgt_bpe.json")
    n = cfg["layers"] // 2 or 1
    model = Seq2SeqTransformer(cfg["src_vocab"], cfg["tgt_vocab"], cfg["d_model"], cfg["heads"], n, n, cfg["d_ff"], 0.0)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # predict_seq2seq tự bỏ dấu đầu vào nên truyền thẳng câu
    return lambda x: predict_seq2seq(model, src_tok, tgt_tok, [x], device)[0]


if __name__ == "__main__":
    main()
