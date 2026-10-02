"""Train tagger hoặc seq2seq. Chạy trên Kaggle T4 (fp16).

python -m vi_diacritics.train --task tagger --data data/splits --out runs/tagger --max-minutes 300
python -m vi_diacritics.train --task seq2seq --data data/splits --out runs/s2s --max-minutes 300

Có --max-minutes vì Kaggle cắt session sau 12h; hết giờ thì lưu checkpoint,
lần sau chạy lại với --resume.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .bpe import BOS_ID, EOS_ID, train_bpe
from .bpe import load as load_bpe
from .dataset import LengthGroupedSampler, Seq2SeqDataset, TaggerDataset, collate_seq2seq, collate_tagger, read_lines
from .labels import N_LABELS, decode, strip
from .metrics import evaluate
from .model import Seq2SeqTransformer, TaggerTransformer, count_params
from .vocab import CharVocab


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def predict_tagger(model, vocab, texts, device, batch_size=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        folded = [strip(t) for t in texts[i : i + batch_size]]
        ids, _ = collate_tagger([(vocab.encode(f), [0] * len(f)) for f in folded])
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(ids.to(device)).argmax(-1).cpu().tolist()
        out += [decode(f, p[: len(f)]) for f, p in zip(folded, pred, strict=True)]
    model.train()
    return out


@torch.no_grad()
def predict_seq2seq(model, src_tok, tgt_tok, texts, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        srcs = [src_tok.encode(strip(t)).ids[:128] for t in texts[i : i + batch_size]]
        src, _, _ = collate_seq2seq([(s, [BOS_ID, EOS_ID]) for s in srcs])
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            ys = model.greedy(src.to(device), BOS_ID, EOS_ID, max_len=160).cpu().tolist()
        for y in ys:
            y = y[: y.index(EOS_ID)] if EOS_ID in y else y
            out.append(tgt_tok.decode(y))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["tagger", "seq2seq"], default="tagger")
    ap.add_argument("--data", default="data/splits")
    ap.add_argument("--out", default="runs/tagger")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--bpe-vocab", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train_lines = read_lines(Path(args.data) / "train.txt", args.train_limit)
    val_lines = read_lines(Path(args.data) / "val.txt", args.eval_n)
    print(f"train={len(train_lines):,} val={len(val_lines):,} device={device}")

    if args.task == "tagger":
        vpath = out / "vocab.json"
        vocab = CharVocab.load(vpath) if vpath.exists() else CharVocab.build(strip(t) for t in train_lines[:500_000])
        vocab.save(vpath)
        mask = torch.tensor(vocab.label_mask())
        model = TaggerTransformer(len(vocab), N_LABELS, mask, args.d_model, args.heads, args.layers, args.d_ff,
                                  args.dropout)
        ds = TaggerDataset(train_lines, vocab)
        collate = collate_tagger
        config = {"task": "tagger", "vocab_size": len(vocab), "n_labels": N_LABELS}
    else:
        sp, tp = out / "src_bpe.json", out / "tgt_bpe.json"
        if not sp.exists():
            sample = train_lines[:500_000]
            train_bpe((strip(t) for t in sample), args.bpe_vocab, sp)
            train_bpe(iter(sample), args.bpe_vocab, tp)
        src_tok, tgt_tok = load_bpe(sp), load_bpe(tp)
        model = Seq2SeqTransformer(src_tok.get_vocab_size(), tgt_tok.get_vocab_size(), args.d_model, args.heads,
                                   args.layers // 2 or 1, args.layers // 2 or 1, args.d_ff, args.dropout)
        ds = Seq2SeqDataset(train_lines, src_tok, tgt_tok)
        collate = collate_seq2seq
        config = {"task": "seq2seq", "src_vocab": src_tok.get_vocab_size(), "tgt_vocab": tgt_tok.get_vocab_size()}

    config.update({k: getattr(args, k) for k in ("d_model", "heads", "layers", "d_ff", "dropout")})
    (out / "config.json").write_text(json.dumps(config, indent=2), "utf-8")
    model.to(device)
    print(f"params: {count_params(model) / 1e6:.2f}M")

    sampler = LengthGroupedSampler(ds, args.batch_size)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    total = args.epochs * len(dl)
    step, best, start_epoch = 0, -1.0, 0

    ckpt = out / "last.pt"
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best, start_epoch = st["step"], st["best"], st["epoch"]
        sampler.epoch = start_epoch
        print(f"resume từ step {step}")

    log = (out / "log.jsonl").open("a", encoding="utf-8")
    t_start = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, total)
            batch = [b.to(device, non_blocking=True) for b in batch]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model.loss(*batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % 100 == 0:
                rec = {"step": step, "epoch": epoch, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"],
                       "min": round((time.time() - t_start) / 60, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 1000 == 0:
                    print(rec)

            if step % args.eval_every == 0:
                if args.task == "tagger":
                    preds = predict_tagger(model, vocab, val_lines, device)
                else:
                    preds = predict_seq2seq(model, src_tok, tgt_tok, val_lines[:1000], device)
                m = evaluate(preds, val_lines[: len(preds)])
                m["step"] = step
                print("eval", m)
                log.write(json.dumps({"eval": m}) + "\n")
                if m["word_acc"] > best:
                    best = m["word_acc"]
                    torch.save({"model": model.state_dict(), "config": config, "step": step, "val": m}, out / "best.pt")

            if (time.time() - t_start) / 60 > args.max_minutes:
                stop = True
                break
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "epoch": epoch + (0 if stop else 1)}, ckpt)
        if stop:
            print("hết thời gian, đã lưu last.pt - chạy lại với --resume")
            break
    log.close()
    print("best word_acc", best)


if __name__ == "__main__":
    main()
