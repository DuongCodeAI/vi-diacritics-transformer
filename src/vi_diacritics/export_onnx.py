"""Xuất tagger sang ONNX (fp32 + int8) để chạy trên CPU và trên trình duyệt.

python -m vi_diacritics.export_onnx --run runs/tagger --out export/

Chỉ xuất tagger: seq2seq cần vòng lặp decode + beam ở ngoài, chạy web phức tạp
mà lại chậm và có thể bịa chữ -> không đáng để deploy.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .model import TaggerTransformer
from .vocab import CharVocab


class _Probs(torch.nn.Module):
    """Xuất thẳng xác suất (softmax) để phía JS/ORT khỏi tự tính, và để lấy độ tự tin."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return self.m(ids).softmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/tagger")
    ap.add_argument("--out", default="export")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "config.json").read_text("utf-8"))
    vocab = CharVocab.load(run / "vocab.json")
    model = TaggerTransformer(len(vocab), cfg["n_labels"], torch.tensor(vocab.label_mask()), cfg["d_model"],
                              cfg["heads"], cfg["layers"], cfg["d_ff"], 0.0)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"])
    model.eval()

    dummy = torch.randint(2, len(vocab), (1, 40))
    fp32 = out / "tagger.onnx"
    torch.onnx.export(_Probs(model), (dummy,), str(fp32), input_names=["ids"], output_names=["probs"],
                      dynamic_axes={"ids": {0: "batch", 1: "seq"}, "probs": {0: "batch", 1: "seq"}},
                      opset_version=args.opset)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out / "tagger.int8.onnx"
    # dynamic quant: trọng số Linear -> int8, activation lượng tử hoá lúc chạy. Không cần dữ liệu calibrate.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    shutil.copy(run / "vocab.json", out / "vocab.json")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), "utf-8")

    for p in (fp32, int8):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
