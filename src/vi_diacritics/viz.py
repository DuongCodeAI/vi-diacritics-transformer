"""Vẽ attention map của tagger cho một câu.

Muốn xem: khi quyết định dấu cho chữ "ma" trong "con ma" vs "mà thôi",
model nhìn vào ký tự nào. Kỳ vọng: layer đầu nhìn quanh trong cùng âm tiết,
layer sau nhìn sang âm tiết bên cạnh (ngữ cảnh).
"""

import matplotlib.pyplot as plt
import torch

from .labels import strip


@torch.no_grad()
def attention_maps(model, vocab, text: str):
    folded = strip(text)
    ids = torch.tensor([vocab.encode(folded)])
    model.eval()
    model.set_keep_attn(True)
    model(ids)
    maps = [layer.attn.last_attn[0].cpu() for layer in model.encoder.layers]  # mỗi cái (H, T, T)
    model.set_keep_attn(False)
    return folded, maps


def plot_attention(model, vocab, text: str, layers=(0, -1), query_char: int | None = None, path=None):
    folded, maps = attention_maps(model, vocab, text)
    chars = [c if c != " " else "␣" for c in folded]
    fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 6))
    axes = axes if len(layers) > 1 else [axes]
    for ax, li in zip(axes, layers, strict=True):
        att = maps[li].mean(0)  # trung bình các head
        if query_char is not None:
            ax.bar(range(len(chars)), att[query_char].numpy())
            ax.set_xticks(range(len(chars)), chars)
            ax.set_title(f"layer {li}: ký tự '{chars[query_char]}' nhìn vào đâu")
        else:
            ax.imshow(att.numpy(), cmap="viridis")
            ax.set_xticks(range(len(chars)), chars)
            ax.set_yticks(range(len(chars)), chars)
            ax.set_title(f"layer {li} (trung bình {maps[li].shape[0]} head)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
