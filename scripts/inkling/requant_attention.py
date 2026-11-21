#!/usr/bin/env python3
"""Re-quantize Inkling 2 Mini's attention projections from 8-bit to fewer bits (default 6).

usage: requant_attention.py SOURCE_DIR OUTPUT_DIR [--bits 6]

SOURCE_DIR is mlx-community/Inkling-2-Mini-oQ4e-mtp. Every linear layer under `.attn.` that the
source stores at 8 bits (group size 64, affine) is dequantized and quantized again at --bits with the
same group size. All other tensors are copied unchanged, and the other files (tokenizer, template, the
MTP draft model) are linked from SOURCE_DIR. With 6 bits, decode on an M5 Max went from about 70.5 to
74.5 tok/s with the fused decode kernels, with no measurable change in output quality.
"""
import argparse
import json
import os
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, default=6)
    args = parser.parse_args()
    src, out = args.source.expanduser(), args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((src / "config.json").read_text())
    quant = config["quantization"]
    targets = {name for name, q in quant.items()
               if isinstance(q, dict) and ".attn." in name and q.get("bits") == 8 and q.get("group_size") == 64}
    print(f"{len(targets)} attention layers: 8 -> {args.bits} bits")

    index = json.loads((src / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        tensors = mx.load(str(src / shard))
        changed = 0
        for name in targets:
            if f"{name}.weight" not in tensors:
                continue
            w, s, b = tensors[f"{name}.weight"], tensors[f"{name}.scales"], tensors[f"{name}.biases"]
            full = mx.dequantize(w, s, b, group_size=64, bits=8)
            w2, s2, b2 = mx.quantize(full, group_size=64, bits=args.bits)
            tensors[f"{name}.weight"], tensors[f"{name}.scales"], tensors[f"{name}.biases"] = w2, s2.astype(s.dtype), b2.astype(b.dtype)
            changed += 1
        mx.save_safetensors(str(out / shard), tensors, metadata={"format": "mlx"})
        print(f"{shard}: {changed} layers re-quantized")

    for name in targets:
        quant[name] = {**quant[name], "bits": args.bits}
        if isinstance(config.get("quantization_config"), dict) and name in config["quantization_config"]:
            config["quantization_config"][name] = {**config["quantization_config"][name], "bits": args.bits}
    (out / "config.json").write_text(json.dumps(config, indent=4))

    for item in src.iterdir():
        if item.name in ("config.json", ".cache") or item.name.startswith("model-0"):
            continue
        link = out / item.name
        if not link.exists():
            os.symlink(item.resolve(), link)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
