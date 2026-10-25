# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

"""Convert a Lightning checkpoint into a release ``.safetensors`` file.

Keeps only the weights training changed (LoRA, head encoder, dense heads,
``fallback_person_embed``) and the embedding rows added for ``<|gaze_pad|>``. Every
backbone tensor it drops is first checked to be bit-identical to the Hugging Face
base model, so loading the release file with ``omnigf.vlm.checkpoint.load_checkpoint``
rebuilds exactly the same weights as the Lightning checkpoint.

    python scripts/export_release_weights.py in.ckpt out.safetensors \
        [--base Qwen/Qwen3-VL-4B-Instruct]
"""

import argparse
import glob
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omnigf.vlm.checkpoint import FORMAT, GAZE_PAD_KEY, is_backbone_key  # noqa: E402

PEFT_PREFIX = "model.base_model.model."
EMBED_KEY = PEFT_PREFIX + "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = PEFT_PREFIX + "lm_head.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    base_dir = snapshot_download(args.base, allow_patterns=["*.safetensors", "*.json"])
    base_files = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            base_files.update({k: f for k in s.keys()})

    def base_tensor(key):
        with safe_open(base_files[key], "pt") as s:
            return s.get_tensor(key)

    keep, n_dropped = {}, 0
    for k, v in state.items():
        if not is_backbone_key(k):
            keep[k] = v.contiguous().clone()
            continue
        if k == LM_HEAD_KEY:
            # Qwen3-VL-4B ties lm_head to embed_tokens; an untied head is not handled.
            if not torch.equal(v, state[EMBED_KEY]):
                sys.exit(f"{k} differs from {EMBED_KEY}; untied lm_head is not supported")
            n_dropped += 1
            continue
        if k == EMBED_KEY:
            base = base_tensor("model.language_model.embed_tokens.weight")
            n = base.shape[0]
            if not torch.equal(v[:n], base):
                sys.exit(f"{k}: rows of the base vocabulary differ from {args.base}")
            if v.shape[0] > n:
                keep[GAZE_PAD_KEY] = v[n:].contiguous().clone()
            n_dropped += 1
            continue
        bk = k[len(PEFT_PREFIX):].replace(".base_layer.", ".")
        if bk not in base_files:
            sys.exit(f"{k} is not in {args.base}; refusing to drop it")
        if not torch.equal(v, base_tensor(bk)):
            sys.exit(f"{k} differs from {args.base}; refusing to drop it")
        n_dropped += 1

    meta = {"format": FORMAT, "base_model": args.base, "source": os.path.basename(args.ckpt)}
    save_file(keep, args.out, metadata=meta)
    size = sum(t.numel() * t.element_size() for t in keep.values()) / 2**20
    print(f"{args.out}: kept {len(keep)} tensors ({size:.1f} MiB), "
          f"dropped {n_dropped} backbone tensors identical to {args.base}")


if __name__ == "__main__":
    main()
