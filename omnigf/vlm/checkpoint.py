# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

"""Load OmniGF weights from a Lightning checkpoint or a released ``.safetensors`` file.

A Lightning ``.ckpt`` stores the whole model, including the frozen Qwen-VL backbone
(about 8.6 GB). The released ``.safetensors`` files store only what training changed:
the LoRA adapters, the head encoder, the dense heads and ``fallback_person_embed``,
plus the embedding row that ``use_gaze_encoder`` appends for ``<|gaze_pad|>``
(``GAZE_PAD_KEY``). The backbone comes from the Hugging Face weights that
``VLMGazeModelGE.__init__`` already loads, so a released file is about 115 MB.
"""

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Embedding rows past the base vocabulary (the <|gaze_pad|> row), shape [n_new, hidden].
GAZE_PAD_KEY = "omnigf.added_embedding_rows"
FORMAT = "omnigf-release"


def is_backbone_key(key: str) -> bool:
    """True for frozen Qwen-VL weights: under ``model.`` and not a LoRA adapter."""
    return key.startswith("model.") and "lora_" not in key


def read_state_dict(path: str) -> Tuple[dict, bool]:
    """Return ``(state_dict, is_release_file)`` for a ``.ckpt`` or ``.safetensors`` path."""
    if str(path).endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not an OmniGF release file (metadata {meta})")
        return load_file(path), True
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"], False


def load_checkpoint(
    model: torch.nn.Module, path: str, drop_mismatched: bool = False
) -> Tuple[List[str], List[str]]:
    """Load ``path`` into ``model`` (non-strict) and return ``(missing, unexpected)``.

    For a release file the backbone keys are expected to be missing, so they are left
    out of ``missing``. ``drop_mismatched`` skips keys whose shape differs from the
    model's (used when evaluating on a different label vocabulary).
    """
    state, is_release = read_state_dict(path)

    added_rows = state.pop(GAZE_PAD_KEY, None)
    if added_rows is not None:
        emb = model.model.get_input_embeddings().weight
        n_new = added_rows.shape[0]
        if emb.shape[1] != added_rows.shape[1] or emb.shape[0] <= n_new:
            raise ValueError(
                f"{path}: added embedding rows {tuple(added_rows.shape)} do not fit the "
                f"model's embedding table {tuple(emb.shape)}; check vlm.use_gaze_encoder"
            )
        with torch.no_grad():
            emb[-n_new:] = added_rows.to(emb.dtype)

    if drop_mismatched:
        model_state = model.state_dict()
        for k in [k for k, v in state.items() if k in model_state]:
            if tuple(state[k].shape) != tuple(model_state[k].shape):
                logger.warning(
                    "Dropping shape-mismatched key '%s' (ckpt %s vs model %s)",
                    k, tuple(state[k].shape), tuple(model_state[k].shape),
                )
                del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_release:
        missing = [k for k in missing if not is_backbone_key(k)]
    return missing, unexpected
