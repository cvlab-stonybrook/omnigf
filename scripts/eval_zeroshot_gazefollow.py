# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Zero-shot evaluation of base Qwen3-VL on the GazeFollow test set.

This script runs **no fine-tuning, no LoRA, no checkpoint, no gaze
encoder, no heatmap decoder, no inout head**. The base model receives
the same point-mode chat prompt as the fine-tuned pipeline (system
instruction + per-person ``P{pid}: {"bbox_2d": [...]}`` user turn) and
must emit the JSON format described in the system prompt. Predictions
are extracted with ``MultiPersonOutputParser``; missing or malformed
per-person entries leave ``gaze_pts_text`` as NaN, and the
``GFTestDistance`` metric layer substitutes ``(0.5, 0.5)`` for those
NaN entries when computing distance (transparent fallback — the saved
JSON shows ``null`` so the parse-success rate is reportable).

What this script measures (the GazeFollow distances reported in the paper,
on the test split's primary subject):

  * ``gf_avg_dist``     — distance from prediction to the centroid of
                          the multi-annotator GT set ("Avg Dist.").
  * ``gf_min_dist``     — distance from prediction to the closest
                          annotator's GT point ("Min Dist.").

Plus diagnostic counters: ``n_total`` (test samples seen), ``n_valid``
(samples with a parseable prediction), ``parse_success_rate``.

Usage::

    python scripts/eval_zeroshot_gazefollow.py \\
        --config_name vlm_config.yaml \\
        --disk_prefix /path/to/data_root \\
        --batch_size 4 \\
        --num_workers 4 \\
        --max_new_tokens 96 \\
        --output_json /tmp/zeroshot_gazefollow.json

The default config (``omnigf/config/vlm_config.yaml``) already enables
GazeFollow as the only source for train/val/test. We additionally force
``vlm.use_gaze_encoder=False`` (the base model has no <|gaze_pad|>
token in its vocab) and ``vlm.prompt_mode=point``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json as _json
import re as _re

from omnigf.performance.metrics import GFTestDistance  # noqa: E402
from omnigf.vlm.collator import VLMGazeCollator  # noqa: E402
from omnigf.vlm.dataset import VLMGazeDataModule  # noqa: E402
from omnigf.vlm.parser import MultiPersonOutputParser  # noqa: E402

# Fallback regex for bare JSON objects without the ``P{pid}:`` prefix —
# the base Qwen3-VL drops the prefix and just emits the inner object,
# e.g. ``{"status": "inside", "point_2d": [238, 629]}``. We only invoke
# this fallback when the prompt has exactly one annotated person (the
# GazeFollow-test default), so the bare object is unambiguously about
# that person.
_BARE_JSON_RE = _re.compile(r"\{[^{}]*\"status\"[^{}]*\}", flags=_re.DOTALL)


def _fallback_single_person_parse(
    text: str, primary_pid: int, coord_bins: int,
) -> Optional[Dict]:
    """Parse a bare ``{"status": ...}`` JSON object (no ``P{pid}:``
    prefix) and return a per-person dict in the same shape as
    ``MultiPersonOutputParser.parse``. Returns None on failure.
    """
    m = _BARE_JSON_RE.search(text)
    if m is None:
        return None
    try:
        obj = _json.loads(m.group(0))
    except _json.JSONDecodeError:
        return None
    status = str(obj.get("status", "")).lower()
    if status == "outside":
        return {"pid": int(primary_pid), "inside": False}
    if status != "inside":
        return None
    pt = obj.get("point_2d")
    if (
        pt is None
        or not isinstance(pt, (list, tuple))
        or len(pt) < 2
        or not isinstance(pt[0], (int, float))
        or not isinstance(pt[1], (int, float))
    ):
        return None
    x = float(pt[0]) / (coord_bins - 1)
    y = float(pt[1]) / (coord_bins - 1)
    return {"pid": int(primary_pid), "inside": True, "x": x, "y": y}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_zeroshot_gazefollow")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config_name", default="vlm_config.yaml",
                   help="Hydra config under omnigf/config/. The default loads "
                        "the GazeFollow-only config used by main_vlm.py.")
    p.add_argument("--config_overrides", nargs="*", default=[],
                   help='Optional ``key=value`` overrides applied via '
                        'OmegaConf, e.g. ``vlm.coord_bins=1000``.')
    p.add_argument("--disk_prefix", default=None,
                   help="Override the config's top-level ``disk_prefix`` "
                        "(used for OmegaConf interpolation in dataset roots).")
    p.add_argument("--model_name", default=None,
                   help="HuggingFace model id for the base VLM. Defaults to "
                        "``cfg.vlm.model_name`` (Qwen/Qwen3-VL-4B-Instruct).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit to first N test samples (None = full split). "
                        "Useful for smoke runs.")
    p.add_argument("--max_new_tokens", type=int, default=96,
                   help="Cap on autoregressive decoding length. Each "
                        "decoded token is one full forward pass; for "
                        "single-person GF-test prompts (default), 96 is "
                        "comfortably above the ~30-token JSON target.")
    p.add_argument("--output_json", default="./zeroshot_gazefollow.json",
                   help="Path to write the metric report.")
    p.add_argument("--save_predictions", default=None,
                   help="Optional path to also dump per-sample predictions "
                        "(parseable text, raw text, parse status). Useful "
                        "for analysing which samples the model failed to "
                        "format correctly.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42,
                   help="Used only for the optional --num_samples subset "
                        "selection (sequential, no shuffle, so the seed "
                        "doesn't actually affect anything yet — kept for "
                        "future-proofing).")
    p.add_argument("--dtype", default="bf16",
                   choices=("bf16", "fp16", "fp32"),
                   help="VLM weight dtype.")
    return p.parse_args()


def _select_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _force_gazefollow_only(cfg) -> None:
    """Disable every non-GazeFollow source in include / val / test; ensure
    the test split actually uses the GazeFollow source.

    The default config already has gazefollow=True for every split, but
    this is a defensive normalisation in case a user passes a VSGaze-style
    config name by mistake.
    """
    for key in ("include_sources", "val_include_sources", "test_include_sources"):
        if key not in cfg.data:
            continue
        for src in cfg.data[key]:
            cfg.data[key][src] = (src == "gazefollow")


def main() -> None:
    args = parse_args()

    # ---- Load + adapt config ----
    config_path = os.path.join(REPO_ROOT, "omnigf", "config", args.config_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if args.disk_prefix is not None:
        cfg.disk_prefix = args.disk_prefix
    if args.config_overrides:
        cfg.merge_with_dotlist(args.config_overrides)

    # Force the zero-shot operating point. The base Qwen3-VL has no
    # <|gaze_pad|> token in its vocab, no LoRA adapters, and no auxiliary
    # heads; we use the "point" prompt+target schema (the system prompt
    # explicitly tells the model what JSON to emit) and approach=lm so
    # the collator skips the heatmap-decoder plumbing entirely.
    cfg.vlm.use_gaze_encoder = False
    cfg.vlm.prompt_mode = "point"
    cfg.vlm.approach = "lm"
    cfg.vlm.use_semantic_labels = False
    cfg.vlm.use_semantic_head = False
    cfg.vlm.max_new_tokens = int(args.max_new_tokens)
    if args.model_name is not None:
        cfg.vlm.model_name = args.model_name
    cfg.data.num_workers = int(args.num_workers)
    _force_gazefollow_only(cfg)

    # ---- Build data module + test loader ----
    logger.info("Building VLMGazeDataModule (gazefollow-only, point mode, "
                "no gaze encoder).")
    dm = VLMGazeDataModule(cfg)
    dm.setup("test")
    test_dataset = dm.test_dataset
    if args.num_samples is not None and args.num_samples < len(test_dataset):
        logger.info("Limiting to first %d / %d test samples.",
                    args.num_samples, len(test_dataset))
        test_dataset = Subset(test_dataset, range(int(args.num_samples)))

    # Per-sample collator: max_people=None lets the collator pad to per-batch
    # max instead of clipping to ``vlm.max_people``. Mirrors the inference
    # path in save_vlm_predictions.py:188-192.
    collator = VLMGazeCollator(
        processor=dm.processor,
        approach=dm.approach,
        max_people=None,
        use_gaze_encoder=False,
        mask_point_block_loss=False,
        prompt_mode="point",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collator,
        pin_memory=False,
        drop_last=False,
    )
    logger.info("Test split size: %d (DataLoader batches: %d, batch_size=%d)",
                len(test_dataset), len(test_loader), args.batch_size)

    # ---- Load base VLM (no LoRA, no checkpoint) ----
    from transformers import AutoModelForImageTextToText
    model_name = cfg.vlm.model_name
    dtype = _select_dtype(args.dtype)
    logger.info("Loading base VLM %r in dtype=%s (no LoRA, no checkpoint).",
                model_name, dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    tokenizer = dm.tokenizer

    # ---- Inference loop ----
    parser = MultiPersonOutputParser(coord_bins=int(getattr(cfg.vlm, "coord_bins", 1000)))
    gf_dist = GFTestDistance().to(device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    n_total = 0
    n_parsed = 0   # samples where the primary subject got a parseable prediction
    n_inside_pred = 0  # samples where the model said "inside" for the primary
    per_sample_records: List[Dict] = []

    logger.info("Running zero-shot inference on %d batches...", len(test_loader))
    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16))
        if device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="zero-shot")):
            if batch is None:
                continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            prompt_ids = batch["prompt_input_ids"]
            prompt_mask = batch["prompt_attention_mask"]
            gen_kwargs = {
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask,
                "pixel_values": batch["pixel_values"],
                "max_new_tokens": int(args.max_new_tokens),
                "pad_token_id": pad_token_id,
                "do_sample": False,
            }
            if "image_grid_thw" in batch:
                gen_kwargs["image_grid_thw"] = batch["image_grid_thw"]

            gen_ids = model.generate(**gen_kwargs)
            prompt_len = prompt_ids.shape[1]
            response_ids = gen_ids[:, prompt_len:]
            generated_texts = tokenizer.batch_decode(
                response_ids, skip_special_tokens=True,
            )

            B = prompt_ids.size(0)
            N = batch["inout"].size(1)
            # GazeFollow test: the primary (annotated) subject is the LAST
            # real pid in the prompt, per ``omnigf/vlm/dataset.py:_build_gf_test_sample``.
            # Pull its index per sample.
            num_annotated = batch["num_annotated"]
            last_idx = (num_annotated - 1).to(device)

            # Parse generated text per sample, fill gaze_pts_text [B, N, 2].
            # Missing / malformed per-pid entries stay NaN; the GFTestDistance
            # metric substitutes (0.5, 0.5) at update time.
            gaze_pts_text = torch.full(
                (B, N, 2), float("nan"), device=device, dtype=torch.float32,
            )
            inout_text = torch.zeros(B, N, device=device, dtype=torch.float32)
            sample_records = [None] * B
            for bi, text in enumerate(generated_texts):
                ctx = batch.get("image_path", [None] * B)
                ctx_str = ctx[bi] if ctx and bi < len(ctx) else f"batch{batch_idx}_idx{bi}"
                preds = parser.parse(text, context=ctx_str)
                gt_pids = batch["person_ids"][bi]
                # Fallback: if the prefix-based parser found nothing AND
                # this sample has exactly one annotated person (the
                # GazeFollow-test default with gf_test_all_people=False),
                # accept a bare ``{"status": ...}`` JSON object — the
                # base Qwen3-VL routinely drops the ``P0:`` prefix.
                if not preds and int(num_annotated[bi].item()) == 1:
                    primary_pid_int = int(gt_pids[last_idx[bi].item()].item())
                    fb = _fallback_single_person_parse(
                        text, primary_pid_int,
                        coord_bins=int(getattr(cfg.vlm, "coord_bins", 1000)),
                    )
                    if fb is not None:
                        preds = [fb]
                primary_pred = None
                for pred in preds:
                    pid = pred["pid"]
                    pid_mask = gt_pids == pid
                    if pid_mask.sum() == 0:
                        continue
                    idx = pid_mask.nonzero(as_tuple=True)[0][0].item()
                    if pred.get("inside"):
                        gaze_pts_text[bi, idx, 0] = pred["x"]
                        gaze_pts_text[bi, idx, 1] = pred["y"]
                        inout_text[bi, idx] = 1.0
                    else:
                        # status="outside": leave NaN (Distance metric
                        # treats NaN as center-fallback, but the sample
                        # is unlikely to land in any bbox either way).
                        # Mark inout=0 so the diagnostic count is right.
                        inout_text[bi, idx] = 0.0
                    if int(idx) == int(last_idx[bi].item()):
                        primary_pred = pred
                # Per-sample record for optional --save_predictions dump.
                if args.save_predictions:
                    sample_records[bi] = {
                        "image_path": ctx_str,
                        "raw_text": text,
                        "parsed_count": len(preds),
                        "primary_pid": int(gt_pids[last_idx[bi].item()].item()),
                        "primary_inside": (
                            bool(primary_pred.get("inside", False))
                            if primary_pred is not None else None
                        ),
                        "primary_xy": (
                            [primary_pred.get("x"), primary_pred.get("y")]
                            if primary_pred is not None and primary_pred.get("inside")
                            else None
                        ),
                    }

            # Update GFTestDistance: pick the primary subject's prediction per sample.
            bi_range = torch.arange(B, device=device)
            primary_pred_xy = gaze_pts_text[bi_range, last_idx]   # [B, 2]
            gaze_gt_all = batch["gaze_pts_all"]                   # [B, A, 2]
            gf_dist.update(primary_pred_xy, gaze_gt_all)
            # Diagnostic counts
            for bi in range(B):
                n_total += 1
                if torch.isfinite(primary_pred_xy[bi, 0]):
                    n_parsed += 1
                if inout_text[bi, last_idx[bi]].item() > 0.5:
                    n_inside_pred += 1
            if args.save_predictions:
                per_sample_records.extend([r for r in sample_records if r is not None])

    # ---- Aggregate metrics ----
    if gf_dist.n_observations > 0:
        avg_dist, min_dist = gf_dist.compute()
        n_valid = float(gf_dist.n_valid.item())
    else:
        avg_dist = min_dist = float("nan")
        n_valid = 0.0

    metrics = {
        "n_total": n_total,
        "n_parsed_primary": n_parsed,
        "n_valid_metric": int(n_valid),
        "parse_success_rate": (n_parsed / n_total) if n_total > 0 else 0.0,
        "n_pred_primary_inside": n_inside_pred,
        "gf_avg_dist": float(avg_dist),
        "gf_min_dist": float(min_dist),
        "model_name": str(cfg.vlm.model_name),
        "dtype": args.dtype,
        "max_new_tokens": int(args.max_new_tokens),
        "config_name": args.config_name,
    }

    logger.info("=" * 60)
    logger.info("Zero-shot GazeFollow test results")
    for k in ("n_total", "n_parsed_primary", "n_valid_metric",
              "parse_success_rate", "n_pred_primary_inside",
              "gf_avg_dist", "gf_min_dist"):
        v = metrics[k]
        if isinstance(v, float):
            logger.info("  %-22s %.4f", k, v)
        else:
            logger.info("  %-22s %s", k, v)
    logger.info("=" * 60)

    # ---- Write report ----
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metric report to %s", args.output_json)

    if args.save_predictions and per_sample_records:
        pred_dir = os.path.dirname(os.path.abspath(args.save_predictions))
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        with open(args.save_predictions, "w") as f:
            json.dump(per_sample_records, f, indent=2)
        logger.info("Wrote %d per-sample prediction records to %s",
                    len(per_sample_records), args.save_predictions)


if __name__ == "__main__":
    main()
