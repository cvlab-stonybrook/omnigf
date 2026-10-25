# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Zero-shot evaluation of a GazeFollow-pretrained VLM on the GazeHOI test set.

Loads a checkpoint trained with ``vlm.prompt_mode=point_semantic`` (joint
gaze-target + category prediction in the text block) and runs it over
GazeHOI's ``gazehoi_test.h5`` to report the three metrics from the
SemGaze paper:

  * **GazeAcc** — predicted gaze point falls inside the GT object bbox.
  * **Acc@1**   — top-1 predicted category matches the GT primary
    ``object`` label (single-label, matching SemGaze's GazeHOI eval —
    SemGaze does not report MultiAccuracy on this dataset).
  * **Acc@3**   — the primary GT label is in the top-3 CLIP candidates.

Strategy (decided after inspecting per-image person counts):
  1. The h5 has one row per image with ``head_bboxes`` of shape ``[K, 4]``
     where ``K`` = number of (person, object) pair rows from the
     source CSV. Multi-target persons appear as duplicated head bboxes
     under different ``pair_id`` slots.
  2. We dedupe heads per image (rounded-tuple key) and run the VLM once
     per unique head. Each unique head's ``(point, category)`` prediction
     is then **broadcast to every pair-slot sharing that head**, so no
     evaluation row is dropped (matches the original SemGaze per-row eval).
  3. For images whose unique-head count exceeds ``--chunk_size``, we
     emit multiple chunks (mirroring ``generate_pseudo_gaze_annotations_vlm.py``).
     Each unique head is in exactly one chunk; predictions are stitched
     per-row at the end.

Free-text predicted categories that don't exactly match the GazeHOI
test vocab are routed via :class:`omnigf.vlm.text_to_id_mapper.TextToTestIdMapper`,
which falls back to CLIP cosine + (optionally) a WordNet-hypernym
reroute over CLIP top-k.

Usage (Hydra-style)::

    cd scripts/
    python eval_gazehoi_semantic.py \\
        --config-name vlm_vsgaze_config.yaml \\
        +eval_gazehoi.h5_path=/data/.../GazeHOI/h5_annotations/gazehoi_test.h5 \\
        +eval_gazehoi.image_root=/data/.../GazeHOI \\
        +eval_gazehoi.test_vocab_embeddings=/data/.../gazehoi/clip_vocab_templated/test_vocab_embeddings.pt \\
        +eval_gazehoi.test_vocab_labels=/data/.../gazehoi/clip_vocab_templated/test_vocab_labels.json \\
        +eval_gazehoi.chunk_size=8 \\
        +eval_gazehoi.batch_size=8 \\
        +eval_gazehoi.num_workers=4 \\
        +eval_gazehoi.use_wordnet=true \\
        +eval_gazehoi.output_json=/tmp/gazehoi_eval.json \\
        test.checkpoint=/path/to/vlm_pointsemantic.ckpt
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from omnigf.train.transforms import Compose, Resize, ToTensor  # noqa: E402
from omnigf.utils.utils import square_bbox  # noqa: E402
from omnigf.vlm.collator import VLMGazeCollator  # noqa: E402
from omnigf.vlm.dataset import SYSTEM_PROMPTS, VLMGazeDataModule  # noqa: E402
from omnigf.vlm.checkpoint import load_checkpoint  # noqa: E402
from omnigf.vlm.models_ge import VLMGazeModelGE  # noqa: E402
from omnigf.vlm.text_to_id_mapper import TextToTestIdMapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eval_gazehoi_semantic")


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #


def _bbox_key(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    """Round-then-tuple key for dedupe. 5 decimals is enough since
    head_bboxes in the h5 are stored as float32 image-fraction coords."""
    return tuple(np.round(bbox.astype(np.float64), 5).tolist())


def _build_prompt_for_chunk(
    bboxes_norm: torch.Tensor, coord_bins: int, use_gaze_encoder: bool,
) -> str:
    """Inline VLMGazeDataset._build_prompt for chunked inference. Pids
    are always relabelled 0..K-1 within a chunk."""
    gaze_marker = "<|gaze_pad|> " if use_gaze_encoder else ""
    lines = []
    for k, bbox in enumerate(bboxes_norm):
        b = (bbox.clamp(0, 1) * (coord_bins - 1)).int()
        bbox_json = (
            f'{{"bbox_2d": [{int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])}]}}'
        )
        lines.append(f"P{k}: {gaze_marker}{bbox_json}")
    return "\n".join(lines)


def _is_point_in_box(pt: Tuple[float, float], box: np.ndarray) -> bool:
    """SemGaze GazeAcc test: point (x, y) inside [xmin, ymin, xmax, ymax].
    All inputs in normalised image coords."""
    if pt is None or any(v is None or not np.isfinite(v) for v in pt):
        return False
    x, y = float(pt[0]), float(pt[1])
    xmin, ymin, xmax, ymax = (float(v) for v in box[:4])
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def _is_bad_pair(
    head_bbox: np.ndarray,
    gaze_pt: np.ndarray,
    object_bbox: Optional[np.ndarray],
) -> bool:
    """Filter pairs whose normalised coords fall outside [0, 1] in any
    field. ``build_gazehoi_h5.py`` divides each pixel coordinate by the
    on-disk PIL size; for ~2% of pairs the CSV pixel coords exceed the
    image's PIL dims (the affected images appear to have been
    rotated/resized when packaged while their annotations were left in
    original-image coordinates). Per-row recovery isn't possible without
    the original dims + orientation, so we drop these rows."""
    if (head_bbox > 1.0).any() or (head_bbox < 0.0).any():
        return True
    if (gaze_pt > 1.0).any() or (gaze_pt < 0.0).any():
        return True
    if object_bbox is not None and (
        (object_bbox > 1.0).any() or (object_bbox < 0.0).any()
    ):
        return True
    return False


# ----------------------------------------------------------------------- #
# Dataset: one sample per (row, chunk-of-unique-heads)
# ----------------------------------------------------------------------- #


class GazeHOIChunkedTestDataset(Dataset):
    """Yields one VLM-ready sample per (h5 row, chunk of unique heads).

    Per row in the h5:
      * Read ``head_bboxes`` ``[K, 4]`` (normalised; one entry per pair row).
      * Dedupe by rounded-tuple key into ``U`` unique heads, recording
        ``unique_to_pairs[u] = [pair indices sharing this head]``.
      * Slice the ``U`` unique heads into chunks of ``chunk_size``. Each
        chunk becomes one sample. ``meta`` carries ``row_idx`` plus
        ``unique_indices_in_chunk`` so the runner can stitch predictions
        back into per-pair slots.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        chunk_size: int,
        coord_bins: int,
        prompt_mode: str,
        use_gaze_encoder: bool,
        image_size: Tuple[int, int],
        head_size: Tuple[int, int],
    ):
        self.df = df
        self.image_root = image_root
        self.chunk_size = int(chunk_size)
        self.coord_bins = int(coord_bins)
        self.prompt_mode = prompt_mode
        self.use_gaze_encoder = bool(use_gaze_encoder)
        # Mirror the trainer: Resize the full image to ``image_size`` AND
        # each head crop to ``head_size``; ToTensor stacks heads into
        # [K, 3, head_size, head_size]. The VLM processor does its own
        # normalize, so we don't apply Normalize here.
        self.transform = Compose([
            Resize(img_size=image_size, head_size=head_size),
            ToTensor(),
        ])

        # Per-row dedupe info: unique_bboxes [U, 4], unique_to_pairs (list
        # of lists of pair indices). Cached so __getitem__ stays O(K). Only
        # bbox-validated pairs (see ``_is_bad_pair``) are recorded.
        self._unique_per_row: List[Dict] = []
        self.index_map: List[Tuple[int, List[int]]] = []
        n_pairs_total = 0
        n_pairs_dropped = 0
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            head_bboxes = np.asarray(row["head_bboxes"], dtype=np.float32)
            gaze_points = np.asarray(row["gaze_points"], dtype=np.float32)
            object_bboxes = np.asarray(row["object_bboxes"], dtype=np.float32)
            if head_bboxes.shape[0] == 0:
                self._unique_per_row.append(None)
                continue
            seen: Dict[Tuple, int] = {}
            unique_bboxes: List[np.ndarray] = []
            unique_to_pairs: List[List[int]] = []
            for pair_idx in range(head_bboxes.shape[0]):
                n_pairs_total += 1
                ob = object_bboxes[pair_idx] if pair_idx < object_bboxes.shape[0] else None
                if _is_bad_pair(head_bboxes[pair_idx], gaze_points[pair_idx], ob):
                    n_pairs_dropped += 1
                    continue
                key = _bbox_key(head_bboxes[pair_idx])
                if key in seen:
                    unique_to_pairs[seen[key]].append(pair_idx)
                else:
                    seen[key] = len(unique_bboxes)
                    unique_bboxes.append(head_bboxes[pair_idx])
                    unique_to_pairs.append([pair_idx])
            U = len(unique_bboxes)
            if U == 0:
                self._unique_per_row.append(None)
                continue
            self._unique_per_row.append({
                "unique_bboxes": np.stack(unique_bboxes, axis=0),
                "unique_to_pairs": unique_to_pairs,
            })
            for c in range(0, U, self.chunk_size):
                chunk = list(range(c, min(c + self.chunk_size, U)))
                self.index_map.append((row_idx, chunk))

        n_rows_with_unique = sum(1 for u in self._unique_per_row if u is not None)
        n_unique_total = sum(
            len(u["unique_bboxes"]) for u in self._unique_per_row if u is not None
        )
        if n_pairs_dropped > 0:
            logger.warning(
                "GazeHOIChunkedTestDataset: dropped %d/%d pairs (%.2f%%) "
                "whose normalised coords overflow [0, 1] — see "
                "_is_bad_pair() for details.",
                n_pairs_dropped, n_pairs_total,
                100.0 * n_pairs_dropped / max(n_pairs_total, 1),
            )
        logger.info(
            "GazeHOIChunkedTestDataset: %d rows (%d non-empty), %d pair rows "
            "(%d valid), %d unique heads, %d chunks (chunk_size=%d)",
            len(df), n_rows_with_unique, n_pairs_total,
            n_pairs_total - n_pairs_dropped, n_unique_total,
            len(self.index_map), self.chunk_size,
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        row_idx, unique_indices_in_chunk = self.index_map[idx]
        row = self.df.iloc[row_idx]
        info = self._unique_per_row[row_idx]
        path = row["path"]
        unique_bboxes_np = info["unique_bboxes"][unique_indices_in_chunk]
        bboxes_chunk = torch.from_numpy(unique_bboxes_np).float()  # normalised

        img_path = os.path.join(self.image_root, path)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning("Failed to open image %s: %s", img_path, e)
            return None

        # Mirror the canonical training-data prep
        # (``gazefollow.py:459-477``, ``videoattentiontarget_temporal.py:371-376``,
        # ``childplay_temporal.py:326-331``):
        #   1. Convert normalised bboxes to pixel space.
        #   2. ``square_bbox`` (no expand, no jitter at val/test).
        #   3. Crop heads from the squared pixel bboxes (PIL pads negatives).
        #   4. Re-normalise + clamp to [0, 1] for the VLM prompt.
        # This guarantees the model sees the exact same head-bbox shape and
        # crop framing it was trained with.
        img_w, img_h = image.size
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        bboxes_chunk_px = bboxes_chunk * scale
        bboxes_chunk_px = square_bbox(bboxes_chunk_px, img_w, img_h)

        head_crops_pil: List[Image.Image] = []
        if self.use_gaze_encoder:
            for k in range(bboxes_chunk_px.shape[0]):
                head_crops_pil.append(
                    image.crop(bboxes_chunk_px[k].int().tolist())
                )

        bboxes_chunk_norm = (bboxes_chunk_px / scale).clamp(0.0, 1.0)

        pre = {"image": image, "heads": head_crops_pil}
        pre = self.transform(pre)
        pil_image = to_pil_image(pre["image"].clamp(0, 1))

        K = bboxes_chunk_norm.shape[0]
        sample = {
            "image": pil_image,
            "system_prompt": SYSTEM_PROMPTS[self.prompt_mode],
            "prompt": _build_prompt_for_chunk(
                bboxes_chunk_norm, self.coord_bins, self.use_gaze_encoder,
            ),
            # Dummy target — model.generate is what runs at inference, the
            # collator just needs SOMETHING to tokenise. The label-fill
            # path is never read because we don't compute LM loss here.
            "target": "P0: {\"status\": \"outside\"}",
            "gaze_pts": torch.full((K, 2), -1.0),
            "inout": torch.full((K,), -1.0),
            "person_ids": torch.arange(K),
            "num_annotated": K,
            "head_bboxes": bboxes_chunk_norm,
            "path": [path],
        }
        if self.use_gaze_encoder:
            sample["head_crops"] = pre["heads"]  # [K, 3, hs, hs]
        meta = {
            "row_idx": row_idx,
            "unique_indices_in_chunk": list(unique_indices_in_chunk),
        }
        return sample, meta


def _make_collate_fn(vlm_collator: VLMGazeCollator):
    def collate(items):
        items = [it for it in items if it is not None]
        if not items:
            return None
        samples = [it[0] for it in items]
        metas = [it[1] for it in items]
        batch = vlm_collator(samples)
        if batch is None:
            return None
        batch["_metas"] = metas
        return batch
    return collate


# ----------------------------------------------------------------------- #
# Metric accumulation
# ----------------------------------------------------------------------- #


class GazeHOIMetricAccumulator:
    """Accumulates per-pair predictions and computes GazeAcc / Acc@k.

    The primary gaze localisation signal is the **heatmap argmax**
    (``preds["gaze_pts"]``) plus the **inout-head sigmoid** (``preds["inout_prob"]``),
    matching the convention used by ``scripts/save_vlm_predictions.py``.
    The text-parsed gaze (``preds["gaze_pts_text"]`` + ``preds["inout_prob_text"]``)
    is recorded as a *supplementary* signal so we can report a side-by-side
    GazeAcc_text as a sanity check.

    The semantic category prediction has no head-based alternative — it
    comes only from the text block (``preds["categories_text"]``), so
    Acc@1 / Acc@3 always derive from the parsed text.
    """

    def __init__(
        self,
        mapper: TextToTestIdMapper,
        topk: int = 3,
    ):
        self.mapper = mapper
        self.topk = topk
        self.entries: List[Dict] = []

    def add_pair(
        self,
        gt_obj_bbox: np.ndarray,
        gt_labels: List[str],
        pred_pt_hm: Optional[Tuple[float, float]],
        pred_inout_prob_hm: Optional[float],
        pred_pt_text: Optional[Tuple[float, float]],
        pred_inout_text: Optional[float],
        pred_category: Optional[str],
    ) -> None:
        gt_label_ids = set()
        for lab in gt_labels:
            if lab is None:
                continue
            lab = str(lab).strip()
            if not lab:
                continue
            tid = self.mapper.label_to_id.get(lab.lower())
            if tid is None:
                tid = self.mapper.label_to_id.get(lab)
            if tid is not None and tid >= 0:
                gt_label_ids.add(int(tid))

        # Predicted top-1 ID and similarity vector (cached by mapper).
        pred_top1_id: int = -1
        pred_sims = None
        if pred_category:
            pred_top1_id = int(self.mapper.text_to_id(pred_category))
            pred_sims = self.mapper.text_to_sims(pred_category)
            if pred_sims is not None:
                pred_sims = pred_sims.detach().cpu()

        self.entries.append({
            "gt_obj_bbox": np.asarray(gt_obj_bbox, dtype=np.float32),
            "gt_label_ids": gt_label_ids,
            # Primary (heatmap + inout head).
            "pred_pt_hm": pred_pt_hm,
            "pred_inout_prob_hm": pred_inout_prob_hm,
            # Supplementary (text-parsed).
            "pred_pt_text": pred_pt_text,
            "pred_inout_text": pred_inout_text,
            # Category (text only).
            "pred_category": pred_category,
            "pred_top1_id": pred_top1_id,
            "pred_sims": pred_sims,
        })

    def compute(self) -> Dict:
        """Compute SemGaze-style metrics.

        Headline numbers ("GazeAcc", "Acc@1", "Acc@k") use the primary
        signals — heatmap argmax for the gaze point and CLIP-routed text
        category for class. Supplementary "GazeAcc_text" reuses the same
        per-pair denominators with the text-parsed gaze for diagnosis.

        Denominators mirror :class:`semgaze.metrics.GazeAccuracy` /
        :class:`semgaze.metrics.MultiAccuracy`: GazeAcc divides by total
        pairs (missing/unparseable predictions count as wrong), Acc@k
        divides by pairs with at least one mappable GT vocab id.
        """
        n_total = len(self.entries)
        if n_total == 0:
            return {"n_total": 0}

        n_gaze_correct_hm = 0
        n_gaze_parseable_hm = 0
        n_pred_inside_hm = 0     # diagnostic: heatmap-head said inside (>=0.5)
        n_gaze_correct_text = 0
        n_gaze_parseable_text = 0
        n_pred_inside_text = 0   # diagnostic: text said inside

        n_acc1_correct = 0
        n_acc1_eligible = 0
        n_acck_correct = 0
        n_pred_with_category = 0
        n_pred_unmapped = 0

        for e in self.entries:
            # ---- Primary: heatmap + inout head ----
            pt_hm = e["pred_pt_hm"]
            if pt_hm is not None and all(np.isfinite(v) for v in pt_hm):
                n_gaze_parseable_hm += 1
                if _is_point_in_box(pt_hm, e["gt_obj_bbox"]):
                    n_gaze_correct_hm += 1
            if (
                e["pred_inout_prob_hm"] is not None
                and float(e["pred_inout_prob_hm"]) >= 0.5
            ):
                n_pred_inside_hm += 1

            # ---- Supplementary: text-parsed ----
            pt_text = e["pred_pt_text"]
            if pt_text is not None and all(np.isfinite(v) for v in pt_text):
                n_gaze_parseable_text += 1
                if _is_point_in_box(pt_text, e["gt_obj_bbox"]):
                    n_gaze_correct_text += 1
            if (
                e["pred_inout_text"] is not None
                and float(e["pred_inout_text"]) >= 0.5
            ):
                n_pred_inside_text += 1

            # ---- Category (text only) ----
            gt_ids = e["gt_label_ids"]
            if not gt_ids:
                continue
            n_acc1_eligible += 1

            if e["pred_category"]:
                n_pred_with_category += 1
            top1 = e["pred_top1_id"]
            if e["pred_category"] and top1 < 0:
                n_pred_unmapped += 1

            if top1 in gt_ids:
                n_acc1_correct += 1

            # Acc@k uses RAW CLIP top-k (no reroute), matching the
            # original ``semgaze.metrics.MultiAccuracy(top_k=3)``.
            sims = e["pred_sims"]
            if sims is not None:
                k = min(self.topk, sims.numel())
                topk_idx = sims.topk(k).indices.tolist()
                if any(int(i) in gt_ids for i in topk_idx):
                    n_acck_correct += 1

        out = {
            "n_total": n_total,
            # Primary GazeAcc (heatmap argmax)
            "GazeAcc": n_gaze_correct_hm / max(n_total, 1),
            "n_gaze_correct": n_gaze_correct_hm,
            "n_gaze_parseable": n_gaze_parseable_hm,
            "n_pred_inside": n_pred_inside_hm,
            # Supplementary GazeAcc (text-parsed)
            "GazeAcc_text": n_gaze_correct_text / max(n_total, 1),
            "n_gaze_correct_text": n_gaze_correct_text,
            "n_gaze_parseable_text": n_gaze_parseable_text,
            "n_pred_inside_text": n_pred_inside_text,
            # Category (text only)
            "Acc@1": n_acc1_correct / max(n_acc1_eligible, 1),
            f"Acc@{self.topk}": n_acck_correct / max(n_acc1_eligible, 1),
            "n_acc1_eligible": n_acc1_eligible,
            "n_acc1_correct": n_acc1_correct,
            "n_acck_correct": n_acck_correct,
            "n_pred_with_category": n_pred_with_category,
            "n_pred_unmapped": n_pred_unmapped,
        }
        return out


# ----------------------------------------------------------------------- #
# Hydra entry
# ----------------------------------------------------------------------- #


@hydra.main(
    config_path="./../omnigf/config/",
    config_name="vlm_vsgaze_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    eg = cfg.get("eval_gazehoi", None)
    if eg is None:
        raise ValueError(
            "Required Hydra group ``eval_gazehoi`` not found. Pass settings "
            "via Hydra ``+eval_gazehoi.*`` overrides."
        )
    h5_path = eg.h5_path
    image_root = eg.image_root
    test_vocab_emb_path = eg.test_vocab_embeddings
    test_vocab_lab_path = eg.test_vocab_labels
    chunk_size = int(eg.get("chunk_size", 8))
    batch_size = int(eg.get("batch_size", 8))
    num_workers = int(eg.get("num_workers", 4))
    use_wordnet = bool(eg.get("use_wordnet", False))
    wordnet_top_k = int(eg.get("wordnet_top_k", 10))
    output_json = eg.get("output_json", None)
    limit = eg.get("limit", None)
    if limit is not None:
        limit = int(limit)
    clip_model = str(eg.get("clip_model", "openai/clip-vit-base-patch16"))

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"GazeHOI h5 not found: {h5_path}")
    logger.info("Loading GazeHOI test annotations from %s", h5_path)
    df = pd.read_hdf(h5_path, "data")
    logger.info("Loaded %d image rows", len(df))
    if limit is not None:
        df = df.iloc[:limit].copy()
        logger.info("Limiting to first %d rows for smoke test", len(df))

    # ---- Build VLM data module (for processor + tokenizer; NOT for data) ----
    logger.info("Building VLMGazeDataModule for processor/tokenizer")
    data_module = VLMGazeDataModule(cfg)
    data_module.setup("test")
    coord_bins = int(getattr(cfg.vlm, "coord_bins", 1000))
    prompt_mode = str(getattr(cfg.vlm, "prompt_mode", "point"))
    if prompt_mode != "point_semantic":
        logger.warning(
            "vlm.prompt_mode=%r — this script expects 'point_semantic' for "
            "joint gaze + category prediction. Continuing anyway.",
            prompt_mode,
        )
    use_gaze_encoder = bool(getattr(cfg.vlm, "use_gaze_encoder", False))
    image_size = cfg.data.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    head_size = getattr(cfg.data, "head_size", (224, 224))
    if isinstance(head_size, int):
        head_size = (head_size, head_size)

    # ---- Load VLM checkpoint ----
    logger.info("Instantiating VLMGazeModelGE")
    model = VLMGazeModelGE(cfg)
    test_ckpt = getattr(cfg.test, "checkpoint", None)
    if not test_ckpt:
        raise ValueError(
            "test.checkpoint is required — point at the GazeFollow-pretrained "
            "VLM checkpoint (point_semantic mode)."
        )
    logger.info("Loading checkpoint: %s", test_ckpt)
    # Cross-vocab safety: the checkpoint was trained against a smaller test
    # vocab (e.g. 346 GF labels), but our eval-time model is built with the
    # GazeHOI vocab (463 labels) loaded from disk. The classification table
    # ``test_vocab_emb`` and any torchmetric internal state shaped by
    # ``num_classes`` therefore won't match. Strip every checkpoint key whose
    # tensor shape disagrees with the freshly-instantiated model — these are
    # all just non-learned lookup tables / metric counters that the eval
    # script reinitialises anyway.
    missing, unexpected = load_checkpoint(model, test_ckpt, drop_mismatched=True)
    if missing:
        logger.warning("Missing keys: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])

    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()
    tokenizer = data_module.tokenizer

    # ---- Build the chunked dataset + DataLoader ----
    eval_dataset = GazeHOIChunkedTestDataset(
        df=df,
        image_root=image_root,
        chunk_size=chunk_size,
        coord_bins=coord_bins,
        prompt_mode=prompt_mode,
        use_gaze_encoder=use_gaze_encoder,
        image_size=image_size,
        head_size=head_size,
    )
    inference_collator = VLMGazeCollator(
        processor=data_module.processor,
        approach=data_module.approach,
        max_people=None,
        use_gaze_encoder=use_gaze_encoder,
        mask_point_block_loss=getattr(data_module, "mask_point_block_loss", False),
        prompt_mode=prompt_mode,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_make_collate_fn(inference_collator),
        pin_memory=False,
        drop_last=False,
    )

    # Per-(row_idx, unique_idx) prediction store. We allocate by row.
    n_rows = len(df)
    unique_count_per_row = []
    for row_idx in range(n_rows):
        info = eval_dataset._unique_per_row[row_idx]
        unique_count_per_row.append(len(info["unique_bboxes"]) if info is not None else 0)

    # Primary signals (heatmap argmax + inout-head sigmoid).
    pred_pt_hm_per_unique: List[List[Optional[Tuple[float, float]]]] = [
        [None] * U for U in unique_count_per_row
    ]
    pred_inout_prob_hm_per_unique: List[List[Optional[float]]] = [
        [None] * U for U in unique_count_per_row
    ]
    # Supplementary signals (text-parsed gaze + inout flag).
    pred_pt_text_per_unique: List[List[Optional[Tuple[float, float]]]] = [
        [None] * U for U in unique_count_per_row
    ]
    pred_inout_text_per_unique: List[List[Optional[float]]] = [
        [None] * U for U in unique_count_per_row
    ]
    # Category (text only).
    pred_category_per_unique: List[List[Optional[str]]] = [
        [None] * U for U in unique_count_per_row
    ]
    has_pred_per_unique: List[List[bool]] = [
        [False] * U for U in unique_count_per_row
    ]

    n_chunks_run = 0
    n_skipped_batches = 0

    precision_str = str(getattr(cfg.train, "precision", "32"))
    use_bf16_autocast = "bf16" in precision_str
    logger.info(
        "Running batched chunked inference: batch_size=%d num_workers=%d "
        "(autocast bfloat16=%s)",
        batch_size, num_workers, use_bf16_autocast,
    )
    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=use_bf16_autocast)
        if device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(loader, total=len(loader), desc="gazehoi/test"):
            if batch is None:
                n_skipped_batches += 1
                continue
            metas = batch.pop("_metas")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            preds = model.forward(batch, tokenizer=tokenizer)
            # Primary heads-based signals (mirrors save_vlm_predictions.py
            # which reads ``preds["gaze_pts"]`` and ``preds["inout_prob"]``):
            #   - ``gaze_pts``    [B, K, 2] heatmap argmax in [0, 1]
            #   - ``inout_prob``  [B, K]    sigmoid of the inout head
            # Supplementary text-parsed signals (Approach-1 style):
            #   - ``gaze_pts_text``    [B, K, 2] (NaN if unparsed)
            #   - ``inout_prob_text``  [B, K]    (0.0 / 1.0 hard from text)
            #   - ``categories_text``  list[B][K] (None if unparsed)
            gaze_pts_hm = preds.get("gaze_pts")
            inout_prob_hm = preds.get("inout_prob")
            gaze_pts_text = preds.get("gaze_pts_text")
            inout_prob_text = preds.get("inout_prob_text")
            categories_text = preds.get("categories_text", None)

            for bi, meta in enumerate(metas):
                row_idx = meta["row_idx"]
                u_in_chunk = meta["unique_indices_in_chunk"]
                for k_local, u_idx in enumerate(u_in_chunk):
                    # Heatmap argmax + inout head (primary).
                    if gaze_pts_hm is not None:
                        x, y = (
                            float(gaze_pts_hm[bi, k_local, 0].item()),
                            float(gaze_pts_hm[bi, k_local, 1].item()),
                        )
                        pt_hm = (x, y) if (np.isfinite(x) and np.isfinite(y)) else None
                    else:
                        pt_hm = None
                    inout_hm = (
                        float(inout_prob_hm[bi, k_local].item())
                        if inout_prob_hm is not None else None
                    )
                    # Text-parsed (supplementary).
                    if gaze_pts_text is not None:
                        xt, yt = (
                            float(gaze_pts_text[bi, k_local, 0].item()),
                            float(gaze_pts_text[bi, k_local, 1].item()),
                        )
                        pt_text = (xt, yt) if (np.isfinite(xt) and np.isfinite(yt)) else None
                    else:
                        pt_text = None
                    inout_text = (
                        float(inout_prob_text[bi, k_local].item())
                        if inout_prob_text is not None else None
                    )
                    # Category (text only).
                    cat = None
                    if categories_text is not None:
                        if (
                            bi < len(categories_text)
                            and k_local < len(categories_text[bi])
                        ):
                            cat = categories_text[bi][k_local]

                    pred_pt_hm_per_unique[row_idx][u_idx] = pt_hm
                    pred_inout_prob_hm_per_unique[row_idx][u_idx] = inout_hm
                    pred_pt_text_per_unique[row_idx][u_idx] = pt_text
                    pred_inout_text_per_unique[row_idx][u_idx] = inout_text
                    pred_category_per_unique[row_idx][u_idx] = cat
                    has_pred_per_unique[row_idx][u_idx] = True
                n_chunks_run += 1

    if n_skipped_batches:
        logger.warning("Skipped %d batches (all samples were unreadable images).", n_skipped_batches)
    logger.info("VLM forward passes complete: %d chunks total.", n_chunks_run)

    # ---- Build the TextToTestIdMapper and accumulate per-pair metrics ----
    logger.info(
        "Loading TextToTestIdMapper (use_wordnet=%s, top_k=%d)",
        use_wordnet, wordnet_top_k,
    )
    mapper = TextToTestIdMapper(
        test_vocab_embeddings_path=test_vocab_emb_path,
        test_vocab_labels_path=test_vocab_lab_path,
        device=device,
        clip_name=clip_model,
        use_wordnet=use_wordnet,
        wordnet_top_k=wordnet_top_k,
    )

    accumulator = GazeHOIMetricAccumulator(mapper=mapper, topk=3)

    n_rows_skipped_no_unique = 0
    n_pairs_with_pred = 0
    n_pairs_total = 0
    for row_idx in range(n_rows):
        info = eval_dataset._unique_per_row[row_idx]
        if info is None:
            n_rows_skipped_no_unique += 1
            continue
        row = df.iloc[row_idx]
        object_bboxes = np.asarray(row["object_bboxes"], dtype=np.float32)
        sem_labels = list(row["semgaze_head_labels"])
        # NOTE: SemGaze on GazeHOI evaluates against the **single** primary
        # ``object`` label per pair (their MultiAccuracy is unused on this
        # dataset). We deliberately ignore ``semgaze_head_aux_labels`` here
        # so this script's headline numbers reproduce the SemGaze paper
        # convention and match the in-training Lightning test report.

        for u_idx, pair_indices in enumerate(info["unique_to_pairs"]):
            pred_pt_hm = pred_pt_hm_per_unique[row_idx][u_idx]
            pred_inout_prob_hm = pred_inout_prob_hm_per_unique[row_idx][u_idx]
            pred_pt_text = pred_pt_text_per_unique[row_idx][u_idx]
            pred_inout_text = pred_inout_text_per_unique[row_idx][u_idx]
            pred_cat = pred_category_per_unique[row_idx][u_idx]
            has_pred = has_pred_per_unique[row_idx][u_idx]
            for pair_idx in pair_indices:
                n_pairs_total += 1
                if has_pred:
                    n_pairs_with_pred += 1
                gt_obj = object_bboxes[pair_idx]
                gt_main = sem_labels[pair_idx] if pair_idx < len(sem_labels) else ""
                accumulator.add_pair(
                    gt_obj_bbox=gt_obj,
                    gt_labels=[gt_main],
                    pred_pt_hm=pred_pt_hm,
                    pred_inout_prob_hm=pred_inout_prob_hm,
                    pred_pt_text=pred_pt_text,
                    pred_inout_text=pred_inout_text,
                    pred_category=pred_cat,
                )

    metrics = accumulator.compute()
    metrics.update({
        "n_pairs_total": n_pairs_total,
        "n_pairs_with_pred": n_pairs_with_pred,
        "parse_success_rate": (
            n_pairs_with_pred / n_pairs_total if n_pairs_total > 0 else 0.0
        ),
        "n_rows_skipped_no_unique": n_rows_skipped_no_unique,
        "chunks_run": n_chunks_run,
        "use_wordnet": use_wordnet,
        "wordnet_top_k": wordnet_top_k,
        "chunk_size": chunk_size,
        "checkpoint": str(test_ckpt),
        "h5_path": str(h5_path),
    })

    logger.info("=" * 70)
    logger.info("GazeHOI test results")
    for k in (
        "GazeAcc", "GazeAcc_text", "Acc@1", "Acc@3",
        "n_total", "n_gaze_correct", "n_gaze_parseable", "n_pred_inside",
        "n_gaze_correct_text", "n_gaze_parseable_text", "n_pred_inside_text",
        "n_acc1_eligible", "n_acc1_correct", "n_acck_correct",
        "n_pred_with_category", "n_pred_unmapped",
        "n_pairs_total", "n_pairs_with_pred", "parse_success_rate",
    ):
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                logger.info("  %-26s %.4f", k, v)
            else:
                logger.info("  %-26s %s", k, v)
    logger.info("=" * 70)

    if output_json:
        out_dir = os.path.dirname(os.path.abspath(output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # JSON-serialise: drop the cached sims, convert sets to lists.
        with open(output_json, "w") as f:
            json.dump(
                {k: (list(v) if isinstance(v, set) else v) for k, v in metrics.items()},
                f, indent=2,
            )
        logger.info("Wrote metrics to %s", output_json)


if __name__ == "__main__":
    main()
