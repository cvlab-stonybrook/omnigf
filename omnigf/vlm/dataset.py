# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""VLM-specific gaze dataset and data module.

Pipeline::

    UnifiedGazeFrameDataset → VLMGazeDataset → VLMGazeCollator → Trainer

:class:`VLMGazeDataset` wraps the unified dataset and builds prompt/target
*strings* plus a PIL image.  Tokenisation happens in the collator (see
:mod:`omnigf.vlm.collator`).
"""

from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.functional import to_pil_image

import lightning.pytorch as pl
from omegaconf import DictConfig

from omnigf.datasets.unified_gaze_frame import (
    UnifiedGazeFrameDataModule,
)

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "point": (
        "You are given an image with head bounding boxes for all people "
        "in the scene. Each person is identified as P0..PN with their head "
        'box given as {"bbox_2d": [x1, y1, x2, y2]} in coordinates [0, 999]. '
        "For each person, predict whether their gaze target is inside or "
        "outside the image. Output one JSON object per person. If the gaze "
        'target is inside, output {"status": "inside", "point_2d": [x, y]}. '
        'If outside, output {"status": "outside"}.'
    ),
    "point_semantic": (
        "You are given an image with head bounding boxes for all people "
        "in the scene. Each person is identified as P0..PN with their head "
        'box given as {"bbox_2d": [x1, y1, x2, y2]} in coordinates [0, 999]. '
        "For each person, predict whether their gaze target is inside or "
        "outside the image, and what they are looking at. Output one JSON "
        "object per person. If the gaze target is inside, output "
        '{"status": "inside", "point_2d": [x, y], "category": "<object>"}. '
        'If outside, output {"status": "outside"}.'
    ),
}

# Unknown person handling policies:
#
# "exclude" (default, recommended):
#   Unknown persons (inout=-1) are excluded from both prompt and target.
#   The model only sees persons it must predict for. This gives a clean
#   1:1 correspondence between prompt persons and target predictions,
#   which is important for autoregressive LLMs — the model learns that
#   EVERY person in the prompt gets a prediction. At inference time all
#   detected heads are listed and all get predictions, matching this
#   training convention. Unknown persons are still visible in the image;
#   the vision encoder captures them implicitly.
#
# "include_marked" (alternative):
#   Unknown persons are included in the prompt with a "(no annotation)"
#   marker and excluded from the target. This provides explicit social
#   context but creates a train/test mismatch: the marker does not exist
#   at inference time, and the target skips persons that appear in the
#   prompt, which can confuse autoregressive generation. Use this only
#   if experiments show that explicit social context outweighs the
#   train/test inconsistency.

UNKNOWN_POLICIES = ("exclude", "include_marked")

# Order in which persons are laid out in the prompt / target before pids are
# relabelled to 0..V-1. Added for the reviewer ablation on person-token order.
#
#   "dataset" — leave the order produced by the source dataset. At TRAIN time
#               that is already a random permutation (``get_shuffle_idx`` in
#               ``omnigf/utils/social_gaze.py``, called by every temporal loader
#               under an ``if split == "train"`` gate), so the model is trained
#               order-agnostic. At VAL/TEST it is the raw H5 ``person_ids``
#               order. This is the default and the historical behaviour.
#   "lr_tb"   — left-to-right by head-bbox left edge, top-to-bottom tiebreak.
#   "rl_bt"   — exact reverse of ``lr_tb``.
#   "random"  — random permutation, seeded per-sample by
#               ``(person_order_seed, index)`` so a given seed is reproducible
#               across runs and DataLoader workers (required for eval).
PERSON_ORDERS = ("dataset", "lr_tb", "rl_bt", "random")


# ---------------------------------------------------------------------------
# VLMGazeDataset
# ---------------------------------------------------------------------------


class VLMGazeDataset(Dataset):
    """Wraps :class:`UnifiedGazeFrameDataset` to produce multi-person VLM
    training samples.

    Each sample contains:

    * ``image`` — PIL Image (un-normalised, as the VLM processor expects).
    * ``prompt`` — system instruction + head-bbox listing.
    * ``target`` — sequential per-person predictions (text).
    * ``head_bboxes`` — ``[V, 4]`` for annotated persons.
    * Plus metadata for loss masking (``gaze_pts``, ``inout``, ``person_ids``,
      ``num_annotated``, and optionally ``gaze_heatmaps``).

    Tokenisation is deferred to :class:`~omnigf.vlm.collator.VLMGazeCollator`.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        coord_bins: int = 1000,
        unknown_policy: str = "exclude",
        prompt_mode: str = "point",
        relabel_ids: bool = True,
        split: str = "train",
        gf_test_all_people: bool = False,
        use_gaze_encoder: bool = False,
        gaze_injection: str = "replace",
        use_semantic_labels: bool = False,
        social_gaze_mode: Optional[bool] = None,
        extract_pair_labels: bool = False,
        show_no_annotation_marker: bool = False,
        sort_persons_lr_tb: bool = False,
        sort_annotated_first: bool = False,
        person_order: str = "dataset",
        person_order_seed: int = 0,
        mask_per_person_lm_loss: bool = False,
    ):
        assert unknown_policy in UNKNOWN_POLICIES
        assert prompt_mode in SYSTEM_PROMPTS
        assert person_order in PERSON_ORDERS, (
            f"person_order must be one of {sorted(PERSON_ORDERS)}, got {person_order!r}"
        )

        self.base = base_dataset
        self.coord_bins = coord_bins
        self.unknown_policy = unknown_policy
        # Auto-select semantic prompt when semantic labels are enabled
        if use_semantic_labels and prompt_mode == "point":
            self.prompt_mode = "point_semantic"
        else:
            self.prompt_mode = prompt_mode
        self.relabel_ids = relabel_ids
        self.split = split
        self.gf_test_all_people = gf_test_all_people
        self.use_gaze_encoder = use_gaze_encoder
        self.gaze_injection = gaze_injection
        # The dedicated <|gaze_pad|> placeholder token is only used by the
        # "replace" injection strategy. The additive strategies (add_span /
        # add_anchor) add the gaze embedding onto real text tokens, so the
        # placeholder is not inserted into the prompt at all.
        self.use_gaze_pad_token = (
            use_gaze_encoder and gaze_injection == "replace"
        )
        self.use_semantic_labels = use_semantic_labels

        # ``extract_pair_labels`` enables the heads-based social-gaze
        # path: the dataset emits per-pair ``lah/laeo/coatt`` label
        # tensors in the relabelled 0..V-1 pid space, which the
        # ``SocialPairHead`` consumes. When enabled, val/test should
        # also keep all real persons (auto-promote ``social_gaze_mode``)
        # so the head sees all candidate-gazer / candidate-receiver
        # positions at eval, matching the MTGS-style metric aggregation.
        self.extract_pair_labels = bool(extract_pair_labels)

        # social_gaze_mode controls whether val/test keeps ``inout == -1``
        # persons in the sample so social-gaze pair labels on them
        # survive into metric aggregation. When unset, auto-enable iff
        # the prompt mode emits a social block (``point_then_social``)
        # OR the heads-based pair-label path is active.
        # For localization-only runs, leave this False: inout=-1 persons
        # would contribute nothing to the gaze metrics (MTGS already
        # masks them via ``inout_gt == 1``) and keeping them creates
        # holes in the ground-truth point block.
        if social_gaze_mode is None:
            social_gaze_mode = self.extract_pair_labels
        self.social_gaze_mode = bool(social_gaze_mode)

        # When True, append ``(no annotation)`` to the prompt line of each
        # person whose ``inout == -1`` (i.e., no GT gaze label). Identifies
        # the unannotated persons by their actual ``inout`` value rather
        # than by position in the prompt list. Default False — train and
        # inference prompts then look identical for every person, and the
        # model is expected to predict for all of them.
        self.show_no_annotation_marker = bool(show_no_annotation_marker)

        # When True, sort the kept persons by head-bbox left edge ascending
        # (left-to-right), with the bbox top edge as tiebreaker (top-to-
        # bottom). The relabelled pid 0 then always denotes the leftmost
        # person, pid 1 the second-from-left, etc. Applies to all per-person
        # tensors and remaps pids inside the social-gaze sets / pair labels
        # so the prompt, target, and per-pair supervision stay consistent.
        # Default False (preserves original H5 / shuffler order).
        # Superseded by ``person_order="lr_tb"``; kept so existing configs and
        # checkpointed runs keep working. When the legacy flag is set and
        # ``person_order`` is left at its default, it maps onto "lr_tb".
        self.sort_persons_lr_tb = bool(sort_persons_lr_tb)

        # Person-token ordering ablation (see PERSON_ORDERS above).
        if person_order == "dataset" and self.sort_persons_lr_tb:
            person_order = "lr_tb"
        self.person_order = person_order
        self.person_order_seed = int(person_order_seed)

        # When True, partition kept persons so that all annotated
        # (``inout != -1``) persons come first and unannotated
        # (``inout == -1``) persons come at the end. Composes with
        # ``sort_persons_lr_tb``: when both are True, persons are split
        # by annotation status first, then sorted left-to-right within
        # each group; when only this flag is True, annotated persons stay
        # in their original order, then unannotated persons. Default False.
        self.sort_annotated_first = bool(sort_annotated_first)

        # When True, keep ``inout == -1`` persons in the training sample
        # (prompt + per-person tensors) and emit a uniform target line per
        # person (with a ``"unknown"`` status placeholder for inout=-1).
        # The collator will mask the inner content of each unknown line
        # from the LM loss, leaving only the ``P{pid}:`` skeleton
        # supervised. The model's heatmap / inout / social-pair heads
        # already mask inout=-1 persons via their own loss masks.
        self.mask_per_person_lm_loss = bool(mask_per_person_lm_loss)

        self._n_skipped = 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict:
        sample = self.base[index]

        # --- Extract center frame ---
        head_bboxes = sample["head_bboxes"][0]  # [N, 4]
        gaze_pts = sample["gaze_pts"][0]        # [N, 2] or [M, 2] for GF test
        pids = sample["person_ids"]             # [N]

        # GazeFollow test has multi-annotator gaze_pts (different shape from
        # head_bboxes) and unpadded inout.  Handle separately.
        is_gf_test = (
            sample.get("dataset") == "gazefollow" and self.split == "test"
        )
        if is_gf_test:
            return self._build_gf_test_sample(sample, head_bboxes, gaze_pts, pids)

        inout = sample["inout"][0]              # [N]

        # Filter out non-person slots in the source H5 annotations:
        #   * ``pid == -1``      — leading dummy/padding slot.
        #   * ``pid >= 1000``    — gaze-target / attention-object bbox
        #                          stored alongside head bboxes in the
        #                          same ``person_ids`` / ``head_bboxes``
        #                          arrays. These always carry ``inout=-1``
        #                          and a sentinel gaze point, but their
        #                          bbox is a real region in the image —
        #                          including them as "persons" would
        #                          render a fake ``P<n>: head at [...]``
        #                          line in the prompt pointing at the
        #                          gaze target itself.
        real_mask = (pids >= 0) & (pids < 1000)
        real_bboxes = head_bboxes[real_mask]
        real_gaze = gaze_pts[real_mask]
        real_inout = inout[real_mask]
        # Per-person pseudo flag (1 = filled by MTGS pseudo-annotation pass,
        # 0 = original GT). Defaults to all-zero when the source h5 lacks
        # the column (pre-pseudo datasets remain numerically unchanged).
        is_pseudo_full = sample.get("is_pseudo")
        if is_pseudo_full is None:
            real_is_pseudo = torch.zeros_like(real_inout, dtype=torch.float)
        else:
            # Same temporal-dim slice the other per-person tensors use above.
            real_is_pseudo = is_pseudo_full[0][real_mask].float()

        # Separate annotated (inout != -1) from unknown (inout == -1)
        annotated_mask = real_inout != -1
        unknown_mask = real_inout == -1
        n_annotated = int(annotated_mask.sum().item())
        n_real = int(real_mask.sum().item())

        # Person-filtering policy:
        #   * social_gaze_mode + val/test  -> keep ALL real persons so
        #     social-gaze pair labels on inout=-1 persons survive into
        #     metric aggregation (MTGS masks pairs at the label level
        #     ``gt != -1``, independently of per-person inout). Gaze /
        #     in-out metrics already mask on ``inout_gt == 1`` at the
        #     metric layer so inout=-1 persons contribute nothing to
        #     the final numbers, and no random sample-replacement is
        #     needed.
        #   * social_gaze_mode + train    -> filter to annotated_mask so
        #     training sees only clean supervision (no "holes" in the
        #     point block).
        #   * localization-only (social_gaze_mode=False), all splits ->
        #     filter to annotated_mask. Unused inout=-1 persons would
        #     contribute nothing to any metric; keeping them would just
        #     pad prompts uselessly.
        keep_for_val_test = (
            self.social_gaze_mode
            and self.split in ("val", "test")
            and n_real > 0
        )
        # Heads-based pair-label path: keep ALL real persons regardless of
        # split, so the SocialPairHead sees multi-person pairs at train
        # time too. 
        # When ``mask_per_person_lm_loss`` is enabled the trainer wants to
        # see ALL real persons (including inout=-1) so the model gets
        # cross-attention exposure to crowded prompts. The LM loss for
        # those persons is masked at the collator level. Without this
        # flag the original behavior holds: annotated-only training.
        keep_for_pair_head = (
            self.extract_pair_labels and n_real > 0
            and self.mask_per_person_lm_loss
        )
        if keep_for_val_test or keep_for_pair_head:
            keep_in_real = torch.ones_like(annotated_mask)
            n_kept = n_real
        else:
            keep_in_real = annotated_mask
            n_kept = n_annotated
        # Kept for backwards-compatible references below.
        is_social_mode = self.social_gaze_mode

        if n_kept == 0:
            if self.split == "train":
                # No meaningful content for this sample at train — LM loss
                # on an empty target would be zero and waste a batch slot.
                # Replace with a random other sample (pre-existing train
                # behaviour).
                self._n_skipped += 1
                return self[torch.randint(len(self), (1,)).item()]
            # val / test: drop without replacement so metric aggregation
            # isn't polluted by evaluating some other sample twice. The
            # collator filters out ``None`` entries; if the whole batch
            # is empty, the collator returns ``None`` and the Lightning
            # {val,test}_step short-circuits.
            self._n_skipped += 1
            return None

        # Person order is already shuffled by the source datasets during
        # training (via get_shuffle_idx), so no additional shuffle needed here.
        ann_bboxes = real_bboxes[keep_in_real]
        ann_gaze = real_gaze[keep_in_real]
        ann_inout = real_inout[keep_in_real]
        ann_is_pseudo = real_is_pseudo[keep_in_real]

        # Optional reordering of kept persons. Two orthogonal knobs:
        #
        #   * ``person_order`` — one of PERSON_ORDERS (see top of file):
        #     "dataset" (no reorder), "lr_tb" (left-to-right by head-bbox
        #     x_min, top-to-bottom tiebreak), "rl_bt" (exact reverse of
        #     "lr_tb"), or "random" (seeded per-sample permutation).
        #     The legacy ``sort_persons_lr_tb=True`` maps onto "lr_tb".
        #   * ``sort_annotated_first`` — partition so annotated persons
        #     (``inout != -1``) come first, unannotated last. Within each
        #     group, secondary order follows ``person_order``. NOTE: this is
        #     a no-op whenever ``show_no_annotation_marker`` is False, since
        #     ``keep_in_real`` then already filters to annotated persons only.
        #
        # Pids stay 0..V-1 after relabelling but now correspond to the
        # post-sort order. ``sort_perm`` is None when no sort is requested
        # or there's nothing to sort (V <= 1).
        sort_perm = None
        do_sort = self.person_order != "dataset" or self.sort_annotated_first
        if do_sort and n_kept > 1:
            if self.person_order in ("lr_tb", "rl_bt"):
                xs = ann_bboxes[:, 0].clamp(0, 1)
                ys = ann_bboxes[:, 1].clamp(0, 1)
                # x dominates; tiny y term breaks ties (y in [0, 1]).
                spatial_key = xs + ys * 1e-6
                if self.person_order == "rl_bt":
                    # Negate rather than flip the argsort result: flipping a
                    # stable argsort would also reverse tied entries, so it is
                    # not the exact inverse of "lr_tb" when x values coincide.
                    spatial_key = -spatial_key
            elif self.person_order == "random":
                # Seeded per-sample so the permutation is a deterministic
                # function of (seed, index): reproducible across re-runs and
                # independent of DataLoader worker sharding. Normalised to
                # [0, 1) so ``sort_annotated_first``'s x10 group key still
                # dominates for any number of persons.
                gen = torch.Generator()
                gen.manual_seed(self.person_order_seed * 1_000_003 + int(index))
                spatial_key = (
                    torch.randperm(n_kept, generator=gen).float() / n_kept
                )
            else:
                # Identity key — argsort(stable=True) preserves original order.
                spatial_key = torch.arange(n_kept, dtype=torch.float)

            if self.sort_annotated_first:
                # group_key: 0 for annotated, 1 for unannotated. Scaled
                # by 10 to dominate ``spatial_key`` (which is in
                # ``[0, ~1.001]``), so all annotated persons come first.
                group_key = (ann_inout == -1).float() * 10.0
                keys = group_key + spatial_key
            else:
                keys = spatial_key

            sort_perm = torch.argsort(keys, stable=True)
            ann_bboxes = ann_bboxes[sort_perm]
            ann_gaze = ann_gaze[sort_perm]
            ann_inout = ann_inout[sort_perm]
            ann_is_pseudo = ann_is_pseudo[sort_perm]

        # --- Relabel IDs to 0..V-1 ---
        relabelled_pids = torch.arange(n_kept)
        # --- Build prompt person list ---
        prompt_pids = relabelled_pids
        prompt_bboxes = ann_bboxes

        # Per-person inout for the prompt. Aligned with prompt_pids /
        # prompt_bboxes so the marker (if enabled) lands on the right
        # persons (those with ``inout == -1``).
        prompt_inout = ann_inout

        # ``include_marked`` only matters for non-social modes — social
        # mode already includes inout=-1 persons in the main list.
        if (not is_social_mode
                and self.unknown_policy == "include_marked"
                and unknown_mask.sum() > 0):
            unk_bboxes = real_bboxes[unknown_mask]
            unk_pids = torch.arange(n_annotated, n_annotated + unknown_mask.sum().item())
            prompt_pids = torch.cat([prompt_pids, unk_pids])
            prompt_bboxes = torch.cat([prompt_bboxes, unk_bboxes])
            # Appended unknowns are by definition ``inout == -1``.
            prompt_inout = torch.cat([
                prompt_inout,
                torch.full((unknown_mask.sum().item(),), -1.0, dtype=prompt_inout.dtype),
            ])

        prompt = self._build_prompt(prompt_pids, prompt_bboxes, inout=prompt_inout)

        # Extract per-person semantic text labels for target building
        ann_sem_labels = None
        if self.use_semantic_labels and "gaze_label_texts" in sample:
            all_labels = sample["gaze_label_texts"]  # list of str, len N
            # Apply same real_mask + keep_in_real filtering
            real_labels = [all_labels[i] for i in range(len(all_labels)) if real_mask[i]]
            ann_sem_labels = [real_labels[i] for i in range(len(real_labels)) if keep_in_real[i]]
            if sort_perm is not None:
                ann_sem_labels = [ann_sem_labels[i] for i in sort_perm.tolist()]

        # Compute social-gaze sets (for point_then_social target and eval).
        # The kept-person order after real_mask + keep_in_real defines
        # the relabelled 0..V-1 pid space used in both the target text
        # and ``social_sets``. In social mode keep_in_real is all-True
        # across real persons so pair labels between inout=-1 persons
        # are retained.
        social_sets = None
        if is_social_mode:
            social_sets = self._compute_social_sets(
                sample, real_mask, keep_in_real,
            )
            if sort_perm is not None:
                # Remap old pids (in keep_in_real-iteration order) to new
                # pids (sort_perm-induced order). inv_perm[old] = new.
                inv_perm = torch.empty_like(sort_perm)
                inv_perm[sort_perm] = torch.arange(sort_perm.shape[0])
                inv_list = inv_perm.tolist()
                social_sets = {
                    "LAH": [[inv_list[a], inv_list[b]] for (a, b) in social_sets.get("LAH", [])],
                    "LAEO": [
                        sorted([inv_list[a], inv_list[b]])
                        for (a, b) in social_sets.get("LAEO", [])
                    ],
                    "SA": [
                        sorted([inv_list[a] for a in group])
                        for group in social_sets.get("SA", [])
                    ],
                }

        target, target_point_prefix = self._build_target(
            relabelled_pids, ann_gaze, ann_inout, ann_sem_labels,
            social_sets=social_sets,
        )

        # --- Convert image tensor to PIL ---
        img_tensor = sample["image"][0]  # [3, H, W]
        # The image comes normalised from the MTGS pipeline; convert back for
        # the VLM processor.  If the pixel range is already ~[0,1] or standard
        # normalised, the VLM's own image_processor will re-normalise.
        pil_image = to_pil_image(img_tensor.clamp(0, 1))

        result = {
            "image": pil_image,
            "system_prompt": SYSTEM_PROMPTS[self.prompt_mode],
            "prompt": prompt,
            "target": target,
            "gaze_pts": ann_gaze,         # [V, 2]
            "inout": ann_inout,           # [V]
            "is_pseudo": ann_is_pseudo,   # [V] (1 = pseudo-labelled, 0 = GT)
            "person_ids": relabelled_pids,  # [V]
            # num_annotated must match the number of per-person tensors
            # (``head_bboxes`` / ``gaze_pts`` / ``inout`` / ``head_crops``),
            # NOT the count of persons with ``inout != -1``. Downstream
            # uses it (a) as the ``P<id>`` position-finding bound in the
            # collator, and (b) as the gaze-token injection bound in
            # ``_inject_gaze_into_hidden``. Keeping it at ``n_kept``
            # ensures every prompt person's ``<|gaze_pad|>`` token gets
            # replaced; inout=-1 persons then naturally disappear from
            # metric totals via the ``inout_gt == 1`` mask in the metric
            # layer.
            "num_annotated": n_kept,
            "head_bboxes": ann_bboxes,    # [V, 4]
            "path": sample.get("path", [""]),
            # Original (pre-resize) image size as (w, h). Needed by the
            # full-resolution GazeFollow AUC protocol (Chong et al. /
            # Gaze-LLE), which upsamples the predicted heatmap to the source
            # image size before scoring. Carried as [2]; the source datasets
            # store it as [1, 2] after center-frame selection.
            "img_size": (
                sample["img_size"][0]
                if "img_size" in sample and sample["img_size"].ndim > 1
                else sample.get("img_size")
            ),
        }
        if target_point_prefix is not None:
            result["target_point_prefix"] = target_point_prefix
        if social_sets is not None:
            # Pass-through metadata for eval. Stored as a plain dict of
            # Python lists; the collator leaves it alone.
            result["social_sets"] = social_sets

        # Heads-based social-gaze pair labels (relabelled 0..V-1 space).
        # Mirrors ``_compute_social_sets`` but emits the tensor form
        # consumed by ``SocialPairHead`` + ``compute_social_pair_bce_loss``
        # rather than the human-readable set form. Only computed when the
        # caller opted in via ``extract_pair_labels=True``.
        if self.extract_pair_labels:
            from omnigf.vlm.social_labels import social_labels_to_pair_tensors
            raw_pids = sample.get("person_ids")
            if raw_pids is not None:
                # Reconstruct keep_mask in raw_pids index space (same
                # logic as ``_compute_social_sets`` at lines 712-717).
                keep_mask_raw = torch.zeros(raw_pids.shape[0], dtype=torch.bool)
                real_idx = torch.nonzero(real_mask, as_tuple=False).flatten()
                ann_in_real = torch.nonzero(keep_in_real, as_tuple=False).flatten()
                for k in ann_in_real.tolist():
                    keep_mask_raw[real_idx[k].item()] = True

                def _get(key):
                    val = sample.get(key)
                    if val is None:
                        return None
                    return val[0] if (val.ndim >= 2 and val.shape[0] >= 1) else val

                pair_tensors = social_labels_to_pair_tensors(
                    _get("lah_labels"),
                    _get("laeo_labels"),
                    _get("coatt_labels"),
                    raw_pids,
                    keep_mask_raw,
                )
                if sort_perm is not None and pair_tensors["lah"].numel() > 0:
                    # Permute pair tensors from old (keep_mask-iteration)
                    # pid space to new (sort_perm) pid space. Each flat
                    # entry corresponds to an (i, j) ordered pair from
                    # ``itertools.permutations(range(V), 2)``; for each new
                    # pair (i', j'), read the old pair label at
                    # (sort_perm[i'], sort_perm[j']).
                    import itertools as _it
                    V = sort_perm.shape[0]
                    perm_list = sort_perm.tolist()
                    pairs = list(_it.permutations(range(V), 2))
                    old_pair_to_idx = {p: k for k, p in enumerate(pairs)}
                    remap = torch.tensor(
                        [old_pair_to_idx[(perm_list[i], perm_list[j])] for (i, j) in pairs],
                        dtype=torch.long,
                    )
                    pair_tensors = {k: v[remap] for k, v in pair_tensors.items()}
                result["lah_pair_labels"] = pair_tensors["lah"]
                result["laeo_pair_labels"] = pair_tensors["laeo"]
                result["coatt_pair_labels"] = pair_tensors["coatt"]

        if "gaze_heatmaps" in sample:
            hm = sample["gaze_heatmaps"][0]  # [N, 64, 64]
            kept = hm[real_mask][keep_in_real]
            if sort_perm is not None:
                kept = kept[sort_perm]
            result["gaze_heatmaps"] = kept

        if self.use_gaze_encoder and "heads" in sample:
            heads = sample["heads"][0]  # [N, 3, H, W]
            # Use the same ``keep_in_real`` mask as ``head_bboxes`` etc.
            # so per-sample shape stays consistent across every
            # per-person tensor the collator batches. Using a different
            # mask here (the old code used ``annotated_mask``) causes a
            # shape mismatch in ``GazeEncoder.forward_head`` because the
            # collator's ``_pad_people_tensor`` pads each tensor to its
            # own batch-max independently.
            kept_heads = heads[real_mask][keep_in_real]
            if sort_perm is not None:
                kept_heads = kept_heads[sort_perm]
            result["head_crops"] = kept_heads  # [V, 3, H, W]

        # Semantic gaze labels — use keep_in_real so size matches the
        # other per-person tensors in the collator.
        if "gaze_label_embs" in sample:
            embs = sample["gaze_label_embs"][0]  # [N, 512]
            ids = sample["gaze_label_ids"][0]     # [N]
            test_ids = sample["gaze_test_label_ids"][0]  # [N]
            kept_embs = embs[real_mask][keep_in_real]
            kept_ids = ids[real_mask][keep_in_real]
            kept_test_ids = test_ids[real_mask][keep_in_real]
            if sort_perm is not None:
                kept_embs = kept_embs[sort_perm]
                kept_ids = kept_ids[sort_perm]
                kept_test_ids = kept_test_ids[sort_perm]
            result["gaze_label_embs"] = kept_embs
            result["gaze_label_ids"] = kept_ids
            result["gaze_test_label_ids"] = kept_test_ids
            result["has_semantic_label"] = (kept_ids >= 0)
            if "gaze_multi_test_label_ids" in sample:
                multi = sample["gaze_multi_test_label_ids"][0]  # [N, 5]
                kept_multi = multi[real_mask][keep_in_real]
                if sort_perm is not None:
                    kept_multi = kept_multi[sort_perm]
                result["gaze_multi_test_label_ids"] = kept_multi

        # Surface the per-kept-person GT category text list. Already
        # built above as ``ann_sem_labels`` (filtered + permuted to match
        # the kept per-person tensors). Off-by-default downstream — only
        # the visualisation script reads it.
        if ann_sem_labels is not None:
            result["gaze_label_texts"] = ann_sem_labels

        # GazeHOI surfaces the per-pair looked-at-object bbox so the
        # visualisation script can draw it next to the GT gaze tip.
        # Filtered through the same ``real_mask`` + ``keep_in_real``
        # + ``sort_perm`` chain as the other per-person tensors so the
        # row order matches ``head_bboxes`` etc. Other sources don't
        # populate this key, so the field is absent from the result
        # for them — no schema impact.
        if "object_bboxes" in sample:
            obj = sample["object_bboxes"][0]  # [N, 4]
            kept_obj = obj[real_mask][keep_in_real]
            if sort_perm is not None:
                kept_obj = kept_obj[sort_perm]
            result["object_bboxes"] = kept_obj

        return result

    # ------------------------------------------------------------------ #
    # GazeFollow test
    # ------------------------------------------------------------------ #

    def _build_gf_test_sample(
        self, sample: Dict, head_bboxes, gaze_pts_raw, pids,
    ) -> Dict:
        """Build a VLM sample for the GazeFollow test split.

        GazeFollow test has ONE annotated person (the **last** real person)
        with multiple annotator gaze points ``[A, 2]``.  Other people have
        ``inout=0`` (outside).  ``inout`` in the source sample is not
        padded/trimmed consistently with ``head_bboxes``, so we reconstruct
        it from the known convention.

        Two modes controlled by ``self.gf_test_all_people``:

        * ``False`` (default): only the annotated person is included in the
          prompt.  Avoids autoregressive error propagation from generating
          many "OUTSIDE" predictions before the target person.
        * ``True``: all real people are included for social context.  The
          person of interest is always the last one.

        The multi-annotator gaze points are stored as ``gaze_pts_all`` for
        use with :class:`~omnigf.performance.metrics.GFTestDistance` and
        :class:`~omnigf.performance.metrics.GFTestAUC`.
        """
        # Real people (non-padding). Same aux-pid filter as the main
        # branch — GazeFollow has no pid>=1000 today (max pid ~200) but
        # keeping the check consistent across branches is cheap and
        # future-proofs against source-dataset convention changes.
        real_mask = (pids >= 0) & (pids < 1000)
        num_real = real_mask.sum().item()
        real_bboxes = head_bboxes[real_mask]  # [K, 4]

        # Multi-annotator gaze points (filter out -1 padding)
        gaze_pts_all = gaze_pts_raw[gaze_pts_raw[:, 0] >= 0]  # [A, 2]
        avg_gaze = gaze_pts_all.mean(0)  # [2]

        if self.gf_test_all_people:
            # Include all real people: outside (0) except last (1)
            out_inout = torch.zeros(num_real)
            out_inout[-1] = 1.0
            out_gaze = torch.full((num_real, 2), -1.0)
            out_gaze[-1] = avg_gaze
            out_bboxes = real_bboxes
            n_annotated = num_real
        else:
            # Only the annotated person (last real person)
            out_inout = torch.tensor([1.0])
            out_gaze = avg_gaze.unsqueeze(0)  # [1, 2]
            out_bboxes = real_bboxes[-1:]      # [1, 4]
            n_annotated = 1

        relabelled_pids = torch.arange(n_annotated)

        prompt = self._build_prompt(relabelled_pids, out_bboxes, inout=out_inout)

        # Semantic text labels for GF test
        gf_sem_labels = None
        if self.use_semantic_labels and "gaze_label_texts" in sample:
            all_labels = sample["gaze_label_texts"]
            real_labels = [all_labels[i] for i in range(len(all_labels)) if real_mask[i]]
            if self.gf_test_all_people:
                gf_sem_labels = real_labels
            else:
                gf_sem_labels = real_labels[-1:]

        target, target_point_prefix = self._build_target(
            relabelled_pids, out_gaze, out_inout, gf_sem_labels,
        )

        img_tensor = sample["image"][0]
        pil_image = to_pil_image(img_tensor.clamp(0, 1))

        result = {
            "image": pil_image,
            "system_prompt": SYSTEM_PROMPTS[self.prompt_mode],
            "prompt": prompt,
            "target": target,
            "gaze_pts": out_gaze,            # [K, 2] or [1, 2]
            "inout": out_inout,              # [K] or [1]
            "person_ids": relabelled_pids,   # [K] or [1]
            "num_annotated": n_annotated,
            "head_bboxes": out_bboxes,       # [K, 4] or [1, 4]
            "gaze_pts_all": gaze_pts_all,    # [A, 2] — all annotator points
            "path": sample.get("path", [""]),
            # Original (pre-resize) image size as (w, h). Needed by the
            # full-resolution GazeFollow AUC protocol (Chong et al. /
            # Gaze-LLE), which upsamples the predicted heatmap to the source
            # image size before scoring. Carried as [2]; the source datasets
            # store it as [1, 2] after center-frame selection.
            "img_size": (
                sample["img_size"][0]
                if "img_size" in sample and sample["img_size"].ndim > 1
                else sample.get("img_size")
            ),
        }
        if target_point_prefix is not None:
            result["target_point_prefix"] = target_point_prefix

        if self.use_gaze_encoder and "heads" in sample:
            heads = sample["heads"][0]  # [N, 3, H, W]
            real_heads = heads[real_mask]  # [K_real, 3, H, W]
            if self.gf_test_all_people:
                result["head_crops"] = real_heads
            else:
                result["head_crops"] = real_heads[-1:]  # last person only

        if "gaze_label_embs" in sample:
            embs = sample["gaze_label_embs"][0][real_mask]
            ids = sample["gaze_label_ids"][0][real_mask]
            test_ids = sample["gaze_test_label_ids"][0][real_mask]
            if self.gf_test_all_people:
                result["gaze_label_embs"] = embs
                result["gaze_label_ids"] = ids
                result["gaze_test_label_ids"] = test_ids
            else:
                result["gaze_label_embs"] = embs[-1:]
                result["gaze_label_ids"] = ids[-1:]
                result["gaze_test_label_ids"] = test_ids[-1:]
        if gf_sem_labels is not None:
            # Per-kept-person GT category text — list of strings whose
            # length matches the per-person tensors above. Visualisation
            # uses this to render the GT label next to the GT gaze tip.
            result["gaze_label_texts"] = gf_sem_labels
            result["has_semantic_label"] = result["gaze_label_ids"] >= 0
            if "gaze_multi_test_label_ids" in sample:
                multi = sample["gaze_multi_test_label_ids"][0][real_mask]
                if self.gf_test_all_people:
                    result["gaze_multi_test_label_ids"] = multi
                else:
                    result["gaze_multi_test_label_ids"] = multi[-1:]

        # ---- Heads-based pair labels (LAH on GazeFollow test) ----
        # Only LAH is annotated for GazeFollow (laeo / coatt are all -1
        # at the source — see ``omnigf/datasets/gazefollow.py:540-542``).
        # We still emit all three tensors (with laeo/coatt as -1 sentinel
        # everywhere) so the collator's pair padding sees a consistent
        # schema regardless of source. The MTGS-style metric module
        # masks ``gt != -1`` per category, so only LAH contributes to
        # the GF test report.
        if self.extract_pair_labels:
            from omnigf.vlm.social_labels import social_labels_to_pair_tensors
            raw_pids = sample.get("person_ids")
            if raw_pids is not None:
                # Build keep_mask in the raw-pids space matching the
                # ``gf_test_all_people`` decision above. With all-people
                # mode we keep every real pid; otherwise we keep only
                # the last real pid (the primary subject).
                keep_mask_raw = torch.zeros(raw_pids.shape[0], dtype=torch.bool)
                real_idx_list = torch.nonzero(real_mask, as_tuple=False).flatten().tolist()
                if self.gf_test_all_people:
                    for i in real_idx_list:
                        keep_mask_raw[i] = True
                elif len(real_idx_list) > 0:
                    keep_mask_raw[real_idx_list[-1]] = True

                def _get(key):
                    val = sample.get(key)
                    if val is None:
                        return None
                    return val[0] if (val.ndim >= 2 and val.shape[0] >= 1) else val

                pair_tensors = social_labels_to_pair_tensors(
                    _get("lah_labels"),
                    _get("laeo_labels"),
                    _get("coatt_labels"),
                    raw_pids,
                    keep_mask_raw,
                )
                result["lah_pair_labels"] = pair_tensors["lah"]
                result["laeo_pair_labels"] = pair_tensors["laeo"]
                result["coatt_pair_labels"] = pair_tensors["coatt"]

        return result

    # ------------------------------------------------------------------ #
    # Prompt / target builders
    # ------------------------------------------------------------------ #

    def _build_prompt(self, pids, bboxes, inout=None) -> str:
        """Build the user-turn prompt (bbox listing only).

        The system instruction is stored separately in ``"system_prompt"``
        and assembled into a proper ``{"role": "system"}`` message by the
        collator.

        When ``self.show_no_annotation_marker`` is True AND ``inout`` is
        provided, persons whose ``inout == -1`` get a trailing
        ``(no annotation)`` tag (identifies unannotated persons by their
        actual ``inout`` value rather than by position — earlier code
        used a position-based slice that was correct only when unknowns
        had been explicitly appended at the end). When False (default),
        every person gets the same line format and the model is expected
        to predict for all of them at inference.

        Format (default, no marker)::

            P0: {"bbox_2d": [120, 340, 200, 480]}
            P1: {"bbox_2d": [560, 100, 650, 250]}

        Format (``show_no_annotation_marker=True``, P1 has ``inout==-1``)::

            P0: {"bbox_2d": [120, 340, 200, 480]}
            P1: {"bbox_2d": [560, 100, 650, 250]} (no annotation)
        """
        gaze_marker = "<|gaze_pad|> " if self.use_gaze_pad_token else ""

        # Materialise the per-person inout values so we can index by ``i``.
        inout_list = None
        if self.show_no_annotation_marker and inout is not None:
            inout_list = (
                inout.tolist() if torch.is_tensor(inout) else list(inout)
            )
        lines = []
        for i, (pid, bbox) in enumerate(zip(pids, bboxes)):
            b = (bbox.clamp(0, 1) * (self.coord_bins - 1)).int()
            bbox_json = f'{{"bbox_2d": [{b[0]}, {b[1]}, {b[2]}, {b[3]}]}}'
            line = f"P{int(pid)}: {gaze_marker}{bbox_json}"
            if inout_list is not None and i < len(inout_list) and int(inout_list[i]) == -1:
                line = f"{line} (no annotation)"
            lines.append(line)
        return "\n".join(lines)

    def _build_target(
        self,
        pids,
        gaze_pts,
        inout,
        sem_labels=None,
        social_sets=None,
    ):
        """Build the assistant-turn target text.

        Single-block modes (``point`` / ``point_semantic``)::

            P0: {"status": "inside", "point_2d": [450, 620]}
            P1: {"status": "outside"}

        Two-block mode (``point_then_category``)::

            P0: {"status": "inside", "point_2d": [450, 620]}
            P1: {"status": "outside"}

            P0: {"category": "laptop"}
            P1: {"category": "none"}

        Two-block mode (``point_then_social``)::

            P0: {"status": "inside", "point_2d": [450, 620]}
            P1: {"status": "outside"}

            {"LAH":[[0,1]],"LAEO":[],"SA":[]}

        Returns:
            ``(target, target_point_prefix)`` — ``target_point_prefix`` is
            ``None`` for single-block modes and a byte-prefix of ``target``
            for ``point_then_category`` / ``point_then_social``. The
            collator uses the prefix only when
            ``vlm.mask_point_block_loss=True`` (default False after the
            drift fix); otherwise LM loss covers both blocks.
        """
        lines = []
        # When per-person LM-loss masking is enabled, emit a uniform line
        # per person so the prompt and target shapes always match (every
        # person mentioned in the prompt gets a ``P{pid}: {...}`` line).
        # Inout in {0, -1} use ``[-1, -1]`` as a sentinel point coord; the
        # ``"unknown"`` status word for inout=-1 persons is a placeholder
        # whose tokens get masked from LM loss in the collator.
        if self.mask_per_person_lm_loss:
            for pid, gp, io in zip(pids, gaze_pts, inout):
                pid_int = int(pid)
                io_val = int(io.item() if torch.is_tensor(io) else io)
                if io_val == 1:
                    mx = int(gp[0].clamp(0, 1) * (self.coord_bins - 1))
                    my = int(gp[1].clamp(0, 1) * (self.coord_bins - 1))
                    lines.append(
                        f'P{pid_int}: {{"status": "inside", "point_2d": [{mx}, {my}]}}'
                    )
                elif io_val == 0:
                    lines.append(
                        f'P{pid_int}: {{"status": "outside", "point_2d": [-1, -1]}}'
                    )
                else:  # io_val == -1
                    lines.append(
                        f'P{pid_int}: {{"status": "unknown", "point_2d": [-1, -1]}}'
                    )
            return "\n".join(lines), None

        for i, (pid, gp, io) in enumerate(zip(pids, gaze_pts, inout)):
            pid_int = int(pid)
            if io == 1:
                mx = int(gp[0].clamp(0, 1) * (self.coord_bins - 1))
                my = int(gp[1].clamp(0, 1) * (self.coord_bins - 1))
                # Include category if available and non-empty
                cat = sem_labels[i] if sem_labels is not None and i < len(sem_labels) else ""
                if cat:
                    lines.append(
                        f'P{pid_int}: {{"status": "inside", "point_2d": [{mx}, {my}], "category": "{cat}"}}'
                    )
                else:
                    lines.append(
                        f'P{pid_int}: {{"status": "inside", "point_2d": [{mx}, {my}]}}'
                    )
            elif io == 0:
                lines.append(f'P{pid_int}: {{"status": "outside"}}')
        return "\n".join(lines), None

    def _compute_social_sets(self, sample, real_mask, annotated_mask):
        """Compute the LAH/LAEO/SA sets for this sample in the
        relabelled 0..V-1 pid space matching the target text.

        Returns an empty-sets dict when the source dataset does not
        populate the relevant label tensors (e.g. GazeFollow has no
        social labels at all).
        """
        from omnigf.vlm.social_labels import social_labels_to_sets

        raw_pids = sample.get("person_ids")
        if raw_pids is None:
            return {"LAH": [], "LAEO": [], "SA": []}

        # Compose the keep mask in the raw_pids index space.
        keep_mask = torch.zeros(raw_pids.shape[0], dtype=torch.bool)
        real_idx = torch.nonzero(real_mask, as_tuple=False).flatten()
        # annotated_mask is over real_mask's indexed space; expand it back.
        ann_in_real = torch.nonzero(annotated_mask, as_tuple=False).flatten()
        for k in ann_in_real.tolist():
            keep_mask[real_idx[k].item()] = True

        def _get(key):
            val = sample.get(key)
            if val is None:
                return None
            # After _select_center_frame, val has shape [1, num_pairs].
            if val.ndim == 2:
                val = val[0]
            return val

        lah = _get("lah_labels")
        laeo = _get("laeo_labels")
        coatt = _get("coatt_labels")
        if lah is None and laeo is None and coatt is None:
            return {"LAH": [], "LAEO": [], "SA": []}

        return social_labels_to_sets(lah, laeo, coatt, raw_pids, keep_mask)


# ---------------------------------------------------------------------------
# VLMGazeDataModule
# ---------------------------------------------------------------------------


class VLMGazeDataModule(pl.LightningDataModule):
    """Lightning data module that wraps :class:`UnifiedGazeFrameDataModule`
    with :class:`VLMGazeDataset` and uses the VLM collator for tokenisation.

    Expects ``cfg.vlm`` to contain at least:

    * ``approach`` — one of ``"lm"``, ``"heatmap"``, ``"patch_dist"``.
    * ``model_name`` — HuggingFace model id (e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``).
    * ``coord_bins`` — quantisation resolution (default 1000).
    * ``unknown_policy`` — ``"exclude"`` or ``"include_marked"``.
    * ``prompt_mode`` — ``"point"`` or ``"point_semantic"`` (semantic prompt is
      selected automatically when semantic labels are enabled).
    * ``max_people`` — maximum people to track positions for.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vlm_cfg = cfg.vlm
        self.approach = vlm_cfg.approach
        self.coord_bins = getattr(vlm_cfg, "coord_bins", 1000)
        self.unknown_policy = getattr(vlm_cfg, "unknown_policy", "exclude")
        self.prompt_mode = getattr(vlm_cfg, "prompt_mode", "point")
        self.max_people = getattr(vlm_cfg, "max_people", 5)
        self.gf_test_all_people = getattr(vlm_cfg, "gf_test_all_people", True)
        self.use_gaze_encoder = getattr(vlm_cfg, "use_gaze_encoder", False)
        # Gaze-token injection strategy (ablation switch): "replace" (default,
        # overwrite the <|gaze_pad|> slot), "add" (add to that slot), or
        # "add_person" (add onto the prompt-side P{pid}: anchor).
        self.gaze_injection = getattr(vlm_cfg, "gaze_injection", "replace")
        # Where the dense heads read each person's hidden state from
        # ("anchor" | "line_last" | "line_mean"). The latter two need the
        # assistant-side per-person span emitted by the collator.
        self.person_extract = getattr(vlm_cfg, "person_extract", "anchor")
        self.mask_point_block_loss = getattr(vlm_cfg, "mask_point_block_loss", False)
        # None -> auto-enable for prompt_mode=point_then_social.
        self.social_gaze_mode = getattr(vlm_cfg, "social_gaze_mode", None)
        # Heads-based pair-label path: emits per-pair lah/laeo/coatt
        # tensors in the relabelled 0..V-1 pid space. Independent of the
        # text-based ``point_then_social`` path; both can be off, one
        # can be on, or both can be on.
        self.use_social_pair_head = getattr(vlm_cfg, "use_social_pair_head", False)
        self.use_all_data = getattr(cfg.train, "use_all_data", False)
        self.batch_size = cfg.train.batch_size
        self.num_workers = getattr(cfg.data, "num_workers", 4)

        # Build dataset_roots dict from config.
        dataset_roots = {}
        for source_name, config in UnifiedGazeFrameDataModule.DATASET_REGISTRY.items():
            root_key = config["root_key"]
            if hasattr(cfg.data, root_key):
                dataset_roots[root_key] = getattr(cfg.data, root_key)

        # The internal unified data module handles building source datasets.
        val_include_sources = getattr(cfg.data, "val_include_sources", None)
        test_include_sources = getattr(cfg.data, "test_include_sources", None)

        # Keep the scalar ``num_people`` at train time (needed as an
        # augmentation cap for variable-size scenes), but use ``"all"`` for
        # val/test so the source-dataset ``num_keep`` truncation at e.g.
        # ``videoattentiontarget_temporal.py:213-219`` does not fire. That
        # truncation (a) picks a random num_keep every __getitem__ and (b)
        # with the H5 layout ``[dummy, real_pids..., aux_pids...]`` keeps
        # the LAST num_keep entries, which silently drops real persons
        # from the front and retains the aux-object pids at the back.
        np_cfg = cfg.data.num_people
        if not isinstance(np_cfg, (dict, DictConfig)):
            np_cfg = {"train": np_cfg, "val": np_cfg, "test": "all"}

        self._inner_dm = UnifiedGazeFrameDataModule(
            dataset_roots=dataset_roots,
            ann_root=cfg.data.ann_root,
            batch_size=1,  # not used directly
            image_size=cfg.model.image_size,
            head_size=getattr(cfg.data, "head_size", (224, 224)),
            heatmap_size=getattr(cfg.data, "heatmap_size", 64),
            num_people=np_cfg,
            include_sources=getattr(cfg.data, "include_sources", {"gazefollow": True}),
            val_include_sources=val_include_sources,
            test_include_sources=test_include_sources,
            gf_include_inout=getattr(cfg.data, "gf_include_inout", False),
            gf_use_semgaze_bbox=getattr(cfg.data, "gf_use_semgaze_bbox", False),
            gf_semgaze_ann_root=getattr(cfg.data, "gf_semgaze_ann_root", None),
            gf_use_semantic_labels=getattr(vlm_cfg, "use_semantic_labels", False),
            gf_semantic_vocab_dir=getattr(vlm_cfg, "semantic_vocab_dir", None),
            pseudo_train_sources=getattr(cfg.data, "pseudo_train_sources", None),
            gazehoi_ann_root=getattr(cfg.data, "gazehoi_ann_root", None),
            gazehoi_vocab2id_path=getattr(cfg.data, "gazehoi_vocab2id_path", None),
            goo_ann_root=getattr(cfg.data, "goo_ann_root", None),
        )

        self.tokenizer = None
        self.processor = None

    def prepare_data(self):
        from transformers import AutoProcessor  # noqa: E402

        model_name = self.cfg.vlm.model_name
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def setup(self, stage: str = "fit"):
        from transformers import AutoProcessor  # noqa: E402
        from omnigf.vlm.collator import VLMGazeCollator  # noqa: E402

        model_name = self.cfg.vlm.model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.padding_side = "left"

            # Add gaze placeholder special token — only for the "replace"
            # injection strategy, which overwrites this token's slot. The
            # additive strategies (add_span / add_anchor) add the gaze
            # embedding onto existing text tokens and never use a placeholder.
            if self.use_gaze_encoder and self.gaze_injection == "replace":
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<|gaze_pad|>"]}
                )
                self.gaze_pad_id = self.tokenizer.convert_tokens_to_ids("<|gaze_pad|>")
                logger.info("Added <|gaze_pad|> token (id=%d)", self.gaze_pad_id)

        # Build internal datasets
        self._inner_dm.setup(stage)

        # Common kwargs for VLMGazeDataset
        use_sem_labels = getattr(self.cfg.vlm, "use_semantic_labels", False)
        ds_kwargs = dict(
            coord_bins=self.coord_bins,
            unknown_policy=self.unknown_policy,
            prompt_mode=self.prompt_mode,
            relabel_ids=True,
            use_gaze_encoder=self.use_gaze_encoder,
            gaze_injection=self.gaze_injection,
            use_semantic_labels=use_sem_labels,
            social_gaze_mode=self.social_gaze_mode,
            extract_pair_labels=self.use_social_pair_head,
            show_no_annotation_marker=getattr(
                self.cfg.vlm, "show_no_annotation_marker", False,
            ),
            sort_persons_lr_tb=getattr(
                self.cfg.vlm, "sort_persons_lr_tb", False,
            ),
            sort_annotated_first=getattr(
                self.cfg.vlm, "sort_annotated_first", False,
            ),
            person_order=getattr(self.cfg.vlm, "person_order", "dataset"),
            person_order_seed=getattr(self.cfg.vlm, "person_order_seed", 0),
            mask_per_person_lm_loss=getattr(
                self.cfg.vlm, "mask_per_person_lm_loss", False,
            ),
        )

        if stage in ("fit",):
            if self.use_all_data:
                base_train = self._inner_dm.train_dataset
                base_val = self._inner_dm.val_dataset
                combined = ConcatDataset([base_train, base_val])
                logger.info(
                    "use_all_data=True: merging train (%d) + val (%d) = %d samples",
                    len(base_train), len(base_val), len(combined),
                )
                self.train_dataset = VLMGazeDataset(
                    combined, split="train", **ds_kwargs,
                )
            else:
                self.train_dataset = VLMGazeDataset(
                    self._inner_dm.train_dataset, split="train", **ds_kwargs,
                )
                self.val_dataset = VLMGazeDataset(
                    self._inner_dm.val_dataset, split="val", **ds_kwargs,
                )
        elif stage == "validate":
            self.val_dataset = VLMGazeDataset(
                self._inner_dm.val_dataset, split="val", **ds_kwargs,
            )
        elif stage == "test":
            base_test = self._inner_dm.test_dataset
            # Single-person ablation: yield one sample per annotated person so
            # the model sees a single ``P0:`` line instead of the whole scene.
            # Person-level metrics stay directly comparable (the same set of
            # persons is scored either way); what changes is the prompt context
            # and the forward-pass count (VAT: 13,127 frames -> 31,978 persons).
            if getattr(self.cfg.vlm, "single_person_eval", False):
                from omnigf.datasets.single_person_view import SinglePersonFrameView
                base_test = SinglePersonFrameView(base_test)
            self.test_dataset = VLMGazeDataset(
                base_test,
                split="test",
                gf_test_all_people=self.gf_test_all_people,
                **ds_kwargs,
            )

        # Build collator — uses the full processor for joint image+text processing
        self.collator = VLMGazeCollator(
            processor=self.processor,
            approach=self.approach,
            max_people=self.max_people,
            use_gaze_encoder=self.use_gaze_encoder,
            mask_point_block_loss=self.mask_point_block_loss,
            mask_per_person_lm_loss=getattr(
                self.cfg.vlm, "mask_per_person_lm_loss", False,
            ),
            prompt_mode=self.prompt_mode,
            gaze_injection=self.gaze_injection,
            person_extract=self.person_extract,
        )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=False,
            drop_last=True,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        from omnigf.vlm.collator import VLMGazeCollator

        # Use a separate collator with no max_people clipping for test
        test_collator = VLMGazeCollator(
            processor=self.processor,
            approach=self.approach,
            max_people=None,
            use_gaze_encoder=self.use_gaze_encoder,
            mask_point_block_loss=self.mask_point_block_loss,
            mask_per_person_lm_loss=getattr(
                self.cfg.vlm, "mask_per_person_lm_loss", False,
            ),
            prompt_mode=self.prompt_mode,
            gaze_injection=self.gaze_injection,
            person_extract=self.person_extract,
        )


        test_batch_size = getattr(self.cfg.test, "batch_size", 1)
        multi_gpu = torch.cuda.device_count() > 1
        return DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=test_collator,
            pin_memory=False,
            drop_last=multi_gpu,
        )
