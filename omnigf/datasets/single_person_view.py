# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Single-person view over a multi-person frame dataset.

Ablation: our model normally receives every person in one prompt and emits one
answer line each (single-pass, multi-person). This wrapper instead yields ONE
sample per annotated person, so the model sees a single ``P0:`` line at a time
-- the per-person protocol used by single-person baselines (e.g. gazelle /
Gaze-LLE, which evaluate one (image, head bbox) pair per annotation).

Two things this is useful for:

1. Accuracy: does removing the multi-person context help or hurt? Multi-person
   prompting could help (scene context, mutual constraints) or hurt (harder
   task, longer generation, cross-person interference).
2. Runtime: it is the empirical per-person baseline cost. VAT test has 13,127
   frames but 31,978 annotated persons, so single-person evaluation costs
   ~2.44x the forward passes -- the ``N passes vs 2 passes`` contrast in the
   scalability analysis, measured rather than modelled.

Nothing downstream needs to change: ``VLMGazeDataset`` already handles V=1
(GazeFollow test does exactly this via ``gf_test_all_people=False``, which
keeps ``real_heads[-1:]``), relabels pids to ``0..V-1``, and the collator
already drops ``None`` samples.

Person layout in the source H5 (verified on ``vat_test.h5``)::

    person_ids : [  -1    0    1    2    3 1001]
    inout      : [-1.   1.   1.   1.   1.   -1.]

index 0 is a dummy, trailing ids >= 1000 are auxiliary target objects, and the
annotated persons are exactly the ``inout != -1`` entries -- the same rule
``VLMGazeDataset`` applies via its ``annotated_mask``.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Keys whose person axis is dim 1 (dim 0 is the temporal axis, T=1 after
# center-frame selection). Anything not listed and not detected by the
# generic shape check below is passed through untouched.
_PERSON_DIM1_KEYS = (
    "heads", "head_centers", "head_masks", "head_bboxes", "inout",
    "gaze_pts", "gaze_vecs", "gaze_heatmaps", "speaking", "is_child",
    "visibility", "is_pseudo",
)
# Per-pair labels laid out as P*(P-1) permutations; a single person has no
# pairs, so these become width 0.
_PAIR_KEYS = ("lah_labels", "laeo_labels", "coatt_labels")
# Flat per-person vectors with no temporal axis.
_PERSON_DIM0_KEYS = ("person_ids", "pids")


def build_single_person_index(unified) -> List[Tuple[int, int]]:
    """Build ``[(frame_index, annotated_ordinal), ...]`` for a unified dataset.

    Counts come from each source dataset's already-loaded pandas annotations
    (one row per frame, array-valued columns), so this costs well under a
    second for VAT -- no image decoding. Reading the source's own
    ``annotations`` rather than re-opening the H5 guarantees the counts match
    the data the dataset will actually serve, and avoids guessing filenames.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for src_name, src in unified.datasets.items():
        ann = getattr(src, "annotations", None)
        if ann is None or not hasattr(ann, "__iter__"):
            logger.warning(
                "Source '%s' exposes no groupby 'annotations'; falling back to "
                "per-frame probing for it.", src_name,
            )
            counts[src_name] = {}
            continue
        per_path = {}
        for path, group in ann:
            io = np.asarray(group["inout"].iloc[0]).reshape(-1)
            per_path[path] = int((io != -1).sum())
        counts[src_name] = per_path

    index: List[Tuple[int, int]] = []
    n_missing = 0
    for j, (src_name, src_i) in enumerate(unified.index):
        src = unified.datasets[src_name]
        paths = getattr(src, "paths", None)
        n = None
        if paths is not None and src_i < len(paths):
            n = counts.get(src_name, {}).get(paths[src_i])
        if n is None:
            # Unknown count -> keep a single slot; __getitem__ returns None if
            # that person does not exist, and the collator drops it.
            n_missing += 1
            n = 1
        for k in range(n):
            index.append((j, k))

    if n_missing:
        logger.warning("%d frames had no cheap person count available", n_missing)
    return index


class SinglePersonFrameView(Dataset):
    """Explodes each frame into one sample per annotated person.

    Sits between the unified frame dataset and ``VLMGazeDataset``.
    """

    def __init__(self, base, index: List[Tuple[int, int]] = None):
        self.base = base
        self.index = index if index is not None else build_single_person_index(base)
        logger.info(
            "SinglePersonFrameView: %d frames -> %d single-person samples "
            "(%.2fx forward passes)",
            len(base), len(self.index),
            len(self.index) / max(len(base), 1),
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        frame_idx, ordinal = self.index[i]
        sample = self.base[frame_idx]
        if sample is None:
            return None

        inout = sample.get("inout")
        if inout is None:
            return None
        io = inout[0].reshape(-1) if inout.ndim >= 2 else inout.reshape(-1)
        annotated = torch.nonzero(io != -1, as_tuple=False).flatten()
        if ordinal >= annotated.numel():
            # Person count changed vs. the cheap index (or index fallback);
            # the collator drops None samples.
            return None
        slot = int(annotated[ordinal])
        n_persons = int(io.numel())

        out = {}
        for key, val in sample.items():
            if not torch.is_tensor(val):
                out[key] = val
                continue
            if key in _PAIR_KEYS:
                # P*(P-1) permutations -> 1 person has none.
                out[key] = val[..., :0] if val.ndim >= 2 else val[:0]
                continue
            if key in _PERSON_DIM0_KEYS:
                out[key] = val[slot:slot + 1]
                continue
            if key in _PERSON_DIM1_KEYS or (
                val.ndim >= 2 and val.shape[1] == n_persons
            ):
                out[key] = val[:, slot:slot + 1]
                continue
            out[key] = val

        if "num_valid_people" in out and torch.is_tensor(out["num_valid_people"]):
            out["num_valid_people"] = torch.ones_like(out["num_valid_people"])
        return out
