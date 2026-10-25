# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Helpers that convert MTGS per-pair social-gaze labels to/from the
human-readable set form emitted by the VLM social-gaze target.

MTGS-side layout (per sample, post-center-frame selection):

* ``person_ids``  : ``[n]`` raw ids, with ``-1`` entries for padding.
* ``lah_labels``  : ``[num_pairs]`` of ``{0, 1, -1}``, indexed by
  ``itertools.permutations(arange(n), 2)``.
* ``laeo_labels`` : same shape / pair indexing.
* ``coatt_labels``: same shape / pair indexing (1 iff the two persons are
  coattending the same target).

VLM-side target layout::

    {"LAH":  [[sender, receiver], ...],   # ordered, relabelled to 0..V-1
     "LAEO": [[i, j], ...],               # unordered, i<j
     "SA":   [[a, b, c], [d, e], ...]}    # groups of >=2 people

Relabelling: raw ``person_ids[i]`` is mapped to a contiguous ``0..V-1``
index over the V **kept** persons (those passing ``keep_mask``). This
matches the ``relabelled_pids = torch.arange(n_annotated)`` convention
in :class:`omnigf.vlm.dataset.VLMGazeDataset`.
"""

from __future__ import annotations

import itertools
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SocialSets = Dict[str, List[List[int]]]


# ---------------------------------------------------------------------------
# MTGS per-pair vectors -> human-readable sets
# ---------------------------------------------------------------------------


def _pair_indices(n: int) -> List[Tuple[int, int]]:
    return list(itertools.permutations(range(n), 2))


def _build_old_to_new(keep_mask: torch.Tensor) -> torch.Tensor:
    """Return a ``[n]`` long tensor mapping old idx -> new idx (``-1`` if
    ``keep_mask[i]`` is False)."""
    n = int(keep_mask.shape[0])
    old_to_new = torch.full((n,), -1, dtype=torch.long)
    nxt = 0
    for i in range(n):
        if bool(keep_mask[i]):
            old_to_new[i] = nxt
            nxt += 1
    return old_to_new


def _connected_components(
    nodes: Iterable[int], edges: Iterable[Tuple[int, int]],
) -> List[List[int]]:
    nodes = list(nodes)
    adj: Dict[int, List[int]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    seen: set = set()
    comps: List[List[int]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp: List[int] = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x])
        comps.append(sorted(comp))
    return comps


def social_labels_to_sets(
    lah: Optional[torch.Tensor],
    laeo: Optional[torch.Tensor],
    coatt: Optional[torch.Tensor],
    raw_pids: torch.Tensor,
    keep_mask: torch.Tensor,
) -> SocialSets:
    """Convert MTGS-style per-pair label vectors to the VLM set format.

    Args:
        lah, laeo, coatt: each ``[num_pairs]`` of ``{0, 1, -1}``, where
            ``num_pairs = len(raw_pids) * (len(raw_pids) - 1)``. Pass
            ``None`` for a channel that is not present in the source
            dataset (e.g. VAT has no ``laeo_labels`` populated for every
            split). Tensors may be on any device; will be moved to CPU.
        raw_pids: ``[n]`` the per-sample ``person_ids`` (padding entries
            allowed). Used only for its length / keep_mask alignment;
            the actual ``person_ids`` values are discarded because the
            VLM target uses relabelled ``0..V-1`` pids.
        keep_mask: ``[n]`` boolean. The kept persons get new indices
            ``0..V-1`` in order of appearance; filtered-out persons (e.g.
            padding or unannotated) are dropped.

    Returns:
        ``{"LAH": [...], "LAEO": [...], "SA": [...]}`` — sorted, deduped
        lists of int pairs/groups indexed over ``0..V-1``.

    Semantics:
        * LAH is **directed**. Both ``[i,j]`` and ``[j,i]`` can appear.
        * LAEO is **symmetric and deduped** to ``i<j`` ordering. If
          either ``(i,j)`` or ``(j,i)`` is labelled 1, the pair is
          emitted once.
        * SA is derived from ``coatt_labels`` by connected components.
          Edge ``(i,j)`` exists iff ``coatt_labels`` is 1 in either
          ordering. Singletons are dropped (only groups of >=2 returned).
    """
    n = int(raw_pids.shape[0])
    expected_num_pairs = n * (n - 1)
    # Bring everything to CPU 1-D.
    def _prep(x):
        if x is None:
            return None
        x = x.detach().cpu()
        if x.ndim == 2:
            x = x[0]
        return x

    lah = _prep(lah)
    laeo = _prep(laeo)
    coatt = _prep(coatt)
    for name, vec in (("lah", lah), ("laeo", laeo), ("coatt", coatt)):
        if vec is None:
            continue
        if int(vec.shape[0]) != expected_num_pairs:
            raise ValueError(
                f"{name}_labels length {int(vec.shape[0])} does not match "
                f"expected num_pairs={expected_num_pairs} for n={n} persons"
            )

    pair_idx = _pair_indices(n)
    old_to_new = _build_old_to_new(keep_mask)

    lah_set: set = set()
    laeo_sym: set = set()  # unordered canonical (a<b)
    coatt_edges: List[Tuple[int, int]] = []

    for k, (i, j) in enumerate(pair_idx):
        ni = int(old_to_new[i].item())
        nj = int(old_to_new[j].item())
        if ni < 0 or nj < 0:
            continue

        if lah is not None and int(lah[k].item()) == 1:
            lah_set.add((ni, nj))

        if laeo is not None and int(laeo[k].item()) == 1:
            a, b = (ni, nj) if ni < nj else (nj, ni)
            laeo_sym.add((a, b))

        if coatt is not None and int(coatt[k].item()) == 1:
            a, b = (ni, nj) if ni < nj else (nj, ni)
            coatt_edges.append((a, b))

    v = int(keep_mask.sum().item())
    sa_groups = _connected_components(range(v), coatt_edges)
    sa_groups = [g for g in sa_groups if len(g) >= 2]

    return {
        "LAH": [list(p) for p in sorted(lah_set)],
        "LAEO": [list(p) for p in sorted(laeo_sym)],
        "SA": sa_groups,
    }


# ---------------------------------------------------------------------------
# MTGS per-pair vectors -> relabelled per-pair vectors
# ---------------------------------------------------------------------------


def social_labels_to_pair_tensors(
    lah: Optional[torch.Tensor],
    laeo: Optional[torch.Tensor],
    coatt: Optional[torch.Tensor],
    raw_pids: torch.Tensor,
    keep_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert MTGS-style per-pair label vectors to the relabelled
    ``0..V-1`` pid space, preserving the per-pair tensor format used by
    :class:`omnigf.vlm.heads.SocialPairHead` and the
    ``compute_social_pair_bce_loss`` loss.

    Args:
        lah, laeo, coatt: each ``[num_pairs]`` of ``{0, 1, -1}``, where
            ``num_pairs = len(raw_pids) * (len(raw_pids) - 1)``. ``None``
            for missing channels (e.g. VAT has no LAEO populated).
        raw_pids: ``[n]`` source-positions vector (padding allowed).
        keep_mask: ``[n]`` boolean mask of kept positions.

    Returns:
        ``{"lah": [V*(V-1)], "laeo": [V*(V-1)], "coatt": [V*(V-1)]}``,
        each a 1-D float tensor in ``{-1.0, 0.0, 1.0}``. Pairs are
        ordered to match :func:`itertools.permutations(range(V), 2)`,
        which is the same convention used in ``SocialPairHead.forward``.

        Returns empty 0-d tensors when ``V < 2`` (no pairs).
    """
    n = int(raw_pids.shape[0])
    v = int(keep_mask.sum().item())
    if v < 2:
        empty = torch.empty(0, dtype=torch.float32)
        return {"lah": empty.clone(), "laeo": empty.clone(), "coatt": empty.clone()}

    pair_idx_old = _pair_indices(n)
    old_pair_to_idx = {pair: k for k, pair in enumerate(pair_idx_old)}

    old_to_new = _build_old_to_new(keep_mask)
    # Inverse map: new position -> old position.
    new_to_old = [-1] * v
    for old_i in range(n):
        ni = int(old_to_new[old_i].item())
        if ni >= 0:
            new_to_old[ni] = old_i

    pair_idx_new = _pair_indices(v)
    p = len(pair_idx_new)

    def _build(src):
        if src is None:
            return torch.full((p,), -1.0)
        out = torch.full((p,), -1.0)
        for k, (ni, nj) in enumerate(pair_idx_new):
            old_pair = (new_to_old[ni], new_to_old[nj])
            old_k = old_pair_to_idx[old_pair]
            out[k] = float(src[old_k].item())
        return out

    return {
        "lah": _build(lah),
        "laeo": _build(laeo),
        "coatt": _build(coatt),
    }


# ---------------------------------------------------------------------------
# Set formatting / parsing helpers (shared by dataset target and parser)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Set <-> ordered-pair expansion (for set-based P/R/F1 and MTGS comparison)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Partition view (for SA-specific ARI)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MTGS logit matrices -> partition (for the MTGS-side SA ARI comparison)
# ---------------------------------------------------------------------------
