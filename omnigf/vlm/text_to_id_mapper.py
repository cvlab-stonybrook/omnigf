# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Standalone text -> test-vocab-ID mapping (CLIP template-based).

Replicates ``SemanticLoRAModel._text_to_test_id(use_clip=True)`` without
requiring the full Qwen VLM to be loaded — only CLIP text encoder +
projection plus the precomputed L2-normalised test vocab embeddings.

Exposes both top-1 (``text_to_id``) and top-k similarities
(``text_to_sims``) so callers can compute acc@1 and acc@3 consistently
with the training-time metric.

An optional WordNet hypernym reroute (``use_wordnet=True``) reranks CLIP
top-k candidates: if any of the predicted category's tokens has one of
the top-k candidate labels in its hypernym chain, that candidate wins
over the raw CLIP argmax.  Concrete effect on our GazeFollow vocab:

  - ``baseball``         -> ``ball``   (CLIP alone: baseball bat)
  - ``baseball player``  -> ``person`` (CLIP alone: baseball glove)
  - ``runner``           -> ``person`` (CLIP alone: person too, unchanged)
  - ``child``/``man``... -> ``person`` (CLIP alone: person too, unchanged)

It does NOT help cases where the relation is not a WordNet hypernym,
e.g. ``baseball team`` -> ``person`` (team is a social-group, not a
hyponym of person).
"""

from typing import Dict, List, Optional

import json

import torch
import torch.nn.functional as F


# Must match the template used to build the vocab embeddings
# (see scripts/regenerate_clip_embeddings.py) and the runtime encoders
# in omnigf/vlm/models_ge.py.
CLIP_TEXT_TEMPLATE = "a photo of a {}"


# Default hard-suffix rules used by both TextToTestIdMapper and the
# Lightning model-side reroute (SemanticLoRAModel / VLMGazeModelGE).
# Keyed by the query's last word (lowercased); value is the target test
# label. Applies only to phrases with >= 2 words.
DEFAULT_HARD_SUFFIX_RULES: Dict[str, str] = {
    "team": "person",
}


# Default hard-word rules: direct remap of a full (single- or multi-word)
# query string to a target label. Applied before WordNet / CLIP. Useful
# for generic-to-specific synonyms absent from both the test vocab and
# WordNet's hypernym graph (e.g. the GazeFollow vocab contains "laptop"
# but not "computer").
DEFAULT_HARD_WORD_RULES: Dict[str, str] = {
    "computer": "laptop",
}


# ---------------------------------------------------------------------------
# Standalone helpers (usable without a TextToTestIdMapper instance so the
# Lightning models can apply the same reroute during val/test).
# ---------------------------------------------------------------------------

_WORDNET_DEPTH_CACHE: Dict[str, Dict[str, int]] = {}


def ensure_wordnet() -> None:
    """Lazy-download WordNet the first time it's needed."""
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("cat")  # force load
    except LookupError:
        nltk.download("wordnet", quiet=True)


def wordnet_depth_map(word: str) -> Dict[str, int]:
    """Return {ancestor: min_hypernym_hops} across every NOUN sense of ``word``."""
    if word in _WORDNET_DEPTH_CACHE:
        return _WORDNET_DEPTH_CACHE[word]
    from nltk.corpus import wordnet as wn
    depths: Dict[str, int] = {word: 0}
    for syn in wn.synsets(word, pos=wn.NOUN):
        for path in syn.hypernym_paths():
            leaf_idx = len(path) - 1
            for k, s in enumerate(path):
                dist = leaf_idx - k
                for lemma in s.lemmas():
                    name = lemma.name().lower().replace("_", " ")
                    if name not in depths or dist < depths[name]:
                        depths[name] = dist
    _WORDNET_DEPTH_CACHE[word] = depths
    return depths


def wordnet_reroute_id(
    text: str,
    sims,  # torch.Tensor [V]
    labels: List[str],
    top_k: int = 10,
) -> Optional[int]:
    """Pick the most-specific WordNet hypernym candidate in CLIP top-k.

    Among top-k candidates whose label appears in the query's union
    hypernym set, return the one with the smallest hypernym distance
    (CLIP rank is only a tie-breaker). Returns None if none qualify.
    """
    query_words = [w for w in text.split() if w]
    if not query_words:
        return None
    depth_map: Dict[str, int] = {}
    for w in query_words:
        for anc, d in wordnet_depth_map(w).items():
            if anc not in depth_map or d < depth_map[anc]:
                depth_map[anc] = d
    if not depth_map:
        return None
    k = min(top_k, sims.numel())
    top_idx = sims.topk(k).indices.tolist()
    best_id: Optional[int] = None
    best_key = None  # (depth, clip_rank) — smaller is better
    for rank, i in enumerate(top_idx):
        cand = labels[i].lower()
        if cand in depth_map:
            key = (depth_map[cand], rank)
            if best_key is None or key < best_key:
                best_key = key
                best_id = int(i)
    return best_id


def resolve_hard_suffix_rules(
    rules: Dict[str, str],
    label_to_id: Dict[str, int],
) -> Dict[str, int]:
    """Resolve {last_word: target_label} to {last_word: target_id},
    silently dropping rules whose target isn't in the vocab."""
    out: Dict[str, int] = {}
    for last_word, target in rules.items():
        tid = label_to_id.get(target)
        if tid is not None:
            out[last_word.lower()] = tid
    return out


def hard_suffix_id(text: str, hard_rule_ids: Dict[str, int]) -> Optional[int]:
    """Return hard-rule target ID if ``text`` is a >=2-word phrase whose
    last token matches a rule key."""
    if not hard_rule_ids:
        return None
    words = text.split()
    if len(words) < 2:
        return None
    return hard_rule_ids.get(words[-1])


def hard_word_id(text: str, hard_rule_ids: Dict[str, int]) -> Optional[int]:
    """Return hard-rule target ID if the full normalised ``text`` matches
    a rule key exactly (single- or multi-word phrase equality)."""
    if not hard_rule_ids:
        return None
    return hard_rule_ids.get(text)


class TextToTestIdMapper:
    """Map a free-text category into a test-vocab ID (with CLIP fallback)."""

    # Module-level default; class alias retained for callers that used it.
    DEFAULT_HARD_SUFFIX_RULES = DEFAULT_HARD_SUFFIX_RULES

    def __init__(
        self,
        test_vocab_embeddings_path: str,
        test_vocab_labels_path: str,
        device: torch.device,
        clip_name: str = "openai/clip-vit-base-patch16",
        template: str = CLIP_TEXT_TEMPLATE,
        use_wordnet: bool = False,
        wordnet_top_k: int = 10,
        hard_suffix_rules: Optional[Dict[str, str]] = None,
        hard_word_rules: Optional[Dict[str, str]] = None,
    ):
        from transformers import CLIPModel, CLIPTokenizerFast

        self.device = device
        self.template = template

        self.test_vocab_emb = torch.load(
            test_vocab_embeddings_path, map_location=device, weights_only=False,
        )
        with open(test_vocab_labels_path) as f:
            self.labels: List[str] = json.load(f)
        self.label_to_id: Dict[str, int] = {l: i for i, l in enumerate(self.labels)}

        _full = CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        self.clip_text = _full.text_model.eval().to(device)
        self.clip_proj = _full.text_projection.eval().to(device)
        self.clip_tok = CLIPTokenizerFast.from_pretrained(clip_name)
        for p in self.clip_text.parameters():
            p.requires_grad = False
        for p in self.clip_proj.parameters():
            p.requires_grad = False

        # Cache full [V] similarity vectors keyed by normalised text.
        self._sim_cache: Dict[str, torch.Tensor] = {}

        # Optional WordNet-based reroute over CLIP top-k.
        self.use_wordnet = use_wordnet
        self.wordnet_top_k = wordnet_top_k
        if self.use_wordnet:
            ensure_wordnet()

        # Hard suffix rules (e.g. "X team" -> person).
        suffix_rules = (
            DEFAULT_HARD_SUFFIX_RULES
            if hard_suffix_rules is None
            else hard_suffix_rules
        )
        self._hard_suffix_ids = resolve_hard_suffix_rules(
            suffix_rules, self.label_to_id,
        )

        # Hard word rules (e.g. "computer" -> "laptop").
        word_rules = (
            DEFAULT_HARD_WORD_RULES
            if hard_word_rules is None
            else hard_word_rules
        )
        self._hard_word_ids = resolve_hard_suffix_rules(
            word_rules, self.label_to_id,
        )

    def _wordnet_reroute(
        self,
        text: str,
        sims: torch.Tensor,
    ) -> Optional[int]:
        if not self.use_wordnet:
            return None
        return wordnet_reroute_id(text, sims, self.labels, self.wordnet_top_k)

    @torch.no_grad()
    def _clip_encode(self, text: str) -> torch.Tensor:
        prompt = self.template.format(text)
        inputs = self.clip_tok(
            [prompt], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        out = self.clip_text(**inputs)
        emb = self.clip_proj(out.pooler_output)
        return F.normalize(emb, dim=-1).squeeze(0)

    def _ensure_sims(self, text: str) -> Optional[torch.Tensor]:
        """Return (and cache) the [V] similarity vector for ``text``."""
        if not text:
            return None
        key = text.strip().lower()
        if not key:
            return None
        if key in self._sim_cache:
            return self._sim_cache[key]
        emb = self._clip_encode(key)
        sims = emb @ self.test_vocab_emb.to(emb.dtype).T
        self._sim_cache[key] = sims
        return sims

    def _hard_suffix_id(self, text: str) -> Optional[int]:
        return hard_suffix_id(text, self._hard_suffix_ids)

    def text_to_id(self, text: Optional[str]) -> int:
        """Top-1 ID: hard word rule -> exact vocab hit -> hard suffix rule
        -> WordNet reroute -> CLIP argmax.

        Hard word rules run first so they can override a same-named vocab
        entry (e.g. GazeFollow has both "computer" and "laptop" as
        distinct classes — remapping merges them).
        """
        if not text:
            return -1
        key = text.strip().lower()
        if not key:
            return -1
        # Hard word rule (override; may shadow a same-named vocab entry).
        word = hard_word_id(key, self._hard_word_ids)
        if word is not None:
            self._ensure_sims(key)
            return word
        # Populate the sim cache either way so acc@3 stays consistent.
        if key in self.label_to_id:
            self._ensure_sims(key)
            return self.label_to_id[key]
        hard = self._hard_suffix_id(key)
        if hard is not None:
            self._ensure_sims(key)
            return hard
        sims = self._ensure_sims(key)
        if sims is None:
            return -1
        if self.use_wordnet:
            rid = self._wordnet_reroute(key, sims)
            if rid is not None:
                return rid
        return int(sims.argmax().item())

    def text_to_sims(self, text: Optional[str]) -> Optional[torch.Tensor]:
        """Full [V] similarity vector used for acc@3."""
        if not text:
            return None
        return self._ensure_sims(text)
