# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""VLM gaze model with GazeEncoder injection.

:class:`VLMGazeModelGE` is the main VLM-based gaze prediction Lightning module.
It injects per-person gaze tokens (from a ResNet18-based GazeEncoder) into the
VLM's token sequence via a forward pre-hook on the first transformer layer.

The gaze encoder takes head crops and bounding boxes, produces one gaze token
per person, and replaces the ``<|gaze_pad|>`` placeholder tokens that were
inserted into the prompt by :class:`~omnigf.vlm.dataset.VLMGazeDataset`.

Supports two approach variants via ``cfg.vlm.approach``:

* ``"lm"`` — Approach 1: language-modelling loss only.
* ``"heatmap"`` — LM loss + heatmap decoder (MSE) + BCE in/out head.
"""

import json
import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torchmetrics import Accuracy

from omnigf.networks.gaze_encoder import GazeEncoder
from omnigf.performance.metrics import Distance, AUC, GFTestDistance, GFTestAUC
from omnigf.vlm.evaluation import extract_predictions
from omnigf.vlm.heads import (
    InOutHead,
    SocialPairHead,
)
from omnigf.vlm.losses import (
    compute_inout_bce_loss, compute_heatmap_mse_loss,
    compute_social_pair_bce_loss,
)
from omnigf.vlm.decoders import (
    ConvTransposeHeatmapDecoder,
)
from omnigf.vlm.parser import MultiPersonOutputParser
from omnigf.vlm.metrics import MultiAccuracy

logger = logging.getLogger(__name__)


# Template used when encoding categories for CLIP-based mapping to the
# test vocabulary. Must match the template used to build
# ``test_vocab_embeddings.pt`` (see ``scripts/regenerate_clip_embeddings.py``).
CLIP_TEXT_TEMPLATE = "a photo of a {}"


class VLMGazeModelGE(pl.LightningModule):
    """VLM-based gaze prediction model with GazeEncoder injection.

    Configuration namespace ``cfg.vlm`` should contain:

    * ``approach``: ``"lm"`` | ``"heatmap"``
    * ``model_name``: HuggingFace model id
    * ``lora_rank``: LoRA rank (default 16)
    * ``lora_alpha``: LoRA alpha (default 32)
    * ``lora_target_modules``: list of module name patterns
    * ``decoder_type``: ``"conv_transpose"``
    * ``lambda_lm``, ``lambda_bce``, ``lambda_hm``: loss weights
    * ``use_gaze_encoder``: inject gaze encoder features per person
    * ``gaze_weights``: path to Gaze360 ResNet18 checkpoint
    """

    APPROACHES = ("lm", "heatmap")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vlm = cfg.vlm
        self.approach = vlm.approach
        assert self.approach in self.APPROACHES

        # ---- Loss weights ----
        self.lambda_lm = getattr(vlm, "lambda_lm", 1.0)
        self.lambda_bce = getattr(vlm, "lambda_bce", 2.0)
        self.lambda_hm = getattr(vlm, "lambda_hm", 1.0)
        # Per-person scale applied to heatmap+inout BCE losses for persons
        # whose annotation came from the MTGS pseudo-fill pass (``is_pseudo``
        # column in the source h5). Default 1.0 = no effect; set <1.0 to
        # down-weight pseudo persons relative to GT.
        self.pseudo_loss_scale = float(getattr(vlm, "pseudo_loss_scale", 1.0))

        # ---- Generation ----
        # Upper bound on autoregressive decoding length at val/test. Each
        # decoded token is a full VLM forward pass, so tightening this is
        # a linear latency win. See ``vlm_config.yaml`` for sensible
        # per-prompt-mode ceilings.
        self.max_new_tokens = int(getattr(vlm, "max_new_tokens", 128))

        # ---- Prompt mode ----
        self.prompt_mode = getattr(vlm, "prompt_mode", "point")

        # ---- LLM hidden-state layer for the dense heads ----
        # Which layer of ``outputs.hidden_states`` the heatmap / in-out /
        # semantic / social heads read, for BOTH the person-token and the
        # vision-token gathers. ``-1`` (default) is the final layer, which is
        # also what the LM head reads — so the person anchor's state is the
        # one that produced the emitted ``point_2d`` and the dense heads stay
        # aligned with the text prediction. Earlier layers decouple them.
        #
        # Indexing follows HF: ``hidden_states[0]`` is the embedding output and
        # ``hidden_states[i]`` (i >= 1) is the output of transformer layer i,
        # so for a 36-layer model ``-1`` == ``36``. Negative indices work.
        #
        # NOTE: this is NOT an eval-time knob. The decoder is trained against
        # one layer's feature distribution, so every value needs its own
        # training run; loading a checkpoint under a different ``hidden_layer``
        # produces meaningless numbers, not an ablation.
        self.hidden_layer = int(getattr(vlm, "hidden_layer", -1))

        # ---- Load VLM backbone ----
        self.model = self._load_vlm(vlm.model_name)
        cfg_obj = self.model.config
        hidden_dim = (
            getattr(cfg_obj, "hidden_size", None)
            or cfg_obj.text_config.hidden_size
        )

        # Validate ``hidden_layer`` against the real depth now that the model
        # is loaded. ``hidden_states`` has num_layers + 1 entries (index 0 is
        # the embedding output), so valid values are [-(L+1), L].
        n_layers = (
            getattr(cfg_obj, "num_hidden_layers", None)
            or cfg_obj.text_config.num_hidden_layers
        )
        if not (-(n_layers + 1) <= self.hidden_layer <= n_layers):
            raise ValueError(
                f"vlm.hidden_layer={self.hidden_layer} is out of range for a "
                f"{n_layers}-layer model (valid: {-(n_layers + 1)}..{n_layers})"
            )
        logger.info(
            "Dense heads read hidden_states[%d] (model has %d layers)",
            self.hidden_layer, n_layers,
        )

        # ---- Apply LoRA ----
        self._apply_lora(vlm)

        # ---- Approach-specific heads ----
        if self.approach == "heatmap":
            self.inout_head = InOutHead(hidden_dim)
            self.heatmap_decoder = self._build_heatmap_decoder(vlm, hidden_dim)

        # ---- Social-pair head (MTGS-style, orthogonal to gaze approach) ----
        # Reads the same per-person hidden states the gaze heads consume
        # and emits 3 logits per directed pair (LAH / LAEO / coatt).
        # Trained jointly via BCE losses with category-specific
        # ``pos_weight`` matching MTGS defaults; evaluated MTGS-style
        # (per-gazer LAH F1, per-pair LAEO F1 with per-gazer argmax,
        # per-pair coatt F1).
        self.use_social_pair_head = bool(getattr(vlm, "use_social_pair_head", False))
        if self.use_social_pair_head:
            self.social_pair_head = SocialPairHead(
                hidden_dim=hidden_dim,
                mid_dim=int(getattr(vlm, "social_pair_mid_dim", 256)),
                dropout=float(getattr(vlm, "social_pair_dropout", 0.1)),
            )
            self.social_pos_weight_lah = float(getattr(vlm, "social_pos_weight_lah", 3.0))
            self.social_pos_weight_laeo = float(getattr(vlm, "social_pos_weight_laeo", 2.0))
            self.social_pos_weight_coatt = float(getattr(vlm, "social_pos_weight_coatt", 2.0))
            self.lambda_social = float(getattr(vlm, "lambda_social", 1.0))
            self.social_f1_threshold = float(getattr(vlm, "social_f1_threshold", 0.5))

        # ---- Fallback embedding for persons not found in generated text ----
        if self.approach == "heatmap":
            self.fallback_person_embed = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * 0.02
            )

        # ---- Gaze encoder (ResNet18 head crop encoder) ----
        # Only built when ``use_gaze_encoder`` is enabled — the dataset /
        # collator / optimizer all branch on the same flag, and the
        # ``<|gaze_pad|>`` embedding row must only exist when the tokenizer
        # has the matching token (otherwise the vocab size diverges from
        # checkpoints saved without the GE).
        self.use_gaze_encoder = bool(getattr(vlm, "use_gaze_encoder", False))
        # ---- Gaze-token injection strategy (ablation switch) ----
        #   "replace"    — overwrite the <|gaze_pad|> slot with the gaze token
        #                  (default / original behaviour).
        #   "add_span"   — add the gaze token to every prompt token of the
        #                  person's line (``P{pid}: <|gaze_pad|> {bbox}``).
        #   "add_anchor" — add the gaze token onto the single prompt-side
        #                  ``P{pid}:`` anchor token.
        # All modes keep the <|gaze_pad|> token in the prompt, so the prompt is
        # identical across modes. ``gaze_add_scale`` scales the added token in
        # the additive modes (raw add by default; lower it if gaze/text norms
        # mismatch).
        self.gaze_injection = getattr(vlm, "gaze_injection", "replace")
        assert self.gaze_injection in ("replace", "add_span", "add_anchor"), (
            f"vlm.gaze_injection={self.gaze_injection!r} invalid"
        )
        self.gaze_add_scale = float(getattr(vlm, "gaze_add_scale", 1.0))

        # ---- Person-token extraction position (ablation switch) ----
        # Where the dense heads (heatmap decoder / in-out / social / semantic)
        # read each person's representation from:
        #   "anchor"    — hidden state at the assistant-side ``P{pid}:`` token
        #                 (default / original behaviour). Sees the full prompt
        #                 and previous persons' answers, but NOT this person's
        #                 own predicted coordinates.
        #   "line_last" — last token of the person's answer line; additionally
        #                 sees this person's own emitted coordinates.
        #   "line_mean" — mean-pool over the person's answer-line tokens.
        # NOTE: under teacher forcing the answer tokens are ground truth, while
        # at inference they are the model's own predictions — so the two
        # line_* modes carry an exposure-bias risk. Judge them on test metrics,
        # not on training loss.
        self.person_extract = getattr(vlm, "person_extract", "anchor")
        assert self.person_extract in ("anchor", "line_last", "line_mean"), (
            f"vlm.person_extract={self.person_extract!r} invalid"
        )
        if self.use_gaze_encoder:
            gaze_feature_dim = getattr(vlm, "gaze_feature_dim", 512)
            self.gaze_encoder = GazeEncoder(
                token_dim=hidden_dim, feature_dim=gaze_feature_dim,
            )

            # The ResNet-18 backbone starts from Gaze360 pre-training; the
            # projection heads are randomly initialised and trained.
            gaze_weights = getattr(vlm, "gaze_weights", None)
            if not gaze_weights:
                raise ValueError(
                    "vlm.gaze_weights must be set when use_gaze_encoder=True. "
                    "Provide the path to a Gaze360 pre-trained checkpoint."
                )
            self._load_gaze_weights(gaze_weights)

            # ---- Freeze the pre-trained backbone (ablation) ----
            # Freezes only the ResNet18 backbone; the projection heads still
            # train (they are random-init and would be meaningless frozen).
            # ``configure_optimizers`` skips ``requires_grad=False`` params.
            self.freeze_gaze_backbone = bool(
                getattr(vlm, "freeze_gaze_backbone", False)
            )
            if self.freeze_gaze_backbone:
                for p in self.gaze_encoder.backbone.parameters():
                    p.requires_grad = False
                self.gaze_encoder.backbone.eval()
                logger.info("Froze GazeEncoder backbone")

            # Resize token embeddings for the new <|gaze_pad|> token — only the
            # "replace" strategy adds that token to the tokenizer. The additive
            # strategies (add_span / add_anchor) add the gaze embedding onto
            # existing text tokens and introduce no new vocab, so the embedding
            # table (and thus the checkpoint vocab size) stays at the base size.
            if self.gaze_injection == "replace":
                self.model.resize_token_embeddings(
                    self.model.get_input_embeddings().num_embeddings + 1
                )

            logger.info(
                "GazeEncoder enabled (feature_dim=%d, hidden_dim=%d, injection=%s)",
                gaze_feature_dim, hidden_dim, self.gaze_injection,
            )

        # ---- Semantic gaze: shared vocab loading ----
        self.use_semantic_labels = getattr(vlm, "use_semantic_labels", False)

        self.register_buffer("test_vocab_emb", None)
        self._test_vocab_size = 346

        if self.use_semantic_labels:
            vocab_path = getattr(vlm, "test_vocab_embeddings", None)
            if vocab_path is not None:
                self.test_vocab_emb = torch.load(
                    vocab_path, map_location="cpu", weights_only=False,
                )
            test_vocab_labels_path = getattr(vlm, "test_vocab_labels", None)
            if test_vocab_labels_path is not None:
                self._test_vocab_size = len(json.load(open(test_vocab_labels_path)))

        # ---- Text-based semantic evaluation ----
        if self.use_semantic_labels:
            # Direct text -> test vocab ID (exact match)
            self._label_to_test_id = {}
            self._test_vocab_labels: list[str] = []   # list indexed by test id
            test_vocab_labels_path = getattr(vlm, "test_vocab_labels", None)
            if test_vocab_labels_path is not None:
                for i, l in enumerate(json.load(open(test_vocab_labels_path))):
                    self._label_to_test_id[l] = i
                    self._test_vocab_labels.append(l)

            # Optional WordNet-based reroute over CLIP top-k test labels.
            # When True, _text_to_test_id (use_clip=True) will pick the most
            # specific WordNet hypernym among the top-K CLIP candidates before
            # falling back to raw argmax. Closes cases like "boxer" -> "person"
            # that pure CLIP may miss. See omnigf/vlm/text_to_id_mapper.py for
            # the rules.
            self.use_wordnet_reroute: bool = bool(getattr(vlm, "use_wordnet_reroute", False))
            self.wordnet_top_k: int = int(getattr(vlm, "wordnet_top_k", 10))
            self._wordnet_reroute_fn = None
            if self.use_wordnet_reroute:
                from omnigf.vlm.text_to_id_mapper import (
                    ensure_wordnet as _ensure_wordnet,
                    wordnet_reroute_id as _wordnet_reroute_id,
                )
                _ensure_wordnet()
                self._wordnet_reroute_fn = _wordnet_reroute_id

            # For non-exact matches during validation: train label -> nearest test ID
            self._train_label2id = {}
            self._train_to_test_id = None
            vocab_dir = getattr(vlm, "semantic_vocab_dir", None)
            if vocab_dir is not None:
                lab_path = os.path.join(vocab_dir, "train_label2id.json")
                if os.path.exists(lab_path):
                    self._train_label2id = json.load(open(lab_path))
                train_emb_path = os.path.join(vocab_dir, "train_vocab_embeddings.pt")
                if os.path.exists(train_emb_path) and self.test_vocab_emb is not None:
                    train_emb = torch.load(train_emb_path, map_location="cpu", weights_only=False)
                    sims = train_emb @ self.test_vocab_emb.cpu().T
                    self._train_to_test_id = sims.argmax(dim=1).tolist()
                    del train_emb, sims

            # CLIP encoder for test-time non-exact matches (loaded at test start)
            self._clip_model = None
            self._clip_text_projection = None
            self._clip_tokenizer = None
            self._clip_text_cache: Dict[str, int] = {}

            self.test_sem_text_acc1 = Accuracy(
                task="multiclass", top_k=1,
                num_classes=self._test_vocab_size, ignore_index=-1,
            )
            self.test_sem_text_acc3 = Accuracy(
                task="multiclass", top_k=3,
                num_classes=self._test_vocab_size, ignore_index=-1,
            )
            self.test_sem_text_multi_acc1 = MultiAccuracy(top_k=1, ignore_index=-1)
            self.val_sem_text_acc1 = Accuracy(
                task="multiclass", top_k=1,
                num_classes=self._test_vocab_size, ignore_index=-1,
            )
            logger.info("Text-based semantic evaluation enabled")

        # ---- Metrics ----
        self.val_dist = Distance()
        self.val_auc = AUC()
        self._val_inout_gt = []
        self._val_inout_pred = []
        self._val_inout_valid = []
        if self.approach != "lm":
            self.val_dist_text = Distance()

        # Test metrics -- GazeFollow uses multi-annotator evaluation
        self.test_gf_dist = GFTestDistance()
        # ``gf_auc_full_res=True`` scores GazeFollow AUC with the reference
        # protocol (predicted heatmap upsampled to the original image size,
        # GT pixels placed at that resolution) instead of the native 64x64
        # grid — for apples-to-apples comparison against Chong / Gaze-LLE.
        self.test_gf_auc = GFTestAUC(
            full_resolution=bool(getattr(vlm, "gf_auc_full_res", False)),
        )
        self.test_dist = Distance()
        self.test_auc = AUC()
        self._test_inout_gt = []
        self._test_inout_pred = []
        self._test_inout_valid = []
        if self.approach != "lm":
            self.test_dist_text = Distance()
            self.test_gf_dist_text = GFTestDistance()

        # ---- Social-pair head record buffers (heads-based, MTGS-style) ----
        # Each entry: per-batch slice of {lah/laeo/coatt}_pred (sigmoid
        # probs), {lah/laeo/coatt}_gt, inout_gt, pair_valid_mask, n_people.
        # Aggregated by ``social_pair_metrics.aggregate_pair_metrics`` in
        # on_*_epoch_end.
        self._val_pair_records: List[Dict] = []
        self._test_pair_records: List[Dict] = []

        # ---- Output parser ----
        coord_bins = getattr(vlm, "coord_bins", 1000)
        self.parser = MultiPersonOutputParser(coord_bins=coord_bins)

        self.save_hyperparameters(ignore=["cfg"])

    def train(self, mode: bool = True):
        """Keep the frozen GazeEncoder backbone in eval mode.

        When ``freeze_gaze_backbone`` is set, Lightning's per-phase
        ``.train()`` calls would otherwise flip the backbone's BatchNorm back
        to training mode and let its running stats keep adapting — defeating
        the "fixed encoder" ablation. Force it back to eval on every switch.
        """
        super().train(mode)
        if getattr(self, "freeze_gaze_backbone", False):
            self.gaze_encoder.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Model loading and LoRA
    # ------------------------------------------------------------------

    def _load_vlm(self, model_name: str):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        # Freeze the entire model (LoRA will unfreeze adapter params)
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Loaded VLM: %s (frozen)", model_name)
        return model

    def _apply_lora(self, vlm_cfg):
        from peft import LoraConfig, get_peft_model

        lora_rank = getattr(vlm_cfg, "lora_rank", 16)
        lora_alpha = getattr(vlm_cfg, "lora_alpha", 32)
        target_modules = list(
            getattr(vlm_cfg, "lora_target_modules", ["q_proj", "v_proj"])
        )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied: %d trainable / %d total (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )

    def _build_heatmap_decoder(self, vlm_cfg, hidden_dim: int):
        decoder_type = getattr(vlm_cfg, "decoder_type", "conv_transpose")
        if decoder_type == "conv_transpose":
            return ConvTransposeHeatmapDecoder(
                hidden_dim=hidden_dim,
                proj_dim=getattr(vlm_cfg, "decoder_proj_dim", 256),
                activation=getattr(vlm_cfg, "decoder_activation", "relu"),
                negative_slope=float(
                    getattr(vlm_cfg, "decoder_negative_slope", 0.01)),
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_gaze_weights(self, gaze_weights_path: str):
        """Load Gaze360 pre-trained ResNet18 backbone weights."""
        try:
            ckpt = torch.load(gaze_weights_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Gaze360 checkpoint from {gaze_weights_path}: {e}"
            ) from e

        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"Gaze360 checkpoint at {gaze_weights_path} missing "
                f"'model_state_dict' key. Found keys: {list(ckpt.keys())}"
            )

        weights = OrderedDict(
            (name.replace("base_head.", ""), value)
            for name, value in ckpt["model_state_dict"].items()
            if "base_head" in name
        )
        if not weights:
            raise ValueError(
                f"No 'base_head.*' keys found in Gaze360 checkpoint at "
                f"{gaze_weights_path}. Cannot initialise GazeEncoder backbone."
            )

        self.gaze_encoder.backbone.load_state_dict(weights, strict=True)
        logger.info("Loaded Gaze360 backbone from %s", gaze_weights_path)
        del ckpt, weights

    ETH_XGAZE_HF_REPO = "hysts/ptgaze-eth-xgaze-resnet18"

    def _load_clip_encoder(self):
        """Lazy-load CLIP text encoder for mapping novel labels to test vocab."""
        if self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPTokenizerFast
        _full = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
        )
        self._clip_model = _full.text_model.eval()
        self._clip_text_projection = _full.text_projection.eval()
        del _full
        self._clip_tokenizer = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-base-patch16",
        )
        # Freeze -- only used for inference
        for p in self._clip_model.parameters():
            p.requires_grad = False
        for p in self._clip_text_projection.parameters():
            p.requires_grad = False
        # Move to same device as test_vocab_emb
        if self.test_vocab_emb is not None:
            self._clip_model = self._clip_model.to(self.test_vocab_emb.device)
            self._clip_text_projection = self._clip_text_projection.to(
                self.test_vocab_emb.device,
            )
        logger.info("Loaded CLIP text encoder for semantic text evaluation")

    @torch.no_grad()
    def _clip_encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text string with CLIP and return L2-normalised embedding."""
        self._load_clip_encoder()
        prompt = CLIP_TEXT_TEMPLATE.format(text)
        inputs = self._clip_tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
        ).to(self._clip_model.device)
        out = self._clip_model(**inputs)
        emb = self._clip_text_projection(out.pooler_output)  # [1, 512]
        return F.normalize(emb, dim=-1).squeeze(0)

    def _text_to_test_id(self, text: str, use_clip: bool = False) -> int:
        """Map a generated category text to a test vocab ID.

        Args:
            text: Predicted category string.
            use_clip: If True (test), fall back to CLIP encoding for
                non-exact matches. If False (val), use pre-computed
                train->test mapping instead.
        """
        if not text:
            return -1
        text = text.strip().lower()
        # 1. Exact match in test vocab
        if text in self._label_to_test_id:
            # When use_clip, also cache CLIP sims for top-3 evaluation
            if use_clip and text not in self._clip_text_cache and self.test_vocab_emb is not None:
                emb = self._clip_encode_text(text)
                self._clip_text_cache[text] = emb @ self.test_vocab_emb.to(emb.dtype).T
            return self._label_to_test_id[text]
        if use_clip:
            # 2a. Test: CLIP encode + nearest neighbour (cache full sims)
            if text in self._clip_text_cache:
                sims = self._clip_text_cache[text]
            elif self.test_vocab_emb is not None:
                emb = self._clip_encode_text(text)
                sims = emb @ self.test_vocab_emb.to(emb.dtype).T  # [V]
                self._clip_text_cache[text] = sims
            else:
                return -1
            # Optional WordNet reroute over CLIP top-k.
            if (
                self.use_wordnet_reroute
                and self._wordnet_reroute_fn is not None
                and self._test_vocab_labels
            ):
                # wordnet_reroute_id runs on CPU torch tensors; move sims to CPU
                # if it lives on GPU.
                sims_cpu = sims.detach().float().cpu() if sims.is_cuda else sims
                reroute = self._wordnet_reroute_fn(
                    text, sims_cpu, self._test_vocab_labels, self.wordnet_top_k,
                )
                if reroute is not None:
                    return int(reroute)
            return int(sims.argmax().item())
        else:
            # 2b. Val: pre-computed train label -> nearest test ID
            if text in self._train_label2id and self._train_to_test_id is not None:
                train_idx = self._train_label2id[text]
                return self._train_to_test_id[train_idx]
        return -1

    # ------------------------------------------------------------------
    # Gaze injection
    # ------------------------------------------------------------------

    def _encode_gaze_tokens(self, batch: Dict) -> torch.Tensor:
        """Run GazeEncoder on head crops and return gaze tokens.

        Returns:
            ``[B, N, hidden_dim]`` gaze token embeddings.
        """
        head_crops = batch["head_crops"]    # [B, N, 3, H, W]
        head_bboxes = batch["head_bboxes"]  # [B, N, 4]

        gaze_emb = self.gaze_encoder.forward_backbone(head_crops)
        gaze_tokens, _ = self.gaze_encoder.forward_head(gaze_emb, head_bboxes)
        return gaze_tokens  # [B, N, hidden_dim]

    def _inject_gaze_into_hidden(self, hidden_states, batch, offset=0):
        """Inject per-person gaze tokens into the hidden states.

        The target and operation depend on ``self.gaze_injection``:

        * ``replace`` — overwrite the ``<|gaze_pad|>`` slots
          (``gaze_token_positions``).
        * ``add_span`` / ``add_anchor`` — add the gaze token onto the
          prompt-side person token span ``[person_inject_starts,
          person_inject_ends)`` (whole line vs. single anchor).

        Args:
            hidden_states: ``[B, seq_len, D]``.
            batch: Must contain ``head_crops``, ``num_annotated`` and the
                position/span tensors selected by the injection mode.
            offset: Position offset (e.g. from a prepended outside token).

        Returns:
            Modified hidden states.
        """
        additive = self.gaze_injection in ("add_span", "add_anchor")
        need_keys = (
            ("person_inject_starts", "person_inject_ends")
            if additive else ("gaze_token_positions",)
        )
        if "head_crops" not in batch or any(k not in batch for k in need_keys):
            return hidden_states

        seq_len = hidden_states.size(1)

        # During generate() decoding steps, hidden_states is [B, 1, D]
        # (single new token). Positions refer to the full prompt and are
        # already baked into the KV cache from prefill -- skip injection.
        if seq_len <= 1:
            return hidden_states

        gaze_tokens = self._encode_gaze_tokens(batch)  # [B, N, D]
        gaze_tokens = gaze_tokens.to(hidden_states.dtype)

        num_annotated = batch["num_annotated"]  # [B]

        if not additive:
            positions = batch["gaze_token_positions"]  # [B, N]
            B, N = positions.shape
            for bi in range(B):
                n_people = min(num_annotated[bi].item(), N)
                for pi in range(n_people):
                    pos = positions[bi, pi].item() + offset
                    if pos > offset:  # original pos > 0 means found
                        hidden_states[bi, pos] = gaze_tokens[bi, pi]
            return hidden_states

        # Additive span injection.
        starts = batch["person_inject_starts"]  # [B, N]
        ends = batch["person_inject_ends"]      # [B, N]
        B, N = starts.shape
        for bi in range(B):
            n_people = min(num_annotated[bi].item(), N)
            for pi in range(n_people):
                s = starts[bi, pi].item()
                if s <= 0:  # anchor not found
                    continue
                e = max(ends[bi, pi].item(), s + 1)
                s_off, e_off = s + offset, min(e + offset, seq_len)
                hidden_states[bi, s_off:e_off] = (
                    hidden_states[bi, s_off:e_off]
                    + self.gaze_add_scale * gaze_tokens[bi, pi].unsqueeze(0)
                )

        return hidden_states

    def _register_gaze_hook(self, batch, offset=0):
        """Register a forward pre-hook on the first transformer layer.

        The hook injects gaze embeddings after Qwen has already done its
        ``embed_tokens`` + vision ``masked_scatter``.  During ``generate()``,
        decoding steps pass a single new token (never ``<|gaze_pad|>``), so
        the hook is naturally a no-op -- no special guarding needed.

        Args:
            batch: Collated batch dict.
            offset: Position offset for prepended tokens (e.g. outside token).

        Returns:
            Hook handle (must be removed after forward/generate).
        """

        def _hook(module, args, kwargs):
            if args:
                hidden_states = args[0]
                modified = self._inject_gaze_into_hidden(
                    hidden_states, batch, offset=offset,
                )
                return (modified,) + args[1:], kwargs
            elif "hidden_states" in kwargs:
                kwargs["hidden_states"] = self._inject_gaze_into_hidden(
                    kwargs["hidden_states"], batch, offset=offset,
                )
            return args, kwargs

        # Navigate to the first transformer layer through PEFT wrapper.
        # Qwen3-VL: base_model.model.model.language_model.layers
        # Qwen2.5-VL: base_model.model.model.model.layers
        base = self.model.base_model.model
        first_layer = None
        for path in [
            ("model", "language_model", "layers"),  # Qwen3-VL
            ("model", "model", "layers"),            # Qwen2.5-VL
        ]:
            obj = base
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                first_layer = obj[0]
                break
            except (AttributeError, IndexError):
                continue

        if first_layer is None:
            raise RuntimeError(
                "Cannot find transformer layers for gaze injection hook"
            )

        handle = first_layer.register_forward_pre_hook(_hook, with_kwargs=True)
        return handle

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _build_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build ``mm_token_type_ids`` for Qwen3-VL: 1 at image-token positions,
        2 at video-token positions, 0 elsewhere.

        Passing this to the model forces ``get_rope_index`` to run every call,
        which refreshes ``self.model.rope_deltas`` for the current batch size.
        Without it, training reuses a stale cache from the previous
        ``generate()`` call and breaks on the last partial batch of an epoch.
        """
        cfg = self.model.config
        image_token_id = getattr(cfg, "image_token_id", None)
        video_token_id = getattr(cfg, "video_token_id", None)
        mm = torch.zeros_like(input_ids, dtype=torch.int32)
        if image_token_id is not None:
            mm[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm[input_ids == video_token_id] = 2
        return mm

    def _vlm_forward(self, batch: Dict) -> Dict:
        """VLM forward with gaze token injection."""
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "pixel_values": batch["pixel_values"],
            "mm_token_type_ids": self._build_mm_token_type_ids(batch["input_ids"]),
        }
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"]

        needs_hidden = self.approach == "heatmap"
        if needs_hidden:
            kwargs["output_hidden_states"] = True

        # Register the gaze-injection hook
        handles = []
        if "head_crops" in batch:
            handles.append(self._register_gaze_hook(batch))

        try:
            outputs = self.model(**kwargs)
        finally:
            for h in handles:
                h.remove()

        return outputs

    def _gather_hidden(
        self,
        hidden_states,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Gather hidden states at given positions.

        Args:
            hidden_states: ``[B, seq_len, D]``.
            positions: ``[B, N]`` or ``[B, V]``.

        Returns:
            ``[B, N, D]`` or ``[B, V, D]``.
        """
        D = hidden_states.size(-1)
        return torch.gather(
            hidden_states, 1,
            positions.unsqueeze(-1).expand(-1, -1, D),
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _find_person_positions_in_generated(
        self,
        gen_ids: torch.Tensor,
        tokenizer,
        num_people: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find ``P0:``, ``P1:``, ... positions in generated token sequences.

        Args:
            gen_ids: ``[B, seq_len]`` -- full generated sequences (prompt + generated).
            tokenizer: HuggingFace tokenizer.
            num_people: Maximum number of people to search for.

        Returns:
            Tuple of:
                - ``positions`` ``[B, num_people]`` -- token indices (0 for not found).
                - ``found_mask`` ``[B, num_people]`` -- ``True`` if that person
                  was found in the generated text.
        """
        from omnigf.vlm.parser import person_anchor_token_variants

        B = gen_ids.size(0)
        positions = torch.zeros(B, num_people, dtype=torch.long, device=gen_ids.device)
        found_mask = torch.zeros(B, num_people, dtype=torch.bool, device=gen_ids.device)

        for bi in range(B):
            ids = gen_ids[bi].tolist()
            for pid in range(num_people):
                # Iterate every anchor variant (BPE-context drift; see
                # ``parser.person_anchor_token_variants``) and track the
                # rightmost overall match — that's the model's most
                # recent "talking-about-this-person" position, which is
                # the generated-side occurrence when the LM produced
                # well-formed output, or the prompt-side occurrence as
                # a fallback.
                last_pos = -1
                for anchor_str in person_anchor_token_variants(pid, self.prompt_mode):
                    p_ids = tokenizer.encode(anchor_str, add_special_tokens=False)
                    if not p_ids:
                        continue
                    for pos in range(len(ids) - len(p_ids) + 1):
                        if ids[pos : pos + len(p_ids)] == p_ids:
                            if pos > last_pos:
                                last_pos = pos
                if last_pos >= 0:
                    positions[bi, pid] = last_pos
                    found_mask[bi, pid] = True

        return positions, found_mask

    def _gather_hidden_with_fallback(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        found_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather hidden states, substituting fallback embedding for missing persons.

        Args:
            hidden_states: ``[B, seq_len, D]``.
            positions: ``[B, N]`` -- token indices (may be 0 for missing persons).
            found_mask: ``[B, N]`` -- ``True`` where person was found.

        Returns:
            ``[B, N, D]`` -- hidden states with fallback for missing persons.
        """
        gathered = self._gather_hidden(hidden_states, positions)  # [B, N, D]

        if not found_mask.all():
            fallback = self.fallback_person_embed.expand(
                gathered.size(0), gathered.size(1), -1,
            )  # [B, N, D]
            gathered = torch.where(
                found_mask.unsqueeze(-1), gathered, fallback,
            )

        return gathered

    def _gather_person_hidden(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        found_mask: Optional[torch.Tensor] = None,
        starts: Optional[torch.Tensor] = None,
        ends: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-person representation, per ``self.person_extract``.

        Args:
            hidden_states: ``[B, seq_len, D]``.
            positions: ``[B, N]`` — ``P{pid}:`` anchor positions.
            found_mask: ``[B, N]`` — where the person was located (eval only);
                missing persons get :attr:`fallback_person_embed`.
            starts / ends: ``[B, N]`` — per-person answer-line span (exclusive
                end), required by the ``line_*`` modes.

        Returns:
            ``[B, N, D]``.
        """
        # Fall back to anchor gathering when spans are unavailable (e.g. an
        # "anchor" run, or a batch collated before the spans existed).
        if self.person_extract == "anchor" or starts is None or ends is None:
            if found_mask is not None:
                return self._gather_hidden_with_fallback(
                    hidden_states, positions, found_mask,
                )
            return self._gather_hidden(hidden_states, positions)

        seq_len = hidden_states.size(1)
        # A person is usable only if the span was actually located.
        valid = starts > 0
        if found_mask is not None:
            valid = valid & found_mask

        if self.person_extract == "line_last":
            # Last token INSIDE the span (end is exclusive), clamped in-range.
            last_pos = (ends - 1).clamp(min=0, max=seq_len - 1)
            last_pos = torch.where(valid, last_pos, positions)
            gathered = self._gather_hidden(hidden_states, last_pos)
        else:  # "line_mean"
            idx = torch.arange(seq_len, device=hidden_states.device)
            idx = idx.view(1, 1, seq_len)
            span_mask = (
                (idx >= starts.unsqueeze(-1))
                & (idx < ends.clamp(max=seq_len).unsqueeze(-1))
                & valid.unsqueeze(-1)
            )
            w = span_mask.to(hidden_states.dtype)
            denom = w.sum(-1, keepdim=True).clamp(min=1.0)
            gathered = torch.einsum("bnl,bld->bnd", w, hidden_states) / denom
            # Empty spans would pool to zero — use the anchor instead.
            empty = span_mask.sum(-1) == 0
            if empty.any():
                anchor_hidden = self._gather_hidden(hidden_states, positions)
                gathered = torch.where(
                    empty.unsqueeze(-1), anchor_hidden, gathered,
                )

        if found_mask is not None and not found_mask.all():
            fallback = self.fallback_person_embed.expand(
                gathered.size(0), gathered.size(1), -1,
            ).to(gathered.dtype)
            gathered = torch.where(
                found_mask.unsqueeze(-1), gathered, fallback,
            )
        return gathered

    def _find_person_spans_in_generated(
        self,
        gen_ids: torch.Tensor,
        positions: torch.Tensor,
        found_mask: torch.Tensor,
        pad_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-person answer-line spans over a generated sequence.

        ``start`` is the person's anchor; ``end`` is the next *found* anchor
        after it, or the end of the real (non-pad) content when this is the
        last person the model emitted. Robust to malformed generations where
        some persons are missing entirely.

        Returns:
            ``(starts, ends)`` each ``[B, N]``, ``end`` exclusive.
        """
        B, N = positions.shape
        starts = positions.clone()
        ends = torch.zeros_like(positions)
        for bi in range(B):
            row = gen_ids[bi]
            nonpad = (row != pad_id).nonzero(as_tuple=True)[0]
            seq_end = int(nonpad[-1].item()) + 1 if nonpad.numel() > 0 else int(row.numel())
            anchors = [
                int(positions[bi, k]) for k in range(N) if bool(found_mask[bi, k])
            ]
            for k in range(N):
                if not bool(found_mask[bi, k]):
                    continue
                s = int(positions[bi, k])
                later = [a for a in anchors if a > s]
                e = min(later) if later else seq_end
                ends[bi, k] = max(e, s + 1)
        return starts, ends

    def _find_vision_positions_in_ids(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Find vision token positions between ``<|vision_start|>`` and ``<|vision_end|>``.

        Args:
            input_ids: ``[B, seq_len]`` -- token sequences.
            tokenizer: HuggingFace tokenizer (for special token IDs).
            image_grid_thw: ``[B, 3]`` -- unused but kept for API compat.

        Returns:
            ``[B, max_vision_tokens]`` -- indices of the vision tokens.
        """
        B = input_ids.size(0)
        device = input_ids.device
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        counts = []
        start_positions = []
        for bi in range(B):
            ids = input_ids[bi]
            s = (ids == vision_start_id).nonzero(as_tuple=True)[0]
            e = (ids == vision_end_id).nonzero(as_tuple=True)[0]
            if len(s) > 0 and len(e) > 0:
                start = s[0].item() + 1
                n = e[0].item() - start
                counts.append(n)
                start_positions.append(start)
            else:
                logger.warning("No <|vision_start|>/<|vision_end|> in sample %d", bi)
                counts.append(0)
                start_positions.append(0)

        max_v = max(counts) if counts else 0
        positions = torch.zeros(B, max_v, dtype=torch.long, device=device)
        for bi in range(B):
            if counts[bi] > 0:
                positions[bi, :counts[bi]] = torch.arange(
                    start_positions[bi],
                    start_positions[bi] + counts[bi],
                    device=device,
                )

        return positions

    # ------------------------------------------------------------------
    # Inference forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict,
        tokenizer=None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference with gaze encoder injection."""
        prompt_ids = batch.get("prompt_input_ids", batch["input_ids"])
        prompt_mask = batch.get("prompt_attention_mask", batch["attention_mask"])
        # Default to the config-driven value set in __init__; callers can
        # still pin a specific length via the kwarg.
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        needs_hidden = self.approach == "heatmap"
        model_cfg = self.model.config
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        pad_id = getattr(text_cfg, "pad_token_id", None) or text_cfg.eos_token_id
        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "pixel_values": batch["pixel_values"],
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "pad_token_id": pad_id,
        }
        if "image_grid_thw" in batch:
            gen_kwargs["image_grid_thw"] = batch["image_grid_thw"]

        # ---- Generate with gaze injection ----
        handles = []
        if "head_crops" in batch:
            gen_batch = dict(batch)
            if "prompt_gaze_token_positions" in batch:
                gen_batch["gaze_token_positions"] = batch["prompt_gaze_token_positions"]
            if "prompt_person_inject_starts" in batch:
                gen_batch["person_inject_starts"] = batch["prompt_person_inject_starts"]
                gen_batch["person_inject_ends"] = batch["prompt_person_inject_ends"]
            handles.append(
                self._register_gaze_hook(gen_batch)
            )

        try:
            gen_output = self.model.generate(**gen_kwargs)
        finally:
            for h in handles:
                h.remove()

        gen_ids = gen_output.sequences

        generated_texts = None
        if tokenizer is not None:
            prompt_len = prompt_ids.shape[1]
            response_ids = gen_ids[:, prompt_len:]
            generated_texts = tokenizer.batch_decode(
                response_ids, skip_special_tokens=True,
            )

        # ---- Head-based predictions via re-forward ----
        pred_heatmaps = None
        inout_pred = None
        person_hidden_last = None
        vision_hidden_last = None

        if needs_hidden:
            assert tokenizer is not None

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            gen_attn_mask = (gen_ids != pad_id).long()

            fwd_kwargs = {
                "input_ids": gen_ids,
                "attention_mask": gen_attn_mask,
                "pixel_values": batch["pixel_values"],
                "output_hidden_states": True,
            }
            if "image_grid_thw" in batch:
                fwd_kwargs["image_grid_thw"] = batch["image_grid_thw"]

            fwd_handles = []
            if "head_crops" in batch:
                fwd_batch = dict(batch)
                if "prompt_gaze_token_positions" in batch:
                    fwd_batch["gaze_token_positions"] = batch["prompt_gaze_token_positions"]
                if "prompt_person_inject_starts" in batch:
                    fwd_batch["person_inject_starts"] = batch["prompt_person_inject_starts"]
                    fwd_batch["person_inject_ends"] = batch["prompt_person_inject_ends"]
                fwd_handles.append(
                    self._register_gaze_hook(fwd_batch)
                )

            try:
                with torch.no_grad():
                    fwd_out = self.model(**fwd_kwargs)
            finally:
                for h in fwd_handles:
                    h.remove()

            all_hidden = fwd_out.hidden_states

            hidden = all_hidden[self.hidden_layer]

            num_people = batch["inout"].size(1)
            person_pos, found_mask = self._find_person_positions_in_generated(
                gen_ids, tokenizer, num_people,
            )
            # Answer-line spans over the GENERATED sequence (line_* modes).
            person_starts = person_ends = None
            if self.person_extract in ("line_last", "line_mean"):
                person_starts, person_ends = self._find_person_spans_in_generated(
                    gen_ids, person_pos, found_mask, pad_id,
                )

            vision_pos = self._find_vision_positions_in_ids(
                gen_ids, tokenizer, batch["image_grid_thw"],
            )

            person_hidden_last = self._gather_person_hidden(
                hidden, person_pos,
                found_mask=found_mask,
                starts=person_starts, ends=person_ends,
            )
            vision_hidden_last = self._gather_hidden(hidden, vision_pos)

            if self.approach == "heatmap":
                pred_heatmaps = self.heatmap_decoder(vision_hidden_last, person_hidden_last)
                inout_pred = self.inout_head(person_hidden_last)

        preds = extract_predictions(
            approach=self.approach,
            outputs=None,
            batch=batch,
            parser=self.parser,
            generated_texts=generated_texts,
            pred_heatmaps=pred_heatmaps,
            inout_pred=inout_pred,
        )
        # Expose the decoded text strings for downstream introspection
        # (visualisation / debugging — diagnose unparseable generations
        # without re-decoding ``gen_ids``). ``None`` when the caller did
        # not pass ``tokenizer``.
        if generated_texts is not None:
            preds["raw_texts"] = generated_texts

        # ---- Social-pair head predictions ----
        # Heads-based MTGS-style social-gaze predictions. Only computed
        # when the head is enabled and we actually have ``person_hidden_last``
        # in scope (true for the heatmap approach; the lm-only approach does
        # not extract hidden states).
        if self.use_social_pair_head and person_hidden_last is not None:
            with torch.no_grad():
                pair_probs = self._social_pair_predict(person_hidden_last)  # [B, P, 3]
            preds["social_pair_probs"] = pair_probs

        return preds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        # VLMGazeCollator returns None when every sample in the batch was
        # skipped (all-inout=-1 frames in val/test non-social mode).
        # Train batches should never be None (VLMGazeDataset replaces at
        # train), but guard defensively anyway.
        if batch is None:
            return None
        outputs = self._vlm_forward(batch)
        lm_loss = outputs.loss
        logs = {"lm_loss": lm_loss.item()}
        total_loss = self.lambda_lm * lm_loss

        inout_gt = batch["inout"]       # [B, N]

        if self.approach == "heatmap":
            total_loss, logs = self._train_heatmap(
                outputs, batch, lm_loss, inout_gt,
            )
        # ---- Logging ----
        B = inout_gt.size(0)
        for key, val in logs.items():
            self.log(
                f"loss/train/{key}", val,
                batch_size=B, prog_bar=(key == "total"),
                on_step=True, on_epoch=True,
            )
        self.log(
            "loss/train", total_loss.item(),
            batch_size=B, prog_bar=True, on_step=True, on_epoch=True,
        )

        return {"loss": total_loss}

    def _train_heatmap(self, outputs, batch, lm_loss, inout_gt):
        """Heatmap approach training."""
        all_hidden = outputs.hidden_states
        last_hidden = all_hidden[self.hidden_layer]
        person_pos = batch["person_token_positions"]
        vision_pos = batch["vision_token_positions"]
        # Assistant-side answer-line spans (only present for the line_* modes).
        person_starts = batch.get("person_span_starts")
        person_ends = batch.get("person_span_ends")

        person_hidden = self._gather_person_hidden(
            last_hidden, person_pos, starts=person_starts, ends=person_ends,
        )
        vision_hidden = self._gather_hidden(last_hidden, vision_pos)

        pred_hm = self.heatmap_decoder(vision_hidden, person_hidden)

        inout_pred = self.inout_head(person_hidden)

        inside = inout_gt == 1
        valid = inout_gt != -1

        # Per-person loss-scale for pseudo-labelled persons. Defaults to all
        # ones (no effect) when ``is_pseudo`` isn't in the batch.
        is_pseudo = batch.get("is_pseudo")
        if is_pseudo is not None and self.pseudo_loss_scale != 1.0:
            pseudo_weight = torch.where(
                is_pseudo.to(inout_gt.device) > 0.5,
                torch.full_like(inout_gt, self.pseudo_loss_scale, dtype=torch.float),
                torch.ones_like(inout_gt, dtype=torch.float),
            )
        else:
            pseudo_weight = None

        bce_loss, bce_logs = compute_inout_bce_loss(
            inout_pred, inout_gt.float(), valid, weight=pseudo_weight,
        )
        gt_hm = batch.get("gaze_heatmaps", torch.zeros_like(pred_hm))
        hm_loss, hm_logs = compute_heatmap_mse_loss(
            pred_hm, gt_hm, inside, weight=pseudo_weight,
        )

        total = (
            self.lambda_lm * lm_loss
            + self.lambda_hm * hm_loss
            + self.lambda_bce * bce_loss
        )
        logs = {"lm_loss": lm_loss.item(), **hm_logs, **bce_logs, "total": total.item()}

        # ---- Social-pair head BCE loss ----
        if self.use_social_pair_head and "lah_pair_labels" in batch:
            social_loss, social_logs = self._compute_social_pair_loss(
                person_hidden, batch,
            )
            total = total + self.lambda_social * social_loss
            logs.update(social_logs)
            logs["total"] = total.item()

        return total, logs

    # ------------------------------------------------------------------
    # Social-pair head helpers
    # ------------------------------------------------------------------

    def _compute_social_pair_loss(self, person_hidden, batch):
        """Run the social-pair head and compute MTGS-style BCE.

        Args:
            person_hidden: ``[B, N, D]`` per-person hidden states.
            batch: must contain ``lah_pair_labels`` / ``laeo_pair_labels``
                / ``coatt_pair_labels`` ``[B, N*(N-1)]`` and
                ``pair_valid_mask`` ``[B, N*(N-1)]`` boolean.

        Returns:
            ``(loss, logs)`` — scalar loss tensor and a dict of per-cat
            float losses for logging.
        """
        pair_logits = self.social_pair_head(person_hidden)  # [B, P, 3]
        loss, logs = compute_social_pair_bce_loss(
            pair_logits,
            batch["lah_pair_labels"],
            batch["laeo_pair_labels"],
            batch["coatt_pair_labels"],
            batch["pair_valid_mask"],
            pos_weight_lah=self.social_pos_weight_lah,
            pos_weight_laeo=self.social_pos_weight_laeo,
            pos_weight_coatt=self.social_pos_weight_coatt,
        )
        return loss, logs

    def _social_pair_predict(self, person_hidden):
        """Forward the social-pair head with sigmoid applied. Used at
        eval; returns ``[B, P, 3]`` of probabilities in [0, 1]."""
        pair_logits = self.social_pair_head(person_hidden)
        return torch.sigmoid(pair_logits)

    # ------------------------------------------------------------------
    # Semantic metric helpers
    # ------------------------------------------------------------------

    def _eval_text_semantic(
        self, preds, batch, acc1_metric, multi_acc1_metric=None,
        acc3_metric=None, use_clip: bool = False,
    ):
        """Update text-based semantic metrics (Approach 1).

        Args:
            use_clip: If True (test), use CLIP encoding for non-exact
                matches. If False (val), use train->test mapping.
            acc3_metric: If provided and use_clip=True, update top-3
                accuracy using cached CLIP similarity logits.
        """
        if not self.use_semantic_labels:
            return
        if "categories_text" not in preds or "gaze_test_label_ids" not in batch:
            return

        is_gazefollow = "gaze_pts_all" in batch
        B = batch["inout"].size(0)
        N = batch["inout"].size(1)
        cats = preds["categories_text"]

        # Map generated text -> test vocab IDs (also populates _clip_text_cache when use_clip)
        pred_test_ids = torch.full((B, N), -1, dtype=torch.long,
                                   device=batch["inout"].device)
        for bi in range(B):
            for ni in range(min(len(cats[bi]), N)):
                if cats[bi][ni] is not None:
                    pred_test_ids[bi, ni] = self._text_to_test_id(
                        cats[bi][ni], use_clip=use_clip,
                    )

        if is_gazefollow:
            last_idx = batch["num_annotated"] - 1
            bi = torch.arange(B, device=last_idx.device)
            gt_test_ids = batch["gaze_test_label_ids"][bi, last_idx]
            pred_ids = pred_test_ids[bi, last_idx]
            valid = gt_test_ids >= 0
        else:
            gt_test_ids = batch["gaze_test_label_ids"]
            inout_gt = batch["inout"]
            valid = (gt_test_ids >= 0) & (inout_gt == 1)
            pred_ids = pred_test_ids

        if not valid.any():
            return

        # All-zero logits for unmapped predictions -> counts as wrong
        one_hot = torch.zeros(
            valid.sum(), self._test_vocab_size, device=pred_ids.device,
        )
        mapped = pred_ids[valid] >= 0
        if mapped.any():
            one_hot[mapped] = one_hot[mapped].scatter_(
                1, pred_ids[valid][mapped].unsqueeze(1), 1.0,
            )
        acc1_metric.update(one_hot, gt_test_ids[valid])

        # Top-3 using CLIP similarity logits from cache
        if use_clip and acc3_metric is not None:
            logits = torch.zeros(
                (B, N, self._test_vocab_size), device=pred_ids.device,
            )
            for bi_idx in range(B):
                for ni in range(min(len(cats[bi_idx]), N)):
                    if cats[bi_idx][ni] is not None:
                        text = cats[bi_idx][ni].strip().lower()
                        if text in self._clip_text_cache:
                            logits[bi_idx, ni] = self._clip_text_cache[text]
            if is_gazefollow:
                logits_sel = logits[bi, last_idx]
            else:
                logits_sel = logits
            acc3_metric.update(logits_sel[valid], gt_test_ids[valid])

        if multi_acc1_metric is not None and is_gazefollow:
            if "gaze_multi_test_label_ids" in batch:
                multi_gt = batch["gaze_multi_test_label_ids"][bi, last_idx]
                multi_acc1_metric.update(one_hot, multi_gt[valid])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict, batch_idx: int):
        """Single forward(), add semantic metrics."""
        # Skip batches whose samples were all dropped by VLMGazeDataset
        # (all-inout=-1 frames in localization-only social_gaze_mode=False).
        if batch is None:
            return None
        # Teacher-forced pass for val LM loss
        outputs = self._vlm_forward(batch)
        lm_loss = outputs.loss
        B = batch["inout"].size(0)

        self.log(
            "loss/val/lm", lm_loss.item(),
            batch_size=B, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,
        )

        # Inference via generate + re-forward
        tokenizer = self.trainer.datamodule.tokenizer
        preds = self.forward(batch, tokenizer=tokenizer)

        # ---- Standard gaze metrics ----
        inout_gt = batch["inout"]
        gaze_gt = batch["gaze_pts"]
        gaze_pred = preds["gaze_pts"]

        self.val_dist.update(gaze_pred, gaze_gt, inout_gt)
        self.log(
            "metric/val/dist", self.val_dist,
            batch_size=B, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )

        if self.approach != "lm" and "gaze_pts_text" in preds:
            self.val_dist_text.update(preds["gaze_pts_text"], gaze_gt, inout_gt)
            self.log(
                "metric/val/dist_text", self.val_dist_text,
                batch_size=B, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
            )
        if "heatmap" in preds and "gaze_heatmaps" in batch:
            # A spatially FLAT heatmap means the decoder has collapsed (see the
            # conv_out init note in decoders.py): argmax then always returns
            # pixel 0, so dist freezes at ~0.73 and AUC at exactly 0.5 for the
            # rest of the run. It is unrecoverable — the dead ReLU has no
            # gradient — so say so immediately instead of after 20 epochs.
            if not getattr(self, "_warned_flat_heatmap", False):
                hm_std = preds["heatmap"].float().flatten(-2).std(dim=-1).max()
                if hm_std < 1e-8:
                    logger.error(
                        "Predicted heatmaps are spatially CONSTANT (max std=%.2e). "
                        "The heatmap decoder has collapsed and cannot recover; "
                        "dist will stay pinned and AUC at 0.5. Kill this run.",
                        float(hm_std),
                    )
                    self._warned_flat_heatmap = True
            self.val_auc.update(
                preds["heatmap"].view(-1, 64, 64),
                batch["gaze_heatmaps"].view(-1, 64, 64),
                inout_gt.view(-1),
            )
            self.log(
                "metric/val/auc", self.val_auc,
                batch_size=B, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
            )

        if "inout_prob" in preds:
            valid = (inout_gt.view(-1) != -1)
            self._val_inout_gt.append(inout_gt.view(-1).cpu())
            self._val_inout_pred.append(preds["inout_prob"].view(-1).cpu())
            self._val_inout_valid.append(valid.cpu())

        # ---- Social-pair head records (heads-based, MTGS-style) ----
        self._collect_pair_records(preds, batch, self._val_pair_records)

        # ---- Semantic metrics ----
        if self.use_semantic_labels:
            self._eval_text_semantic(preds, batch, self.val_sem_text_acc1)
            self.log("metric/val/sem_text_acc@1", self.val_sem_text_acc1,
                     batch_size=B, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Social-gaze record collection / aggregation
    # ------------------------------------------------------------------

    # Subset of ``aggregate_set_metrics`` keys we surface as Lightning
    # metrics. Drops macro variants and the ``ari_mean_all`` /
    # ``ari_n_withGT`` diagnostics; keeps per-cat micro P/R/F1 + the
    # SA-bearing-only ARI + the JSON validity rate.
    _SOCIAL_LOG_KEYS = (
        "json_validity_rate",
        "LAH/precision_micro", "LAH/recall_micro", "LAH/f1_micro",
        # MTGS-style per-gazer LAH metric: micro-aggregated binary
        # classification per ``inout==1`` person — does the model
        # correctly predict whether each candidate gazer is looking at
        # any head? Matches MTGS's ``performance/compute_metrics.py``.
        "LAH/precision_per_gazer", "LAH/recall_per_gazer", "LAH/f1_per_gazer",
        "LAEO/precision_micro", "LAEO/recall_micro", "LAEO/f1_micro",
        "SA/precision_micro", "SA/recall_micro", "SA/f1_micro",
        "SA/ari_mean_withGT",
    )

    def _collect_pair_records(
        self, preds: Dict, batch: Dict, bucket: List[Dict],
    ) -> None:
        """Accumulate per-batch social-pair head outputs + GT labels.

        Skipped when the head isn't active or the batch lacks pair
        labels (e.g. GazeFollow-only batches that don't carry social
        annotations through the collator).
        """
        if "social_pair_probs" not in preds:
            return
        if "lah_pair_labels" not in batch or "pair_valid_mask" not in batch:
            return
        probs = preds["social_pair_probs"].detach().cpu()  # [B, P, 3]
        lah_p, laeo_p, coatt_p = probs[..., 0], probs[..., 1], probs[..., 2]
        lah_gt = batch["lah_pair_labels"].detach().cpu()
        laeo_gt = batch["laeo_pair_labels"].detach().cpu()
        coatt_gt = batch["coatt_pair_labels"].detach().cpu()
        valid = batch["pair_valid_mask"].detach().cpu()
        inout = batch["inout"].detach().cpu()
        num_ann = batch["num_annotated"].detach().cpu()
        # GazeFollow test batches carry ``gaze_pts_all`` as a marker.
        # Per MTGS's ``performance/compute_metrics.py:96-97``, MTGS forces
        # ``io = 1`` for GF when filtering candidate gazers, so the
        # metric module overrides our standard ``inout==1`` gate for
        # records flagged ``is_gazefollow=True``.
        is_gazefollow = bool("gaze_pts_all" in batch)
        B = lah_p.size(0)
        for b_i in range(B):
            n = int(num_ann[b_i].item())
            bucket.append({
                "lah_pred":   lah_p[b_i],
                "laeo_pred":  laeo_p[b_i],
                "coatt_pred": coatt_p[b_i],
                "lah_gt":     lah_gt[b_i],
                "laeo_gt":    laeo_gt[b_i],
                "coatt_gt":   coatt_gt[b_i],
                "inout_gt":   inout[b_i],
                "pair_valid_mask": valid[b_i],
                "n_people":   n,
                "is_gazefollow": is_gazefollow,
            })

    def _aggregate_and_log_pair_metrics(
        self, records: List[Dict], split: str,
    ) -> List[str]:
        """Run MTGS-style pair-metric aggregation and log per-category
        AP/AUC/P/R/F1 keys. Returns parts for the epoch-end log line."""
        if not records:
            return []
        from omnigf.performance.social_pair_metrics import aggregate_pair_metrics

        report = aggregate_pair_metrics(
            records, threshold=self.social_f1_threshold,
        )
        parts: List[str] = []
        # Log every key from the report — caller can prune via
        # tensorboard-side filters; keys are already namespaced by category.
        for key, value in report.items():
            try:
                self.log(
                    f"metric/{split}/{key}", float(value),
                    prog_bar=key.endswith("/f1"),
                    sync_dist=True,
                )
                parts.append(f"{key}={float(value):.4f}")
            except (TypeError, ValueError):
                continue
        logger.info(
            "%s social-pair (%d records): %s",
            split, len(records),
            {k: round(float(v), 4) for k, v in report.items()
             if isinstance(v, (int, float))},
        )
        return parts

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        parts = []
        for key in ("loss/val/lm", "metric/val/dist", "metric/val/dist_text", "metric/val/auc"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")

        # Log valid / total parsed predictions
        n_valid = int(self.val_dist.n_valid.item())
        n_total = int(self.val_dist.n_observations.item())
        parts.append(f"parsed={n_valid}/{n_total}")

        if self.approach != "lm" and hasattr(self, "val_dist_text"):
            n_valid_t = int(self.val_dist_text.n_valid.item())
            n_total_t = int(self.val_dist_text.n_observations.item())
            parts.append(f"parsed_text={n_valid_t}/{n_total_t}")

        if self._val_inout_gt:
            ap = self._compute_ap(self._val_inout_gt, self._val_inout_pred, self._val_inout_valid)
            if ap is not None and self.global_rank == 0:
                self.log("metric/val/inout_ap", ap, prog_bar=True, rank_zero_only=True)
                parts.append(f"inout_ap={ap:.4f}")
            self._val_inout_gt.clear()
            self._val_inout_pred.clear()
            self._val_inout_valid.clear()

        # Social-pair head metrics (heads-based, MTGS-style)
        pair_parts = self._aggregate_and_log_pair_metrics(
            self._val_pair_records, split="val",
        )
        parts.extend(pair_parts)
        self._val_pair_records.clear()

        # Semantic metrics
        key = "metric/val/sem_text_acc@1"
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")

        logger.info("Val epoch %d: %s", self.current_epoch, " | ".join(parts))

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step with semantic predictions."""
        # Same skip contract as validation_step — VLMGazeCollator returns
        # None when every frame in the batch had no annotated persons.
        if batch is None:
            return None
        tokenizer = self.trainer.datamodule.tokenizer
        preds = self.forward(batch, tokenizer=tokenizer)

        # ---- Standard gaze metrics ----
        is_gazefollow = "gaze_pts_all" in batch
        B = batch["inout"].size(0)

        if is_gazefollow:
            last_idx = batch["num_annotated"] - 1
            bi = torch.arange(B, device=last_idx.device)
            gaze_pred = preds["gaze_pts"][bi, last_idx]
            gaze_gt_all = batch["gaze_pts_all"]
            self.test_gf_dist.update(gaze_pred, gaze_gt_all)
            if "heatmap" in preds:
                hm_pred = preds["heatmap"][bi, last_idx]
                self.test_gf_auc.update(
                    hm_pred, gaze_gt_all, img_size=batch.get("img_size"),
                )
            if self.approach != "lm" and "gaze_pts_text" in preds:
                gaze_pred_text = preds["gaze_pts_text"][bi, last_idx]
                self.test_gf_dist_text.update(gaze_pred_text, gaze_gt_all)
        else:
            gaze_pred = preds["gaze_pts"]
            gaze_gt = batch["gaze_pts"]
            inout_gt = batch["inout"]
            self.test_dist.update(gaze_pred, gaze_gt, inout_gt)
            if "heatmap" in preds and "gaze_heatmaps" in batch:
                self.test_auc.update(
                    preds["heatmap"].view(-1, 64, 64),
                    batch["gaze_heatmaps"].view(-1, 64, 64),
                    inout_gt.view(-1),
                )
            if "inout_prob" in preds:
                valid = (inout_gt.view(-1) != -1)
                self._test_inout_gt.append(inout_gt.view(-1).cpu())
                self._test_inout_pred.append(preds["inout_prob"].view(-1).cpu())
                self._test_inout_valid.append(valid.cpu())
            if self.approach != "lm" and "gaze_pts_text" in preds:
                self.test_dist_text.update(
                    preds["gaze_pts_text"], gaze_gt, inout_gt,
                )
        # ---- Social-pair head records (heads-based, MTGS-style) ----
        # Collected for BOTH GF and non-GF test batches when the head is
        # active and the dataset emits pair labels. ``_collect_pair_records``
        # is a no-op for batches missing either ``social_pair_probs`` or
        # ``lah_pair_labels``, so calling it unconditionally is safe.
        self._collect_pair_records(preds, batch, self._test_pair_records)
        if self.use_semantic_labels:
            self._eval_text_semantic(
                preds, batch,
                self.test_sem_text_acc1, self.test_sem_text_multi_acc1,
                acc3_metric=self.test_sem_text_acc3, use_clip=True,
            )

    def on_test_epoch_start(self):
        """Load CLIP text encoder for test-time semantic evaluation."""
        if self.use_semantic_labels:
            self._load_clip_encoder()

    def on_test_epoch_end(self):
        """Log all metrics."""
        # ---- GazeFollow metrics (multi-annotator) ----
        if self.test_gf_dist.n_observations > 0:
            avg_dist, min_dist = self.test_gf_dist.compute()
            self.log("metric/test/avg_dist", avg_dist, prog_bar=True)
            self.log("metric/test/min_dist", min_dist, prog_bar=True)
            self.log("metric/test/n_obs", self.test_gf_dist.n_observations, prog_bar=True)
            logger.info(
                "metric/test/avg_dist=%.4f  min_dist=%.4f  n_obs=%d",
                float(avg_dist), float(min_dist),
                int(self.test_gf_dist.n_observations),
            )

        if self.test_gf_auc.n_observations > 0:
            self.log("metric/test/auc", self.test_gf_auc.compute(), prog_bar=True)
            logger.info(
                "metric/test/auc=%.4f  n_obs=%d  full_resolution=%s",
                float(self.test_gf_auc.compute()),
                int(self.test_gf_auc.n_observations),
                self.test_gf_auc.full_resolution,
            )

        if self.approach != "lm" and self.test_gf_dist_text.n_observations > 0:
            a_dist, m_dist = self.test_gf_dist_text.compute()
            self.log("metric/test/avg_dist_text", a_dist, prog_bar=True)
            self.log("metric/test/min_dist_text", m_dist)

        # ---- Non-GazeFollow metrics (per-person) ----
        if self.test_dist.n_observations > 0:
              self.log("metric/test/dist", self.test_dist.compute(), prog_bar=True)
              self.log("metric/test/n_obs", self.test_dist.n_observations, prog_bar=True)
              self.log("metric/test/n_valid", self.test_dist.n_valid, prog_bar=True)
              logger.info(
                  "metric/test/dist=%.4f  n_obs=%d  n_valid=%d",
                  float(self.test_dist.compute()),
                  int(self.test_dist.n_observations),
                  int(self.test_dist.n_valid),
              )

        if self.test_auc.n_observations > 0:
            self.log("metric/test/auc", self.test_auc.compute(), prog_bar=True)
            # Mirrored to the file logger so each Hydra run dir's
            # ``vlm_train.log`` is self-contained. Without this the AUC only
            # reaches wandb / the progress bar, which makes sweeps that compare
            # runs from their logs (e.g. scripts/rebuttal/*) unreadable.
            logger.info(
                "metric/test/auc=%.4f  n_obs=%d",
                float(self.test_auc.compute()),
                int(self.test_auc.n_observations),
            )

        if self.approach != "lm" and self.test_dist_text.n_observations > 0:
            self.log("metric/test/dist_text", self.test_dist_text.compute(), prog_bar=True)

        # ---- In/out AP (non-GazeFollow) ----
        if self._test_inout_gt:
            ap = self._compute_ap(self._test_inout_gt, self._test_inout_pred, self._test_inout_valid)
            if ap is not None and self.global_rank == 0:
                self.log("metric/test/inout_ap", ap, prog_bar=True, rank_zero_only=True)
                logger.info("In-out AP: %.4f", ap)
                self.log("metric/test/n_obs_inout", int(torch.cat(self._test_inout_valid).sum()), prog_bar=True)
            self._test_inout_gt.clear()
            self._test_inout_pred.clear()
            self._test_inout_valid.clear()

        # ---- Text-based semantic metrics ----
        if self.use_semantic_labels:
            if self.test_sem_text_acc1.update_count > 0:
                acc1 = self.test_sem_text_acc1.compute()
                self.log("metric/test/sem_text_acc@1", acc1, prog_bar=True)
                logger.info("Semantic text acc@1: %.4f", acc1)
                self.test_sem_text_acc1.reset()
            if self.test_sem_text_acc3.update_count > 0:
                acc3 = self.test_sem_text_acc3.compute()
                self.log("metric/test/sem_text_acc@3", acc3, prog_bar=True)
                logger.info("Semantic text acc@3: %.4f", acc3)
                self.test_sem_text_acc3.reset()

            if self.test_sem_text_multi_acc1.total > 0:
                multi_acc1 = self.test_sem_text_multi_acc1.compute()
                self.log("metric/test/sem_text_multi_acc@1", multi_acc1, prog_bar=True)
                logger.info("Semantic text multi_acc@1: %.4f", multi_acc1)
                self.test_sem_text_multi_acc1.reset()

        # ---- Social-pair head metrics (heads-based, MTGS-style) ----
        self._aggregate_and_log_pair_metrics(
            self._test_pair_records, split="test",
        )
        self._test_pair_records.clear()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        vlm = self.cfg.vlm
        lr = self.cfg.optimizer.lr
        head_lr = getattr(vlm, "head_lr", lr * 10)
        wd = self.cfg.optimizer.weight_decay

        lora_params = []
        head_params = []
        gaze_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "gaze_encoder" in name:
                gaze_encoder_params.append(param)
            elif "model" in name and "lora" in name.lower():
                lora_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr, "name": "lora"})
        if gaze_encoder_params:
            param_groups.append({"params": gaze_encoder_params, "lr": head_lr, "name": "gaze_encoder"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "name": "heads"})

        optimizer = optim.AdamW(param_groups, weight_decay=wd)

        scheduler_type = getattr(self.cfg.scheduler, "type", "CosineAnnealingWarmRestarts")
        if scheduler_type == "constant":
            return optimizer

        for pg in optimizer.param_groups:
            pg["base_lr"] = pg["lr"]

        num_samples = len(self.trainer.datamodule.train_dataset)
        accum = self.cfg.train.accumulate_grad_batches
        steps_per_epoch = math.ceil(num_samples / (self.cfg.train.batch_size * accum))
        T_0 = getattr(self.cfg.scheduler, "t_0_epochs", 3) * steps_per_epoch

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        num_samples = len(self.trainer.datamodule.train_dataset)
        accum = self.cfg.train.accumulate_grad_batches
        warmup_steps = getattr(self.cfg.scheduler, "warmup_epochs", 1) * math.ceil(
            num_samples / (self.cfg.train.batch_size * accum)
        )
        if self.trainer.global_step < warmup_steps:
            # Linear warmup: ramp from 0 to base_lr, skip cosine scheduler
            lr_scale = float(self.trainer.global_step + 1) / warmup_steps
            for pg in scheduler.optimizer.param_groups:
                pg["lr"] = lr_scale * pg["base_lr"]
        else:
            # After warmup: let cosine scheduler control LR
            scheduler.step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_ap(self, gt_list, pred_list, valid_list):
        """Compute average precision, gathering across GPUs if needed."""
        from sklearn.metrics import average_precision_score

        gt_local = torch.cat(gt_list).float()
        pred_local = torch.cat(pred_list).float()
        valid_local = torch.cat(valid_list).bool()

        # Gather across all GPUs under DDP
        if self.trainer.world_size > 1:
            gt_gathered = self.all_gather(gt_local).view(-1)
            pred_gathered = self.all_gather(pred_local).view(-1)
            valid_gathered = self.all_gather(valid_local).view(-1)
        else:
            gt_gathered = gt_local
            pred_gathered = pred_local
            valid_gathered = valid_local

        # Filter to valid entries only
        gt_all = gt_gathered[valid_gathered].cpu().numpy()
        pred_all = pred_gathered[valid_gathered].cpu().numpy()

        if len(gt_all) == 0:
            return None
        return average_precision_score(gt_all, pred_all)
