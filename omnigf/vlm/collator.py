# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

from typing import Dict, List, Optional, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import logging

logger = logging.getLogger(__name__)

class VLMGazeCollator:
    """Tokenise and batch VLMGazeDataset samples for Qwen VLM training."""

    APPROACHES = ("lm", "heatmap")

    def __init__(
        self,
        processor, # Input is now the unified processor
        approach: str = "lm",
        max_people: int = 5,
        use_gaze_encoder: bool = False,
        mask_point_block_loss: bool = False,
        mask_per_person_lm_loss: bool = False,
        prompt_mode: str = "point",
        gaze_injection: str = "replace",
        person_extract: str = "anchor",
    ):
        assert approach in self.APPROACHES, f"approach={approach!r} not in {self.APPROACHES}"
        assert person_extract in ("anchor", "line_last", "line_mean"), (
            f"person_extract={person_extract!r} not in "
            "{'anchor', 'line_last', 'line_mean'}"
        )
        # Where the dense heads read each person's representation from (see
        # ``models_ge.VLMGazeModelGE._gather_person_hidden``):
        #   anchor    — hidden state at the assistant-side ``P{pid}:`` token
        #               (default / original behaviour).
        #   line_last — last token of the person's answer line.
        #   line_mean — mean-pool over the person's answer-line tokens.
        # The latter two need the assistant-side per-person span.
        self.person_extract = person_extract
        assert gaze_injection in ("replace", "add_span", "add_anchor"), (
            f"gaze_injection={gaze_injection!r} not in "
            "{'replace', 'add_span', 'add_anchor'}"
        )
        # Injection strategy for gaze-encoder tokens (see
        # ``omnigf.vlm.models_ge.VLMGazeModelGE._inject_gaze_into_hidden``):
        #   replace    — overwrite the <|gaze_pad|> slot.
        #   add_span   — add the gaze token to every prompt token of the
        #                person's line (``P{pid}: <|gaze_pad|> {bbox}``).
        #   add_anchor — add the gaze token to the single ``P{pid}:`` anchor.
        # The additive modes emit prompt-side per-person spans (start/end)
        # alongside the ``<|gaze_pad|>`` positions.
        self.gaze_injection = gaze_injection
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.approach = approach
        self.max_people = max_people
        self.use_gaze_encoder = use_gaze_encoder
        # Used by ``_find_person_positions`` to pick the right per-person
        # anchor token sequence (``P{pid}:`` for JSON modes,
        # ``Person {pid}`` for natural-language). See
        # ``omnigf.vlm.parser.person_anchor_token`` for the mapping.
        self.prompt_mode = prompt_mode
        # When True: mask the point block so LM loss fires only on the
        # appended category block (original lm_two_block behaviour). When
        # False: LM loss covers the full assistant turn (both blocks); the
        # point-block tokens are GT so this just regularises the localisation
        # distribution while the category block is learned on top. False is
        # the recommended default because masking leads to LoRA drift on the
        # point block at inference (observed: status values leak category
        # strings like "climbing wall", "frisbee").
        self.mask_point_block_loss = mask_point_block_loss

        # When True: for each person whose ``inout == -1`` in the sample,
        # mask the LM loss on the inner content of their ``P{pid}: {...}``
        # line (everything between the ``P{pid}:`` prefix and the start of
        # the next person's prefix or end of the assistant turn). The
        # ``P{pid}:`` skeleton stays supervised so the model learns to emit
        # one line per person mentioned in the prompt. The dataset emits a
        # placeholder ``"unknown"`` line for those persons; the placeholder
        # tokens flow through teacher forcing but contribute no gradient.
        self.mask_per_person_lm_loss = mask_per_person_lm_loss

        # Cache special token IDs
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.vision_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # Used to bound the last person's line span (``add_span`` injection):
        # the ``<|im_end|>`` that closes the user turn sits right after the
        # person listing. Falls back to eos if the token is absent.
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_end_id is None or self.im_end_id < 0:
            self.im_end_id = self.tokenizer.eos_token_id

        # The <|gaze_pad|> placeholder exists only for the "replace" strategy.
        self.use_gaze_pad_token = use_gaze_encoder and gaze_injection == "replace"
        if self.use_gaze_pad_token:
            self.gaze_pad_id = self.tokenizer.convert_tokens_to_ids("<|gaze_pad|>")

    def __call__(self, batch_samples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Drop ``None`` samples emitted by ``VLMGazeDataset.__getitem__``
        # when a val/test frame had no annotated persons (localization-only
        # social_gaze_mode=False path). If the whole batch is empty after
        # filtering, return ``None`` and let the LightningModule's
        # {validation,test}_step short-circuit.
        batch_samples = [s for s in batch_samples if s is not None]
        if not batch_samples:
            return None

        batch: Dict[str, torch.Tensor] = {}
        all_input_ids: List[torch.Tensor] = []
        all_prompt_ids: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pixel_values = []
        all_image_grid_thw = []

        for sample in batch_samples:
            # ---- Step 1: Process images and text together via Processor ----
            system_text = sample["system_prompt"]
            prompt_text = sample["prompt"]
            target_text = sample["target"]

            # Full Conversation
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]

            # Step A: Get formatted text via chat template (no tokenization yet)
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            prompt_text_only = self.tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            # Step B: Process text + image jointly via processor.
            # This expands <|image_pad|> to the correct number of vision tokens
            # based on the actual image dimensions.
            inputs = self.processor(
                text=[full_text], images=[sample["image"]],
                return_tensors="pt", padding=False,
            )
            prompt_inputs = self.processor(
                text=[prompt_text_only], images=[sample["image"]],
                return_tensors="pt", padding=False,
            )

            full_ids = inputs["input_ids"][0]
            prompt_ids = prompt_inputs["input_ids"][0]

            # ---- Step 2: Labels and Padding ----
            # For two-block prompt_mode="point_then_category" samples carry a
            # ``target_point_prefix``. Masking the point block (point-only
            # LM loss on the category block) was the original design, but it
            # causes LoRA drift on the point block at inference. Setting
            # ``mask_point_block_loss=False`` keeps LM loss on the full
            # assistant turn; the point-block tokens are GT so this acts as
            # a regulariser while the category block is learned on top.
            if self.mask_point_block_loss and "target_point_prefix" in sample:
                point_ids = self.tokenizer(
                    sample["target_point_prefix"],
                    add_special_tokens=False,
                )["input_ids"]
                middle_len = len(prompt_ids) + len(point_ids)
            else:
                middle_len = len(prompt_ids)

            labels = torch.full_like(full_ids, -100)
            # Only calculate loss on the category block (or the full assistant
            # turn when no target_point_prefix is provided).
            labels[middle_len:] = full_ids[middle_len:]

            # ---- Per-person LM-loss masking for inout==-1 persons ----
            # For each unknown person, mask everything from the end of
            # their ``P{pid}:`` prefix up to (but not including) the next
            # person's ``P{pid}:`` prefix (or end of sequence for the last
            # person). The ``P{pid}:`` skeleton stays supervised; the
            # inner content (``{...}`` plus trailing ``\n`` if any) gets
            # ``-100``. Anchoring on actual ``P{pid}:`` token positions
            # makes this robust to BPE merges that combine ``{`` with
            # surrounding chars or ``}\n`` into a single token.
            if self.mask_per_person_lm_loss and "inout" in sample:
                inout_vec = sample["inout"]
                pids_vec = sample["person_ids"]
                n_kept = int(sample.get("num_annotated", len(inout_vec)))
                # Find each person's P{pid}: start in full_ids (assistant
                # side) and the prefix length. Backward scan = same
                # convention as ``_find_person_positions``.
                full_ids_list = full_ids.tolist()
                pid_to_start: Dict[int, int] = {}
                pid_to_prefix_len: Dict[int, int] = {}
                for k in range(n_kept):
                    pid = int(pids_vec[k])
                    p_str = f"P{pid}:"
                    p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
                    pid_to_prefix_len[pid] = len(p_ids)
                    p_len = len(p_ids)
                    for pos in range(len(full_ids_list) - p_len, -1, -1):
                        if full_ids_list[pos : pos + p_len] == p_ids:
                            pid_to_start[pid] = pos
                            break
                # Sort persons by their pid_start position to find the
                # next-person boundary for each unknown.
                sorted_starts = sorted(pid_to_start.values())
                end_of_assistant = full_ids.size(0)
                for k in range(n_kept):
                    if int(inout_vec[k]) != -1:
                        continue
                    pid = int(pids_vec[k])
                    start = pid_to_start.get(pid)
                    if start is None:
                        # Backward scan didn't find P{pid}: — should not
                        # happen unless the dataset's prompt/target are
                        # mis-built. Skip this person silently.
                        continue
                    prefix_len = pid_to_prefix_len[pid]
                    next_starts = [s for s in sorted_starts if s > start]
                    next_start = min(next_starts) if next_starts else end_of_assistant
                    mask_start = start + prefix_len
                    mask_end = next_start
                    if mask_end > mask_start:
                        labels[mask_start:mask_end] = -100

            all_input_ids.append(full_ids)
            all_prompt_ids.append(prompt_ids)
            all_labels.append(labels)
            
            # Extract vision tensors (pixel_values and grid_thw)
            # Qwen3-VL processor returns these in the dict
            all_pixel_values.append(inputs["pixel_values"])
            if "image_grid_thw" in inputs:
                all_image_grid_thw.append(inputs["image_grid_thw"])

        # ---- Step 3: Left-pad to uniform length ----
        # Decoder-only models need left-padding for correct generate() behaviour.
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        B = len(all_input_ids)
        max_len = max(ids.size(0) for ids in all_input_ids)

        input_ids_padded = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels_padded = torch.full((B, max_len), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(all_input_ids, all_labels)):
            seq_len = ids.size(0)
            input_ids_padded[i, max_len - seq_len:] = ids
            labels_padded[i, max_len - seq_len:] = lab
        attention_mask = (input_ids_padded != pad_id).long()

        # Prompt-only ids for generate() (no ground-truth target leakage)
        max_prompt_len = max(ids.size(0) for ids in all_prompt_ids)
        prompt_ids_padded = torch.full((B, max_prompt_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(all_prompt_ids):
            seq_len = ids.size(0)
            prompt_ids_padded[i, max_prompt_len - seq_len:] = ids
        prompt_attention_mask = (prompt_ids_padded != pad_id).long()

        batch.update({
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "prompt_input_ids": prompt_ids_padded,
            "prompt_attention_mask": prompt_attention_mask,
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_grid_thw": torch.cat(all_image_grid_thw, dim=0) if all_image_grid_thw else None
        })

        # Debug Check
        # num_pad_tokens = (batch["input_ids"] == self.vision_pad_id).sum().item()
        # logger.info(f"Batch vision pad tokens: {num_pad_tokens}")

        # ---- Step 4: Metadata Padding (gaze_pts, etc.) ----
        # (This logic remains largely the same as your original)
        batch["inout"] = self._pad_people_tensor([s["inout"] for s in batch_samples], -1, self.max_people)
        batch["gaze_pts"] = self._pad_people_tensor([s["gaze_pts"] for s in batch_samples], -1.0, self.max_people)
        batch["person_ids"] = self._pad_people_tensor([s["person_ids"] for s in batch_samples], -1, self.max_people)
        batch["head_bboxes"] = self._pad_people_tensor([s["head_bboxes"] for s in batch_samples], 0.0, self.max_people)
        batch["num_annotated"] = torch.tensor([s["num_annotated"] for s in batch_samples], dtype=torch.long)
        # Per-person pseudo-fill flag (0 = GT, 1 = MTGS pseudo). Pad value 0
        # treats padded slots as non-pseudo, matching their effective
        # exclusion via the existing ``inout != -1`` masks.
        if "is_pseudo" in batch_samples[0]:
            batch["is_pseudo"] = self._pad_people_tensor(
                [s["is_pseudo"] for s in batch_samples], 0.0, self.max_people,
            )

        # ---- Ground-truth social-gaze sets (point_then_social mode) ----
        # Pass through as a Python list of dicts; one entry per sample.
        # Cannot be a tensor because each sample has a variable number of
        # LAH/LAEO pairs and SA groups in the relabelled 0..V-1 pid space.
        if "social_sets" in batch_samples[0]:
            batch["social_sets"] = [s["social_sets"] for s in batch_samples]

        # ---- Per-pair social-gaze labels (heads-based path) ----
        # Each sample emits ``[V_i*(V_i-1)]`` tensors in
        # ``itertools.permutations(range(V_i), 2)`` order. We pad to
        # ``[max_V*(max_V-1)]`` where ``max_V`` is the per-batch max
        # number of kept persons (= ``self._pad_people_tensor`` would
        # use). Pairs in the padded space that touch a position
        # ``>= V_i`` get label -1 (annotated-mask drops them); pairs
        # within ``[0, V_i)`` are remapped to the new permutation index.
        if "lah_pair_labels" in batch_samples[0]:
            inout_tensors = [s["inout"] for s in batch_samples]
            max_v = (
                self.max_people
                if self.max_people is not None
                else max(t.size(0) for t in inout_tensors)
            )
            batch["lah_pair_labels"] = self._pad_pair_tensor(
                [s["lah_pair_labels"] for s in batch_samples],
                [int(t.size(0)) for t in inout_tensors],
                max_v,
            )
            batch["laeo_pair_labels"] = self._pad_pair_tensor(
                [s["laeo_pair_labels"] for s in batch_samples],
                [int(t.size(0)) for t in inout_tensors],
                max_v,
            )
            batch["coatt_pair_labels"] = self._pad_pair_tensor(
                [s["coatt_pair_labels"] for s in batch_samples],
                [int(t.size(0)) for t in inout_tensors],
                max_v,
            )
            batch["pair_valid_mask"] = self._build_pair_valid_mask(
                [int(t.size(0)) for t in inout_tensors], max_v,
            )

        # ---- Original image sizes (w, h) — full-resolution GazeFollow AUC ----
        if batch_samples[0].get("img_size") is not None:
            batch["img_size"] = torch.stack(
                [s["img_size"] for s in batch_samples]
            )  # [B, 2]

        # ---- Multi-annotator gaze points (GazeFollow test) ----
        if "gaze_pts_all" in batch_samples[0]:
            all_gaze_pts = [s["gaze_pts_all"] for s in batch_samples]
            max_a = max(g.size(0) for g in all_gaze_pts)
            batch["gaze_pts_all"] = self._pad_people_tensor(
                all_gaze_pts, pad_value=-1.0, fixed_size=max_a,
            )

        # ---- Gaze heatmaps (needed for heatmap approach) ----
        if self.approach == "heatmap" and "gaze_heatmaps" in batch_samples[0]:
            batch["gaze_heatmaps"] = self._pad_people_tensor(
                [s["gaze_heatmaps"] for s in batch_samples],
                pad_value=0.0, fixed_size=self.max_people,
            )

        # ---- Step 5: Dynamic Position Finding ----
        if self.approach == "heatmap":
            batch["person_token_positions"] = self._find_person_positions(
                input_ids_padded, [s["num_annotated"] for s in batch_samples]
            )

        if self.approach == "heatmap":
            batch["vision_token_positions"] = self._find_vision_positions(
                input_ids_padded, batch["image_grid_thw"]
            )

        # ---- Assistant-side per-person spans (person_extract ablation) ----
        # ``line_last`` / ``line_mean`` read the model's ANSWER line for each
        # person, so they need [start, end) over the assistant-side occurrence
        # (take="last"). At eval the spans are re-derived from the generated
        # sequence by ``models_ge._find_person_spans_in_generated``.
        if (
            self.approach == "heatmap"
            and self.person_extract in ("line_last", "line_mean")
        ):
            span_starts, span_ends = self._find_person_spans(
                input_ids_padded,
                [s["num_annotated"] for s in batch_samples],
                whole_line=True,
                take="last",
            )
            batch["person_span_starts"] = span_starts
            batch["person_span_ends"] = span_ends

        # ---- Step 5b: Semantic gaze label data ----
        if "gaze_label_embs" in batch_samples[0]:
            batch["gaze_label_embs"] = self._pad_people_tensor(
                [s["gaze_label_embs"] for s in batch_samples],
                pad_value=0.0, fixed_size=self.max_people,
            )  # [B, max_N, 512]
            batch["gaze_label_ids"] = self._pad_people_tensor(
                [s["gaze_label_ids"] for s in batch_samples],
                pad_value=-1, fixed_size=self.max_people,
            )  # [B, max_N]
            batch["gaze_test_label_ids"] = self._pad_people_tensor(
                [s["gaze_test_label_ids"] for s in batch_samples],
                pad_value=-1, fixed_size=self.max_people,
            )  # [B, max_N]
            batch["has_semantic_label"] = self._pad_people_tensor(
                [s["has_semantic_label"] for s in batch_samples],
                pad_value=False, fixed_size=self.max_people,
            )  # [B, max_N]
            if "gaze_multi_test_label_ids" in batch_samples[0]:
                batch["gaze_multi_test_label_ids"] = self._pad_people_tensor(
                    [s["gaze_multi_test_label_ids"] for s in batch_samples],
                    pad_value=-1, fixed_size=self.max_people,
                )  # [B, max_N, 5]

        # ---- Step 6: Gaze encoder data ----
        if self.use_gaze_encoder:
            num_ann = [s["num_annotated"] for s in batch_samples]
            if self.gaze_injection == "replace":
                # ``replace`` overwrites the <|gaze_pad|> slot; find those
                # positions for the full (training) and prompt-only (generate)
                # sequences.
                batch["gaze_token_positions"] = self._find_gaze_positions(
                    input_ids_padded, num_ann,
                )
                batch["prompt_gaze_token_positions"] = self._find_gaze_positions(
                    prompt_ids_padded, num_ann,
                )
            else:
                # Additive modes: no <|gaze_pad|> token exists. Add the gaze
                # embedding onto the prompt-side person token span [start, end)
                # for both the full (training-forward) and prompt-only
                # (generate) sequences.
                #   add_span   — whole line: [anchor, next anchor / turn end)
                #   add_anchor — the ``P{pid}:`` identifier tokens only
                whole_line = self.gaze_injection == "add_span"
                s_full, e_full = self._find_person_spans(
                    input_ids_padded, num_ann, whole_line,
                )
                s_pr, e_pr = self._find_person_spans(
                    prompt_ids_padded, num_ann, whole_line,
                )
                batch["person_inject_starts"] = s_full
                batch["person_inject_ends"] = e_full
                batch["prompt_person_inject_starts"] = s_pr
                batch["prompt_person_inject_ends"] = e_pr
            if "head_crops" in batch_samples[0]:
                batch["head_crops"] = self._pad_people_tensor(
                    [s["head_crops"] for s in batch_samples],
                    pad_value=0.0,
                    fixed_size=self.max_people,
                )  # [B, max_N, 3, H, W]

        # ---- Step 7: Non-tensor metadata passthrough ----
        if "path" in batch_samples[0]:
            # path is a list (from temporal dim); take first element
            batch["image_path"] = [
                s["path"][0] if isinstance(s["path"], list) else s["path"]
                for s in batch_samples
            ]

        return batch

    def _find_person_positions(
        self,
        input_ids: torch.Tensor,
        num_annotated_list: List[int],
        take: str = "last",
    ) -> torch.Tensor:
        """Locate the ``P{pid}:`` anchor token position per person.

        Args:
            take: ``"last"`` returns the rightmost (assistant-side) occurrence
                — used for reading per-person hidden states for the decoder
                heads. ``"first"`` returns the leftmost (prompt-side)
                occurrence — used for gaze injection at prefill (mode
                ``add_person``), where the anchor must sit in the prompt so it
                conditions generation. On a prompt-only sequence there is a
                single occurrence, so both agree.
        """
        from omnigf.vlm.parser import person_anchor_token_variants

        assert take in ("first", "last")
        B = input_ids.size(0)
        max_n = self.max_people or max(num_annotated_list)
        positions = torch.zeros(B, max_n, dtype=torch.long)
        for bi in range(B):
            ids = input_ids[bi].tolist()
            n_people = num_annotated_list[bi]
            for pid in range(min(n_people, max_n)):
                # Try each anchor variant (BPE-context-dependent — see
                # ``person_anchor_token_variants`` docstring) and take the
                # leftmost/rightmost match overall depending on ``take``.
                best = -1
                for anchor_str in person_anchor_token_variants(pid, self.prompt_mode):
                    p_ids = self.tokenizer.encode(anchor_str, add_special_tokens=False)
                    if not p_ids:
                        continue
                    if take == "last":
                        rng = range(len(ids) - len(p_ids), -1, -1)  # backward
                    else:
                        rng = range(0, len(ids) - len(p_ids) + 1)   # forward
                    for pos in rng:
                        if ids[pos : pos + len(p_ids)] == p_ids:
                            if take == "last":
                                best = max(best, pos)
                            elif best < 0 or pos < best:
                                best = pos
                            break
                if best >= 0:
                    positions[bi, pid] = best
        return positions

    def _find_person_spans(
        self,
        input_ids: torch.Tensor,
        num_annotated_list: List[int],
        whole_line: bool,
        take: str = "first",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-person token span ``[start, end)``.

        ``start`` is the ``P{pid}:`` anchor on the side selected by ``take``:
        ``"first"`` = prompt-side (used by the additive gaze-injection modes),
        ``"last"`` = assistant-side (used by the ``line_last`` / ``line_mean``
        person-extraction modes, which read the model's answer line).

        When ``whole_line`` is True, ``end`` extends to the next person's anchor
        on the same side, or — for the last person — to the ``<|im_end|>`` that
        closes that turn (so a prompt-side span never bleeds into the assistant
        target, and an assistant-side span stops at the end of the answer).
        When False, the span is just the ``P{pid}:`` identifier tokens.

        Returns:
            ``(starts, ends)`` each ``[B, max_people]`` (0 for missing persons;
            ``end`` is exclusive).
        """
        from omnigf.vlm.parser import person_anchor_token_variants

        starts = self._find_person_positions(
            input_ids, num_annotated_list, take=take,
        )
        max_n = starts.size(1)
        ends = torch.zeros_like(starts)
        B = input_ids.size(0)
        for bi in range(B):
            ids = input_ids[bi].tolist()
            seq_len = len(ids)
            n = min(num_annotated_list[bi], max_n)
            anchors = [int(starts[bi, pi]) for pi in range(n)]
            for pi in range(n):
                s = anchors[pi]
                if s <= 0:  # anchor not found for this person
                    continue
                if not whole_line:
                    # Cover the full ``P{pid}:`` identifier tokens (the anchor
                    # variant that matches at ``s``), not just the first token.
                    alen = 1
                    for anchor_str in person_anchor_token_variants(pi, self.prompt_mode):
                        p_ids = self.tokenizer.encode(anchor_str, add_special_tokens=False)
                        if p_ids and ids[s : s + len(p_ids)] == p_ids:
                            alen = len(p_ids)
                            break
                    ends[bi, pi] = s + alen
                    continue
                # Next person's anchor (smallest anchor strictly after s).
                later = [a for a in anchors if a > s]
                if later:
                    e = min(later)
                else:
                    # Last person: first <|im_end|> after the anchor (user-turn
                    # close), else end of sequence.
                    e = seq_len
                    for pos in range(s + 1, seq_len):
                        if ids[pos] == self.im_end_id:
                            e = pos
                            break
                ends[bi, pi] = max(e, s + 1)
        return starts, ends

    def _find_gaze_positions(
        self, input_ids: torch.Tensor, num_annotated_list: List[int],
    ) -> torch.Tensor:
        """Find ``<|gaze_pad|>`` token positions in the input sequence.

        Returns:
            ``[B, max_people]`` — positions of gaze placeholder tokens
            (0 for missing/padding persons).
        """
        B = input_ids.size(0)
        max_n = self.max_people or max(num_annotated_list)
        positions = torch.zeros(B, max_n, dtype=torch.long)
        for bi in range(B):
            ids = input_ids[bi]
            gaze_positions = (ids == self.gaze_pad_id).nonzero(as_tuple=True)[0]
            n_people = min(num_annotated_list[bi], max_n)
            for pi in range(min(len(gaze_positions), n_people)):
                positions[bi, pi] = gaze_positions[pi].item()
        return positions

    def _find_vision_positions(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Dynamically finds all vision tokens between <|vision_start|> and <|vision_end|>.

        Note: ``image_grid_thw`` reports the **pre-spatial-merge** grid for
        Qwen3-VL (e.g. ``[1, 28, 28]`` = 784), but the processor inserts
        **post-merge** ``<|image_pad|>`` tokens (e.g. 196 for merge_size=2).
        To be robust, we count the actual tokens in the sequence instead of
        computing from the grid.
        """
        B = input_ids.size(0)
        counts = []
        starts = []
        for bi in range(B):
            ids = input_ids[bi]
            s = (ids == self.vision_start_id).nonzero(as_tuple=True)[0]
            e = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]
            if len(s) > 0 and len(e) > 0:
                start = s[0].item() + 1  # skip <|vision_start|>
                n = e[0].item() - start  # tokens between start and end
                counts.append(n)
                starts.append(start)
            else:
                counts.append(0)
                starts.append(0)

        max_v = max(counts) if counts else 0
        positions = torch.zeros(B, max_v, dtype=torch.long)
        for bi in range(B):
            if counts[bi] > 0:
                positions[bi, :counts[bi]] = torch.arange(
                    starts[bi], starts[bi] + counts[bi],
                )
        return positions

    def _pad_people_tensor(self, tensors, pad_value, fixed_size=None):
        max_v = fixed_size or max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            pad_size = max_v - t.size(0)
            if pad_size > 0:
                t = torch.cat([t, torch.full((pad_size, *t.shape[1:]), pad_value, dtype=t.dtype, device=t.device)])
            elif pad_size < 0:
                t = t[:max_v]
            padded.append(t)
        return torch.stack(padded)

    @staticmethod
    def _pair_idx_in_max_v(i: int, j: int, max_v: int) -> int:
        """Position of ordered pair ``(i, j)`` in
        ``itertools.permutations(range(max_v), 2)``.

        permutations enumerate as ``(0, 1), (0, 2), …, (0, max_v-1),
        (1, 0), (1, 2), …``. The position is ``i*(max_v-1) + j_offset``
        where ``j_offset = j`` if ``j < i`` else ``j - 1`` (skips j=i).
        """
        return i * (max_v - 1) + (j if j < i else j - 1)

    @classmethod
    def _pad_pair_tensor(cls, tensors, v_per_sample, max_v):
        """Pad per-sample pair tensors ``[V_i*(V_i-1)]`` (in
        ``permutations(range(V_i), 2)`` order) to a uniform
        ``[B, max_v*(max_v-1)]`` (in ``permutations(range(max_v), 2)``
        order). Pairs that touch a padded position get label ``-1``.
        """
        import itertools as _it
        out_pairs = max_v * (max_v - 1)
        padded = []
        for t, v in zip(tensors, v_per_sample):
            row = torch.full((out_pairs,), -1.0, dtype=torch.float32)
            if v >= 2:
                pair_idx_v = list(_it.permutations(range(v), 2))
                for old_k, (i, j) in enumerate(pair_idx_v):
                    new_k = cls._pair_idx_in_max_v(i, j, max_v)
                    row[new_k] = float(t[old_k].item())
            padded.append(row)
        return torch.stack(padded)

    @classmethod
    def _build_pair_valid_mask(cls, v_per_sample, max_v):
        """Boolean mask ``[B, max_v*(max_v-1)]`` — True for pairs whose
        both endpoints are real (kept) persons in that sample, i.e. both
        positions ``< v_per_sample[bi]``."""
        out_pairs = max_v * (max_v - 1)
        B = len(v_per_sample)
        mask = torch.zeros(B, out_pairs, dtype=torch.bool)
        import itertools as _it
        pair_idx_max = list(_it.permutations(range(max_v), 2))
        for bi, v in enumerate(v_per_sample):
            for k, (i, j) in enumerate(pair_idx_max):
                if i < v and j < v:
                    mask[bi, k] = True
        return mask