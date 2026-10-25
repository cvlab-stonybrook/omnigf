#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Evaluate semantic gaze on GazeFollow (Table 4).
#
#   bash scripts/test_semantic_gazefollow.sh
#   CKPT=/path/to/your.ckpt bash scripts/test_semantic_gazefollow.sh
#
# A generated category is mapped onto the 346-label test vocabulary by exact match,
# then CLIP text similarity ("a photo of a {}", ViT-B/16), then a WordNet hypernym
# reroute over the CLIP top-10 (WORDNET=True). Reports "Semantic text acc@1" (Acc@1)
# and "multi_acc@1" (MultiAcc: correct if any annotator's label matches), plus the
# gaze distances. Released checkpoint: 0.649 / 0.739; with WORDNET=False
# (CLIP only): 0.634 / 0.722.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

CKPT="${CKPT:-$CKPT_SEMANTIC_GAZEFOLLOW}"
WORDNET="${WORDNET:-True}"
BATCH_SIZE="${BATCH_SIZE:-8}"
mapfile -t SOURCE_ARGS < <(source_args gazefollow)

python main_vlm.py \
    experiment.task=test \
    experiment.name="Test-semantic-gazefollow" \
    experiment.output_folder="$OUTPUT_ROOT/test_semantic_gazefollow" \
    "test.checkpoint='$CKPT'" \
    train.devices=1 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.prompt_mode=point_semantic \
    vlm.use_semantic_labels=True \
    vlm.use_semantic_head=False \
    vlm.use_wordnet_reroute="$WORDNET" \
    vlm.wordnet_top_k=10 \
    vlm.max_new_tokens=192 \
    vlm.semantic_vocab_dir="$GF_VOCAB_DIR" \
    vlm.test_vocab_embeddings="$GF_VOCAB_DIR/test_vocab_embeddings.pt" \
    vlm.test_vocab_labels="$GF_VOCAB_DIR/test_vocab_labels.json" \
    data.gf_include_inout=False \
    data.gf_use_semgaze_bbox=False \
    data.gf_semgaze_ann_root="$GF_SEMGAZE_ANN_ROOT" \
    test.batch_size="$BATCH_SIZE" \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
