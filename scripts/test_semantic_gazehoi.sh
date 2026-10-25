#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Evaluate semantic gaze on GazeHOI (Table 5): 4,934 test (person, object) pairs.
#
#   bash scripts/test_semantic_gazehoi.sh
#   CKPT=/path/to/your.ckpt bash scripts/test_semantic_gazehoi.sh
#
# GazeAcc: the predicted gaze point falls inside the target object's box.
# Acc@1: the generated category, mapped onto the 463-label vocabulary (exact match,
# CLIP similarity, WordNet reroute over the CLIP top-10), equals the GT label.
# Released checkpoint on this code: GazeAcc 0.779, Acc@1 0.719.
# LIMIT=50 runs a quick check on the first 50 pairs.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

CKPT="${CKPT:-$CKPT_SEMANTIC_GAZEHOI}"
WORDNET="${WORDNET:-True}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_ROOT/test_semantic_gazehoi/metrics.json}"
EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=("+eval_gazehoi.limit=$LIMIT")

python eval_gazehoi_semantic.py \
    --config-name vlm_config.yaml \
    "test.checkpoint='$CKPT'" \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.prompt_mode=point_semantic \
    vlm.use_semantic_labels=True \
    vlm.use_semantic_head=False \
    vlm.max_new_tokens=256 \
    vlm.semantic_vocab_dir="$GAZEHOI_VOCAB_DIR" \
    vlm.test_vocab_embeddings="$GAZEHOI_VOCAB_DIR/test_vocab_embeddings.pt" \
    vlm.test_vocab_labels="$GAZEHOI_VOCAB_DIR/test_vocab_labels.json" \
    +eval_gazehoi.h5_path="$GAZEHOI_ANN_ROOT/gazehoi_test.h5" \
    +eval_gazehoi.image_root="$ROOT_GAZEHOI" \
    +eval_gazehoi.test_vocab_embeddings="$GAZEHOI_VOCAB_DIR/test_vocab_embeddings.pt" \
    +eval_gazehoi.test_vocab_labels="$GAZEHOI_VOCAB_DIR/test_vocab_labels.json" \
    +eval_gazehoi.chunk_size=4 \
    +eval_gazehoi.batch_size="$BATCH_SIZE" \
    +eval_gazehoi.num_workers=8 \
    +eval_gazehoi.use_wordnet="$WORDNET" \
    +eval_gazehoi.wordnet_top_k=10 \
    +eval_gazehoi.output_json="$OUTPUT_JSON" \
    "${EXTRA[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
