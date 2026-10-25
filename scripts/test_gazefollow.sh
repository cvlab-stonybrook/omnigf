#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Evaluate gaze target estimation on GazeFollow (Table 1).
#
#   bash scripts/test_gazefollow.sh
#   CKPT=/path/to/your.ckpt bash scripts/test_gazefollow.sh
#
# Reports metric/test/avg_dist (distance to the annotators' mean point, the paper's
# "Avg Dist.") and metric/test/min_dist (distance to the closest annotator), plus AUC.
# Released checkpoint: 0.091 / 0.040.
# FULL_RES=True (default) scores AUC at the original image size, as in the reference
# implementations; FULL_RES=False scores it on the 64x64 heatmap grid (about 0.002 lower).
# The distances do not depend on FULL_RES.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

CKPT="${CKPT:-$CKPT_GAZEFOLLOW}"
FULL_RES="${FULL_RES:-True}"
BATCH_SIZE="${BATCH_SIZE:-8}"
mapfile -t SOURCE_ARGS < <(source_args gazefollow)

python main_vlm.py \
    experiment.task=test \
    experiment.name="Test-gazefollow" \
    experiment.output_folder="$OUTPUT_ROOT/test_gazefollow" \
    "test.checkpoint='$CKPT'" \
    train.devices=1 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.gf_test_all_people=False \
    vlm.gf_auc_full_res="$FULL_RES" \
    data.gf_include_inout=False \
    test.batch_size="$BATCH_SIZE" \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
