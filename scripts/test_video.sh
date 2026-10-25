#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Evaluate gaze target estimation on VideoAttentionTarget (Table 1) or ChildPlay (Table 2).
#
#   DATASET=vat bash scripts/test_video.sh
#   DATASET=childplay bash scripts/test_video.sh
#   DATASET=vat CKPT=/path/to/your.ckpt bash scripts/test_video.sh
#
# Reports metric/test/dist (L2 distance on in-frame gaze) and metric/test/inout_ap.
# Released checkpoints: VAT 0.096 / AP 0.922, ChildPlay 0.090 / AP 0.996.
# Extra Hydra overrides can be appended to the command line.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

DATASET="${DATASET:?set DATASET=vat or DATASET=childplay}"
case "$DATASET" in
    vat)       CKPT="${CKPT:-$CKPT_VAT}" ;;
    childplay) CKPT="${CKPT:-$CKPT_CHILDPLAY}" ;;
    *) echo "unknown DATASET=$DATASET (vat | childplay)" >&2; exit 1 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-16}"
mapfile -t SOURCE_ARGS < <(source_args "$DATASET")

# main_vlm_vsgaze.py evaluates the weights given by train.pretrained_ckpt.
python main_vlm_vsgaze.py \
    experiment.task=test \
    experiment.name="Test-${DATASET}" \
    experiment.output_folder="$OUTPUT_ROOT/test_${DATASET}" \
    "train.pretrained_ckpt='$CKPT'" \
    train.devices=1 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.max_new_tokens=128 \
    data.num_people=4 \
    test.batch_size="$BATCH_SIZE" \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
