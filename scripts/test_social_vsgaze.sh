#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Evaluate gaze following + social gaze on the pooled VSGaze test set (Table 6).
#
#   bash scripts/test_social_vsgaze.sh
#   CKPT=/path/to/your.ckpt bash scripts/test_social_vsgaze.sh
#
# Reports metric/test/dist and inout_ap (gaze), LAH/f1 (per-gazer), LAEO/f1 (per-gazer
# argmax) and coatt/ap (shared attention, per pair), following the MTGS protocol.
# Released checkpoint: Dist 0.099, AP_IO 0.992, F1_LAH 0.816, F1_LAEO 0.756,
# AP_SA 0.758. About 3 h on one GPU.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

CKPT="${CKPT:-$CKPT_SOCIAL}"
BATCH_SIZE="${BATCH_SIZE:-16}"
mapfile -t SOURCE_ARGS < <(source_args vat childplay videocoatt uco_laeo)

# main_vlm_vsgaze.py evaluates the weights given by train.pretrained_ckpt.
python main_vlm_vsgaze.py \
    experiment.task=test \
    experiment.name="Test-social-vsgaze" \
    experiment.output_folder="$OUTPUT_ROOT/test_social_vsgaze" \
    "train.pretrained_ckpt='$CKPT'" \
    train.devices=1 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.prompt_mode=point \
    vlm.mask_per_person_lm_loss=True \
    vlm.use_social_pair_head=True \
    vlm.sort_annotated_first=True \
    vlm.max_new_tokens=256 \
    vlm.max_people=10 \
    data.num_people=10 \
    data.num_workers=8 \
    test.batch_size="$BATCH_SIZE" \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
