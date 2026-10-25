#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Fine-tune OmniGF for gaze following + social gaze on VSGaze (Table 6): VAT, ChildPlay,
# VideoCoAtt and UCO-LAEO pooled, starting from the GazeFollow model trained with the
# in/out loss. Adds the social pair head (LAH / LAEO / shared attention) on the
# per-person tokens. One GPU, about 25 h for 10 epochs.
#
#   bash scripts/finetune_social_vsgaze.sh
#
# Up to 10 persons per frame; annotated persons come first in the prompt, and persons
# without a gaze label are written as "unknown" with their LM loss masked.
# The run ends with a test of the checkpoint with the best validation distance.
# The released model is epoch 9 of this recipe with SEED=42; the run's own test gave
# Dist 0.099, AP_IO 0.992, F1_LAH 0.815, F1_LAEO 0.755, AP_SA 0.758. Over six seeds
# (best-validation checkpoint each) the test metrics ranged Dist 0.097-0.101,
# AP_IO 0.991-0.992, F1_LAH 0.805-0.815, F1_LAEO 0.721-0.763 and AP_SA 0.716-0.758.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CKPT_GAZEFOLLOW_INOUT}"
mapfile -t SOURCE_ARGS < <(source_args vat childplay videocoatt uco_laeo)

python main_vlm_vsgaze.py \
    experiment.task=train+test \
    experiment.name="Social-VSGaze-seed${SEED}" \
    experiment.output_folder="$OUTPUT_ROOT/finetune_social_vsgaze" \
    wandb.project_name=OmniGF-social \
    "train.pretrained_ckpt='$PRETRAINED_CKPT'" \
    train.seed="$SEED" \
    train.devices=1 \
    train.epochs="$EPOCHS" \
    train.batch_size=8 \
    train.accumulate_grad_batches=2 \
    optimizer.lr=5e-5 \
    vlm.head_lr=5e-5 \
    scheduler.type=CosineAnnealingWarmRestarts \
    scheduler.warmup_epochs=1 \
    scheduler.t_0_epochs=10 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.train_lora=True \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.lambda_lm=1.0 \
    vlm.lambda_hm=10.0 \
    vlm.lambda_bce=0.1 \
    vlm.lambda_kl=0.2 \
    vlm.lambda_social=1.0 \
    vlm.heatmap_loss=mse \
    vlm.prompt_mode=point \
    vlm.mask_point_block_loss=False \
    vlm.mask_per_person_lm_loss=True \
    vlm.use_social_pair_head=True \
    vlm.sort_annotated_first=True \
    vlm.max_new_tokens=256 \
    vlm.max_people=10 \
    data.num_people=10 \
    data.num_workers=8 \
    val.batch_size=8 \
    test.batch_size=16 \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
