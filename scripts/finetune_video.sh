#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Fine-tune OmniGF on VideoAttentionTarget (Table 1) or ChildPlay (Table 2), starting
# from the GazeFollow model trained WITH the in/out loss ($CKPT_GAZEFOLLOW_INOUT, see
# train_gazefollow.sh VARIANT=inout). One GPU, about 1-1.5 h for the 3 epochs.
#
#   DATASET=vat bash scripts/finetune_video.sh
#   DATASET=childplay bash scripts/finetune_video.sh
#
# The run ends with a test of the checkpoint with the best validation distance.
# Both datasets peak at epoch 0. Settings follow the released checkpoints:
#   vat        LoRA LR 2.5e-5, encoder + heads 5e-5, constant LR
#   childplay  2.5e-5 for all groups, 1 warm-up epoch, then cosine over 3 epochs
# Re-runs of this recipe scored VAT 0.099-0.100 / AP 0.922-0.926 and ChildPlay
# 0.090-0.092 / AP 0.995-0.997 over seeds 1 and 42 (the released checkpoints:
# 0.096 / 0.922 and 0.090 / 0.996). SEED defaults to 1, the closer of the two.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

DATASET="${DATASET:?set DATASET=vat or DATASET=childplay}"
case "$DATASET" in
    vat)
        LR=2.5e-5; HEAD_LR=5e-5
        SCHEDULER_TYPE=constant; WARMUP_EPOCHS=0 ;;
    childplay)
        LR=2.5e-5; HEAD_LR=2.5e-5
        SCHEDULER_TYPE=CosineAnnealingWarmRestarts; WARMUP_EPOCHS=1 ;;
    *) echo "unknown DATASET=$DATASET (vat | childplay)" >&2; exit 1 ;;
esac
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-1}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CKPT_GAZEFOLLOW_INOUT}"
mapfile -t SOURCE_ARGS < <(source_args "$DATASET")

python main_vlm_vsgaze.py \
    experiment.task=train+test \
    experiment.name="FT-${DATASET}-seed${SEED}" \
    experiment.output_folder="$OUTPUT_ROOT/finetune_${DATASET}" \
    wandb.project_name=OmniGF-video \
    "train.pretrained_ckpt='$PRETRAINED_CKPT'" \
    train.seed="$SEED" \
    train.devices=1 \
    train.epochs="$EPOCHS" \
    train.batch_size=8 \
    train.accumulate_grad_batches=2 \
    optimizer.lr=$LR \
    vlm.head_lr=$HEAD_LR \
    scheduler.type=$SCHEDULER_TYPE \
    scheduler.warmup_epochs=$WARMUP_EPOCHS \
    scheduler.t_0_epochs=3 \
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
    vlm.max_new_tokens=128 \
    data.num_people=4 \
    val.batch_size=8 \
    test.batch_size=16 \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
