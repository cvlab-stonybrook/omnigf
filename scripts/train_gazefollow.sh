#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Train OmniGF on GazeFollow from the Qwen3-VL-4B base model (one GPU, 20 epochs).
#
#   bash scripts/train_gazefollow.sh                  # Table 1 model (omnigf_gazefollow)
#   VARIANT=inout bash scripts/train_gazefollow.sh    # warm start for VAT / ChildPlay / VSGaze
#
# VARIANT=paper  lambda_bce 0, standard annotations. The in/out head of this model is
#                not trained (GazeFollow test persons are all in-frame).
# VARIANT=inout  lambda_bce 0.1 and gazefollow_train_extended.h5, which adds
#                out-of-frame persons; this is omnigf_gazefollow_inout.
#
# The run ends with a test of the checkpoint with the best validation distance
# (min over epochs; the released models are epoch 19 and epoch 14). Retraining with
# the defaults below gave 0.093 / 0.041 avg / min distance, against 0.091 / 0.040
# for the released model. That model's run name says it used the SemGaze head boxes
# (SEMGAZE_BOX=True); whether this accounts for the gap has not been re-measured.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

VARIANT="${VARIANT:-paper}"
case "$VARIANT" in
    paper) GF_INCLUDE_INOUT=False; LAMBDA_BCE=0.0 ;;
    inout) GF_INCLUDE_INOUT=True;  LAMBDA_BCE=0.1 ;;
    *) echo "unknown VARIANT=$VARIANT (paper | inout)" >&2; exit 1 ;;
esac
EPOCHS="${EPOCHS:-20}"
SEMGAZE_BOX="${SEMGAZE_BOX:-False}"
SEED="${SEED:-42}"

python main_vlm.py \
    experiment.task=train+test \
    experiment.name="GazeFollow-${VARIANT}" \
    experiment.output_folder="$OUTPUT_ROOT/train_gazefollow" \
    wandb.project_name=OmniGF-gazefollow \
    train.seed="$SEED" \
    train.devices=1 \
    train.epochs="$EPOCHS" \
    train.batch_size=8 \
    train.accumulate_grad_batches=2 \
    optimizer.lr=1e-4 \
    vlm.head_lr=2.5e-4 \
    scheduler.type=CosineAnnealingWarmRestarts \
    scheduler.warmup_epochs=1 \
    scheduler.t_0_epochs="$EPOCHS" \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.train_lora=True \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.decoder_activation=relu \
    vlm.use_gaze_encoder=True \
    vlm.lambda_lm=1.0 \
    vlm.lambda_hm=10.0 \
    vlm.lambda_bce=$LAMBDA_BCE \
    vlm.lambda_kl=0.0 \
    vlm.heatmap_loss=mse \
    vlm.gf_test_all_people=False \
    data.gf_include_inout=$GF_INCLUDE_INOUT \
    data.gf_use_semgaze_bbox=$SEMGAZE_BOX \
    data.gf_semgaze_ann_root="$GF_SEMGAZE_ANN_ROOT" \
    val.batch_size=8 \
    test.batch_size=8 \
    "${COMMON_ARGS[@]}" \
    "$@"
