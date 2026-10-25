#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Fine-tune OmniGF for semantic gaze on GazeFollow (Table 4), starting from the
# Table 1 GazeFollow model. Each person's target becomes
#   P0: {"status": "inside", "point_2d": [x, y], "category": "<label>"}
# trained on the refined GazeFollow semantic labels. One GPU, about 2 h per epoch.
#
#   bash scripts/finetune_semantic_gazefollow.sh
#
# The LR follows a 15-epoch cosine schedule; the released model is epoch 3 (best
# validation Acc@1), and the run that produced it was stopped after epoch 7, so
# EPOCHS defaults to 8 (epochs 0-7) on the same schedule. A re-run of this recipe
# picked epoch 3 again and scored Acc@1 0.649 / MultiAcc 0.736 (released: 0.649 /
# 0.739). The test that ends this run maps labels with CLIP only; for the paper's
# protocol (CLIP + WordNet reroute) evaluate the checkpoint with
# test_semantic_gazefollow.sh.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-42}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CKPT_GAZEFOLLOW}"
mapfile -t SOURCE_ARGS < <(source_args gazefollow)

python main_vlm.py \
    experiment.task=train+test \
    experiment.name="Semantic-GazeFollow" \
    experiment.output_folder="$OUTPUT_ROOT/finetune_semantic_gazefollow" \
    wandb.project_name=OmniGF-semantic \
    "train.pretrained_ckpt='$PRETRAINED_CKPT'" \
    train.seed="$SEED" \
    train.devices=1 \
    train.epochs="$EPOCHS" \
    train.batch_size=8 \
    train.accumulate_grad_batches=2 \
    optimizer.lr=5e-5 \
    vlm.head_lr=1e-4 \
    scheduler.type=CosineAnnealingWarmRestarts \
    scheduler.warmup_epochs=1 \
    scheduler.t_0_epochs=15 \
    vlm.approach=heatmap \
    vlm.model_name=Qwen/Qwen3-VL-4B-Instruct \
    vlm.train_lora=True \
    vlm.lora_rank=16 \
    vlm.lora_alpha=32 \
    vlm.decoder_type=conv_transpose \
    vlm.use_gaze_encoder=True \
    vlm.lambda_lm=1.0 \
    vlm.lambda_hm=10.0 \
    vlm.lambda_bce=0.0 \
    vlm.lambda_kl=0.04 \
    vlm.heatmap_loss=mse \
    vlm.prompt_mode=point_semantic \
    vlm.use_semantic_labels=True \
    vlm.use_semantic_head=False \
    vlm.semantic_vocab_dir="$GF_VOCAB_DIR" \
    vlm.test_vocab_embeddings="$GF_VOCAB_DIR/test_vocab_embeddings.pt" \
    vlm.test_vocab_labels="$GF_VOCAB_DIR/test_vocab_labels.json" \
    data.gf_include_inout=False \
    data.gf_use_semgaze_bbox=False \
    data.gf_semgaze_ann_root="$GF_SEMGAZE_ANN_ROOT" \
    val.batch_size=8 \
    test.batch_size=8 \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
