#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Fine-tune OmniGF for semantic gaze on GazeHOI (Table 5), one (person, object) pair
# per sample, with the 463-label GazeHOI vocabulary. One GPU.
#
#   bash scripts/finetune_semantic_gazehoi.sh
#
# Checkpoints are selected by validation Acc@1; the released model is epoch 6.
# Settings are read from the released checkpoint (LoRA LR 5e-5, encoder + heads 1e-4,
# constant LR, weight decay 0.01) and the cluster job that trained it (one person per
# sample, lambda_bce 0, 256 decoding tokens). Its weights are closest to the Table 1
# GazeFollow model, so that is the default warm start. The epoch count was not
# recorded (at least 7); this recipe has not been re-run end to end.
# Evaluate the result with test_semantic_gazehoi.sh.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CKPT_GAZEFOLLOW}"
mapfile -t SOURCE_ARGS < <(source_args gazehoi)

python main_vlm.py \
    experiment.task=train \
    experiment.name="Semantic-GazeHOI" \
    experiment.output_folder="$OUTPUT_ROOT/finetune_semantic_gazehoi" \
    wandb.project_name=OmniGF-semantic \
    "train.pretrained_ckpt='$PRETRAINED_CKPT'" \
    train.seed="$SEED" \
    train.devices=1 \
    train.epochs="$EPOCHS" \
    train.batch_size=8 \
    train.accumulate_grad_batches=2 \
    optimizer.lr=5e-5 \
    vlm.head_lr=1e-4 \
    scheduler.type=constant \
    scheduler.warmup_epochs=0 \
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
    vlm.lambda_kl=0.0 \
    vlm.heatmap_loss=mse \
    vlm.prompt_mode=point_semantic \
    vlm.use_semantic_labels=True \
    vlm.use_semantic_head=False \
    vlm.use_wordnet_reroute=True \
    vlm.wordnet_top_k=10 \
    vlm.max_new_tokens=256 \
    vlm.semantic_vocab_dir="$GAZEHOI_VOCAB_DIR" \
    vlm.test_vocab_embeddings="$GAZEHOI_VOCAB_DIR/test_vocab_embeddings.pt" \
    vlm.test_vocab_labels="$GAZEHOI_VOCAB_DIR/test_vocab_labels.json" \
    data.root_gazehoi="$ROOT_GAZEHOI" \
    data.gazehoi_ann_root="$GAZEHOI_ANN_ROOT" \
    data.gazehoi_vocab2id_path="$GAZEHOI_VOCAB_DIR/vocab2id.json" \
    data.num_people=1 \
    data.num_workers=8 \
    val.batch_size=8 \
    test.batch_size=8 \
    "${SOURCE_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
