#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Shared paths for every launcher in scripts/. Sourced, not run.
# Each variable can be overridden from the environment, e.g.
#   DATA_ROOT=/mnt/gaze CKPT_ROOT=/mnt/omnigf bash scripts/test_gazefollow.sh
# See README.md ("Data" and "Checkpoints") for the expected layout.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# ---- Annotations ----
ANN_ROOT="${ANN_ROOT:-$DATA_ROOT/VSGaze}"                       # <source>_{train,val,test}.h5
GF_SEMGAZE_ANN_ROOT="${GF_SEMGAZE_ANN_ROOT:-$ANN_ROOT/semgaze_merge}"  # GazeFollow with semantic labels
GAZEHOI_ANN_ROOT="${GAZEHOI_ANN_ROOT:-$DATA_ROOT/GazeHOI/h5_annotations}"
GF_VOCAB_DIR="${GF_VOCAB_DIR:-$DATA_ROOT/vocab/gazefollow}"      # CLIP label embeddings
GAZEHOI_VOCAB_DIR="${GAZEHOI_VOCAB_DIR:-$DATA_ROOT/vocab/gazehoi}"

# ---- Images ----
ROOT_GAZEFOLLOW="${ROOT_GAZEFOLLOW:-$DATA_ROOT/gazefollow}"
ROOT_VAT="${ROOT_VAT:-$DATA_ROOT/videoattentiontarget}"
ROOT_CHILDPLAY="${ROOT_CHILDPLAY:-$DATA_ROOT/ChildPlay}"
ROOT_VIDEOCOATT="${ROOT_VIDEOCOATT:-$DATA_ROOT/VideoCoAtt_Dataset}"
ROOT_UCO_LAEO="${ROOT_UCO_LAEO:-$DATA_ROOT/ucolaeodb}"
ROOT_GAZEHOI="${ROOT_GAZEHOI:-$DATA_ROOT/GazeHOI}"

# ---- Weights ----
GAZE_WEIGHTS="${GAZE_WEIGHTS:-$CKPT_ROOT/gaze360_resnet18.pt}"   # Gaze360 ResNet-18 head encoder init
CKPT_GAZEFOLLOW="${CKPT_GAZEFOLLOW:-$CKPT_ROOT/omnigf_gazefollow.safetensors}"              # Table 1
CKPT_GAZEFOLLOW_INOUT="${CKPT_GAZEFOLLOW_INOUT:-$CKPT_ROOT/omnigf_gazefollow_inout.safetensors}"  # warm start for VSGaze
CKPT_VAT="${CKPT_VAT:-$CKPT_ROOT/omnigf_vat.safetensors}"                                    # Table 1
CKPT_CHILDPLAY="${CKPT_CHILDPLAY:-$CKPT_ROOT/omnigf_childplay.safetensors}"                  # Table 2
CKPT_SEMANTIC_GAZEFOLLOW="${CKPT_SEMANTIC_GAZEFOLLOW:-$CKPT_ROOT/omnigf_semantic_gazefollow.safetensors}"  # Table 4
CKPT_SEMANTIC_GAZEHOI="${CKPT_SEMANTIC_GAZEHOI:-$CKPT_ROOT/omnigf_semantic_gazehoi.safetensors}"          # Table 5
CKPT_SOCIAL="${CKPT_SOCIAL:-$CKPT_ROOT/omnigf_social_vsgaze.safetensors}"                    # Table 6

# ---- Logging ----
# wandb is off unless WANDB=True (then set WANDB_API_KEY, or run `wandb login` once).
# WANDB_ENTITY picks the team/user; unset uses your default entity.
WANDB="${WANDB:-False}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"

# Hydra arguments shared by every run: data roots, head-encoder init, logging.
COMMON_ARGS=(
    data.ann_root="$ANN_ROOT"
    data.root_gazefollow="$ROOT_GAZEFOLLOW"
    data.root_vat="$ROOT_VAT"
    data.root_childplay="$ROOT_CHILDPLAY"
    data.root_videocoatt="$ROOT_VIDEOCOATT"
    data.root_uco_laeo="$ROOT_UCO_LAEO"
    data.root_gazehoi="$ROOT_GAZEHOI"
    data.gazehoi_ann_root="$GAZEHOI_ANN_ROOT"
    vlm.gaze_weights="$GAZE_WEIGHTS"
    wandb.log="$WANDB"
    wandb.username="$WANDB_ENTITY"
)

# Print Hydra arguments that enable exactly the given sources for train, val and test.
#   mapfile -t SOURCE_ARGS < <(source_args vat)
source_args() {
    local split src on
    for split in include_sources val_include_sources test_include_sources; do
        for src in gazefollow vat childplay videocoatt uco_laeo gazehoi; do
            on=False
            [[ " $* " == *" $src "* ]] && on=True
            echo "data.$split.$src=$on"
        done
    done
}

# Hydra writes its run directory under $REPO_ROOT/experiments/ (relative to scripts/).
cd "$REPO_ROOT/scripts" || exit 1
