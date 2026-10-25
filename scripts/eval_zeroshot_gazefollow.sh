#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2026 Qiaomu Miao, Stony Brook University
# SPDX-License-Identifier: GPL-3.0-only

# Zero-shot Qwen3-VL-4B-Instruct on the GazeFollow test set (Table 3, first row):
# no fine-tuning, no gaze encoder, text output only. The base model gets the same
# point-mode prompt as OmniGF; outputs that do not parse are scored as the image
# centre (65.3% of outputs parsed in the paper's run).
#
#   bash scripts/eval_zeroshot_gazefollow.sh
#   NUM_SAMPLES=100 bash scripts/eval_zeroshot_gazefollow.sh   # quick check
#
# Paper's run: Avg Dist. 0.340.

source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_ROOT/zeroshot_gazefollow/metrics.json}"
mkdir -p "$(dirname "$OUTPUT_JSON")"
EXTRA=()
[ -n "${NUM_SAMPLES:-}" ] && EXTRA+=(--num_samples "$NUM_SAMPLES")

python eval_zeroshot_gazefollow.py \
    --config_name vlm_config.yaml \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --max_new_tokens 96 \
    --batch_size "${BATCH_SIZE:-4}" \
    --output_json "$OUTPUT_JSON" \
    --config_overrides \
        data.ann_root="$ANN_ROOT" \
        data.root_gazefollow="$ROOT_GAZEFOLLOW" \
    "${EXTRA[@]}" \
    "$@"
