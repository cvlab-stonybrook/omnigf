# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""VLM gaze prediction training entry point.

Usage::

    cd scripts/
    python main_vlm_vsgaze.py                                          # defaults from vlm_vsgaze_config.yaml
    python main_vlm_vsgaze.py --config_name vlm_videoatt_config.yaml   # use a different config
    python main_vlm_vsgaze.py vlm.approach=heatmap                     # override specific values

Or via the shell launcher::

    bash train_vlm_vsgaze.sh
"""

import argparse
import os
import logging
import sys

import hydra
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from omnigf.vlm.checkpoint import load_checkpoint
from omnigf.vlm.models_ge import VLMGazeModelGE
from omnigf.vlm.dataset import VLMGazeDataModule
from omnigf.utils import create_train_logger

logger = logging.getLogger("VLM-Experiment")


def _parse_config_name(default="vlm_vsgaze_config.yaml"):
    """Extract --config_name from argv before Hydra processes it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_name", default=default)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.config_name


_CONFIG_NAME = _parse_config_name()


def build_callbacks(cfg):
    """Build Lightning callbacks for VLM training."""
    callbacks = []

    # Checkpoint callback
    ckpt_folder = os.path.join(cfg.experiment.output_folder, "train/checkpoints/")
    os.makedirs(ckpt_folder, exist_ok=True)

    # auto_insert_metric_name=False renders placeholders as bare values
    # (``{epoch:02d}`` -> ``07``) instead of Lightning's default
    # ``epoch=07`` format — which, combined with the ``/`` in the
    # monitored metric ``metric/val/dist``, would otherwise produce
    # filenames like ``vlm-gaze-epoch=07-metric/val/dist=0.0863.ckpt``
    # (two subdirs + two ``=`` signs). The ``=`` breaks Hydra override
    # parsing downstream, and the ``/`` scatters checkpoints into
    # subdirectories. Spell out the tag prefixes manually.
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_folder,
            filename="vlm-gaze-epoch{epoch:02d}-dist{metric/val/dist:.4f}",
            monitor="metric/val/dist",
            mode="min",
            save_top_k=2,
            save_last=False,
            auto_insert_metric_name=False,
        )
    )

    # LR monitor (for wandb logging)
    if cfg.wandb.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    # SWA (optional)
    if cfg.train.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(
                swa_lrs=cfg.train.swa.lr,
                swa_epoch_start=cfg.train.swa.epoch_start,
                annealing_epochs=cfg.train.swa.annealing_epochs,
            )
        )

    return callbacks


def build_trainer(cfg, callbacks, train_logger):
    """Build Lightning Trainer for VLM training."""
    return pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        devices=getattr(cfg.train, "devices", 1),
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=train_logger if train_logger else None,
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        default_root_dir=cfg.experiment.output_folder,
    )


@hydra.main(
    config_path="./../omnigf/config/",
    config_name=_CONFIG_NAME,
    version_base=None,
)
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("VLM Gaze Prediction Experiment")
    logger.info("=" * 60)
    logger.info("Approach: %s", cfg.vlm.approach)
    logger.info("Model: %s", cfg.vlm.model_name)
    logger.info("Dataset: %s", cfg.experiment.dataset)
    logger.info("Task: %s", cfg.experiment.task)
    logger.info("Output: %s", cfg.experiment.output_folder)

    # Build run name: {name}_{approach}_{lr}_{bs}_{lora_rank}_{decoder_type}
    run_name = (
        f"{cfg.experiment.name}"
        f"_{cfg.vlm.approach}"
        f"_lr{cfg.optimizer.lr}_head{cfg.vlm.head_lr}"
        f"_bs{cfg.train.batch_size}x{cfg.train.accumulate_grad_batches}"
        f"_lambda_lm{cfg.vlm.lambda_lm}"
        f"_lambda_bce{cfg.vlm.lambda_bce}"
        f"_lambda_kl{cfg.vlm.lambda_kl}"
        f"_lambda_hm{cfg.vlm.lambda_hm}"
        f"_lora{cfg.vlm.lora_rank}"
        f"_{cfg.vlm.decoder_type}"
    )
    OmegaConf.update(cfg, "experiment.name", run_name)
    OmegaConf.update(
        cfg, "experiment.output_folder",
        os.path.join(cfg.experiment.output_folder, run_name),
    )
    logger.info("Run name: %s", run_name)
    logger.info("Output: %s", cfg.experiment.output_folder)

    # Create output folder
    os.makedirs(cfg.experiment.output_folder, exist_ok=True)

    # Precision and seed
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    if cfg.train.seed is not None:
        pl.seed_everything(cfg.train.seed)

    # Parse tasks
    tasks = cfg.experiment.task.split("+")

    # Build data module
    data = VLMGazeDataModule(cfg)

    # Build model
    use_gaze_encoder = getattr(cfg.vlm, "use_gaze_encoder", False)
    model = VLMGazeModelGE(cfg)

    # Load pretrained weights (model only, no optimizer/scheduler state)
    pretrained_ckpt = getattr(cfg.train, "pretrained_ckpt", None)
    if pretrained_ckpt is not None:
        logger.info("Loading pretrained weights from: %s", pretrained_ckpt)
        missing, unexpected = load_checkpoint(model, pretrained_ckpt)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        logger.info("Pretrained weights loaded successfully")

    # Optionally freeze LoRA adapters (heads-only fine-tuning)
    if not getattr(cfg.vlm, "train_lora", True):
        n_frozen = 0
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = False
                n_frozen += 1
        logger.info("Froze %d LoRA parameters (heads-only training)", n_frozen)

    # Build logger
    train_logger = False
    if "train" in tasks and cfg.wandb.log:
        train_logger = create_train_logger(cfg)

    # Build callbacks and trainer
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks, train_logger)

    trained = False
    for task in tasks:
        if task == "test" and not trained:
            test_ckpt = pretrained_ckpt
            assert test_ckpt is not None, "Test checkpoint (pretrained_ckpt) is not set"
            logger.info("Starting test with checkpoint: %s", test_ckpt)
            trainer.test(model, data)
        elif task == "val" and not trained:
            logger.info("Starting validation")
            trainer.validate(model, data)
        elif task == "train":
            logger.info("Starting training")
            ckpt = cfg.train.resume_from if cfg.train.resume else None
            trainer.fit(model, data, ckpt_path=ckpt)
            trained = True
        elif task == "test" and trained:
            # Post-training test: use best checkpoint from training
            logger.info("Starting post-training test (last checkpoint)")
            trainer.test(model, data, ckpt_path="best")


if __name__ == "__main__":
    main()
