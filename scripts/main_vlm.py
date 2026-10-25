# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""VLM gaze prediction training entry point.

Usage::

    cd scripts/
    python main_vlm.py                                 # defaults from vlm_config.yaml
    python main_vlm.py vlm.approach=patch_dist         # Approach 3D
    python main_vlm.py vlm.approach=heatmap            # Approach 3C

Or via the shell launcher::

    bash train_vlm_gazefollow.sh
"""
import torch
import os
import logging
import wandb
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from omnigf.vlm.checkpoint import load_checkpoint
from omnigf.vlm.models_ge import VLMGazeModelGE
from omnigf.vlm.dataset import VLMGazeDataModule
from omnigf.utils import create_train_logger

logger = logging.getLogger("VLM-Experiment")


def build_callbacks(cfg):
    """Build Lightning callbacks for VLM training."""
    callbacks = []
    use_all_data = getattr(cfg.train, "use_all_data", False)

    # Checkpoint callback
    ckpt_folder = os.path.join(cfg.experiment.output_folder, "train/checkpoints/")
    os.makedirs(ckpt_folder, exist_ok=True)

    # auto_insert_metric_name=False keeps the filename a single flat
    # string — Lightning's default otherwise renders ``{epoch:02d}`` as
    # ``epoch=07`` and ``{metric/val/dist:.4f}`` as
    # ``metric/val/dist=0.0863`` (with real path separators and ``=``
    # signs), which breaks Hydra override parsing when the checkpoint
    # is passed as ``train.pretrained_ckpt`` to a test script. We spell
    # out the tag prefixes manually below.
    if use_all_data:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_folder,
                filename="vlm-gaze-epoch{epoch:02d}",
                save_top_k=0,
                save_last=True,
                auto_insert_metric_name=False,
            )
        )
    else:
        use_semantic = getattr(cfg.vlm, "use_semantic_labels", False)
        if use_semantic:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=ckpt_folder,
                    filename="vlm-gaze-epoch{epoch:02d}-semacc{metric/val/sem_text_acc@1:.4f}",
                    monitor="metric/val/sem_text_acc@1",
                    mode="max",
                    save_top_k=1,
                    save_last=False,
                    auto_insert_metric_name=False,
                )
            )
        else:
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
    from lightning.pytorch.strategies import DDPStrategy

    use_all_data = getattr(cfg.train, "use_all_data", False)
    strategy_name = getattr(cfg.train, "strategy", "auto")
    num_devices = getattr(cfg.train, "devices", 1)

    if cfg.device == "cuda":
        if isinstance(num_devices, (list, tuple)):
            n_gpu = len(num_devices)
        elif num_devices == -1:
            n_gpu = torch.cuda.device_count()
        else:
            n_gpu = int(num_devices)
    else:
        n_gpu = 0

    # Multi-GPU: DDP needs find_unused_parameters for approach-specific heads.
    if n_gpu > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = strategy_name

    trainer_kwargs = dict(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        devices=num_devices,
        strategy=strategy,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=train_logger if train_logger else None,
        enable_progress_bar=True,
        log_every_n_steps=10,
        #val_check_interval=1.0,
        check_val_every_n_epoch=1,
        default_root_dir=cfg.experiment.output_folder,
    )
    if use_all_data:
        trainer_kwargs["limit_val_batches"] = 0
    return pl.Trainer(**trainer_kwargs)


@hydra.main(
    config_path="./../omnigf/config/",
    config_name="vlm_config.yaml",
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
    logger.info("Use all data: %s", cfg.train.use_all_data)

    # Build run name: {name}_{approach}_{lr}_{bs}_{lora_rank}_{decoder_type}
    run_name = (
        f"{cfg.experiment.name}"
        f"_{cfg.vlm.approach}_decoder{cfg.vlm.decoder_type}"
        f"_lr{cfg.optimizer.lr}_head{cfg.vlm.head_lr}"
        f"_bs{cfg.train.batch_size}x{cfg.train.accumulate_grad_batches}"
        f"_lambda_lm{cfg.vlm.lambda_lm}"
        f"_lambda_kl{cfg.vlm.lambda_kl}"
        f"_lambda_hm{cfg.vlm.lambda_hm}"
        f"_lambda_bce{cfg.vlm.lambda_bce}"
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
    # Build logger
    train_logger = False
    if "train" in tasks and cfg.wandb.log:
        wandb.login()
        train_logger = create_train_logger(cfg)

    # 1. Build the main trainer unconditionally (no matter if train, val, or test)
    callbacks = build_callbacks(cfg)
    trainer = build_trainer(cfg, callbacks, train_logger)

    # 2. Execute tasks in order using the main DDP trainer
    use_all_data = getattr(cfg.train, "use_all_data", False)
    trained = False
    
    for task in tasks:
        if (task == "test" or task == "val") and not trained:
            test_ckpt = getattr(cfg.test, "checkpoint", None)
            if test_ckpt:
                logger.info("Loading test checkpoint: %s", test_ckpt)
                missing, unexpected = load_checkpoint(model, test_ckpt)
                if missing:
                    logger.warning("Missing keys: %s", missing)
                if unexpected:
                    logger.warning("Unexpected keys: %s", unexpected)
                logger.info("Test checkpoint loaded successfully")
            if task == "test":
                logger.info(f"Starting pre-training test on {trainer.world_size} GPUs")
                trainer.test(model, data)
            elif task == "val":
                logger.info(f"Starting validation on {trainer.world_size} GPUs")
                trainer.validate(model, data)
            
        elif task == "train":
            logger.info(f"Starting training on {trainer.world_size} GPUs")
            if use_all_data:
                logger.info("use_all_data=True: training on train+val, validation disabled")
            ckpt = cfg.train.resume_from if cfg.train.resume else None
            trainer.fit(model, data, ckpt_path=ckpt)
            trained = True
            
        elif task == "test" and trained:
            ckpt_path = "last" if use_all_data else "best"
            logger.info("Starting post-training test (%s checkpoint)", ckpt_path)
            trainer.test(model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
