# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import omegaconf

import wandb

from lightning.pytorch.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def create_train_logger(cfg):
    if cfg.wandb.log:
        id = wandb.util.generate_id()
        logger = WandbLogger(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.username,
            group=cfg.wandb.group,
            log_model=False,
            id=id,
            name=cfg.experiment.name,
            save_dir="./",
            allow_val_change=True,
        )

        cfg_dict = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if rank_zero_only.rank == 0:
            logger.experiment.config.update(cfg_dict)
    else:
        logger = False

    return logger
