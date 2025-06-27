"""Training script."""

import argparse
import os
import random
import time
from datetime import timedelta
from typing import Optional

import lightning.pytorch as pl
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities import rank_zero_only

from spec_mamba.exp.params import Params
from spec_mamba.training import TrainModule

load_dotenv(os.path.join(os.getcwd(), ".env"))
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


@rank_zero_only
def _log_seed(logger: WandbLogger, seed: int) -> None:
    logger.experiment.config.update({"seed": seed})


def train(
    params: Params, seed: Optional[int] = None, profile: bool = False
) -> tuple[Trainer, TrainModule]:
    """Training function."""

    if (WANDB_API_KEY is not None) and (not params.train_args.fast_dev_run):
        wandb.login(relogin=True, verify=True, key=WANDB_API_KEY)
        offline = False
    else:
        offline = True

    logger = WandbLogger(
        project=params.train_args.project,
        name=params.train_args.run,
        offline=offline,
    )

    seed = seed if seed is not None else random.randint(0, 1000)
    pl.seed_everything(seed)
    _log_seed(logger=logger, seed=seed)

    model = params.train_module_type(
        model_type=params.model_type,
        model_args=params.model_args,
        data_args=params.data_args,
        train_args=params.train_args,
    )

    profiler = (
        AdvancedProfiler(dirpath=os.getcwd(), filename="profiler.out")
        if profile
        else None
    )

    trainer_kwargs = (
        params.train_args.trainer_kwargs
        if params.train_args.trainer_kwargs is not None
        else {}
    )
    trainer = Trainer(
        max_epochs=params.train_args.max_epochs,
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=params.train_args.devices,
        strategy=params.train_args.strategy,
        fast_dev_run=params.train_args.fast_dev_run,
        profiler=profiler,
        **trainer_kwargs,
    )

    torch.cuda.synchronize()
    start_time = time.time()

    trainer.fit(model)

    torch.cuda.synchronize()
    end_time = time.time()

    print("Training time:", timedelta(seconds=int(end_time - start_time)))

    return trainer, model


def run_exp(params: Params, seed: Optional[int] = None, profile: bool = False) -> None:
    """Train, test and log all metrics and results."""
    trainer, model = train(params=params, seed=seed, profile=profile)
    trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cfg",
        type=str,
        help="Configuration file name (without .py extension).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling.",
    )
    args = parser.parse_args()

    params = Params.from_cfg(args.cfg)
    run_exp(params=params, seed=args.seed, profile=args.profile)
