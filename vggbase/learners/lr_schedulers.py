"""
Returns a learning rate scheduler according to the configuration.

Copy from the Plato platform https://github.com/TL-System/plato.

Most of the code is copied from the Plato.

We made some modifications.

"""

import sys
from types import SimpleNamespace
from typing import List

from timm import scheduler
from torch import optim


class LRSchedulerBuilder(object):
    """A builder to create the lr scheduler."""

    def extract_params(self, trainer_config: dict, target_lr_scheduler: None):
        """Get an optimizer based on the required opt_name."""
        if target_lr_scheduler is None:
            target_lr_scheduler = "lr_scheduler"
        scheduler_name = trainer_config[target_lr_scheduler]
        lr_params = trainer_config["parameters"][target_lr_scheduler]
        return scheduler_name, lr_params

    def get(
        self,
        optimizer: optim.Optimizer,
        trainer_config: dict,
        target_lr_scheduler: str = "lr_scheduler",
    ):
        """Return a learning rate scheduler according to the configuration."""

        scheduler_name, lr_params = self.extract_params(
            trainer_config, target_lr_scheduler
        )

        lr_params["sched"] = scheduler_name
        lr_params["epochs"] = trainer_config["epochs"]
        scheduler_args = SimpleNamespace(**lr_params)
        lr_scheduler, _ = scheduler.create_scheduler(
            args=scheduler_args, optimizer=optimizer
        )

        if lr_scheduler is None:
            sys.exit("Error: Unknown learning rate scheduler.")

        return lr_scheduler

    def get_multiple(
        self, optimizers: List[optim.Optimizer], trainer_configs: List[dict]
    ):
        """Get multiple optimizers."""
        return [
            self.get(opt, opt_config)
            for opt, opt_config in zip(optimizers, trainer_configs)
        ]
