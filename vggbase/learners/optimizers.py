"""
Optimizers for training workloads.
"""

import torch_optimizer as torch_optim
from torch import optim


class OptimizerBuilder(object):
    """A builder to create Optimizers."""

    def extract_params(self, trainer_config: dict, target_optimizer: None):
        """Get an optimizer based on the required opt_name."""
        if target_optimizer is None:
            target_optimizer = "optimizer"
        optim_name = trainer_config[target_optimizer]
        optim_params = trainer_config["parameters"][target_optimizer]
        return optim_name, optim_params

    def assign_params(self, named_parameters, optimizer_parameters: dict):
        """Customize the configuration parameters."""

        param_dicts = [
            {
                "params": [
                    param for _, param in named_parameters if param.requires_grad
                ],
                "lr": optimizer_parameters["lr"],
            }
        ]

        return param_dicts

    def get(
        self,
        model_named_parameters,
        trainer_config: dict,
        target_optimizer: str = "optimizer",
    ) -> optim.Optimizer:
        """Get an optimizer with its name and parameters obtained from the configuration file."""
        registered_optimizers = {
            "Adam": optim.Adam,
            "Adadelta": optim.Adadelta,
            "Adagrad": optim.Adagrad,
            "AdaHessian": torch_optim.Adahessian,
            "AdamW": optim.AdamW,
            "SparseAdam": optim.SparseAdam,
            "Adamax": optim.Adamax,
            "ASGD": optim.ASGD,
            "LBFGS": optim.LBFGS,
            "NAdam": optim.NAdam,
            "RAdam": optim.RAdam,
            "RMSprop": optim.RMSprop,
            "Rprop": optim.Rprop,
            "SGD": optim.SGD,
        }

        optim_name, optim_params = self.extract_params(trainer_config, target_optimizer)
        model_parameters = self.assign_params(
            model_named_parameters, optimizer_parameters=optim_params
        )
        optimizer = registered_optimizers.get(optim_name)
        if optimizer is not None:
            return optimizer(model_parameters, **optim_params)

        raise ValueError(f"No such optimizer: {optim_name}")
