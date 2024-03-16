"""
The implementation of generic components for the diffusion model.
"""

from typing import Optional
from dataclasses import dataclass

import torch

from vggbase.utils.generic_components import FieldFrozenContainer


@dataclass
class BaseDiffusionOutput(FieldFrozenContainer):
    """
    Base class for diffusion model's outputs.
    Args:
        prediction_type (`str`) presenting what has been
         predicted by the diffusion head. Generally, there
         are three options:
            - noise: predicted noise of step t.
            - start: predicted sample of start, i.e., t=0
            - customization: predicted customized one
                        defined by the user.

        predictions (`torch.FloatTensor`) with shape
         For image-relative tasks, [batch_size, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [batch_size, B, 4]
         where B should be same as the input bounding boxes.

        diffusion_targets (`torch.FloatTensor`) with shape
         same as the predictions.
         Generally, the user should create targets to compute
         the loss afterwards. There are typically three options
         depended on the prediction_type.
         ---------------------------
         prediction         |   targets
         noise              |   noise
         start              |   start
         customization      |   customization

        predictions_weight (`torch.FloatTensor`) with shape
         [batch_size, *(1,)*D]
         where D denotes the dimension of the predictions.

        time_steps (`torch.IntTensor`) with shape
         [batch_size, *]
         where the `*` means that the #dimensions of time_steps
         should be consistent with `predictions`.

    """

    prediction_type: Optional[str] = None
    predictions: Optional[torch.FloatTensor] = None
    diffusion_targets: Optional[torch.FloatTensor] = None
    predictions_weight: Optional[torch.FloatTensor] = None
    time_steps: Optional[torch.IntTensor] = None


@dataclass
class BaseDiffusionHeadOutput(FieldFrozenContainer):
    """
    Base class for outputs of the diffusion model head.
    Args:
        prediction_type (`str`) presenting what has been
         predicted by the diffusion head. Generally, there
         are three options:
            - noise: predicted noise of step t.
            - start: predicted sample of start, i.e., t=0
            - velocity: predicted velocity defined by
                        work [3].
        predictions (`torch.FloatTensor`) with shape
         For image-relative tasks, [M, batch_size, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [M, batch_size, B, 4]
         where B should be same as the input bounding boxes.

         where `M` denotes how many models contained in the
         head as a series of model.
    """

    prediction_type: Optional[str] = None
    predictions: Optional[torch.FloatTensor] = None


@dataclass
class BaseDiffusionReversePredictions(FieldFrozenContainer):
    """
    Base class for the reverse's predictions based on the
    diffusion head model.

    The diffusion head model can either predict noises
    or the start samples `x_0`.
    Once noises is predicted, the `x_0` can be predicted
    by applying the ``get_diffusion_forward_start`` function.
    Once start samples is predicted, the noises can be predicted
    by applying the ``get_diffusion_forward_noises`` function.

    Args:
        predicted_noises (`torch.FloatTensor`) with shape
         For image-relative tasks, [batch_size, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [batch_size, B, 4]
         where B should be same as the input bounding boxes.

        predicted_start_samples (`torch.FloatTensor`) with shape
         For image-relative tasks, [batch_size, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [batch_size, B, 4]
         where B should be same as the input bounding boxes.
        time_steps (`torch.IntTensor`) with shape
         [batch_size, *]
         where the `*` means that the #dimensions of time_steps
         should be consistent with `predictions`.
    """

    predicted_noises: Optional[torch.FloatTensor] = None
    predicted_start_samples: Optional[torch.FloatTensor] = None
    time_steps: Optional[torch.IntTensor] = None


@dataclass
class BaseDiffusionPosterior(FieldFrozenContainer):
    """
    Base class for the posterior of diffusion models.

    Args:
        posterior_mean (`torch.FloatTensor`) with shape
         For image-relative tasks, [batch_size, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [batch_size, B, 4]
         where B should be same as the input bounding boxes.

        posterior_variance_betas (`torch.FloatTensor`) with shape
         that is the same as the `posterior_mean`.

        posterior_log_variance_betas (`torch.FloatTensor`) with shape
         that is the same as the `posterior_mean`.
    """

    posterior_mean: Optional[torch.FloatTensor] = None
    posterior_variance_betas: Optional[torch.FloatTensor] = None
    posterior_log_variance_betas: Optional[torch.FloatTensor] = None


@dataclass
class BaseDiffusionReverseSamples(FieldFrozenContainer):
    """
    Base class for reverse sampling process of diffusion models.

    Args:
        reverse_samples (`torch.FloatTensor`) with shape
         For image-relative tasks, [batch_size, N, C, H, W]
         where C, H, W are generally same as the input.
         For box-relative tasks, [batch_size, N, B, 4]
         where B should be same as the input bounding boxes.

         where N denotes how many samples is generated
         based on the steps, making each sample correspond to
         one time step .

        time_steps (`torch.IntTensor`) with shape
         [batch_size, N].
         where N is the number of time steps, making
         each value corresponds to one time step id.
    """

    time_steps: Optional[torch.IntTensor] = None
    reverse_samples: Optional[torch.FloatTensor] = None
