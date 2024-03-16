"""
The implementation of the generalized head used for diffusion model.

In general, diffusion model relies on an external model to make prediction
for each time step in the reverse process. For example, in the image generation
task, this external model can be a <UNet> that outputs a 3-channel feature map as
the predicted noise, which has the same shape as the input image.

However, in this implementation, we aim to make it more flexible with the
target to supporting more scenarios, such as object detection and visual grounding.

To be specific, this generalize head is responsible for
1). mapping time (discrete value) to a feature space
2). operating the models to make task-oriented predictions

Such an external model corresponds to the `theta` in the math equation.
"""

from typing import Type
import logging

import torch
from torch import nn

from vggbase.models.position_encoding import SinusoidalPosition1DEncoder
from vggbase.config import Config
from vggbase.models.module_utils import get_clones

from .UNets import UNet
from .diffusion_components import BaseDiffusionHeadOutput


class GeneralizedDiffusionHead(nn.Module):
    """
    Main class for Generalized Head for the Diffusion Model.

    :param time_embed_n_features: A `int` denoting the time's feature size after
        embedding.
    :param head_model_config: A `dict` containing the detailed configuration of
        the head model.
    """

    def __init__(
        self,
        time_embedding_config: Type[Config],
        time_projection_config: Type[Config],
        head_model_config: Type[Config],
    ) -> None:
        super().__init__()

        # obtain the prediction_type of the head
        self.prediction_type = head_model_config.prediction_type

        # whether return the outputs of a series_outputs head models
        self.return_series_outputs = head_model_config.return_series_outputs

        # the time embedder is used to map the `time_steps`
        # from an integer to a continuous space.
        self.time_embedder = self.init_time_embedder(time_embedding_config)
        # the project aims to map the embedded time to
        # a dense space containing possibly more semantics
        # facilitating subsequent learning.
        self.time_projector = self.init_time_projector(time_projection_config)

        # by default, the head contains a series of models
        self.head_model_series = self.init_head_model_series(head_model_config)

    def init_time_embedder(self, time_embedding_config):
        """Initialize the time embedder."""
        out_features = time_embedding_config.out_features
        logging.info("Initialized time embedder as Sinusoidal Positional Encoding.")
        return SinusoidalPosition1DEncoder(
            enc_features=out_features, is_custom_position_ids=True
        )

    def init_time_projector(self, time_projection_config):
        """Initialize the time projector."""
        logging.info("Initialized time projector as Identity.")
        return nn.Identity()

    def init_head_model_series(self, head_model_config):
        """Initialize the model for this diffusion head."""
        n_repeat = 1
        if hasattr(head_model_config, "n_repeat"):
            n_repeat = head_model_config.n_repeat

        module = UNet()
        logging.info("Initialized diffusion head as UNet.")
        return get_clones(module, n_repeat)

    def forward(self, ts_samples: torch.Tensor, time_steps: torch.Tensor, **kwargs):
        """Forward a series of head models.

        :param ts_samples: A `torch.FloatTensor` with shape
         [bs, h, w, 3] for general image generation
         or [bs, N, 4] for boundingbox-bases tasks.
         where `ts_` here denotes the time step, meaning that the
         samples come from one time step

        :param time_steps: A `torch.IntTensor` with shape
            [bs, ] a 1 dimension tensor.

        Based on Algorithm presented in paper [1], the `ts_samples`
        can be two  types.

        For Algorithm 1 Training, `ts_samples` should be the
        noised raw sample computed based on `x_0` and `t` by applying
        equation shown in Forward diffusion process section of [2].
        Also see the equation in <Algorithm 1 Training> table, which
        is less direct than the one in [2]. But, they are the same.
        Forward diffusion process:
            `X_{0}` -> `X_{1}`-> ... -> `X_{t-1}` -> `X_{t}` -> ... -> `X_{T}`
        `ts_samples` should be computed within this process. But based on math
        derivation, `ts_samples` can be computed directly as what mentioned above.

        For Algorithm 2 Sampling, `ts_samples` should be the
        noised reverse sample computed based on `x_{t+1}` and `t` by applying
        equation shown in in <Algorithm 2 Sampling> table.
        Reverse diffusion process:
            `X_T` -> `X_{T-1}`-> ... -> `X_{t+1}` -> `X_{t}` -> ... -> `X_{0}`
        `ts_samples` should be computed iteratively from iteratively
        from `X_T` to `x_{t+1}` to `ts_samples`. Thus, this is a `for` loop.

        where `ts_samples` should be `x_t` in equation. `x_0` is the raw input sample
        without any diffusions. `T` is the Markov chain steps.

        Reference:
        [1]. Denoising Diffusion Probabilistic Models.
        [2]. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.
        """

        # convert to the shape [batch_size, 1]
        time_steps = time_steps.reshape(-1, 1)
        # embed t to be one feature space
        # shape, [batch_size, 1, embed_n_features]
        time_steps = self.time_embedder(time_steps)
        # convert to [batch_size, embed_n_features]
        time_steps = time_steps.squeeze(1)
        time_steps = self.time_projector(time_steps)
        # forward the embedded t and ts_samples
        head_series_outputs = []
        for _, head_model in enumerate(self.head_model_series):
            # output of one module,
            # shape, [batch_size, *]
            # where `*` is the shape related to head model
            head_model_output = head_model(ts_samples, time_steps)

            if self.return_series_outputs:
                head_series_outputs.append(head_model_output)
            else:
                head_series_outputs = head_model_output

        if self.return_series_outputs:
            # obtain the outputs of all series of models
            # shape, [n_repeat, batch_size, *]
            # where `n_repeat` can be accessed in
            # the function "init_head_model_series".
            head_series_outputs = torch.stack(head_series_outputs)
        else:
            # add to first dimension to make the shape to
            # [1, *]
            head_series_outputs = head_series_outputs[None]

        return BaseDiffusionHeadOutput(
            prediction_type=self.prediction_type, predictions=head_series_outputs
        )
