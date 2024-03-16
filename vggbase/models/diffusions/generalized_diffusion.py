"""
The basic diffusion models who serves as a basic
structure for subsequent implementations.

The original paper for diffusion model is [1].

The mathematical definition used in this code follows the
Appendix B of paper [2].

The best toturial:
"https://lilianweng.github.io/posts/2021-07-11-diffusion-models/".

References:
    [1]. Denoising Diffusion Probabilistic Models.
    [2]. Diffusion Models Beat GANs on Image Syntheis.
    [3]. Perception Prioritized Training of Diffusion Models.
    [4]. https://lilianweng.github.io/posts/2021-07-11-diffusion-models.
    [5]. https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
"""

from typing import Optional, Dict, Type

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from vggbase.config import Config

from .generalized_diffusion_head import GeneralizedDiffusionHead
from .noise_variance_schedules import get as get_schedule
from .diffusion_components import BaseDiffusionOutput
from .diffusion_components import (
    BaseDiffusionReversePredictions,
    BaseDiffusionPosterior,
)
from .diffusion_utils import extract_items, process_samples_fn, identity
from .diffusion_utils import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one


class GeneralizedDiffusionModel(nn.Module):
    """
    Main class for Generalized Diffusion Model.

    :param chain_steps: A `int` denoting how many steps contained in the
     Markov chain. Represented in mathematics as `t`.
    :param noise_variance_schedule_config: A `dict` containing the configuration
     of the noise_variance_schedule, which is the `beta` in [1] and [2].
     the key `noise_variance_schedule` must be included.
    :param diffusion_head_config: A `dict` containing the configuration for the
     diffusion head.
    :param out_weights_config: A `dict` containing the configuration for the
     weights added to the model's outputs.
    :param normalization_config: A `dict` containing the configuration for normalizing
     the samples.
    :param reverse_sampling_config: A `str` presenting which sampling to be used.
     Here are two basic options: posterior and ddpm
    :param device: A `torch.device` denoting the deploy device.
    """

    def __init__(
        self,
        chain_steps: int,
        noise_variance_schedule_config: Type[Config],
        diffusion_head_config: Type[Config],
        out_weights_config: Type[Config],
        normalization_config: Type[Config],
        reverse_sampling_config: str,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.device = device

        self.chain_steps = chain_steps

        self.reverse_sampling_config = reverse_sampling_config
        self.reverse_sampling_schema = self.reverse_sampling_config.schema

        self.noise_variance_scheduler = (
            noise_variance_schedule_config.noise_variance_scheduler
        )
        self.noise_variance_schedule_setup = (
            noise_variance_schedule_config.noise_variance_schedule_setup
        )

        # head of the diffusion model
        self.diffusion_head = None
        self.init_diffusion_head(diffusion_head_config)

        ## mainly for Forward diffusion process
        # prepare items for q(x_t | x_{t-1}) and others
        self.prepare_forward_equations()

        ## mainly for Reverse diffusion process
        # posterior q(x_{t-1} | x_t, x_0)
        self.prepare_reverse_equations()

        ## mainly for the subsequent loss computation
        self.prepare_output_weight(out_weights_config)

        ## mainly for normalizing the generated or sampled samples
        # in the reverse process
        self.scale = None
        self.prepare_normalizations(normalization_config)

    def prepare_forward_equations(self):
        """Prepare equations mainly for Forward diffusion process
        prepare items for q(x_t | x_{t-1}) and others."""

        # for simplify, we utilize `betas` to denote the
        # noise_variance_scheduler
        # prepare the `betas` in advance for the whole chain steps
        # `betas`, tensor with shape [chain_steps]
        # where `betas` in Eq. 15 of [2].
        betas = get_schedule(
            scheduler=self.noise_variance_scheduler,
            chain_steps=self.chain_steps,
            **Config().items_to_dict(self.noise_variance_schedule_setup._asdict()),
        )
        # based on the reparameterization trick in the math derivation,
        # the `alpha_t = 1 - beta_t` should be prepared in advance.
        # and, `alpha_t_hat = \prod_i^t alpha_i`
        # alphas, tensor with shape [chain_steps]
        # alphas_hat, tensor with shape [chain_steps]
        # alphas_hat_prev, tensor with shape [chain_steps]
        # for each step `t`, alphas_hat_prev[t] = alphas_hat[t-1]
        # where `alphas` in Eq. 16 of [2].
        alphas = 1.0 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        alphas_hat_prev = F.pad(alphas_hat[:-1], (1, 0), value=1.0)

        betas = betas.to(self.device)
        alphas = alphas.to(self.device)
        alphas_hat = alphas_hat.to(self.device)
        alphas_hat_prev = alphas_hat_prev.to(self.device)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_hat", alphas_hat)
        self.register_buffer("alphas_hat_prev", alphas_hat_prev)

        # prepare terms for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_hat", torch.sqrt(alphas_hat))
        self.register_buffer("sqrt_one_minus_alphas_hat", torch.sqrt(1.0 - alphas_hat))
        self.register_buffer("log_one_minus_alphas_hat", torch.log(1.0 - alphas_hat))
        self.register_buffer("sqrt_recip_alphas_hat", torch.sqrt(1.0 / alphas_hat))
        self.register_buffer(
            "sqrt_recipm1_alphas_hat", torch.sqrt(1.0 / alphas_hat - 1)
        )

    def prepare_reverse_equations(self):
        """mainly for Reverse diffusion process
        posterior q(x_{t-1} | x_t, x_0)."""
        betas = self.betas
        alphas = self.alphas
        alphas_hat = self.alphas_hat
        alphas_hat_prev = self.alphas_hat_prev
        # 1. Please access `\widetilde{\beta_t}` on the website
        #   or Eq. 19 of [2].
        posterior_variance_betas = betas * (1.0 - alphas_hat_prev) / (1.0 - alphas_hat)
        # 2. Please access `\widetilde{\mu_t}` on the website
        #   or Eq. 18 of [2].
        posterior_mean_coef1 = betas * torch.sqrt(alphas_hat_prev) / (1.0 - alphas_hat)
        posterior_mean_coef2 = (
            (1.0 - alphas_hat_prev) * torch.sqrt(alphas) / (1.0 - alphas_hat)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance_betas.clamp(min=1e-20)),
        )
        self.register_buffer("posterior_variance_betas", posterior_variance_betas)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def prepare_output_weight(self, out_weight_config: dict):
        """Prepare the weights for the output of diffusion head.

        Currently, we only support the p2 weighting mechanism.
            p2_weighting_gamma: A `float` denoting the `gamma` term
             of the p2 weighting mechanism [3].
            p2_weighting_k: A `float` denoting the `k` term in Eq. 8
            of the p2 weighting mechanism [3].
        See Eq. 8 in the work [3].

        """
        p2_weighting_gamma = out_weight_config.p2_weighting_gamma
        p2_weighting_k = out_weight_config.p2_weighting_k

        # calculate p2 reweighting
        self.register_buffer(
            "p2_weighting",
            (p2_weighting_k + self.alphas_hat / (1 - self.alphas_hat))
            ** -p2_weighting_gamma,
        )

    def prepare_normalizations(self, normalization_config):
        """Prepare the normalizations to process the samples of the
        diffusion model."""

        # auto-normalization of data [0, 1] ->
        # [-1, 1] - can turn off by setting it to be False
        self.normalize_samples = (
            normalize_to_neg_one_to_one
            if normalization_config.auto_normalize
            else identity
        )
        self.unnormalize_samples = (
            unnormalize_to_zero_to_one
            if normalization_config.auto_normalize
            else identity
        )

    def get_out_weight(self, extract_ids, out_n_dim):
        """Extract the weight for the output."""
        return extract_items(self.p2_weighting, extract_ids, out_n_dim)

    def init_diffusion_head(self, diffusion_head_config):
        """Define the diffusion head."""

        self.diffusion_head = GeneralizedDiffusionHead(
            diffusion_head_config.time_embedding,
            diffusion_head_config.time_projection,
            diffusion_head_config.head_model,
        )

    def get_diffusion_forward_samples(self, samples_start, steps_id, noises=None):
        """Compute the noised sample q(x_t | x_0) along the forward chain,
        which is given by Eq. 17 of [2]."""

        noises = (
            noises if noises is not None else lambda: torch.randn_like(samples_start)
        )
        n_dims = samples_start.dim()

        return (
            extract_items(self.sqrt_alphas_hat, steps_id, n_dims) * samples_start
            + extract_items(self.sqrt_one_minus_alphas_hat, steps_id, n_dims) * noises
        )

    def get_diffusion_forward_start(self, noised_ts_samples, steps_id, noises):
        """Compute the start samples from the diffusion forward process.

        As the q(x_t | x_0) along the forward chain, which is
        given by Eq. 17 of [2], can be computed directly, there exists a
        math relation shown as `x_t = (*)x_0 + (*)noises` as shown in [4],
        where the term `(*)` presents equations related to alphas_hat.

        Then, given noised_ts_samples `x_t` and `noises`, we can computed the
        start samples `x_0`.

        """
        n_dims = noised_ts_samples.dim()
        return (
            extract_items(self.sqrt_recip_alphas_hat, steps_id, n_dims)
            * noised_ts_samples
            - extract_items(self.sqrt_recipm1_alphas_hat, steps_id, n_dims) * noises
        )

    def get_diffusion_forward_noises(self, noised_ts_samples, samples_start, steps_id):
        """Compute the noises from the diffusion forward process.

        Similarly, using the same logic of ``get_diffusion_forward_start``, the noises
        can be computed given  `x_0` and `x_t`.
        """
        n_dims = noised_ts_samples.dim()
        return (
            extract_items(self.sqrt_recip_alphas_hat, steps_id, n_dims)
            * noised_ts_samples
            - samples_start
        ) / extract_items(self.sqrt_recipm1_alphas_hat, steps_id, n_dims)

    def get_diffusion_forward_posterior(
        self, samples_start, noised_ts_samples, steps_id
    ):
        """Compute the forward posterior `q(x_t-1 | x_t, x_0)` which is
        given by Eq. 18 and Eq. 19 of [2].

        The reverse conditional probability is tractable when
        conditioned on `x_0`.

        We utilize terminology 'get' here because this posterior can be
        computed directly based on Bayesian theorem.

        :param samples_start: A `torch.FloatTensor` presenting the
         raw sample when `t=0`, i.e. `x_0` in Eq. 18.
         Shape can be [batch_size, c, h, w]
        :param noised_ts_samples: A `torch.FloatTensor` presenting the noised
         sample in time steps (`steps_id`).
         Shape is the same as `samples_start`
        :param steps_id: A `torch.IntTensor` containing the time step id
         for each sample in one batch.
        """
        # compute posterior for mean based on Eq. 18.
        # shape, [batch_size, c, h, w]
        n_dims = samples_start.dim()
        posterior_mean = (
            extract_items(self.posterior_mean_coef1, steps_id, n_dims) * samples_start
            + extract_items(self.posterior_mean_coef2, steps_id, n_dims)
            * noised_ts_samples
        )
        # get the pre-computed `posterior_variance_betas`
        # shape, [batch_size, 1, 1, 1]
        # where each obtained item is `beta` for the
        # corresponding sample in the batch.
        posterior_variance_betas = extract_items(
            self.posterior_variance_betas, steps_id, n_dims
        )
        # get the pre-computed `posterior_log_variance_clipped`
        # shape, [batch_size, 1, 1, 1]
        # where each obtained item is `beta` for the
        # corresponding sample in the batch.
        posterior_log_variance_clipped = extract_items(
            self.posterior_log_variance_clipped, steps_id, n_dims
        )
        return BaseDiffusionPosterior(
            posterior_mean=posterior_mean,
            posterior_variance_betas=posterior_variance_betas,
            posterior_log_variance_betas=posterior_log_variance_clipped,
        )

    def perform_diffusion_reverse_predictions(
        self, noised_ts_samples, steps_id, **kwargs
    ):
        """Apply the diffusion head model to make predictions for the
        diffusion reverse process.

        This operation can be regarded as the inner term of Eq. 21
        of [2].

        """
        # make prediction by applying the diffusion head
        # this is to approximate the
        head_opts = self.diffusion_head(
            ts_samples=noised_ts_samples, time_steps=steps_id, **kwargs
        )
        # predict the corresponding noists and start samples
        if head_opts.prediction_type == "noise":
            pred_noise = head_opts.predictions[-1]
            pred_start_samples = self.get_diffusion_forward_start(
                noised_ts_samples, steps_id, pred_noise
            )
            pred_start_samples = process_samples_fn(**kwargs)(pred_start_samples)

        elif head_opts.prediction_type == "start":
            pred_start_samples = head_opts.predictions[-1]
            pred_start_samples = process_samples_fn(**kwargs)(pred_start_samples)

            pred_noise = self.get_diffusion_forward_noises(
                noised_ts_samples, steps_id, pred_start_samples
            )

        return BaseDiffusionReversePredictions(
            predicted_noises=pred_start_samples,
            predicted_start_samples=pred_noise,
            time_steps=steps_id,
        )

    def approximate_diffusion_reverse_posterior(
        self, noised_ts_samples, steps_id, **kwargs
    ):
        """Approximate the reverse posterior `p_θ(x_t-1 | x_t)` which is
        given by Eq. 21 of [2].

        In the reverse process, the `x_0` is unknown, making the `p(x_t-1 | x_t)`
        cannot be computed directly. Thus, a model (the diffusion head model) is
        required to make a prediction.

        For the posterior, the mean and variance are expected to be computed.
        """
        reverse_predictions = self.perform_diffusion_reverse_predictions(
            noised_ts_samples, steps_id
        )
        pred_start_samples = reverse_predictions.predicted_start_samples

        if "clip_denoised" in kwargs:
            pred_start_samples = process_samples_fn(**kwargs)(pred_start_samples)

        posteriors = self.get_diffusion_forward_posterior(
            samples_start=pred_start_samples,
            noised_ts_samples=noised_ts_samples,
            steps_id=steps_id,
        )
        return (
            posteriors,
            pred_start_samples,
        )

    @torch.no_grad()
    def reverse_sample_via_posterior(self, noised_ts_samples, steps_id, **kwargs):
        """Sample the data from the diffusion reverse process relying on the
        computed reverse posterior."""

        # compute the posterior from the reverse process relying
        # on the predictions from the diffusion model
        posteriors, pred_start_samples = self.approximate_diffusion_reverse_posterior(
            noised_ts_samples=noised_ts_samples, steps_id=steps_id, **kwargs
        )
        # generate gaussian noises
        # when steps id is 0, there should be no noise, i.e., 0.
        noises = torch.randn_like(noised_ts_samples)
        noises[steps_id.squeeze() == 0] = 0.0

        # as the x_t-1 ~ N(posterior_mean, posterior_variance)
        # thus, based on the computed posterior, the x_t-1 can be
        # generated directly.
        pred_samples = (
            posteriors.posterior_mean
            + (0.5 * posteriors.posterior_log_variance_betas).exp() * noises
        )
        return pred_samples, pred_start_samples

    @torch.no_grad()
    def reverse_sample_via_ddpm(self, noised_ts_samples, steps_id, **kwargs):
        """Sample the data from the diffusion reverse process relying on the
        original DDPM method [1].

        See the Table <Algorithm 2 Sampling> and Eq. 11 of [1].
        """
        # To utilize the ddmp as the sampling method, the diffusion
        # head model has to predict noises mandatory

        assert self.diffusion_head.prediction_type == "noise"
        predicted_noise = self.diffusion_head(noised_ts_samples, steps_id)

        # generate gaussian noises
        # when steps id is 0, there should be no noise, i.e., 0.
        # this is the `z` in Eq. 11.
        noises = torch.randn_like(noised_ts_samples)
        noises[steps_id.squeeze() == 0] = 0.0

        # obtain alphas, alphas_hat, betas
        n_dims = noised_ts_samples.dim()
        steps_betas = extract_items(self.betas, steps_id, n_dims)
        steps_alphas = extract_items(self.alphas, steps_id, n_dims)
        steps_sqrt_one_minus_alphas_hat = extract_items(
            self.sqrt_one_minus_alphas_hat, steps_id, n_dims
        )

        recipm1_sqrt_alphas = 1 / torch.sqrt(steps_alphas)
        weights_one_minus_steps_alphas = (
            1 - steps_alphas
        ) / steps_sqrt_one_minus_alphas_hat

        return (
            recipm1_sqrt_alphas
            * (noised_ts_samples - weights_one_minus_steps_alphas * predicted_noise)
            + torch.sqrt(steps_betas) * noises
        ), None

    @torch.no_grad()
    def diffusion_reverse_ddim_sampling(self, noised_ts_samples, steps_id, **kwargs):
        """Sample the data from the diffusion reverse process relying on the
        original DDPM method [1].

        See the Table <Algorithm 2 Sampling> and Eq. 11 of [1].
        """
        pass

    @torch.no_grad()
    def diffusion_reverse_sampling(self, target_shape, **kwargs):
        """The sampling process that performs the reverse diffusion process
        of the diffusion model.

            Operating with time steps id T, T-1, ..., 3, 2, 1.
            Thus, starting from a guassina noise in step T, the raw sample
            can be generated by using p(x_t-1 | x_t) iteratively.
        """
        device = self.betas.device

        reverse_step_samples = torch.randn(target_shape, device=device)
        reverse_out_samples = [reverse_step_samples]

        for tiem_step_id in tqdm(
            range(self.chain_steps - 1, -1, -1),
            desc="Reverse Process Sampling",
            total=self.chain_steps,
        ):
            # perform reverse sampling once, p(x_t-1 | x_t)
            # shape, [batch_size, C, H, W]
            reverse_step_samples, _ = self.reverse_sample_via_ddpm(
                reverse_step_samples, tiem_step_id
            )
            if "all_timesteps" in kwargs and kwargs["all_timesteps"]:
                reverse_out_samples.append(reverse_step_samples)
            else:
                reverse_out_samples[0] = reverse_step_samples

        # stack the appended samples to be on tensor with shape
        # [batch_size, N, *target_shape]
        reverse_out_samples = torch.stack(reverse_out_samples, dim=1)
        return self.unnormalize_samples(reverse_out_samples)

    def create_diffusion_targets(
        self, prediction_type, samples_start, noises, **kwargs
    ):
        """Create diffusion models' targets, which will be
        utilized for the subsequent loss computation."""
        if prediction_type == "noise":
            targets = noises
        if prediction_type == "start":
            targets = samples_start
        if prediction_type == "customization":
            raise NotImplementedError(
                "Customized diffusion targets should be supported by users."
            )

        return targets

    def forward(
        self,
        x_samples: torch.Tensor,
        x_noises: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, torch.Tensor],
    ) -> BaseDiffusionOutput:
        """The learning forward of the diffusion models.

        :param: x_samples: A `torch.FloatTensor` denoting the
         input samples.
         Shape should be, [batch_size, n_channels, height, width]
         where `n_channels` is the number of channels.
        :param x_noises: A `torch.FloatTensor` presenting the
         noises generated for input samples.
         It should have the same shape as input samples.
        """

        # obtain what the head will predict as the output
        prediction_type = self.diffusion_head.prediction_type

        batch_size, _, _, _ = x_samples.shape
        device = x_samples.device

        # randomly generate the time steps for each sample of
        # the batch
        time_steps_id = torch.randint(
            0, self.chain_steps, (batch_size,), device=device
        ).long()

        x_noises = x_noises if x_noises is not None else torch.randn_like(x_samples)

        # forward diffusion process
        # to noise the samples of generated time steps
        # noised_ts_samples, [batch_size, C, H, W]
        noised_samples = self.get_diffusion_forward_samples(
            samples_start=x_samples, steps_id=time_steps_id, noises=x_noises
        )

        # make prediction by applying the diffusion head
        #
        head_outputs = self.diffusion_head(
            ts_samples=noised_samples, time_steps=time_steps_id, **kwargs
        )

        # this generalized version tends to utilize the output from
        # the final head
        head_outputs.predictions = head_outputs.predictions[-1]

        # obtain the output weight
        head_outputs_weights = self.get_out_weight(
            time_steps_id, head_outputs.predictions.dim()
        )

        # create the targets for the subsequent loss computation
        targets = self.create_diffusion_targets(prediction_type, x_samples, x_noises)

        return BaseDiffusionOutput(
            prediction_type=prediction_type,
            predictions=head_outputs.predictions,
            diffusion_targets=targets,
            predictions_weight=head_outputs_weights,
            time_steps=time_steps_id,
        )
