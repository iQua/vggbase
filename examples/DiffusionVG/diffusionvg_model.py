"""
The implementation of diffusion models for visual grounding.

"""

import logging
from typing import Optional, Tuple, List, OrderedDict, Type

import torch
from torchvision.ops.boxes import box_convert

from diffusionvg_head import LGDiffusionHead
from diffusionvg_generic import (
    LGDiffusionVGTarget,
    AdditionalDiffusionOutput,
    DiffusionVGReversePredictions,
)

from vggbase.models.diffusions import generalized_diffusion
from vggbase.models.diffusions import diffusion_utils
from vggbase.utils.generic_components import BaseVGList
from vggbase.boxes.bbox_generic import BaseVGModelBBoxes, BaseVGBBoxes
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.boxes import bbox_extension, bbox_utils
from vggbase.config import Config

bbox_prepare_registry = {
    "random": bbox_extension.pad_random_bboxes,
    "repeat": bbox_extension.pad_repeat_bboxes,
    "balance": bbox_extension.pad_balance_bboxes,
}


def normalize_to_scale_range(samples, scale):
    """Normalize bbox to [-scale, scale]"""
    return torch.clamp((samples * 2.0 - 1.0) * scale, min=-1 * scale, max=scale)


def unnormalize_to_zero_to_one(samples, scale):
    """Unnormalize bbox to [0, 1]"""

    samples = torch.clamp(samples, min=-1 * scale, max=scale)
    return ((samples / scale) + 1) / 2.0


class LGDiffusionVG(generalized_diffusion.GeneralizedDiffusionModel):
    """The language guided diffusion for visual grounding."""

    def __init__(
        self,
        n_proposals: Optional[int],
        forward_proposal_manager: Optional[str],
        reverse_proposal_manager: Optional[str],
        **default_congis,
    ) -> None:
        super().__init__(**default_congis)

        self.n_proposals = n_proposals
        self.forward_proposal_manager = forward_proposal_manager
        self.reverse_proposal_manager = reverse_proposal_manager

    def prepare_normalizations(self, normalization_config):
        """Prepare the normalizations to process the samples of the
        diffusion model."""
        self.scale = normalization_config.scale_value
        # auto-normalization of bboxes from
        # [0, 1] -> [-scale_value, scale_value]

        self.normalize_samples = lambda x: (
            normalize_to_scale_range(x, self.scale)
            if normalization_config.auto_normalize
            else diffusion_utils.identity
        )
        self.unnormalize_samples = lambda x: (
            unnormalize_to_zero_to_one(x, self.scale)
            if normalization_config.auto_normalize
            else diffusion_utils.identity
        )

    def init_diffusion_head(self, diffusion_head_config):
        """Define the diffusion head."""

        self.diffusion_head = LGDiffusionHead(
            diffusion_head_config.time_embedding,
            diffusion_head_config.time_projection,
            diffusion_head_config.text_projection,
            diffusion_head_config.rois,
            diffusion_head_config.box_projection,
            diffusion_head_config.head_model,
        )

        logging.info("Initialized LGDiffusionHead for LGDiffusionVG.")

    def forward(
        self,
        x_samples: List[BaseVGBBoxes],
        x_rgbs: OrderedDict[str, torch.Tensor],
        rgbs_hw: List[Tuple[int, int]],
        tquery: torch.FloatTensor,
        rgb_mask: Optional[torch.BoolTensor] = None,
        tquery_mask: Optional[torch.BoolTensor] = None,
        x_noises: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        """The learning forward of the diffusion models.

        :param: x_samples: A List of groundtruth bounding boxes for one
         batch of samples,
         Each item is a `VGGbaseBBoxes` holding the bboxes with VGGbase style
         for the corresponding sample,
         of length, len(x_samples) == batch_size
        :param x_noises: (Default) A List of noises for bounding boxes
         of one batch of samples.
         Each item is a `torch.FloatTensor` holding the noise for the
         corresponding sample.
         of shape, [bs, 4].
        :param x_rgbs: A `torch.FloatTensor` holding features of rgbs
         data,
         of shape, [bs, C, H, W].
        :param rgbs_hw: A `List` holding the original rgb sizes before
         being padded as nested tensor,
         of length, batch_size,
         of format, [h, w].
        :param tquery: A `torch.FloatTensor` holding the query tokens
         for one batch of samples,
         of shape, [bs, P, D]
         where `D` is the lenght of language embeding features.
         It should have the same shape as input samples.
        :param rgb_mask: (Default) A `torch.BoolTensor` holding the mask
         for the `rgbs`, this is generally the padding mask.
         of shape, [bs, H, W]
        :param tquery_mask: (Default) A `torch.BoolTensor` holding the mask
         for the `tquery`,
         of shape, [bs, P].
        """

        # obtain what the head will predict as the output
        prediction_type = self.diffusion_head.prediction_type

        # forward diffusion process
        # to obtain the targets for diffusion visual grounding
        diff_targets = self.create_diffusion_targets(
            prediction_type=prediction_type,
            samples_start=x_samples,
            noises=x_noises,
        )

        # concat bboxes,
        # of shape, [bs, N, 4]
        # of format, albumentations
        noised_boxes = torch.stack(
            [
                sample_targets.diffused_vg_bboxes.bboxes
                for sample_targets in diff_targets
            ]
        )

        noised_boxes_id = [
            sample_targets.diffused_vg_bboxes.bbox_ids
            for sample_targets in diff_targets
        ]

        noised_boxes_label = [
            sample_targets.diffused_vg_bboxes.labels for sample_targets in diff_targets
        ]
        if noised_boxes_id[0] is not None:
            noised_boxes_id = torch.stack(noised_boxes_id)
            noised_boxes_label = torch.stack(noised_boxes_label)
        else:
            noised_boxes_id = None
            noised_boxes_label = None

        # get the time steps for one batch
        # of shape, [bs, ]
        time_steps_id = torch.stack(
            [sample_targets.time_steps_id for sample_targets in diff_targets]
        ).squeeze(-1)
        # get the boards where the bboxes exist
        # of length, len(board_hws) == batch_size
        # of format, each item is a tuple holding h, w
        board_hws = torch.tensor(
            [
                sample_targets.diffused_vg_bboxes.board_hw
                for sample_targets in diff_targets
            ],
            device=tquery.device,
        )

        # of shape, [bs, 4]
        # of format, tensor
        boards_whwh = bbox_utils.convert_hw_whwh(board_hws)

        # convert bboxes from normalized [0,1]
        # i.e, format, albumentations
        # to bboxes of unnormalized
        # of shape, [bs, N, 4]
        # of format, pascal_voc
        noised_boxes = noised_boxes * boards_whwh[:, None, :]

        head_outputs = self.diffusion_head(
            ts_samples=noised_boxes,
            time_steps=time_steps_id,
            x_rgbs=x_rgbs,
            rgbs_hw=rgbs_hw,
            x_tquery=tquery,
            tquery_mask=tquery_mask,
            init_samples_features=None,
        )

        return BaseVGModelOutput(
            **BaseVGModelBBoxes(
                bboxes=head_outputs.predictions,
                similarity_scores=head_outputs.matching_scores,
                bbox_ids=noised_boxes_id,
                class_logits=noised_boxes_label,
                board_hws=board_hws,
                bbox_type="pascal_voc",
            ),
            additional_output=AdditionalDiffusionOutput(
                prediction_type=prediction_type,  # lgdiffusion_targets=diff_targets
            ),
        )

    def create_diffusion_targets(
        self, prediction_type, samples_start, noises, **kwargs
    ):
        """Create the diffusion targets for subsequent learning."""

        diffusionvg_targets = BaseVGList([])

        for sample_idx, sample in enumerate(samples_start):
            sample_bboxes = sample.bboxes
            bbox_ids = sample.bbox_ids
            bboxes_label = sample.labels

            board_h, board_w = sample.board_hw
            board_size_whwh = torch.as_tensor(
                [board_w, board_h, board_w, board_h],
                dtype=torch.float,
                device=sample_bboxes.device,
            )
            # to yolo type
            gt_boxes = sample_bboxes / board_size_whwh
            gt_boxes = box_convert(gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")

            # prepare the gt_bboxes based on desired #proposal
            # of shape, [self.n_proposal, 4]
            # of format, [ctr_x, ctr_y, width, height], i.e., Yolo type.
            prepared_vg_bboxes = bbox_prepare_registry[
                self.forward_proposal_manager.pad
            ](
                BaseVGBBoxes(
                    bboxes=gt_boxes,
                    labels=bboxes_label,
                    bbox_ids=bbox_ids,
                    bbox_type="yolo",
                ),
                target_n_proposals=self.n_proposals,
            )
            # replace the bboxes id and bboxes label once we have
            diffused_boxes_id = prepared_vg_bboxes.bbox_ids
            diffused_boxes_label = prepared_vg_bboxes.labels

            sample_noises = noises[sample_idx] if noises is not None else None
            # obtain the diffused bboxes
            # of shape, [n_proposals, 4]
            # of format, albumentations
            d_boxes, d_noises, d_ts = self.bboxes_forward_diffusion(
                gt_boxes=prepared_vg_bboxes.bboxes,
                x_noises=sample_noises,
                **kwargs,
            )

            diffusionvg_targets.append(
                LGDiffusionVGTarget(
                    diffused_vg_bboxes=BaseVGBBoxes(
                        bboxes=d_boxes,
                        labels=diffused_boxes_label,
                        bbox_ids=diffused_boxes_id,
                        board_hw=(board_h, board_w),
                        bbox_type="albumentations",
                    ),
                    noises=d_noises,
                    time_steps_id=d_ts,
                )
            )

        return diffusionvg_targets

    def bboxes_forward_diffusion(self, gt_boxes, x_noises, **kwargs):
        """Perform the forward diffusion process to obtain the
        diffused bboxes.

        :param gt_boxes: A `torch.FloatTensor` holding the normalized
         bboxes for one sample,
         of shape, [n_proposals, 4]
         of format, [ctr_x, ctr_y, width, height], i.e., Yolo type.

        """

        # generate time steps for bboxes
        # of shape, [1, ]
        # because of the broadcast property, this
        # time steps id can be direclty utilized,
        # leading to all bboxes share the same time step.
        time_steps_id = torch.randint(
            0, self.chain_steps, (1,), device=gt_boxes.device
        ).long()

        # generate noises for bboxes
        # of shape, [num_bboxes, 4]
        x_noises = (
            x_noises
            if x_noises is not None
            else torch.randn(self.n_proposals, 4, device=gt_boxes.device)
        )

        # normalize the input bboxes
        # to [-self.scale, self.scale]
        x_samples = self.normalize_samples(gt_boxes)

        # obtain the noised samples by performing the forward
        # diffusion process
        noised_samples = self.get_diffusion_forward_samples(
            samples_start=x_samples, steps_id=time_steps_id, noises=x_noises
        )
        # convert to [0, 1]
        noised_samples = self.unnormalize_samples(noised_samples)

        diffused_boxes = box_convert(noised_samples, in_fmt="cxcywh", out_fmt="xyxy")

        return diffused_boxes, x_noises, time_steps_id

    def perform_diffusion_reverse_predictions(
        self, noised_ts_samples, steps_id, tquery, x_rgbs, rgbs_hw, tquery_mask
    ):
        """Apply the diffusion head model to make predictions for the
        diffusion reverse process.

        """
        # unnormalized to [0, 1]
        noised_ts_samples = self.unnormalize_samples(noised_ts_samples)
        # convert from cxcywh to xyxy - in range [0, 1]
        noised_ts_samples = box_convert(
            noised_ts_samples, in_fmt="cxcywh", out_fmt="xyxy"
        )
        # convert to unnormalized ones
        rgbs_hw = torch.tensor(rgbs_hw, device=tquery.device)
        rgbs_whwh = bbox_utils.convert_hw_whwh(rgbs_hw)
        noised_ts_samples = noised_ts_samples * rgbs_whwh[:, None, :]

        head_outputs = self.diffusion_head(
            ts_samples=noised_ts_samples,
            time_steps=steps_id,
            x_tquery=tquery,
            tquery_mask=tquery_mask,
            x_rgbs=x_rgbs,
            rgbs_hw=rgbs_hw,
            init_samples_features=None,
        )

        # predict the corresponding noists and start samples
        if head_outputs.prediction_type == "noise":
            pred_noise = head_outputs.predictions[-1]
            pred_start_samples = self.get_diffusion_forward_start(
                noised_ts_samples, steps_id, pred_noise
            )
            # convert box to [0, 1]
            pred_start_samples = pred_start_samples / rgbs_whwh[:, None, :]
            # convert from xyxy to cxcywh
            pred_start_samples = box_convert(
                pred_start_samples, in_fmt="xyxy", out_fmt="cxcywh"
            )
            # convert to [-scale, scale]
            pred_start_samples = self.normalize_samples(pred_start_samples)

        elif head_outputs.prediction_type == "start":
            pred_start_samples = head_outputs.predictions[-1]
            # convert box to [0, 1]
            pred_start_samples = pred_start_samples / rgbs_whwh[:, None, :]
            # convert from xyxy to cxcywh
            pred_start_samples = box_convert(
                pred_start_samples, in_fmt="xyxy", out_fmt="cxcywh"
            )
            # convert to [-scale, scale]
            pred_start_samples = self.normalize_samples(pred_start_samples)

            pred_noise = self.get_diffusion_forward_noises(
                noised_ts_samples, steps_id, pred_start_samples
            )

        return DiffusionVGReversePredictions(
            predicted_noises=pred_start_samples,
            predicted_start_samples=pred_noise,
            time_steps=steps_id,
            vg_head_outputs=head_outputs,
        )

    @torch.no_grad()
    def diffusion_reverse_ddim_sampling(self, x_rgbs, rgbs_hw, tquery, tquery_mask):
        """Sample the data from the diffusion reverse process relying on the
        DDIM method.
        """

        batch_size = len(rgbs_hw)
        target_shape = (batch_size, self.n_proposals, 4)
        # generate the noisy with random gaussian distribution
        #  for time step T
        noisy_samples = torch.randn(target_shape, device=tquery.device)

        total_timesteps = self.chain_steps
        sampling_timesteps = self.reverse_sampling_config.sampling_steps

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times_step = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )
        times_step = list(reversed(times_step.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_step_pairs = list(zip(times_step[:-1], times_step[1:]))

        x_start = None
        for time, time_next in time_step_pairs:
            time_cond = torch.full(
                (batch_size,), time, device=tquery.device, dtype=torch.long
            )

            reverse_perdictions = self.perform_diffusion_reverse_predictions(
                noised_ts_samples=noisy_samples,
                steps_id=time_cond,
                x_rgbs=x_rgbs,
                rgbs_hw=rgbs_hw,
                tquery=tquery,
                tquery_mask=tquery_mask,
            )
            pred_noise = reverse_perdictions.predicted_noises
            x_start = reverse_perdictions.predicted_start_samples
            if time_next < 0:
                noisy_samples = x_start
                continue

            alpha_hat = self.alphas_hat[time]
            alpha_hat_next = self.alphas_hat[time_next]

            sigma = (
                self.reverse_sampling_config.eta
                * (
                    (1 - alpha_hat / alpha_hat_next)
                    * (1 - alpha_hat_next)
                    / (1 - alpha_hat)
                ).sqrt()
            )
            c_value = (1 - alpha_hat_next - sigma**2).sqrt()

            noises = torch.randn_like(noisy_samples)

            noisy_samples = (
                x_start * alpha_hat_next.sqrt() + c_value * pred_noise + sigma * noises
            )
            # many bounding boxes make no sense as their similairity is small, thus
            # replace them with noisy bboxes.
            # [bs, n_boxes, P]
            matching_scores = reverse_perdictions.vg_head_outputs.matching_scores
            if self.reverse_proposal_manager == "box_renewal":
                keep_mask, n_keep = bbox_utils.judge_similarity(matching_scores)
                noisy_samples = bbox_extension.replace_bboxes_randomness(
                    BaseVGBBoxes(bboxes=noisy_samples), keep_mask=keep_mask
                ).bboxes

        return BaseVGModelOutput(
            **BaseVGModelBBoxes(
                bboxes=reverse_perdictions.vg_head_outputs.predictions,
                similarity_scores=reverse_perdictions.vg_head_outputs.matching_scores,
                bbox_ids=None,
                class_logits=None,
                board_hws=rgbs_hw,
                bbox_type="pascal_voc",
            ),
            additional_output=AdditionalDiffusionOutput(
                prediction_type=self.diffusion_head.prediction_type,
                lgdiffusion_targets=None,
            ),
        )


def build_model(model_config: Type[Config], **kwargs):
    """Build the grounding model."""

    grounding_config = model_config.grounding
    text_n_features = kwargs["text_n_features"]
    rgb_n_channels = kwargs["rgb_n_channels"]

    # replace the features
    text_config_in_fetures = grounding_config.diffusion_head.text_projection.in_features

    if text_config_in_fetures != text_n_features:
        raise ValueError(
            "The in_features of text_projection in config should equal to the obtained text_n_features."
        )

    rgb_config_in_channels = grounding_config.diffusion_head.rois.in_channels
    if rgb_config_in_channels != rgb_n_channels:
        raise ValueError(
            "The in_channels of diffusion_head.rois in config should equal to the obtained rgb_n_channels."
        )

    return LGDiffusionVG(
        n_proposals=grounding_config.n_proposals,
        forward_proposal_manager=grounding_config.forward_proposal_manager,
        reverse_proposal_manager=grounding_config.reverse_proposal_manager,
        chain_steps=grounding_config.chain_steps,
        noise_variance_schedule_config=grounding_config.noise_variance_schedule,
        diffusion_head_config=grounding_config.diffusion_head,
        out_weights_config=grounding_config.out_weights,
        normalization_config=grounding_config.normalization,
        reverse_sampling_config=grounding_config.reverse_sampling,
    )
