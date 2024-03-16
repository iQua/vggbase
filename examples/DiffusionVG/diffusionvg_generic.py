"""
The general componenets for diffusion visual grounding.
"""

from typing import Optional
from dataclasses import dataclass

import torch

from vggbase.models.diffusions import diffusion_components
from vggbase.utils.generic_components import BaseVGList, FieldFrozenContainer
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.boxes.bbox_generic import BaseVGBBoxes


@dataclass
class LGDiffusionVGTarget(FieldFrozenContainer):
    """The targets of the samples.

    Args::
     diffused_vg_bboxes: A `VGGbaseBBoxes` holding items
      described in `VGGbaseBBoxes`.

     noises: A `torch.FloatTensor` holding the diffusion noises
      for ``diffused_boxes`` of the forward diffusion process,
      of shape, [N, 4]
     time_steps_id: A `torch.IntTensor` holding the time steps
      for ``diffused_boxes`` of the forward diffusion process,
      of shape, [1, ]
     where `N` is the `n_proposals` defined in the configuraton file.
    """

    diffused_vg_bboxes: BaseVGBBoxes = None
    # fields for diffusion model
    noises: Optional[torch.Tensor] = None
    time_steps_id: Optional[torch.Tensor] = None


@dataclass
class LGDiffusionVGHeadOutput(diffusion_components.BaseDiffusionHeadOutput):
    """
    Outputs of the diffusion visual grounding model's head.
    Args:
     matching_scores: A `torch.Tensor` holding the matching scores
         between bboxes and text queries,
         of shape, [M, bs, n_bboxes, n_ph]

      where `n_ph` is the maximum number of phrases in one batch of
      samples. `M` denotes how many models contained in the head
      as a series of model.
    """

    matching_scores: torch.FloatTensor = None


@dataclass
class DiffusionVGReversePredictions(
    diffusion_components.BaseDiffusionReversePredictions
):
    """Predictions of the reverse diffusion process."""

    vg_head_outputs: Optional[LGDiffusionVGHeadOutput] = None


@dataclass
class LGVGRCNNOutput(FieldFrozenContainer):
    """
    Outputs of the RCNN model for visual grounding.
    Args:

    predicted_bboxes: A `torch.FloatTensor` holding the
       predicted un-normalized bboxes,
       of shape, [bs, n_boxes, 4]
       of format, xyxy
     matching_scores:  A `torch.FloatTensor` holding the the
      matching scores between text queries and bboxes,
      of shape, [bs, n_boxes, n_ph]
     vg_attentions: A `torch.FloatTensor` holding the
       attentions from transformers of text queries and bboxes,
       for all blocks,
       of shape, [bs, n_blks, n_boxes, n_ph]
    pre_rois_features: A `torch.FloatTensor` holding the
       matching scores between text queries and bboxes,
       of shape, [bs, n_boxes, n_features]
    pre_text_features: A `torch.FloatTensor` holding the
       matching scores between text queries and bboxes,
       of shape, [bs, n_boxes, n_features]
    """

    predicted_bboxes: torch.FloatTensor
    matching_scores: torch.FloatTensor = None
    vg_attentions: torch.FloatTensor = None
    pre_rois_features: torch.FloatTensor = None
    pre_text_features: torch.FloatTensor = None


@dataclass
class AdditionalDiffusionOutput(FieldFrozenContainer):
    """
    Outputs of the language-guided diffusion visual grounding model.
    Args:
      prediction_type: A `str` presenting what has been
      predicted by the diffusion head. Generally, there
      are three options:
         - noise: predicted noise of step t.
         - start: predicted sample of start, i.e., t=0
         - customization: predicted customized one
                     defined by the user.
      lgdiffusion_targets: A `BaseVGList` in which each
       one is a `DiffusionVGTarget` holding the targets for the
       corresponding sample.
    """

    prediction_type: Optional[str] = None
    lgdiffusion_targets: BaseVGList[LGDiffusionVGTarget] = None
