"""
The implementation of loss criterion to obtain different various types of losses.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import generalized_box_iou

from vggbase.boxes.bbox_convertion import (
    convert_bbox_format,
    convert_model_bbox_format,
    convert_bbox_type,
)
from vggbase.datasets.data_generic import BaseInputTarget
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.learners.learn_generic import BaseVGMatchOutput
from vggbase.utils.generic_components import BaseVGList


def get_box_losses(
    model_outputs: BaseVGModelOutput,
    match_outputs: BaseVGMatchOutput,
    targets: BaseVGList[BaseInputTarget],
):
    """
    Compute the losses related to the bounding boxes; thus including L1 loss and the GIoU loss.
    """

    # Get matched results,
    # with shape, [batch_size, n_groups, n_bboxes]
    match_bbox_flags = match_outputs.match_bbox_flags
    match_gt_bbox_indexes = match_outputs.match_gt_bbox_indexes

    batch_size = match_bbox_flags.shape[0]

    match_bbox_flags = match_bbox_flags.view(batch_size, -1)
    match_gt_bbox_indexes = match_gt_bbox_indexes.view(batch_size, -1)

    # Get the model output bboxes,
    # with shape, [batch_size, n_groups * n_bboxes, 4]
    src_bboxes = model_outputs.bboxes.view(batch_size, -1, 4)

    # Get the matched bboxes and gt bboxes
    matched_indexs = torch.where(match_bbox_flags > 0)
    # src_bboxes, with shape [n_batch_bboxes, 4]
    src_bboxes = src_bboxes[matched_indexs]
    # tgt_bboxes, with shape [n_batch_bboxes, 4]
    tgt_bboxes = torch.cat(
        [
            target.vg_bboxes.bboxes[
                match_gt_bbox_indexes[idx][match_bbox_flags[idx] > 0]
            ]
            for idx, target in enumerate(targets)
        ],
        dim=0,
    )

    # Compute the l1 loss on the yolo format
    l1_loss = F.l1_loss(
        convert_bbox_type(
            src_bboxes, source_type=model_outputs.bbox_type, target_type="yolo"
        ),
        convert_bbox_type(
            tgt_bboxes, source_type=model_outputs.bbox_type, target_type="yolo"
        ),
        reduction="none",
    )

    giou_loss = 1 - generalized_box_iou(src_bboxes, tgt_bboxes)

    losses = {}
    losses["bbox_l1"] = l1_loss.mean()
    losses["bbox_giou"] = giou_loss.mean()
    return losses


def get_alignment_losses(
    model_outputs: BaseVGModelOutput,
    match_outputs: BaseVGMatchOutput,
    targets: BaseVGList[BaseInputTarget],
    eos_coef: float = 0.1,
):
    """Get the losses related to the cross-modal alignment.

    :param eos_coef: The relative classification weight applied to the no-object category.
    """
    # Extract the similarity scores between src bboxes and
    # input language query, [bs, n_groups, n_bboxes, P]
    src_similiarity_scores = model_outputs.similarity_scores
    batch_size, n_groups, n_bboxes, n_phrases = src_similiarity_scores.shape
    src_similiarity_scores = src_similiarity_scores.view(
        batch_size, n_groups * n_bboxes, -1
    )

    # Get matched results,
    # with shape, [batch_size, n_groups, n_bboxes]
    match_bbox_flags = match_outputs.match_bbox_flags
    match_gt_bbox_indexes = match_outputs.match_gt_bbox_indexes
    # Convert to [batch_size, n_groups * n_bboxes]
    match_bbox_flags = match_bbox_flags.view(batch_size, -1)
    match_gt_bbox_indexes = match_gt_bbox_indexes.view(batch_size, -1)

    # Get the matched bboxes and gt bboxes
    matched_indexs = torch.where(match_bbox_flags > 0)

    # Get ground truth phrases ids of those matched bboxes,
    # with shape, [n_batch_bboxes]
    tgt_bbox_ids = torch.cat(
        [
            target.vg_bboxes.bbox_ids[
                match_gt_bbox_indexes[idx][match_bbox_flags[idx] > 0]
            ]
            for idx, target in enumerate(targets)
        ],
        dim=0,
    )

    # [n_batch_bboxes, n_phrases]
    tgt_bbox_ids = tgt_bbox_ids.to(torch.int64)
    matched_onehot_ids = F.one_hot(tgt_bbox_ids, num_classes=n_phrases)

    # Create a all zero one-hot tensor with shape, [batch_size, n_groups * n_bboxes, n_phrases]
    tgt_onehot_ids = torch.zeros(
        size=(batch_size, n_groups * n_bboxes, n_phrases),
        dtype=torch.int64,
        device=src_similiarity_scores.device,
    )
    tgt_onehot_ids[matched_indexs] = matched_onehot_ids

    # By directly minimizing the similarity scores and the gt one-hot ids, we force
    # 1. The similarity scores of matched bboxes reach the highest in the groundtruth phrase.
    # 2. The similarity scores of unmatched bboxes reach zeros on all phrases.
    # Get the score distances with shape, [batch_size, n_groups * n_bboxes, n_phrases]
    score_distances = torch.square(src_similiarity_scores - tgt_onehot_ids)
    # Sum over the phrase dimension, [batch_size, n_groups * n_bboxes]
    score_distances = torch.sum(score_distances, dim=-1)

    # Create weights for matched and unmatched bboxes
    # with shape [batch_size, n_groups * n_bboxes]
    box_score_weights = eos_coef * torch.ones_like(
        match_bbox_flags, dtype=torch.float32
    )
    box_score_weights[matched_indexs] = 1

    weighted_score_distances = score_distances * box_score_weights
    loss = weighted_score_distances.mean()

    return {"align_loss": loss}


class LossCriterion(nn.Module):
    """This class computes the loss for visual grounding methods."""

    def __init__(
        self,
        eval_config: dict,
        num_classes: Optional[int] = None,
    ):
        """
        A base Criterion to compute the loss for visual grounding.

        :param matcher: A `HungarianMatcher` instance to match targets and proposals
        :param num_classes: The number of object categories, omitting the special no-object category.
            This term can be ignored in the visual grounding task. Default None.
        :param eos_coef: The relative classification weight applied to the no-object category.
            This can also be ignored in the VG tasks. Default None.
        """
        super().__init__()
        self.num_classes = num_classes

        # Extract the loss name and the parameters
        self.loss_names = eval_config["losses"]["names"]
        self.loss_weights = eval_config["losses"]["weights"]

        self.eos_coef = (
            0.1
            if "eos_coef" not in eval_config["losses"]
            else eval_config["losses"]["eos_coef"]
        )

        # A backup for creating the weight for the non-object category
        # once the number of classes is known
        if self.num_classes is not None:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef

            self.register_buffer("empty_weight", empty_weight)

        # Force the box format used by the matcher to
        # be yolo, [x1, y1, x2, y2] \in [0, 1]
        self.box_format = "albumentations"

    def forward(
        self,
        model_outputs: BaseVGMatchOutput,
        match_outputs: BaseVGMatchOutput,
        targets: BaseVGList[BaseInputTarget],
        **kwargs,
    ):
        """Perform the losses computation."""
        # Convert the bboxes to the format used in the matcher
        convert_bbox_format(
            BaseVGList([groundtruth.vg_bboxes for groundtruth in targets]),
            format_type=self.box_format,
        )
        convert_model_bbox_format(
            model_outputs,
            format_type=self.box_format,
        )

        # Compute a series of losses
        box_losses = get_box_losses(model_outputs, match_outputs, targets)
        align_losses = get_alignment_losses(
            model_outputs, match_outputs, targets, eos_coef=self.eos_coef
        )
        # Collect all the losses
        computed_losses = {}
        computed_losses.update(box_losses)
        computed_losses.update(align_losses)

        # Collect required losses and scale them
        valid_losses = {}
        total_loss = 0.0
        for idx, name in enumerate(self.loss_names):
            valid_losses[name] = computed_losses[name]
            valid_losses[f"scaled_{name}"] = valid_losses[name] * self.loss_weights[idx]
            total_loss += valid_losses[f"scaled_{name}"]

        valid_losses["loss"] = total_loss

        return valid_losses
