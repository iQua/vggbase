"""
The implementation for the bounding boxes matching.
"""

from typing import Tuple

import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

from torchvision.ops.boxes import remove_small_boxes, generalized_box_iou, box_iou

from vggbase.boxes.bbox_convertion import (
    convert_bbox_format,
    convert_model_bbox_format,
)
from vggbase.boxes.bbox_utils import remove_outbound_bboxes
from vggbase.learners.learn_generic import BaseVGMatchOutput
from vggbase.datasets.data_generic import BaseInputTarget
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.utils.generic_components import BaseVGList


class BaseMatcher:
    """
    A base matcher to align the target boxes and the predicted boxes following
    the mechanism of FasterRCNN.

    1. Positive bboxes are those 1) have the highest IoU with the groundtruth; 2)
    have IoU with the groundtruth > 0.5. label = 1
    2. Negative bboxes are those that have IoU with the groundtruth < 0.5. label = 0
    3. Invalid bboxes are those that are neither positive nor negative. label = -1
    """

    def __init__(self, matcher_config: dict):
        # Extract the cost when matching the boxes
        metric_names = matcher_config["metrics"]
        metric_weights = matcher_config["metric_weights"]
        self.metrics = dict(zip(metric_names, metric_weights))

        self.positive_threshold = matcher_config["positive_threshold"]
        self.negative_threshold = matcher_config["negative_threshold"]

        # Force the box format used by the matcher to
        # be pascal_voc, [x1, y1, x2, y2] \in [w, h]
        self.box_format = "pascal_voc"

    def convert_format(
        self, inputs: BaseVGModelOutput, targets: BaseVGList[BaseInputTarget]
    ) -> torch.Tensor:
        """Convert the box to the desired format,"""
        convert_bbox_format(
            BaseVGList([groundtruth.vg_bboxes for groundtruth in targets]),
            format_type=self.box_format,
        )
        convert_model_bbox_format(
            inputs,
            format_type=self.box_format,
        )

    def compute_match_scores(
        self, sample_bboxes: torch.Tensor, sample_gt_bboxes: torch.Tensor
    ):
        """Compute the total scores used to match boxes."""

        # Compute IOU scores and GIOU scores
        # of shape, [n_groups * n_boxes, N_i]
        total_scores = 0.0
        if "bbox_iou" in self.metrics:
            ious = box_iou(sample_bboxes, sample_gt_bboxes)
            total_scores += ious * self.metrics["bbox_iou"]
        # of shape [n_groups * n_boxes, N_i]
        if "bbox_giou" in self.metrics:
            g_ious = generalized_box_iou(sample_bboxes, sample_gt_bboxes)
            total_scores += g_ious * self.metrics["bbox_giou"]

        return total_scores

    def match_sample_bboxes(
        self, sample_bboxes: torch.Tensor, sample_groundtruth: BaseInputTarget, **kwargs
    ) -> Tuple[torch.IntTensor, torch.IntTensor]:
        """
        Perform the matching mechanism within one batch.

        By default, only the iou and giou metrics are used for matching. Therefore, each box will be matched with the groundtruth box that has the highest iou or giou score.

        :param sample_bboxes: Hold a batch of predicted bounding boxes.
            with shape [n_groups, N, 4]
        """
        # Get the height and width of the sample
        sample_hw = sample_groundtruth.vg_bboxes.board_hw
        # Extract groundtruth bboxes
        # with shape, [N_i, 4]
        sample_gt_bboxes = sample_groundtruth.vg_bboxes.bboxes

        # Convert to [n_groups * N, 4]
        sample_bboxes = sample_bboxes.view(-1, 4)

        # Get n_groups * n_boxes
        n_group_boxes = sample_bboxes.shape[0]
        # Get N_i
        n_gt_bboxes = sample_gt_bboxes.shape[0]

        # Create the matching results for bboxes
        # match_flags for showing positive, negative, invalid bboxes
        # match_gt_idx for showing the corresponding groundtruth box index
        #  of each bboxes
        match_flags = torch.full((1, n_group_boxes), -1, dtype=torch.int64).to(
            sample_bboxes.device
        )
        match_gt_idx = torch.full((1, n_group_boxes), -1, dtype=torch.int64).to(
            sample_bboxes.device
        )

        keep_idxs1 = remove_small_boxes(sample_bboxes, min_size=10)
        keep_idxs2 = remove_outbound_bboxes(sample_bboxes, sample_hw)
        # Record the indexs of invalid bboxes
        invalid_bbox_idxs = [
            idx
            for idx in range(n_group_boxes)
            if idx not in keep_idxs1 or idx not in keep_idxs2
        ]

        # Compute Total match scores
        # of shape, [n_groups * n_boxes, N_i]
        total_scores = self.compute_match_scores(sample_bboxes, sample_gt_bboxes)

        # Positive bboxes
        # Case 1: Get bboxes who have the highest IoU with groundtruth
        #
        # For each ground-truth, get bbox indexes reach the highest IoU
        # Get bbox indexes with the highest Intersection-over-Union (IoU) overlap with ground-truth boxes
        # of shape [N_i]
        gt_argmax_ious = total_scores.argmax(axis=0)
        # Extract these iou scores
        # of shape [N_i]
        gt_max_ious = total_scores[gt_argmax_ious, np.arange(n_gt_bboxes)]
        # Get all bbox indexes that obtain these highest iou scores
        # of shape unknown but ranges [0, n_groups * n_boxes]
        bbox_argmax_ious = torch.where(total_scores == gt_max_ious)[0]

        # Case 2: Get bboxes who have IoU with groundtruth > positive_threshold
        # Get the index of the highest iou for each box and its corresponding ground truth box
        # of shape [n_groups * n_boxes]
        argmax_ious = total_scores.argmax(axis=1)
        max_ious = total_scores[np.arange(n_group_boxes), argmax_ious]

        # Assign negative labels to those negative bboxes
        match_flags[0, max_ious < self.negative_threshold] = 0

        # Assign positive labels to those positive bboxes
        match_flags[0, bbox_argmax_ious] = 1
        match_flags[0, max_ious >= self.positive_threshold] = 1

        # Assign -2 to invalid bboxes
        match_flags[0, invalid_bbox_idxs] = -2

        # Assign the corresponding groundtruth box index with the largest
        # iou score to each bbox
        match_gt_idx[0, :] = argmax_ious

        return match_flags, match_gt_idx, total_scores

    @torch.no_grad()
    def forward(
        self, inputs: BaseVGModelOutput, targets: BaseVGList[BaseInputTarget], **kwargs
    ) -> BaseVGMatchOutput:
        """
        Match the boxes between the predicted boxes and the groundtruth boxes.

        :param inputs: A `torch.FloatTensor` holding predicted bboxes,
         of shape, [bs, n_groups, N, 4]
         of format, self.box_format of evaluation
        :param targets: A `List` holding groundtruth bboxes for one batch
         of samples,
         with length, len(tgt_bboxes) == bs
         of format, each item's shape [N_i, 4] with format self.box_format
         used in the evaluation
        """
        # Convert the bboxes to the format used in the matcher
        self.convert_format(inputs, targets)

        # Extract the bounding boxes
        # src_bboxes, with shape [bs, n_groups, n_boxes, 4]
        src_bboxes = inputs.bboxes
        batch_size, n_groups, n_boxes = src_bboxes.shape[:3]

        # Convert to [bs, n_groups * n_boxes, 4]
        src_bboxes = src_bboxes.view(batch_size, -1, 4)

        # Collect one batch of matched results
        matched_batches = [
            self.match_sample_bboxes(src_bboxes[batch_idx], groundtruth)
            for batch_idx, groundtruth in enumerate(targets)
        ]

        # Get match_flags, with shape [batch_size, n_groups * n_boxes]
        # Get match_gt_indexes, with shape [batch_size, n_groups * n_boxes]
        # Get match_scores, a list with length batch_size
        # with each term be shape [n_groups, n_boxes, N_i]
        match_flags = torch.cat([batch[0] for batch in matched_batches], dim=0)
        match_gt_indexes = torch.cat([batch[1] for batch in matched_batches], dim=0)
        match_scores = [
            batch[2].view(n_groups, n_boxes, -1) for batch in matched_batches
        ]
        # Reshape to [batch_size, n_groups, n_boxes]
        match_flags = match_flags.view(batch_size, n_groups, -1)
        match_gt_indexes = match_gt_indexes.view(batch_size, n_groups, -1)

        return BaseVGMatchOutput(
            match_bbox_flags=match_flags,
            match_gt_bbox_indexes=match_gt_indexes,
            match_scores=match_scores,
        )


class HungarianMatcher(BaseMatcher):
    """
    A base matcher to align the target boxes and the predicted boxes following the Hungarian matching mechanism of DETR [1].

    [1]. End-to-End Object Detection with Transformers.
    """

    def match_sample_bboxes(
        self, sample_bboxes: torch.Tensor, sample_groundtruth: BaseInputTarget, **kwargs
    ):
        """Perform the Hungarian matching on one batch of bboxes"""
        # Get the height and width of the sample
        sample_hw = sample_groundtruth.vg_bboxes.board_hw
        # Extract groundtruth bboxes
        # with shape, [N_i, 4]
        sample_gt_bboxes = sample_groundtruth.vg_bboxes.bboxes

        # Convert to [n_groups * N, 4]
        sample_bboxes = sample_bboxes.view(-1, 4)

        # Get n_groups * n_boxes
        n_group_boxes = sample_bboxes.shape[0]

        # Create the matching results for bboxes
        # match_flags for showing positive, negative, invalid bboxes
        # match_gt_idx for showing the corresponding groundtruth box index
        #  of each bboxes
        match_flags = torch.full((1, n_group_boxes), -1, dtype=torch.int64).to(
            sample_bboxes.device
        )
        match_gt_idx = torch.full((1, n_group_boxes), -1, dtype=torch.int64).to(
            sample_bboxes.device
        )

        keep_idxs1 = remove_small_boxes(sample_bboxes, min_size=10)
        keep_idxs2 = remove_outbound_bboxes(sample_bboxes, sample_hw)

        # Record the indexs of invalid bboxes
        invalid_bbox_idxs = [
            idx
            for idx in range(n_group_boxes)
            if idx not in keep_idxs1 or idx not in keep_idxs2
        ]

        # Compute the alignment cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        #  The 1 is a constant that doesn't change the matching, it can be omitted.

        # Compute the L1 cost between boxes and gt bboxes
        # of shape, [n_groups * N, N_i]
        cost_l1_bbox = torch.cdist(sample_bboxes, sample_gt_bboxes, p=1)

        # Compute the giou cost betwen boxes and gt bboxes
        # of shape, [n_groups * N, N_i]
        cost_giou = -generalized_box_iou(
            sample_bboxes,
            sample_gt_bboxes,
        )

        # Get the total cost,
        # of [n_groups * N, N_i]
        whole_costs = (
            self.metrics["bbox_l1"] * cost_l1_bbox
            + self.metrics["bbox_giou"] * cost_giou
        )
        whole_costs = whole_costs.cpu()

        # Get the best matching pairs
        # bbox_idxs, 1D with shape [n_groups * N,]
        # gt_bbox_idxs, 1D with shape [n_groups * N,]
        bbox_idxs, gt_bbox_idxs = linear_sum_assignment(whole_costs)
        # Assign the matched gt bbox index to each bbox
        match_flags[0, bbox_idxs] = 1
        match_gt_idx[0, bbox_idxs] = torch.tensor(gt_bbox_idxs, dtype=torch.int64).to(
            sample_bboxes.device
        )

        # Assign -2 to invalid bboxes
        match_flags[0, invalid_bbox_idxs] = -2

        return match_flags, match_gt_idx, -whole_costs


class HungarianOverAllMatcher(BaseMatcher):
    """
    A base matcher to align the target boxes and the predicted boxes following
    the Hungarian matching mechanism of DETR [1].

    This can be regarded as a variant of the HungarianMatcher, where the matching is performed not batch-by-batch but over all the batches.

    [1]. End-to-End Object Detection with Transformers.
    """

    @torch.no_grad()
    def forward(
        self, inputs: BaseVGModelOutput, targets: BaseVGList[BaseInputTarget], **kwargs
    ) -> BaseVGMatchOutput:
        # Convert the bboxes to the format used in the matcher
        self.convert_format(inputs, targets)

        # Extract the bounding boxes
        # src_bboxes, with shape [bs, n_groups, n_boxes, 4]
        src_bboxes = inputs.bboxes

        # Extract groundtruth bboxes contained in a list
        # with length batch_size
        # with shape of i-th item be [N_i, 4]
        tgt_bboxes = [groundtruth.vg_bboxes.bboxes for groundtruth in targets]

        batch_size, n_groups, n_bboxes = src_bboxes.shape[:3]

        # [bs * n_groups * N, 4]
        src_bboxes = src_bboxes.flatten(start_dim=0, end_dim=2)

        # [N_1, N_2, ..., N_i, ..., N_P]
        n_gt_bboxes = [len(bboxes) for bboxes in tgt_bboxes]

        # [N_1+N_2+...+N_i+..+N_P, 4]
        tgt_bboxes = torch.cat([bboxes for bboxes in tgt_bboxes])

        # Compute the L1 cost between boxes
        # of shape, [bs * n_groups * N, N_1+N_2+...+N_i+..+N_P]
        cost_l1_bbox = torch.cdist(src_bboxes, tgt_bboxes, p=1)

        # Compute the giou cost betwen boxes
        # of shape, [bs * n_groups * N, N_1+N_2+...+N_i+..+N_P]
        # in range [0, 1]
        cost_giou = -generalized_box_iou(src_bboxes, tgt_bboxes)

        # Final cost matrix
        # [bs * n_groups * N, N_1+N_2+...+N_i+..+N_P]
        whole_costs = (
            self.metrics["bbox_l1"] * cost_l1_bbox
            + self.metrics["bbox_giou"] * cost_giou
        )
        whole_costs = whole_costs.view(batch_size, n_groups * n_bboxes, -1).cpu()

        # Get match_flags, with shape [batch_size, n_groups * n_boxes]
        # Get match_gt_idx, with shape [batch_size, n_groups * n_boxes]
        match_flags = torch.full(
            (batch_size, n_groups * n_bboxes), -1, dtype=torch.int64
        ).to(src_bboxes.device)
        match_gt_idx = torch.full(
            (batch_size, n_groups * n_bboxes), -1, dtype=torch.int64
        ).to(src_bboxes.device)

        match_scores = list()
        for bs_idx, cost in enumerate(whole_costs.split(n_gt_bboxes, -1)):
            bbox_idxs, gt_bbox_idxs = linear_sum_assignment(cost[bs_idx])
            match_flags[bs_idx, bbox_idxs] = 1
            match_gt_idx[bs_idx, bbox_idxs] = torch.tensor(
                gt_bbox_idxs, dtype=torch.int64
            ).to(src_bboxes.device)
            match_scores.append(-cost[bs_idx])
        # Reshape to [bs, n_groups, n_boxes]
        match_flags = match_flags.view(batch_size, n_groups, -1)
        match_gt_idx = match_gt_idx.view(batch_size, n_groups, -1)

        return BaseVGMatchOutput(
            match_bbox_flags=match_flags,
            match_gt_bbox_indexes=match_gt_idx,
            match_scores=match_scores,
        )
