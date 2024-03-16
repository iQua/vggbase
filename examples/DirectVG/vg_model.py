"""
A model to generate bounding boxes to approach the target progressively.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np


from vggbase.boxes import bbox_extension
from vggbase.boxes import bbox_convertion
from vggbase.boxes import bbox_generic
from vggbase.boxes.bbox_utils import box_iou
from vggbase.models.model_generic import BaseVGModelInput, BaseVGModelOutput


def generate_phrase_bboxes(n_boxes: int, n_phrases: int):
    """Generate boxes for each phrase."""
    # Generate boxes with shape
    # [n_boxes, 4], [ctr_x, ctr_y, width, height]
    # and the label with shape [n_boxes,]
    return bbox_extension.generate_random_bboxes(
        n_bboxes=n_boxes, device=None
    ), torch.rand(size=(n_boxes, n_phrases))


def box_adjustment(
    vg_bboxes: bbox_generic.BaseVGBBoxes,
    ground_truth_boxes: bbox_generic.BaseVGBBoxes,
    iterations=5,
    lr_pos=0.45,
    lr_size=0.4,
) -> Tuple[List[bbox_generic.BaseVGBBoxes], List[torch.Tensor]]:
    """
    Adjust the bounding boxes to approach the ground truth progressively.

    This functions behave as a guidance for those approaches that aim to
    generate bounding boxes to approach the ground truth progressively.
    """
    # Convert bounding boxes and ground truth to pascal_voc format for IoU calculation
    bbox_convertion.convert_bbox_format([vg_bboxes], format_type="pascal_voc")
    bbox_convertion.convert_bbox_format([ground_truth_boxes], format_type="pascal_voc")

    all_iter_vg_boxes = [vg_bboxes.copy()]
    all_iter_similarity = [box_iou(vg_bboxes.bboxes, ground_truth_boxes.bboxes)[0]]
    # Perform the iterative adjustment
    # For each iterative, the box will be perform one by one
    for cur_iter_i in range(iterations):
        latest_vg_bboxes = all_iter_vg_boxes[-1]

        latest_boxes = latest_vg_bboxes.bboxes
        updated_boxes = torch.zeros_like(latest_boxes)
        for i, i_box in enumerate(latest_boxes):
            # Compute IoU with each ground truth box
            iou_scores, _ = box_iou(i_box.unsqueeze(0), ground_truth_boxes.bboxes)
            # Find the ground truth box with the highest IoU score
            best_gt_index = torch.argmax(iou_scores)
            best_gt_box = ground_truth_boxes.bboxes[best_gt_index]

            # Calculate adjustments for position (center) and size
            current_center = (i_box[:2] + i_box[2:]) / 2
            gt_center = (best_gt_box[:2] + best_gt_box[2:]) / 2
            delta_center = gt_center - current_center

            current_size = i_box[2:] - i_box[:2]
            gt_size = best_gt_box[2:] - best_gt_box[:2]
            delta_size = gt_size - current_size

            # Adjust the lr_pos and lr_size
            adj_lr_pos = max(cur_iter_i / iterations - 1, lr_pos)
            adj_lr_size = max(cur_iter_i / iterations - 1, lr_size)

            # Apply adjustments
            # Adjust size and compensate for center movement
            updated_boxes[i][:2] = latest_boxes[i][:2] + adj_lr_pos * delta_center
            updated_boxes[i][2:] = latest_boxes[i][2:] + adj_lr_pos * delta_center
            updated_boxes[i][2:] = (
                updated_boxes[i][2:]
                + adj_lr_size * delta_size
                - adj_lr_size * delta_center
            )
        # create vg bboxes
        iter_vg_bboxes = bbox_generic.BaseVGBBoxes(
            bboxes=updated_boxes,
            board_hw=latest_vg_bboxes.board_hw,
            bbox_type="pascal_voc",
        )
        iter_similarity, _ = box_iou(updated_boxes, ground_truth_boxes.bboxes)
        all_iter_vg_boxes.append(iter_vg_bboxes)
        all_iter_similarity.append(iter_similarity)

    # Convert a list of similarities to a tensor
    # of shape, [n_iterations, n_proposals, 4]
    all_iter_similarity = torch.stack(all_iter_similarity, dim=0)
    return all_iter_vg_boxes, all_iter_similarity


class DirectVG(nn.Module):
    """The VG algorithm to generate bounding boxes progressively."""

    def __init__(
        self,
        n_proposals: int,
    ):
        super().__init__()
        # Set how many proposals to generate
        self.n_proposals = n_proposals

    def generate_bboxes(self, n_boxes_per_phrase: int, n_phrases: int):
        """Generate boxes for one batch of samples."""

        generated_bboxes = [
            generate_phrase_bboxes(n_boxes, n_phrases) for n_boxes in n_boxes_per_phrase
        ]

        # Get boxes with shape [n_proposals, 4]
        return torch.cat([bboxes[0] for bboxes in generated_bboxes], dim=0), torch.cat(
            [bboxes[1] for bboxes in generated_bboxes], dim=0
        )

    def generate_batch_bboxes(self, n_boxes_per_phrase, batch_size, n_phrases):
        """Generate boxes for one batch of samples."""
        batch_bboxes = [
            self.generate_bboxes(n_boxes_per_phrase, n_phrases)
            for _ in range(batch_size)
        ]

        # Get the batch of boxes with shape [batch_size, n_proposals, 4]
        # of format, [ctr_x, ctr_y, width, height]
        # Get the batch of box scores with shape [batch_size, n_proposals, n_phrases]
        return torch.stack([bboxes[0] for bboxes in batch_bboxes], dim=0), torch.stack(
            [bboxes[1] for bboxes in batch_bboxes], dim=0
        )

    def forward(self, inputs: BaseVGModelInput):
        """Forward the model to generate bounding boxes progressively."""
        # Get how many boxes to generate for each phrase
        batch_size, n_phrases = inputs.text_samples.shape[:2]
        # Get the targets
        targets = inputs.targets

        pieces = np.array_split(np.arange(0, self.n_proposals), n_phrases)
        n_boxes_per_phrase = [len(piece) for piece in pieces]

        # Generate bounding boxes progressively
        # Get boxes with shape [batch_size, n_proposals, 4]
        # of format, [ctr_x, ctr_y, width, height]
        batch_boxes, similarity_scores = self.generate_batch_bboxes(
            n_boxes_per_phrase, batch_size, n_phrases
        )

        # For sample in the batch, perform the progressive adjustment

        # Convert the boxes to the pascal_voc format
        progressive_results = []
        progressive_board_hws = []
        progressive_similarities = []
        for i in range(batch_size):
            target_vg_bboxes = targets[i].vg_bboxes
            # Create the BaseVGBBoxes
            vg_bboxes = bbox_generic.BaseVGBBoxes(
                bboxes=batch_boxes[i],
                board_hw=target_vg_bboxes.board_hw,
                bbox_type="yolo",
            )
            iter_vg_bboxes, iter_scores = box_adjustment(vg_bboxes, target_vg_bboxes)

            progressive_results.append(
                torch.stack([vg_bboxes.bboxes for vg_bboxes in iter_vg_bboxes], dim=0)
            )
            progressive_board_hws.append(
                torch.Tensor([vg_bboxes.board_hw for vg_bboxes in iter_vg_bboxes])
            )
            progressive_similarities.append(iter_scores)

        # Convert a list of bounding boxes into a tensor
        # of shape, [batch_size, n_iterations, n_proposals, 4]
        progressive_results = torch.stack(progressive_results, dim=0)
        progressive_board_hws = torch.stack(progressive_board_hws, dim=0)
        progressive_similarities = torch.stack(progressive_similarities, dim=0)

        # Return the output
        return BaseVGModelOutput(
            bboxes=progressive_results,
            similarity_scores=progressive_similarities,
            class_logits=None,
            bbox_ids=None,
            board_hws=progressive_board_hws,
            bbox_type="pascal_voc",
        )
