"""
Utilities for bounding box manipulation and GIoU.

The basic assumption:
    The bboxes denote the bounding boxes in torch.tensor type
    The boxes denote the bounding boxes in list type.

"""

from typing import Union, Tuple, List

import torch
from torchvision.ops.boxes import box_area


def convert_hw_whwh(board_hws: torch.IntTensor):
    """Convert the board_hws to boards_whwh.

    :param board_hws: Holding height and width for one batch of samples,
     of shape, [bs, 2]
    """
    return torch.concat(
        [board_hws[:, 1:], board_hws[:, :1], board_hws[:, 1:], board_hws[:, :1]], dim=1
    )


def box_iou(
    bboxes1: torch.Tensor, bboxes2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    """Compute the iou scores and the union."""
    area1 = box_area(bboxes1)
    area2 = box_area(bboxes2)

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N,M,2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def judge_similarity(
    similarity_scores: torch.FloatTensor,
    threshold: float = 0.5,
) -> Tuple[torch.BoolTensor, torch.IntTensor]:
    """
    Judge those acceptable ones via similarity.

    :param similarity_scores: A `torch.FloatTensor` holding the
     similarity scores between bboxes and target objects, such as phrases,
     of shape, [bs, n_bboxes, n_objects]
    :param threshold: A `float` behaving as a bound of the similarity for
     refreshment.
     of range, [0, 1]
    """
    batch_size = similarity_scores.shape[0]
    n_bboxes = similarity_scores.shape[1]
    flat_scores = similarity_scores.view(batch_size * n_bboxes, -1)

    flat_scores = torch.sigmoid(flat_scores)
    value, _ = torch.max(flat_scores, -1, keepdim=False)
    # [bs * n_bboxes]
    keep_mask = value > threshold
    # [bs, n_bboxes]
    keep_mask = keep_mask.reshape(batch_size, n_bboxes)
    # [bs,]
    n_keep = torch.sum(keep_mask, dim=1)
    return keep_mask, n_keep


def remove_outbound_bboxes(bboxes: torch.Tensor, hw: Tuple[int, int]):
    """Remove bboxes that are out of the boundary."""

    keep = (
        (bboxes[:, 0] >= 0)
        & (bboxes[:, 1] >= 0)
        & (bboxes[:, 2] >= 0)
        & (bboxes[:, 3] >= 0)
        & (bboxes[:, 2] <= hw[1])
        & (bboxes[:, 3] <= hw[0])
    )

    return torch.where(keep)[0]
