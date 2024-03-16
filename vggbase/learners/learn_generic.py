"""
Implementations of generic components for the learning process.
"""

from typing import Optional
from dataclasses import dataclass

import torch


from vggbase.utils.generic_components import FieldFrozenContainer


@dataclass
class BaseVGMatchOutput(FieldFrozenContainer):
    """
    Base class for outputs of visual grounding models.

    Args:
        match_bbox_flags: Match flags assigned to bboxes.
         of shape, [batch_size, n_groups, n_bboxes]
         of values:
          -2: Invalid bboxes should not be considered.
          -1: Not matched bboxes.
          0: Negative bboxes.
          1: Positive bboxes.
        match_gt_bbox_indexes: Indexes of the ground truth bboxes that the
         predicted boxes are matched to. Each box will be assigned the id of the ground truth box.
         of shape, [batch_size, n_groups, n_bboxes]
         of values range from 0 to N_i, where N_i is the number of gt bboxes
            for the i-th sample in the batch.
        match_scores: The matching scores of the bboxes. A list
         of length batch_size while each term is a tensor
         of shape, [n_groups, n_bboxes, N_i]
         of values range from 0 to 1.

        additional_output: A `FieldFrozenContainer` for including the
         possible additional output.
    """

    match_bbox_flags: Optional[torch.IntTensor] = None
    match_gt_bbox_indexes: Optional[torch.IntTensor] = None
    match_scores: Optional[torch.Tensor] = None
    additional_output: Optional[FieldFrozenContainer] = None
