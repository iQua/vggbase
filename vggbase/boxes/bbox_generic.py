"""
The implementations for generic structure of bounding boxes (bboxes).

Following the description in [1], here are four types of bboxes,

For the board with size [640, 480] where width = 640, height = 480,

1- pascal_voc, [x_min, y_min, x_max, y_max], (unnormalized)
    denotes as `xyxy`
    for example, [98, 345, 420, 462]

2- albumentations, [x_min, y_min, x_max, y_max], (normalized)
    denotes as `normalized_xyxy`
    for example, [98 / 640, 345 / 480, 420 / 640, 462 / 480]
                 = [0.153125, 0.71875, 0.65625, 0.9625]

3- coco, [x_min, y_min, width, height],(unnormalized)
    denotes as `xywh`              
    for example, [98, 345, 322, 117]

4- yolo, [x_center, y_center, width, height], (normalized)
    denotes as `cxcywh`
    for example, [0.4046875, 0.840625, 0.503125, 0.24375]

[1]. https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from vggbase.utils.generic_components import FieldFrozenContainer

BboxesFormatTypePool = ("pascal_voc", "albumentations", "coco", "yolo")
BboxesCoordinateTypePool = {
    "pascal_voc": "xyxy",
    "albumentations": "xyxy",
    "coco": "xywh",
    "yolo": "cxcywh",
}


class BaseSampleBBoxes:
    """
    The basic bounding boxes (bboxes) container for the sample of VGGbase.
    It Holds bboxes for one sample.
    Args:
        box_type: A `str` presenting the format of bboxes,
         such as
         yolo, normalized [ctr_x, ctr_y, w, h]
         pascal_voc, un-normalized [x1, y1, x2, y2]

        bboxes: A `List` in which each item is
         a `List` containing multiple bounding boxes for the
         corresponding phrase,
         For example,
            [
                [[281, 106, 464, 329]],
                [[337, 200, 429, 295], [332, 204, 434, 304]],
                [[270, 250, 390, 332]]
            ]
        board_hw: A `tuple` holding the image size where
         bboxes exist,
         of shape, [2, ]
         of format, [h, w]

        Then, the generated `bbox_ids` can be:
        [0, 1, 1, 2]
        Note, the `bbox_ids` here is actually the index of the bboxe's
        phrase query in the sentence.

    """

    def __init__(
        self,
        bboxes: List[List[Tuple[float, float, float, float]]],
        board_hw: Tuple[int, int],
        bbox_type: str,
        phrases_label: Optional[List[List[int]]] = None,
    ):
        self.bbox_type = bbox_type

        # flatten input `bboxes` from nested list
        # to 1d list
        self.bboxes = self.to_standard_bboxes(bboxes)
        # generate bboxes id
        # which is the item index of the bboxes in the
        # input `bboxes`
        self.bbox_ids = self.generate_bbox_ids(bboxes)

        self.board_hw = board_hw

        # the bbox label corresponding to the phrase
        # label
        self.bboxes_label = None
        if phrases_label is not None:
            self.assign_labels(phrases_label)

    def to_standard_bboxes(self, bboxes):
        """Convert the bboxes to the standard format.
        1. flatten the bboxes.
        2. convert to np.ndarray
        """
        return np.array([bbox for group_bboxes in bboxes for bbox in group_bboxes])

    def generate_bbox_ids(self, boxes):
        """Generate ids for each bbox."""
        return np.concatenate(
            [np.full(len(group_b), i) for i, group_b in enumerate(boxes)], axis=0
        )

    def assign_labels(self, phrases_label: List[List[int]]):
        """
        Assign the label to bbox.

        :param: phrases_label: A `List` holding the sub-list which
         includes the corresponding phrase label,
         of shape, len(phrases_label) = bs
         of format, [[], [], []]
         For example, [[6], [2], [5]]

        :return A `List` holding the phrase label of each bbox,
         of shape, length == len(self.bboxes)
         of format, int
         For example, [6, 2, 2, 5]
        """
        self.bboxes_label = np.array(
            [phrases_label[bbox_id][0] for bbox_id in self.bbox_ids]
        )
        return self.bboxes_label


@dataclass
class BaseVGBBoxes(FieldFrozenContainer):
    """
    Base class for bounding boxes (bboxes) of VGGbase.

    Args:
        bboxes: A `tensor.FloatTensor` holding the
         bboxes of the sample `i`,
         of shape, [N_i, 4]
         of format, any formats described by `bbox_type`.
         such as, [x1, y1, x2, y2], pascal_voc.
        labels: A `IntTensor` holding bboxes' phrase
         labels,
         of shape, [N_i]
         of format, int, in range [0, CLASS]
         where the CLASS the total number of categories
         for phrases.

        bbox_ids: A `IntTensor` holding bboxes' id
         within the sample. The `id` should be the index of
         phrase that bbox corresponds to.
         of shape, [N_i]
         of format, int, in range [0, num_phrases-1]

        board_hw: A `tuple` holding the height and
         width of the image board that bboxes exist,
         of shape, [2, ]
         of format, [h, w]
         This is the image's original size without any padding
         or resizing.

        bbox_type: A `str` holding the coordinate style of
         bboxes,
         As described above (top of this file).
    """

    bboxes: Optional[torch.FloatTensor] = None
    labels: Optional[torch.IntTensor] = None
    bbox_ids: Optional[torch.IntTensor] = None
    board_hw: Optional[Tuple[int, int]] = None
    bbox_type: Optional[str] = None


@dataclass
class BaseVGModelBBoxes(FieldFrozenContainer):
    """
    Base class for bounding boxes (bboxes) of VGGbase.

    Args:
        bboxes: A `tensor.FloatTensor` holding the
         bboxes of the sample `i`,
         of shape, [bs, n_groups, N, 4]
         of format, any formats described by `bbox_type`.
         such as, [x1, y1, x2, y2], pascal_voc.
        similarity_scores: A `tensor.FloatTensor` holding matching
         scores between bboxes and text queries.
         of shape, [bs, n_groups, N, P]
         of format, int, in range [0, 1]
         where `P` is the number of phrases after padding
        class_logits: A `IntTensor` holding bboxes' predicted logits
         in terms of phrases,
         of shape, [bs, n_groups, N]
         of format, int, in range [0, CLASS]
         where the CLASS the total number of categories
         for phrases.
        bbox_ids: A `tensor.IntTensor` holding bboxes' id
         within the sample. The `id` should be the index of
         phrase that bbox corresponds to.
         of shape, [bs, N]
         of format, int, in range [0, num_phrases]
        board_hws: A `tensor.IntTensor` holding the height
         and width of the image board that bboxes exist,
         of shape, [bs, 2]
         of format, 4 means [h, w]
        bbox_type: A `str` holding the coordinate style of
         bboxes,
         As described above (top of this file).

        where `n_groups` means that there are multiple groups
        of predicted bboxes for one sample. Generally, this should
        be the multiple heads utilized in grounding model because
        each head will output predictions. Besides, `n_groups` can
        also be the number of iterations for the progressive bounding
        box generation.

    """

    bboxes: Optional[torch.FloatTensor] = None
    similarity_scores: Optional[torch.FloatTensor] = None
    class_logits: Optional[torch.FloatTensor] = None
    bbox_ids: Optional[torch.IntTensor] = None
    board_hws: Optional[torch.IntTensor] = None
    bbox_type: Optional[str] = None

    def to(self, device) -> None:
        """Assign the tensor item into the specific device."""
        for k, k_data in self.items():
            if hasattr(k_data, "to"):
                k_data.to(device)
                super().__setitem__(k, k_data)
