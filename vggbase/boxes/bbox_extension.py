"""
Implementation of bbox extension, which generates, pads, replaces, or 
removes bboxes.
"""

import random

import torch

from vggbase.boxes.bbox_generic import BaseVGBBoxes


def generate_random_bboxes(n_bboxes: int, device) -> torch.FloatTensor:
    """Generate `n_bboxes` randomly"""
    # 3sigma = 1/2 --> sigma: 1/6
    generated_bboxes = torch.randn(n_bboxes, 4, device=device) / 6.0 + 0.5
    # ensure the range [0, 1]
    generated_bboxes = torch.where(
        generated_bboxes < 0, generated_bboxes + 0.5, generated_bboxes
    )
    generated_bboxes = torch.where(
        generated_bboxes > 1, generated_bboxes - 0.5, generated_bboxes
    )
    return generated_bboxes


def pad_random_bboxes(vg_bboxes: BaseVGBBoxes, target_n_proposals: int) -> BaseVGBBoxes:
    """
    Randomly pad or remove bboxes, until the #bboxes reaches to
    `target_n_proposals`.

    :param bboxes: A `torch.FloatTensor` holding bounding boxes,
     of shape, [N_i, 4]
     of format, [ctr_x, ctr_y, width, height], i.e., Yolo type.
    :param target_n_proposals: A `Int` denoting target number of
     bboxes.

    :return A `torch.FloatTensor` holding the padded bounding boxes,
     of shape, [target_n_proposals, 4].
     of format, [ctr_x, ctr_y, width, height], i.e., Yolo type.
    """
    # bboxes must be in yolo type
    assert vg_bboxes.bbox_type == "yolo"

    bboxes = vg_bboxes.bboxes
    num_bboxes = bboxes.shape[0]

    # generate fake bboxes if empty bboxes
    if not num_bboxes:
        bboxes = torch.as_tensor(
            [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=bboxes.device
        )
        num_bboxes = 1

    if num_bboxes < target_n_proposals:
        box_placeholder = generate_random_bboxes(
            target_n_proposals - num_bboxes, device=bboxes.device
        )
        prepared_bboxes = torch.cat((bboxes, box_placeholder), dim=0)

    elif num_bboxes > target_n_proposals:
        select_mask = [True] * target_n_proposals + [False] * (
            num_bboxes - target_n_proposals
        )
        random.shuffle(select_mask)
        prepared_bboxes = bboxes[select_mask]
    else:
        prepared_bboxes = bboxes

    return BaseVGBBoxes(
        bboxes=prepared_bboxes,
        labels=None,
        bbox_ids=None,
        board_hw=vg_bboxes.board_hw,
        bbox_type=vg_bboxes.bbox_type,
    )


def pad_repeat_bboxes(
    vg_bboxes: BaseVGBBoxes,
    target_n_proposals: int,
) -> BaseVGBBoxes:
    """
    Preparation mechanism in this function is interesting,
    as it will pad the `bboxes` by
        1. repeating the `bboxes`
        2. repeating each bboxes by multiples times.
        2. the number of repeats decreases from last ones to first ones,
         meaning that boxes with lower index in `bboxes` will be repeated
         less.
        For example, when
            target_n_proposals = 10
            n_bboxes = 6
        The #repeat: [1, 1, 2, 2, 2, 2]
    """
    # bboxes must be in yolo type
    assert vg_bboxes.bbox_type == "yolo"

    bboxes = vg_bboxes.bboxes
    bbox_ids = vg_bboxes.bbox_ids
    bboxes_label = vg_bboxes.labels

    n_bboxes = bboxes.shape[0]
    if not n_bboxes:  # generate fake gt boxes if empty gt boxes
        bboxes = torch.as_tensor(
            [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=bboxes.device
        )
        n_bboxes = 1

    # number of repeat except the last gt box in one image
    n_repeat = target_n_proposals // n_bboxes
    n_remainders = target_n_proposals % n_bboxes
    repeat_tensor = [n_repeat] * (n_bboxes - n_remainders) + [n_repeat + 1] * (
        n_remainders
    )
    assert sum(repeat_tensor) == target_n_proposals
    random.shuffle(repeat_tensor)
    repeat_tensor = torch.tensor(repeat_tensor, device=bboxes.device)

    padded_bboxes = torch.repeat_interleave(bboxes, repeat_tensor, dim=0)
    padded_bbox_ids = torch.repeat_interleave(bbox_ids, repeat_tensor, dim=0)
    padded_bboxes_label = torch.repeat_interleave(bboxes_label, repeat_tensor, dim=0)

    return BaseVGBBoxes(
        bboxes=padded_bboxes,
        labels=padded_bboxes_label,
        bbox_ids=padded_bbox_ids,
        board_hw=vg_bboxes.board_hw,
        bbox_type=vg_bboxes.bbox_type,
    )


def pad_balance_bboxes(
    vg_bboxes: BaseVGBBoxes,
    target_n_proposals: int,
) -> BaseVGBBoxes:
    """
    Preparation mechanims in this function is interesting,
    as it will pad the `bboxes` by
        1. repeating the `bboxes`
        2. repeating each bboxes by multiples times.
        2. the number of repeats decreases from last ones to first ones,
         meaning that boxes with lower index in `bboxes` will be repeated
         less.
        For example, when
            target_n_proposals = 10
            n_bboxes = 6
        The #repeat: [1, 1, 2, 2, 2, 2]
    """
    # bboxes must be in yolo type
    assert vg_bboxes.bbox_type == "yolo"

    bboxes = vg_bboxes.bboxes
    bbox_ids = vg_bboxes.bbox_ids
    bboxes_label = vg_bboxes.labels

    n_bboxes = bboxes.shape[0]
    if not n_bboxes:  # generate fake gt boxes if empty gt boxes
        bboxes = torch.as_tensor(
            [[0.5, 0.5, 1.0, 1.0]], dtype=torch.float, device=bboxes.device
        )
        n_bboxes = 1

    # number of repeat except the last gt box in one image
    n_repeat = target_n_proposals // n_bboxes
    n_remainders = target_n_proposals % n_bboxes
    repeat_tensor = [n_repeat] * (n_bboxes - n_remainders) + [n_repeat + 1] * (
        n_remainders
    )
    assert sum(repeat_tensor) == target_n_proposals
    random.shuffle(repeat_tensor)
    repeat_tensor = torch.tensor(repeat_tensor, device=bboxes.device)

    padded_bboxes = torch.repeat_interleave(bboxes, repeat_tensor, dim=0)
    padded_bbox_ids = torch.repeat_interleave(bbox_ids, repeat_tensor, dim=0)
    padded_bboxes_label = torch.repeat_interleave(bboxes_label, repeat_tensor, dim=0)

    return BaseVGBBoxes(
        bboxes=padded_bboxes,
        labels=padded_bboxes_label,
        bbox_ids=padded_bbox_ids,
        board_hw=vg_bboxes.board_hw,
        bbox_type=vg_bboxes.bbox_type,
    )


def replace_bboxes_randomness(
    vg_bboxes: BaseVGBBoxes, keep_mask: torch.BoolTensor
) -> BaseVGBBoxes:
    """
    Replace bboxes based on the keep_mask with random ones.

    :param keep_mask: A `torch.BoolTensor` holding mask for
     which bboxes will be kept - True
     of shape, [N_i]
     of format, if keep, True
    """

    bboxes = vg_bboxes.bboxes
    num_bboxes = bboxes.shape[0]
    left_num_bboxes = torch.sum(keep_mask)

    # generate random ones
    box_placeholder = generate_random_bboxes(
        num_bboxes - left_num_bboxes, device=bboxes.device
    )
    # replace
    bboxes[keep_mask] = box_placeholder

    return BaseVGBBoxes(bboxes=bboxes)
