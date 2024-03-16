"""
Implementation the bboxes conversion, which contains the type adjustment and 
label mapping.

The type adjustment aims to convert the bboxes from `source` type to the `target`
type. The label mapping operates on the ids and labels for bounding boxes (bboxes).
"""

import torch
from torchvision.ops.boxes import box_convert

from vggbase.boxes.bbox_generic import BboxesFormatTypePool, BboxesCoordinateTypePool
from vggbase.utils.generic_components import BaseVGList
from vggbase.boxes.bbox_generic import BaseVGModelBBoxes, BaseVGBBoxes
from vggbase.boxes.bbox_utils import convert_hw_whwh


def convert_bbox_type(
    bboxes: torch.Tensor,
    source_type: str,
    target_type: str,
    board_hws: torch.IntTensor = None,
) -> torch.Tensor:
    """
    Convert one batch of bboxes's format from source_type to target_type.

    :param bboxes: A `torch.Tensor` holding bboxes for one batch of samples,
     of shape, [bs, N, 4],
     of format, determined by the `source_type`
    :param source_type: A string denoting the format for input
     'bboxes'.
    :param target_type: A string denoting the desired format for input
     'bboxes'.
    :param board_hws: a `torch.IntTensor` with shape [bs, 2] containing the height
     and width
    """

    assert source_type in BboxesFormatTypePool
    assert target_type in BboxesFormatTypePool

    if source_type == target_type:
        return bboxes

    # Convert to [bs, 4] where 4 means whwh
    board_whwhs = convert_hw_whwh(board_hws) if board_hws is not None else None

    # Convert to the 'albumentations' when
    # the source_type is normalized xyxy,
    # i.e., unnormalized xyxy -> normalized xyxy
    if source_type == "pascal_voc":
        # Normalize the bboxes
        bboxes = bboxes / board_whwhs[:, None, :]

        source_type = "albumentations"

    src_coordinate_type = BboxesCoordinateTypePool[source_type]

    if target_type == "pascal_voc":
        # Must first convert to albumentations type,
        # i.e., normalized coordinates, xyxy
        bboxes = box_convert(
            bboxes,
            in_fmt=src_coordinate_type,
            out_fmt=BboxesCoordinateTypePool["albumentations"],
        )

        # Convert to albumentations back to pascal_voc
        return bboxes * board_whwhs[:, None, :]

    tgt_coordinate_type = BboxesCoordinateTypePool[target_type]

    # Convert the box type to the desried fomat
    # under the normalized coordinates
    bboxes = box_convert(
        bboxes,
        in_fmt=src_coordinate_type,
        out_fmt=tgt_coordinate_type,
    )

    return bboxes


def convert_bbox_format(
    target_bboxes: BaseVGList[BaseVGBBoxes],
    format_type: str,
):
    """
    Convert bboxes from model's ouput or sample target to the
    format accepted by evaluator.

    The potention issue of this function is that:
    as the argument `target_bboxes` is a list, which is mutable,
    the function receives a reference to the original list object,
    rather than a copy of the list. If the function modifies the
    list object through this reference, the changes will be reflected
    in the original list variable outside the function.

    Therefore, no returns are added to this function.
    """
    batch_size = len(target_bboxes)
    for bs_idx in range(batch_size):
        vg_bboxes = target_bboxes[bs_idx]
        bbox_type = vg_bboxes.bbox_type
        # Get bboxes, of shape [N_i, 4]
        bboxes = vg_bboxes.bboxes
        # Convert to the desired format
        # [1, N_i, 4] where 1 is the batch size
        bboxes = bboxes.view(1, -1, 4)
        board_hw = vg_bboxes.board_hw
        board_hw = torch.Tensor(board_hw).view(1, -1)
        board_hw = board_hw.to(bboxes.device)

        format_bboxes = convert_bbox_type(
            bboxes=bboxes,
            source_type=bbox_type,
            target_type=format_type,
            board_hws=board_hw,
        )
        vg_bboxes["bboxes"] = format_bboxes.view(-1, 4)
        vg_bboxes["bbox_type"] = format_type


def convert_model_bbox_format(
    model_bboxes: BaseVGModelBBoxes,
    format_type: str,
) -> BaseVGModelBBoxes:
    """
    Convert the model's output to be format.

    For the VGGbase's BaseVGModelBBoxes, any changes made within this
    function will change the corresponding variable outside the function.
    """

    # Get bboxes, of shape [bs, n_groups, N, 4]
    bboxes = model_bboxes.bboxes
    batch_size, n_bboxes = bboxes.shape[0], bboxes.shape[2]

    # Get hws, of shape [N, 4]
    board_hws = model_bboxes.board_hws
    bbox_type = model_bboxes.bbox_type

    # [bs, n_groups * N, 4]
    bboxes = bboxes.reshape(batch_size, -1, 4)
    bboxes = convert_bbox_type(
        bboxes=bboxes,
        source_type=bbox_type,
        target_type=format_type,
        board_hws=board_hws,
    )
    bboxes = bboxes.reshape(batch_size, -1, n_bboxes, 4)
    # [bs, n_groups, N, 4]
    model_bboxes["bboxes"] = bboxes
    model_bboxes["bbox_type"] = format_type


def convert_mapper(
    bbox_ids: torch.IntTensor, bbox_labels: torch.IntTensor
) -> torch.IntTensor:
    """
    Convert the ids and labels to the mapper.

    :param bbox_ids: A `torch.IntTensor` holding bboxes id.
     of shape, [N]
     of format, in range [0, P-1], where P is the number of phrases.
    :param bbox_labels: A `torch.IntTensor` holding bboxes label.
     of shape, [N]
     of format, in range [0, CLASS]
     where CLASS is the number of categories of phrases.

    :return id_label_mapper: A `torch.IntTensor`
     of shape, [N, 2]
     of format, first column contains the `id` as key while the second
     column contains the corresponding `label`
    """
    id_label = torch.stack([bbox_ids, bbox_labels], dim=1)
    # Remove the duplicate rows from the tensor
    # the left rows will be the mapper
    # of shape, [P, 2]
    id_label_mapper, _ = torch.unique(id_label, dim=0, return_inverse=True)

    return id_label_mapper


def convert_ids_labels(
    search_ids: torch.IntTensor,
    id_label_mapper: torch.IntTensor,
    outside_id_label: int,
) -> torch.IntTensor:
    """
    Convert ids to labels based on the obtained mapper.

    :param search_ids: A `torch.IntTensor` holding bboxes id that will be mapped to
     labels.
     of shape, [n_ids]

    :param outside_id_label: This label will be assigned to ids who is not included in the
     `ids` kye of `id_label_mapper`.
    :return searched_labels: Holding mapped labels with same shape as search_ids.
    """
    searched_labels = torch.zeros_like(search_ids, device=search_ids.device)
    for i, srch_id in enumerate(search_ids):
        # if the srch_id is not inclued in `ids` key of the mapper
        id_key_mask = id_label_mapper[:, 0] == srch_id
        if not torch.any(id_key_mask):
            searched_labels[i] = outside_id_label
        else:
            searched_labels[i] = id_label_mapper[id_key_mask, 1]

    return searched_labels
