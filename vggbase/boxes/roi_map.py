"""
Here we implement the roi mapping in the general CNN architecture.

Mapping the ROI in one feature to another feature map in the CNN layers.

Previous layer to the latter layer

Generally, the ROI denotes the region in the original image.

However, in this code, the ROI means the region in one feature map.

Following the tutorial in https://zhuanlan.zhihu.com/p/24780433.

For the box transformation, the detectron2 should be a great package as
it contains structures.boxes.BoxMode

"""

import torch

from .bbox_utils import box_format_src_to_tgt


class CNNRoiMapper(object):
    """The mapper to map the roi among CNN layers."""

    def __init__(self):
        pass

    def bottom_roi_to_top(
        self,
        roi_coords: torch.Tensor,
        total_strides: int,
        height: int,
        weight: int,
        roi_format_mode: str = "pascal_voc",
    ):
        """Map the roi_coords (x_min, y_min, x_max, y_max) in the previous layer's feature map (bottom)
        to the region of latter layer's feature map (top).

        :param roi_coords: A tensor with shape
                batch_size, num_boxes, 4
        :param total_strides: The multiply of strides between the source layer
                and the target layer,
                including the cnn and pooling layer.
        :param height: A integar showing the height of the 2D board
            where the rois exist.
        :param weight: A integar showing the height of the 2D board
            where the rois exist.
        :param roi_format_mode: The coordinates type of the rois.

        """
        # (x0, y0, x1, y1) inabsolute floating points coordinates.
        # The coordinates in range [0, width or height].
        B = roi_coords.shape[0]

        # to batch_size * num_boxes, 4
        roi_coords = roi_coords.view(-1, 4)

        # conver the rois from the source input
        # type to the target type.
        # (x0, y0, x1, y1) inabsolute floating points coordinates.
        # The coordinates in range [0, width or height].
        roi_coords = box_format_src_to_tgt(
            roi_coords,
            source_type=roi_format_mode,
            target_type="pascal_voc",
            h=height,
            w=weight,
        )

        # torch.div(caption_bboxes, scale_fct)
        top_left_xs_ys = roi_coords[:, :2]
        bottom_right_xs_ys = roi_coords[:, 2:4]

        target_top_left_xs_ys = (
            torch.floor(torch.div(top_left_xs_ys, total_strides)) + 1
        )

        target_bottom_right_xs_ys = (
            torch.ceil(torch.div(bottom_right_xs_ys, total_strides)) - 1
        )

        # to batch_size * num_boxes, 4
        target_roi_coords = torch.stack(
            target_top_left_xs_ys, target_bottom_right_xs_ys, dim=1
        )

        # convert back to the input format
        target_roi_coords = box_format_src_to_tgt(
            roi_coords,
            source_type="pascal_voc",
            target_type=roi_format_mode,
            h=height,
            w=weight,
        )
        target_roi_coords = target_roi_coords[:, 4]
        # convert to batch_size start
        target_roi_coords = target_roi_coords.view(B, -1, 4)

        return target_roi_coords

    def top_roi_to_bottom(
        self,
        roi_coords,
        total_strides,
        height: int,
        weight: int,
        roi_format_mode: str = "pascal_voc",
    ):
        """Map the roi_coords (x_min, y_min, x_max, y_max) in the latter layer's feature map (bottom)
        to the region of previous layer's feature map (top).

        :param roi_coords: A tensor with shape
                batch_size, num_boxes, 4
        :param total_strides: The multiply of strides between the source layer
                and the target layer,
                including the cnn and pooling layer.
        :param height: A integar showing the height of the 2D board
            where the rois exist.
        :param weight: A integar showing the height of the 2D board
            where the rois exist.
        :param roi_format_mode: The coordinates type of the rois.
        """

        B = roi_coords.shape[0]

        # convert to batch_size * num_boxes, 4
        roi_coords = roi_coords.view(-1, 4)

        # conver the rois from the source input
        # type to the target type, pascal_voc.
        # (x0, y0, x1, y1) inabsolute floating points coordinates.
        # The coordinates in range [0, width or height].
        roi_coords = box_format_src_to_tgt(
            roi_coords,
            source_type=roi_format_mode,
            target_type="pascal_voc",
            h=height,
            w=weight,
        )

        # torch.div(caption_bboxes, scale_fct)
        top_left_xs_ys = roi_coords[:, :2]
        bottom_right_xs_ys = roi_coords[:, 2:4]

        target_top_left_xs_ys = torch.mul(top_left_xs_ys, total_strides)
        target_bottom_right_xs_ys = torch.mul(bottom_right_xs_ys, total_strides)

        # to batch_size * num_boxes, 4
        target_roi_coords = torch.stack(
            target_top_left_xs_ys, target_bottom_right_xs_ys, dim=1
        )

        # convert back to the input format
        target_roi_coords = box_format_src_to_tgt(
            roi_coords,
            source_type="pascal_voc",
            target_type=roi_format_mode,
            h=height,
            w=weight,
        )

        target_roi_coords = target_roi_coords[:, 4]

        # convert to batch_size start
        target_roi_coords = target_roi_coords.view(B, -1, 4)

        return target_roi_coords
