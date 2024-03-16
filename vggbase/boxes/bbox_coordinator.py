"""
The coordinates coder of bounding boxes based on 
`torchvision`.

Our implementation refer to the one in torchvision, i.e.,
    torchvision.models.detection._utils import BoxCoder.

Notations:
    ctr_x, ctr_y       - centers x of bounding boxes
    ref_ctr_x, ref_ctr_y - centers of reference boxes, 
    bb_ctr_x, bb_ctr_y - centers of bounding boxes
    ref_w, ref_h   - width and height of reference boxes
    bb_w, bb_h   - width and height of bounding boxes

    reference boxes generally are the ground truth boxes
    or the desired output boxes given predicted
    `offsets`.


Generally, there are four types of Coordinaters.

(1.) Delta-based bbox coding.
Follow RCNN/Fast RCNN./Faster RCNN [0][1][2], [3].

The boxes regression outputs the `offsets`
`offsets = [offset_ctr_x, offset_ctr_y, offset_w, offset_h]` computed as:

    ref_ctr_x  = offset_ctr_x * bb_w + bb_ctr_x
    ref_ctr_y  = offset_ctr_y * bb_h + bb_ctr_y
    ref_w   = exp(offset_w) * bb_w
    ref_h   = exp(offset_h) * bb_h


(2. ) YoLo 3 [8]
The boxes regression outputs the `offsets`  
`offsets = [offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h]` computed as:

    ref_ctr_x  = sigmoid(offsets_ctr_x) + ctr_x
    ref_ctr_y  = sigmoid(offsets_ctr_y) + ctr_y
    ref_w   = exp(offsets_w) *  bb_w
    ref_h   = exp(offsets_h) *  bb_h


(2. ) YoLo 4 [8]
The boxes regression outputs the `offsets`  
`offsets = [offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h]` computed as:

    ref_ctr_x  = sigmoid(offsets_ctr_x) * 1.1 - 0.05 + ctr_x
    ref_ctr_y  = sigmoid(offsets_ctr_y) * 1.1 - 0.05 + ctr_y
    ref_w   = exp(offsets_w) *  bb_w
    ref_h   = exp(offsets_h) *  bb_h
    
(2. ) Scaled-YoLo4 and YoLo5 [8]

The boxes regression outputs the `offsets`  
`offsets = [offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h]` computed as:

    ref_ctr_x  = sigmoid(offsets_ctr_x) * 2 - 0.5 + ctr_x
    ref_ctr_y  = sigmoid(offsets_ctr_y) * 2 - 0.5 + ctr_y
    ref_w   = (sigmoid(offsets_w) * 2)**2 * bb_w
    ref_h   = (sigmoid(offsets_h) * 2)**2 * bb_h

References:

[0]. (RCNN) Rich feature hierarchies for accurate object \
    detection and semantic segmentation.
[1]. Fast RCNN.
[2]. Faster RCNN. 2017.

[3].https://towardsdatascience.com/understanding-and-implementing\
    -faster-r-cnn-a-step-by-step-guide-11acfff216b0

[4]. YOLO9000: Better, Faster, Stronger.
[5]. YOLOv3: An Incremental Improvement

[7]. https://blog.csdn.net/zhicai_liu/article/details/113631706
[8]. https://github.com/ultralytics/yolov5/issues/4373
"""

import math
from typing import Tuple, Optional

import torch


from torchvision.ops.boxes import box_convert


class BBoxBaseCoordinater:
    """The coordinater for offsets of bounding boxes.

    Args:
        coord_weights: A List containing weights used to
         scale coordinates (ctr_x, ctr_y, w, h).
         Each weight corresponds to one coordinate.
         For instance, coord_weights[2]: ctr_y.

        scale_calmp: A float to clip the `offset_w` and `offset_h`
         to prevent sending too large values into torch.exp().
    """

    def __init__(
        self,
        coord_weights: Optional[Tuple[float, float, float, float]] = None,
        wh_scale_calmp: Optional[float] = None,
        protect_eps=1e-7,
    ):
        _default_coord_weights = [2.0, 2.0, 1.0, 1.0]
        _default_scale_calmp = math.log(100000.0 / 16)
        self.coord_weights = (
            coord_weights if coord_weights is not None else _default_coord_weights
        )
        self.wh_scale_calmp = (
            wh_scale_calmp if wh_scale_calmp is not None else _default_scale_calmp
        )

        # added to log function to avoid the 0 value
        self.save_log = lambda x: torch.log(x + protect_eps)

    def before_compute_offsets(
        self,
        ref_bboxes: torch.FloatTensor,
        prop_bboxes: torch.FloatTensor,
        coord_weights: Tuple[float, float, float, float],
    ) -> torch.FloatTensor:
        """Prepare the variables for offsets computation.

        :param ref_bboxes: A `torch.FloatTensor` containing bounding boxes
         of shape [N, 4],
         of format (x1, y1, x2, y2), i.e., xyxy of albumentations.

        :param prop_bboxes: A `torch.FloatTensor` containing bounding boxes
         of shape [N, 4],
         of format (x1, y1, x2, y2), i.e., xyxy of albumentations.

        :param coord_weights: A tuple containg the weights for coordinates.

        :return ref_bboxes: A `torch.FloatTensor` containing bounding boxes
         of shape [N, 4], with format (ctr_x, ctr_y, w, h), i.e., cxcywh.
        :return prop_bboxes: A `torch.FloatTensor` containing bounding boxes
         of shape [N, 4], with format (ctr_x, ctr_y, w, h), i.e., cxcywh.
        :return coord_weights: A coordinates weights `torch.FloatTensor`
         of shape [4].
        """
        dtype = ref_bboxes.dtype
        device = ref_bboxes.device

        coord_weights = torch.as_tensor(coord_weights, dtype=dtype, device=device)

        # convert the xyxy to cxcywh
        ref_bboxes = box_convert(ref_bboxes, in_fmt="xyxy", out_fmt="cxcywh")
        prop_bboxes = box_convert(prop_bboxes, in_fmt="xyxy", out_fmt="cxcywh")

        return ref_bboxes, prop_bboxes, coord_weights

    def compute_offsets(
        self,
        ref_bboxes: torch.FloatTensor,
        prop_bboxes: torch.FloatTensor,
        coord_weights: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute offsets

        :param ref_bboxes: See `return` of ```before_compute_offsets```.
        :param prop_bboxes: See `return` of ```before_compute_offsets```.
        :param coord_weights: See `return` of ```before_compute_offsets```.

        :return A `torch.FloatTensor` containing the computed offsets for each
         coordinates,
         of shape same as `ref_bboxes`.
         of format, (offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h).
        """
        raise NotImplementedError("compute_offset() should be implemented.")

    def after_compute_offsets(self, offsets: torch.FloatTensor) -> torch.FloatTensor:
        """Perform operations after computing the offset.

        :param offsets: See `return` of ```compute_offsets```.

        :return offsets: A `torch.FloatTensor` containing the post-processed
         offsets for bounding boxes.
         of shape same as the input `offsets`.
         of format, (offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h).
        """

        return offsets

    def obtain_offsets(
        self, reference_bboxes: torch.FloatTensor, proposal_bboxes: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute offsets `proposal_bboxes` related to `reference_bboxes`.

        :param reference_bboxes: A `torch.FloatTensor`containing the
          `reference` bounding boxes for one batch of samples
         of shape, [bs, N, 4]

        :param proposal_bboxes:  A `torch.FloatTensor`containing
          the `proposal` bounding boxes for one batch of samples
         of shape, [bs, N, 4]

         where the `reference` generally means the groundtruth.
         The coordinate of bboxes should be (x1, y1, x2, y2), i.e.
          (x1, y1, x2, y2) of format albumentations.

        :return A `torch.FloatTensor` with shape same as `proposal_bboxes`
        """
        # prepare items for offsets computation
        reference_bboxes, proposal_bboxes, coord_weights = self.before_compute_offsets(
            ref_bboxes=reference_bboxes,
            prop_bboxes=proposal_bboxes,
            coord_weights=self.coord_weights,
        )

        offsets = self.compute_offsets(
            ref_bboxes=reference_bboxes,
            prop_bboxes=proposal_bboxes,
            coord_weights=coord_weights,
        )

        offsets = self.after_compute_offsets(offsets)

        return offsets

    def before_amendment(
        self,
        bboxes: torch.FloatTensor,
        offsets: torch.FloatTensor,
        coord_weights: Tuple[float, float, float, float],
    ) -> Tuple[
        torch.FloatTensor,
        Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
    ]:
        """Pre-process items before amend bounding boxes
        based on offsets.

        :param bboxes: A `torch.FloatTensor` of bounding boxes
         of shape (N, 4),
         of format (x1, y1, x2, y2), i.e. xyxy of albumentations.

        :param offsets: A `torch.FloatTensor` of coordinates' offsets
         of shape (N, k*4).
         of format (offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h).

        :return bboxes: A `torch.FloatTensor` containing processed `bboxes`
         of shape same as the input `bboxes`.
         of format (ctr_x, ctr_y, width, height), i.e., cxcywh

        :return deltas: A `Tuple` containing processed `offsets`
         of shape same as the input `offsets`.
        """
        bboxes = box_convert(bboxes, in_fmt="xyxy", out_fmt="cxcywh")
        bboxes = bboxes.to(offsets.dtype)

        # execute the weights for each coord of offsets
        delta_dx = offsets[..., 0::4] / coord_weights[0]
        delta_dy = offsets[..., 1::4] / coord_weights[1]
        delta_dw = offsets[..., 2::4] / coord_weights[2]
        delta_dh = offsets[..., 3::4] / coord_weights[3]

        # prevent sending too large values into torch.exp()
        delta_dx = torch.clamp(delta_dx, max=self.wh_scale_calmp)
        delta_dy = torch.clamp(delta_dy, max=self.wh_scale_calmp)

        deltas = (delta_dx, delta_dy, delta_dw, delta_dh)
        return bboxes, deltas

    def amend_via_deltas(
        self,
        bboxes: torch.FloatTensor,
        deltas: Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
        ],
    ) -> torch.FloatTensor:
        """Amend the bounding boxes based on the offsets.

        :param bboxes: See the ``return`` of ```before_amendment```
        :param deltas: See the ``return`` of ```before_amendment```

        :return amd_bboxes: A `torch.FloatTensor`
         of shape (N, 4 * k),
         of format (ctr_x, ctr_y, w, h), i.e., cxcywh.
         where 4 should be the amend coordinates, i.e., (ctr_x, ctr_y, w, h).
        """
        raise NotImplementedError("compute_offsets() should be implemeted!")

    def after_amendment(self, amd_bboxes: torch.FloatTensor) -> torch.FloatTensor:
        """Post-processing the amend bounding boxes.

        :param amd_bboxes: A `torch.FloatTensor` containing the amend coordinates
         of shape see ``return`` of the function ```amend_via_deltas```.
         of format see ``return`` of the function ```amend_via_deltas```.
         where the meaning of `k` is described in the function ```apply_offsets```.

        :return amd_boxes: A `torch.FloatTensor`
         of shape (N, 4 * k),
         of format (x1, y1, x2, y2), i.e., xyxy of albumentations.
        """

        amd_bboxes[:, 0::4] = amd_bboxes[:, 0::4] - 0.5 * amd_bboxes[:, 2::4]
        amd_bboxes[:, 1::4] = amd_bboxes[:, 1::4] - 0.5 * amd_bboxes[:, 3::4]
        amd_bboxes[:, 2::4] = amd_bboxes[:, 0::4] + 0.5 * amd_bboxes[:, 2::4]
        amd_bboxes[:, 3::4] = amd_bboxes[:, 1::4] + 0.5 * amd_bboxes[:, 3::4]

        return amd_bboxes

    def apply_offsets(
        self,
        bboxes: torch.FloatTensor,
        offsets: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Apply the `offsets` to `bboxes`.

        This is the one proposed by the Faster RCNN [1] as descripted in the
        above (1.).

        :param bboxes: A `torch.FloatTensor` holding bounding boxes,
         of shape [N, 4].
         of format (x1, y1, x2, y2), i.e., xyxy of albumentations

        :param offsets: A `torch.FloatTensor` denoting `offsets`
         of shape [N, k * 4].
         of format, (offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h).
         where 4 is (offsets_ctr_x, offsets_ctr_y, offsets_w, offsets_h).
         where k >= 1. offsets[i] represents k potentially
         different query/class-specific box transformations for the
         single box boxes[i].

        :return A `torch.FloatTensor` with shape same as `offsets`.
        """

        bboxes, deltas = self.before_amendment(
            bboxes=bboxes, offsets=offsets, coord_weights=self.coord_weights
        )

        amend_bboxes = self.amend_via_deltas(bboxes, deltas)

        amend_boxes = self.after_amendment(amend_bboxes)

        return amend_boxes


class BBoxFasterRCNNCoordinater(BBoxBaseCoordinater):
    """The coordinater for offsets of bounding boxes."""

    def compute_offsets(self, ref_bboxes, prop_bboxes, coord_weights):
        offsets = torch.zeros_like(ref_bboxes)

        wx, wy, ww, wh = (
            coord_weights[0],
            coord_weights[1],
            coord_weights[2],
            coord_weights[3],
        )

        offsets[..., 0::4] = (
            wx
            * (ref_bboxes[..., 0::4] - prop_bboxes[..., 0::4])
            / prop_bboxes[..., 2::4]
        )
        offsets[..., 1::4] = (
            wy
            * (ref_bboxes[..., 1::4] - prop_bboxes[..., 1::4])
            / prop_bboxes[..., 3::4]
        )

        offsets[..., 2::4] = ww * self.save_log(
            ref_bboxes[..., 2::4] / prop_bboxes[..., 2::4]
        )

        offsets[..., 3::4] = wh * self.save_log(
            ref_bboxes[..., 3::4] / prop_bboxes[..., 3::4]
        )

        return offsets

    def amend_via_deltas(self, bboxes, deltas):
        """Amend the bounding boxes based on the offsets."""
        (dx, dy, dw, dh) = deltas

        N, k = dx.shape
        amend_bboxes = torch.zeros((N, 4 * k), dtype=dx.dtype, device=bboxes.device)
        amend_bboxes[..., 0::4] = dx * bboxes[..., 2::4] + bboxes[..., 0::4]
        amend_bboxes[..., 1::4] = dy * bboxes[..., 3::4] + bboxes[..., 1::4]
        amend_bboxes[..., 2::4] = torch.exp(dw) * bboxes[..., 2::4]
        amend_bboxes[..., 3::4] = torch.exp(dh) * bboxes[..., 3::4]

        return amend_bboxes


class BBoxYoLoCoordinater(BBoxBaseCoordinater):
    """YoLo1's bounding box coordinater."""

    def compute_offsets(self, ref_bboxes, prop_bboxes, coord_weights):
        offsets = torch.zeros_like(ref_bboxes)
        wx, wy, ww, wh = (
            coord_weights[0],
            coord_weights[1],
            coord_weights[2],
            coord_weights[3],
        )

        offsets[..., 0::4] = wx * torch.logit(
            ref_bboxes[..., 0::4] - prop_bboxes[..., 0::4]
        )
        offsets[..., 1::4] = wy * torch.logit(
            ref_bboxes[..., 1::4] - prop_bboxes[..., 1::4]
        )
        offsets[..., 2::4] = ww * self.save_log(
            ref_bboxes[..., 2::4] / prop_bboxes[..., 2::4]
        )
        offsets[..., 3::4] = wh * self.save_log(
            ref_bboxes[..., 3::4] / prop_bboxes[..., 3::4]
        )

        return offsets

    def amend_via_deltas(self, bboxes, deltas):
        """Amend the bounding boxes based on the offsets."""
        (dx, dy, dw, dh) = deltas

        N, k = dx.shape
        amend_bboxes = torch.zeros((N, 4 * k), dtype=dx.dtype, device=bboxes.device)

        amend_bboxes[..., 0::4] = torch.sigmoid(dx) + bboxes[..., 0::4]
        amend_bboxes[..., 2::4] = torch.sigmoid(dy) + bboxes[..., 1::4]
        amend_bboxes[..., 3::4] = torch.exp(dw) * bboxes[..., 2::4]
        amend_bboxes[..., 3::4] = torch.exp(dh) * bboxes[..., 3::4]

        return amend_bboxes


class BBoxYoLo4Coordinater(BBoxBaseCoordinater):
    """The coordinater for offsets of bounding boxes."""

    def compute_offsets(self, ref_bboxes, prop_bboxes, coord_weights):
        offsets = torch.zeros_like(ref_bboxes)
        wx, wy, ww, wh = (
            coord_weights[0],
            coord_weights[1],
            coord_weights[2],
            coord_weights[3],
        )

        offsets[..., 0::4] = wx * torch.logit(
            (ref_bboxes[..., 0::4] - prop_bboxes[..., 0::4] + 0.5) / 1.1
        )
        offsets[..., 2::4] = wy * torch.logit(
            (ref_bboxes[..., 1::4] - prop_bboxes[..., 1::4] + 0.5) / 1.1
        )
        offsets[..., 3::4] = ww * self.save_log(
            ref_bboxes[..., 2::4] / prop_bboxes[..., 2::4]
        )
        offsets[..., 3::4] = wh * self.save_log(
            ref_bboxes[..., 3::4] / prop_bboxes[..., 3::4]
        )

        return offsets

    def amend_via_deltas(self, bboxes, deltas):
        """Amend the bounding boxes based on the deltas."""
        (dx, dy, dw, dh) = deltas

        N, k = dx.shape
        amend_bboxes = torch.zeros((N, 4 * k), dtype=dx.dtype, device=bboxes.device)

        amend_bboxes[..., 0::4] = torch.sigmoid(dx) * 1.1 - 0.5 + bboxes[..., 0::4]
        amend_bboxes[..., 1::4] = torch.sigmoid(dy) * 1.1 - 0.5 + bboxes[..., 1::4]
        amend_bboxes[..., 2::4] = torch.exp(dw) * bboxes[..., 2::4]
        amend_bboxes[..., 3::4] = torch.exp(dh) * bboxes[..., 3::4]

        return amend_bboxes


class BBoxYoLo5Coordinater(BBoxBaseCoordinater):
    """The coordinater for offsets of bounding boxes."""

    def compute_offsets(self, ref_bboxes, prop_bboxes, coord_weights):
        offsets = torch.zeros_like(ref_bboxes)
        wx, wy, ww, wh = (
            coord_weights[0],
            coord_weights[1],
            coord_weights[2],
            coord_weights[3],
        )

        offsets[..., 0::4] = wx * torch.logit(
            (ref_bboxes[..., 0::4] - prop_bboxes[..., 0::4] + 0.5) / 1.1
        )
        offsets[..., 1::4] = wy * torch.logit(
            (ref_bboxes[..., 1::4] - prop_bboxes[..., 1::4] + 0.5) / 1.1
        )
        offsets[..., 2::4] = ww * self.save_log(
            ref_bboxes[..., 2::4] / prop_bboxes[..., 2::4]
        )
        offsets[..., 3::4] = wh * self.save_log(
            ref_bboxes[..., 3::4] / prop_bboxes[..., 3::4]
        )

        return offsets

    def amend_via_deltas(self, bboxes, deltas):
        """Amend the bounding boxes based on the offsets."""
        (dx, dy, dw, dh) = deltas

        N, k = dx.shape
        amend_bboxes = torch.zeros((N, 4 * k), dtype=dx.dtype, device=bboxes.device)

        amend_bboxes[..., 0::4] = torch.sigmoid(dx) * 2 - 0.5 + bboxes[..., 0::4]
        amend_bboxes[..., 1::4] = torch.sigmoid(dy) * 2 - 0.5 + bboxes[..., 1::4]
        amend_bboxes[..., 2::4] = (
            torch.square(torch.sigmoid(dw * 2)) * bboxes[..., 2::4]
        )
        amend_bboxes[..., 3::4] = (
            torch.square(torch.sigmoid(dh * 2)) * bboxes[..., 3::4]
        )

        return amend_bboxes
