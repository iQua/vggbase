"""
Useful functions for tensor operations

In the nested tensor here, the mask should
always be `bool` in which True means masked
while False means unmasked.
    
"""

from typing import Optional, List, Union, Tuple
import torch

from torch import Tensor
import torchvision


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class BaseNestedTensor:
    """Create nested tensor from a series of 3D tensor with various shapes.

    :param tensors: A `torch.FloatTensor` tensor containing the data,
     of shape [B, ...].
     For 3D tensor, such as image, the shape: [B, C, H, W]
     For 2D tensor, such as text, the shape: [B, P, L]
    :param mask: A `torch.BoolTensor` tensor containing padding indicators,
     of shape [B, ...].
     For 3D tensor, such as image, the shape: [B, H, W]
     For 2D tensor, such as text, the shape: [B, L]
    :param unmaksed_hws: A `torch.IntTensor` tensor containing
     sizes of a series of tensors who are used to generate the ``tensor`.`
     i.e, before being padding.
     of shape [B, ...]
     For 3D tensor, such as image, the shape: [B, 2], where 2 for h, w.
     For 2D tensor, such as text, the shape: [B, 1], where 1 for length.

     Note, the mask here are a torch.BoolTensor containing True, False.
     True means masked while False means unmasked
    """

    def __init__(
        self,
        tensors: torch.FloatTensor,
        mask: torch.BoolTensor,
        unmaksed_hws: Optional[List[Tuple[int, int]]] = None,
    ):
        self.tensors = tensors
        self.mask = mask
        # obtain the sizes for a series of tensors who
        # are used to generated the tensors
        # if not provided, obtain from the masks
        self.unmaksed_hws = (
            unmaksed_hws if unmaksed_hws is not None else self.unmask_sizes()
        )

    def unmask_sizes(self) -> List[Tuple[int, int]]:
        """the size of region in the mask that is not masked."""
        unmasked = ~self.mask

        batch_size = self.mask.shape[0]
        sizes = []

        for batch_idx in range(batch_size):
            sample_unmasked = unmasked[batch_idx]
            mask_n_dim = sample_unmasked.ndim
            sample_unmasked_sizes = []
            for dim_idx in range(mask_n_dim):
                dim_unmasked = sample_unmasked.sum(dim=dim_idx)
                sample_unmasked_sizes.append(torch.max(dim_unmasked).item())

            sizes.append(tuple(sample_unmasked_sizes))

        return sizes

    def to(self, device):
        """Convert the tensor to the desired device to
        create the nested tensor."""
        self.tensors = self.tensors.to(device)
        self.mask = self.mask.to(device)

    @property
    def device(self):
        """Get the device of the tensor."""
        return self.tensors.device

    @property
    def shape(self):
        """Get the shape of the nested tensor."""
        return self.tensors.shape

    @property
    def unnested_shapes(self):
        """Get shapes of unnested tensors."""

        return self.unmaksed_hws

    def decompose_via_mask(self):
        """
        Decompose the nested tensor by removing the masked part, thus
        output a tensor list.

        Thus, the padded nested tensor will be converted to the unmaksed size.
        """
        batch_size = len(self.unmaksed_hws)

        tensors_list = []
        for batch_idx in range(batch_size):
            unpadded_tensor_size = self.unmaksed_hws[batch_idx]
            padded_tensor = self.tensors[batch_idx]

            mask = ~self.mask[batch_idx].unsqueeze(0).expand_as(padded_tensor)
            unpadded_tensor = padded_tensor[mask].view(-1, *unpadded_tensor_size)
            tensors_list.append(unpadded_tensor)

        return tensors_list

    def __repr__(self):
        return str(self.tensors)


class DynamicMaskNestedTensor(BaseNestedTensor):
    """A special nested tensor in which the mask size is
    different in each part of one dimension.

    This generally appears in text-related nested tensor case,
    in which for different phrases of one sentence, the number of
    padded words is different.

    :param tensor: A `torch.FloatTensor` tensor containing the data,
     of shape [B, P, L].
    :param mask: A `torch.BoolTensor` tensor containing padding indicators,
     of shape [B, P, L].
    :param mask_p: A `torch.BoolTensor` tensor containing padding indicators
     for the second dimension,
     of shape [B, P].
    """

    def __init__(
        self,
        *base_config,
        mask_p: Optional[torch.BoolTensor] = None,
    ):
        super().__init__(*base_config)

        self.mask_p = mask_p

    def unmask_sizes(self) -> List[int]:
        """the size of region in the mask that is not masked."""
        unmasked = ~self.mask

        batch_size = self.mask.shape[0]

        # Obtain the sizes
        # [B, P]
        return [
            unmasked[batch_idx].sum(dim=-1).tolist() for batch_idx in range(batch_size)
        ]

    def to(self, device):
        """Convert the tensor to the desired device to
        create the nested tensor."""
        super().to(device)
        self.mask_p = self.mask_p.to(device)


def nested_3d_tensor_from_list(
    tensor_list: List[Tensor], external_masks: List[Tensor] = None
) -> Union[Tensor, Tensor]:
    """Create a 3d nested Tensor from a list of 3D tensors with variable sizes.

    :param tensor_list: A `List` containing a series of 3D tensors
     with variable sizes, of each shape [C, H_i, W_i]
    :param external_masks: A `List` containing the external maskes for
     these 3D tensors, of each shape [H, W].

     where for masking, 1 means masking while 0 means non-masking.
    """

    if torchvision._is_tracing():
        # nested_tensor_from_tensor_list() does not export well to ONNX
        # call _onnx_nested_tensor_from_tensor_list() instead
        return _onnx_nested_tensor_from_tensor_list(tensor_list)

    # TODO make it support different-sized images
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size

    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], : img.shape[2]] = False

    if external_masks is not None:
        for m, e_m in zip(mask, external_masks):
            m[: e_m.shape[0], : e_m.shape[1]].copy_(e_m)
    return tensor, mask


def nested_2d_tensor_from_list(
    tensor_list: List[Tensor], external_masks: List[Tensor] = None
) -> Union[Tensor, Tensor]:
    """Create a 2d nested Tensor from a list of 2D tensors with variable sizes.

    :param tensor_list: A `List` containing a series of 2D tensors
     with variable sizes, of each shape [N, L_i]
    :param external_masks: A `List` containing the external maskes for
     these 2D tensors, of each shape [L].

     where for masking, 1 means masking while 0 means non-masking.
    """

    if torchvision._is_tracing():
        # nested_tensor_from_tensor_list() does not export well to ONNX
        # call _onnx_nested_tensor_from_tensor_list() instead
        return _onnx_nested_tensor_from_tensor_list(tensor_list)

    max_size = _max_by_axis([list(tensor.shape) for tensor in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    b, p, l = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, p, l), dtype=torch.bool, device=device)

    # padding the external mask
    for token, pad_token, m in zip(tensor_list, tensor, mask):
        pad_token[: token.shape[0], : token.shape[1]].copy_(token)
        m[: token.shape[1]] = False
    if external_masks is not None:
        for m, e_m in zip(mask, external_masks):
            m[: e_m.shape[0]].copy_(e_m)

    return tensor, mask


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: List[Tensor],
) -> Union[Tensor, Tensor]:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return tensor, mask
