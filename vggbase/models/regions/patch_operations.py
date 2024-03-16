"""
The operations on patches.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

# pylint: disable=invalid-name
class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        channels (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.channels = channels
        self.reduction = nn.Linear(4 * channels, 2 * channels, bias=False)
        self.norm = norm_layer(4 * channels)

    def forward(self, xs):
        """
        xs: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = xs.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"xs size ({H}*{W}) are not even."

        xs = xs.view(B, H, W, C)

        xs0 = xs[:, 0::2, 0::2, :]  # B H/2 W/2 C
        xs1 = xs[:, 1::2, 0::2, :]  # B H/2 W/2 C
        xs2 = xs[:, 0::2, 1::2, :]  # B H/2 W/2 C
        xs3 = xs[:, 1::2, 1::2, :]  # B H/2 W/2 C
        xs = torch.cat([xs0, xs1, xs2, xs3], -1)  # B H/2 W/2 4*C
        xs = xs.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        xs = self.norm(xs)
        xs = self.reduction(xs)

        return xs

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, channels={self.channels}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.channels
        flops += (H // 2) * (W // 2) * 4 * self.channels * 2 * self.channels
        return flops


# pylint: disable=invalid-name
class PartialPatchMerging(nn.Module):
    """Partial Patch Merging Layer

    Parameters:
        channels (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, channels, norm_layer=nn.LayerNorm, expand=True):
        super().__init__()
        self.channels = channels

        # if expand is True, the channel size will be expanded, otherwise, return 256 size of channel
        expand_channels = 2 * channels if expand else 256
        self.reduction = nn.Linear(4 * channels, expand_channels, bias=False)
        self.norm = norm_layer(4 * channels)

        # added for detection token [please ignore, not used for training]
        # not implemented yet.
        self.expansion = nn.Linear(channels, expand_channels, bias=False)
        self.norm2 = norm_layer(channels)

    def forward(self, xs, H, W):
        """Forward function.

        Parameters:
            xs: Input feature, tensor size (B, H*W+det_token_num, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.

        Returns:
            xs: merged [PATCH, DET] tokens;
            only [PATCH] tokens are reduced in spatial dim, while [DET] tokens is fix-scale
        """

        B, _, C = xs.shape

        xs, det = xs[:, : H * W, :], xs[:, H * W :, :]
        xs = xs.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            xs = F.pad(xs, (0, 0, 0, W % 2, 0, H % 2))

        xs0 = xs[:, 0::2, 0::2, :]  # B H/2 W/2 C
        xs1 = xs[:, 1::2, 0::2, :]  # B H/2 W/2 C
        xs2 = xs[:, 0::2, 1::2, :]  # B H/2 W/2 C
        xs3 = xs[:, 1::2, 1::2, :]  # B H/2 W/2 C
        xs = torch.cat([xs0, xs1, xs2, xs3], -1)  # B H/2 W/2 4*C
        xs = xs.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # simply repeating for DET tokens
        # before, B, self.det_token_num, C
        det = det.repeat(1, 1, 4)

        xs = torch.cat([xs, det], dim=1)
        xs = self.norm(xs)
        xs = self.reduction(xs)

        return xs


# pylint: disable=invalid-name
class VisualTextPartialPatchMerging(nn.Module):
    """Partial Patch Merging Layer for the visual patches and the text windows

        the visual patches merging mechanism follows that in the swin transformer
        the text windows mergeing mechanism is to merge the neighbor two windows to
            make the number of windows along height and width decreases two times.


    Parameters:
        channels (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, channels, norm_layer=nn.LayerNorm, expand=True):
        super().__init__()
        self.channels = channels

        # if expand is True, the channel size will be expanded, otherwise, return 256 size of channel
        expand_channels = 2 * channels if expand else 256
        self.reduction = nn.Linear(4 * channels, expand_channels, bias=False)
        self.norm = norm_layer(4 * channels)

        # added for detection token [please ignore, not used for training]
        # not implemented yet.
        self.expansion = nn.Linear(channels, expand_channels, bias=False)
        self.norm2 = norm_layer(channels)

    def forward(self, xs, H, W, h_num_windows, w_num_windows):
        """Forward function.

        Parameters:
            xs: Input feature, tensor size (B, H*W+det_token_num, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.
            h_num_windows: Number of windows along the height
            w_num_windows: Number of windows along the width
        Returns:
            xs: merged [PATCH, DET] tokens;
            only [PATCH] tokens are reduced in spatial dim, while [DET] tokens is fix-scale
        """

        B, _, C = xs.shape
        num_queries = self.num_queries_token
        xs, det = xs[:, : H * W, :], xs[:, H * W :, :]
        num_per_sample_queries_windows = det.shape[1]
        num_per_sample_query_windows = num_per_sample_queries_windows // num_queries

        xs = xs.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            xs = F.pad(xs, (0, 0, 0, W % 2, 0, H % 2))

        xs0 = xs[:, 0::2, 0::2, :]  # B H/2 W/2 C
        xs1 = xs[:, 1::2, 0::2, :]  # B H/2 W/2 C
        xs2 = xs[:, 0::2, 1::2, :]  # B H/2 W/2 C
        xs3 = xs[:, 1::2, 1::2, :]  # B H/2 W/2 C
        xs = torch.cat([xs0, xs1, xs2, xs3], -1)  # B H/2 W/2 4*C
        xs = xs.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if num_per_sample_query_windows == 1:

            # simply repeating for DET tokens
            # before, B, num_queries, C
            # after, B, num_queries, C * 4
            det = det.repeat(1, 1, 4)
        else:

            det = det.view(B, h_num_windows * w_num_windows, num_queries, C)
            det = det.view(B, h_num_windows, w_num_windows, num_queries, C)

            query_pad_input = (h_num_windows % 2 == 1) or (w_num_windows % 2 == 1)
            if query_pad_input:
                det = F.pad(
                    det, (0, 0, 0, 0, 0, w_num_windows % 2, 0, h_num_windows % 2)
                )
                w_num_windows = w_num_windows + w_num_windows % 2
                h_num_windows = h_num_windows + h_num_windows % 2

            # # padding
            # pad_input = (h_num_windows % 2 == 1) or (w_num_windows % 2 == 1)
            # if pad_input:
            #     # padding det
            #     #  from: batch_size,
            #     det = F.pad(
            #         det,
            #         (0, 0, 0, 0, 0, w_num_windows % 2, 0, h_num_windows % 2))
            #
            # B h_num_windows/2 w_num_windows/2 num_queries C
            det0 = det[:, 0::2, 0::2, :, :]
            det1 = det[:, 1::2, 0::2, :, :]
            det2 = det[:, 0::2, 1::2, :, :]
            det3 = det[:, 1::2, 1::2, :, :]

            # B h_num_windows/2 w_num_windows/2 num_queries 4*C
            det = torch.cat([det0, det1, det2, det3], -1)

            # convert to B, h_num_windows/2 * w_num_windows/2 * num_queries, 4 * C
            det = det.view(
                B, (h_num_windows // 2) * (w_num_windows // 2), num_queries, 4 * C
            )
            det = det.view(B, -1, 4 * C)

        xs = torch.cat([xs, det], dim=1)
        xs = self.norm(xs)
        xs = self.reduction(xs)

        return xs


# pylint: disable=invalid-name
class AdaptivePatchMapper(nn.Module):
    """Image to Patch Mapping adaptively.

        no prior image size is assigned to compute the #patches and to obtain the embedding
        Otherwise, #patches will be computed adaptively based on the size of the input
        image in the forward process.

    Parameters:
        patch_size (int): Patch token size. Default: 4.
        in_n_channels (int): Number of input image channels. Default: 3.
        embed_n_channels (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, patch_size=4, in_n_channels=3, embed_n_channels=96, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_n_channels = in_n_channels
        self.embed_n_channels = embed_n_channels

        self.padding_size = [0, 0]
        self.patches_resolution = [0, 0]
        self.n_patches = None

        self.proj = nn.Conv2d(
            in_n_channels, embed_n_channels, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_n_channels)
        else:
            self.norm = None

    def forward(self, xs):
        """Forward function.

        Args:
            xs (torch.tensor): the input tensor with shape B, C, H, W

        Output:
            xs (torch.tensor): the output tensor with shape B, embed_n_channels, Ph, Pw
                where Ph is the number of patches along height.
                    Pw is the number of patches along width.

                For example, input shape: 2, 3, 888, 1336; patch size: 4
                    the Ww is 222, Ww is 334.
        """

        # padding to make an integer number of patches
        _, _, H, W = xs.size()

        if W % self.patch_size[1] != 0:
            self.padding_size[1] = self.patch_size[1] - W % self.patch_size[1]
            xs = F.pad(xs, (0, self.padding_size[1]))

        if H % self.patch_size[0] != 0:
            self.padding_size[0] = self.patch_size[0] - H % self.patch_size[0]
            xs = F.pad(xs, (0, 0, 0, self.padding_size[0]))

        self.patches_resolution[1] = (W + self.padding_size[1]) // self.patch_size[1]
        self.patches_resolution[0] = (H + self.padding_size[0]) // self.patch_size[0]

        self.n_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # utilize the convolution network to achieve the patch embedding
        # just set the kernel_size and stride to be the patch size
        # brefore xs: B, 3, H, W
        # after xs: B, embed_n_channels, Ph, Pw
        xs = self.proj(xs)

        if self.norm is not None:
            Ph, Pw = xs.size(2), xs.size(3)
            xs = xs.flatten(2).transpose(1, 2)
            xs = self.norm(xs)
            xs = xs.transpose(1, 2).view(-1, self.embed_n_channels, Ph, Pw)

        return xs

    def freeze_mapper(self):
        """Freeze the mapper."""
        for param in self.proj.parameters():
            param.requires_grad = False


# pylint: disable=invalid-name
class PatchMapper(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_n_channels (int): Number of input image channels. Default: 3.
        embed_n_channels (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_n_channels=3,
        embed_n_channels=96,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.n_patches = patches_resolution[0] * patches_resolution[1]

        self.in_n_channels = in_n_channels
        self.embed_n_channels = embed_n_channels

        self.proj = nn.Conv2d(
            in_n_channels, embed_n_channels, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_n_channels)
        else:
            self.norm = None

    def forward(self, xs):
        _, _, H, W = xs.shape
        # relax size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # proj the input xs
        # from B, 3, Ph, Pw
        # to B, embed_n_channels, Ph, Pw
        # then, if norm required
        # flat and transpose xs
        # to B, Ph*Pw, embed_n_channels for the norm operation
        # finally if convert xs back
        # to B, embed_n_channels, Ph, Pw
        xs = self.proj(xs).flatten(2).transpose(1, 2)
        if self.norm is not None:
            Ph, Pw = xs.size(2), xs.size(3)
            xs = self.norm(xs)
            xs = xs.transpose(1, 2).view(-1, self.embed_n_channels, Ph, Pw)
        return xs

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_n_channels
            * self.in_n_channels
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_n_channels
        return flops

    def freeze_mapper(self):
        """Freeze the mapper."""
        for param in self.proj.parameters():
            param.requires_grad = False
