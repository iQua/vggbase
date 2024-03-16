"""
The implementation of the basic layer for the visual grounding.

"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vaq_blocks import BasicVaQBlock


# pylint: disable=invalid-name
class BasicVaQLayer(nn.Module):
    """A basic layer for one stage of the VaQ.
        The layer contains multiple blocks.

    Parameters:
        n_channels (int): Number of feature channels
        depth (int): Depths of this stage.
        n_heads (int): Number of attention head.
        mlp_ratio (float): Ratio of mlp hidden channels to embedding channels. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_channels ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
                    This is the general dropout rate for the mlp and some outputs.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        n_channels: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: torch.nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()

        self.depth = depth
        self.n_channels = n_channels
        self.use_checkpoint = use_checkpoint

        self.n_queries_token = None

        self.blocks = nn.ModuleList()

        # build blocks
        self.init_blocks(
            depth,
            n_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            norm_layer,
        )

    def freeze_layer(self):
        """Freeze layer containing multiple blocks."""
        for block in self.blocks:
            block.eval()
            block.freeze_block()

    def prepare_layer_structure(self, structure_config):
        """Prepare the structure settings for the layer."""
        pass

    def init_blocks(
        self,
        depth,
        n_heads,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        norm_layer,
    ):
        """Customize the blocks for this layer."""
        for i in range(depth):
            self.blocks.append(
                BasicVaQBlock(
                    n_channels=self.n_channels,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
            )

    def init_layer_preprocess(self):
        """Customize the layer postprocessor that process variables before forwarding."""
        pass

    def init_layer_intermediateprocess(self):
        """Customize the layer intermediateprocessor that process variables during forwarding."""
        pass

    def init_layer_postprocess(self):
        """Customize the layer postprocessor that process variables after forwarding."""
        pass

    def init_prepare_attention_op(self):
        """Customize the preparation for operating the attention mechanism."""
        pass

    def init_vq_mask(self, rgb_mask, tquery_mask, **kwargs):
        """Customize the visual and text mask based on the personal requirement.

        By default, the visual mask and query mask are combined as a whole.

        Args:
            rgb_mask (torch.tensor): the visual mask with shape
                                    B, layer_Ph, layer_Pw
            tquery_mask (torch.tensor): the query mask with shape
                                    B, N

        Output:
            vq_mask (torch.tensor): a combined visual and query mask
                with shape, B, 1, 1, Ph * Pw + number_of_queries

        """
        [B, layer_Ph, layer_Pw] = rgb_mask.shape

        # convert to
        # rgb_mask, [B, 1, 1 layer_Ph, layer_Pw]
        rgb_mask = rgb_mask.view(B, layer_Ph * layer_Pw).unsqueeze(1).unsqueeze(2)
        # convert to
        # tquery_mask, [B, 1, 1, N]
        tquery_mask = tquery_mask[:, None, None, :]

        # combine the two to obtain
        # vq_mask, [B, 1, 1 layer_Ph * layer_Pw + N]
        vq_mask = torch.cat((rgb_mask, tquery_mask), dim=-1)

        return vq_mask

    def convert_rgb_mask(self, rgb_mask, patch_resolution):
        """Convert the input visual mask to be consistent with the
        visual input."""
        # convert the rgb mask
        # from rgb_mask, [B, ori_Ph, ori_Pw] containing elements: True, False
        # to rgb_mask, [B, Ph, Pw] containing elements: True, False
        [Ph, Pw] = patch_resolution
        rgb_mask = F.interpolate(rgb_mask[None].float(), size=(Ph, Pw)).to(torch.bool)[
            0
        ]
        return rgb_mask

    def set_n_queries_token(self, n_queries_token):
        """Set the number of query tokens for this layer"""
        self.n_queries_token = n_queries_token
        for block in self.blocks:
            block.n_queries_token = n_queries_token
            block.attn_module.n_queries_token = n_queries_token

    def forward(self, tvq, patch_resolution, rgb_mask, tquery_mask, **kwargs):
        """The forward pass of the layer.

        Args:
            tvq (torch.tensor): Input visual and text feature,
                            Tensor with size (B, Ph*Pw+number_of_text_queries, layer_n_channels).
            patch_resolution (list[int]): Spatial resolution of the patches. It contains
                       Ph, Pw, where  Ph, Pw are the #patches along height and width

            rgb_mask: padding mask for inputs, B, layer_Ph, layer_Pw
                        where the layer_Ph and layer_Pw are the corresponding patches
                        along height and width in this layer, respectively.
                        It contains bool values, [True: masked, 1: unmasked]
            tquery_mask (torch tensor): input text padding mask with shape B,N
                    where N is the max number of phrases in one batch
                        or N is the max number of words in queries of one batch.
                    It contains bool values, [True: masked, 1: unmasked]

        """

        # process the variables before pssing the blocks
        self.init_layer_preprocess()

        vq_mask = self.init_vq_mask(rgb_mask, tquery_mask)

        for n_blk, layer_block in enumerate(self.blocks):
            tvq = layer_block(tvq, vq_mask, patch_resolution)

        # process the outputs of the final block
        self.init_layer_postprocess()

        return {"patch_resolution": patch_resolution, "output_tvq": tvq}
