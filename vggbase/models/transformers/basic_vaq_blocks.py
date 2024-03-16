"""
The implementation of the basic blocks for the visual grounding.


"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from timm.models.layers import DropPath

from vggbase.models.attentions import BiDirectionalCrossAttention
from vggbase.models.general_mlps import build_mlp_from_config


# pylint: disable=invalid-name
class BasicVaQBlock(nn.Module):
    """The basic  Block.
        The block module does not change the #`n_channels` of the input.
        It is built by declearing the input `n_channels`, i.e., `n_channels`.

        It only learns the self or cross relations by the attention mechanism.

        The channels of all q, k, v are `n_channels`.
        Thus, with n_heads, each channels of head `head_n_channels` is
            n_channels // n_heads.


        For each block, the strucutre is:
              x -> norm1 -> attention_module ->  +  -> x -> norm2 -> mlp -> + -> x
              ↓                                  ↑     ↓                    ↑
              ↓__________________________________↑     ↓____________________↑

    Parameters:
        n_channels (int): Number of input channels.
        n_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden features to embedding features.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_n_channels ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "grelu",
        norm_layer: torch.nn.Module = nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(n_channels)
        self.norm2 = norm_layer(n_channels)

        self.attn_module = None

        self.init_attention_module(
            n_channels, n_heads, qkv_bias, qk_scale, attn_drop, drop, **kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_features = int(n_channels * mlp_ratio)
        self.mlp = build_mlp_from_config(
            mlp_configs=dict(
                type="FullyConnectedHead",
                output_n_features=n_channels,
                input_n_features=n_channels,
                hidden_layers_n_features=[mlp_hidden_features],
                batch_norms=[None, None],
                activations=[act_layer, None],
                dropout_ratios=[drop, drop],
            )
        )

        # H, W here denote the input
        #  number of patches size along
        #  the heigth and width
        #  holding the #patches along height and width
        #   before padding!!
        self.Ph = None
        self.Pw = None

    def init_attention_module(
        self,
        n_channels: int,
        n_heads: int,
        qkv_bias: float,
        qk_scale: float,
        attn_drop: float,
        drop: float,
        **kwargs
    ):
        """Customize the attention module of the block."""

        self.attn_module = BiDirectionalCrossAttention(
            xa_n_features=n_channels,
            xb_n_features=n_channels,
            n_heads=n_heads,
            mapped_a2b_qk_n_features=n_channels,
            mapped_b2a_qk_n_features=n_channels,
            mapped_xa_v_n_features=n_channels,
            mapped_xb_v_n_features=n_channels,
            xa_proj_n_features=n_channels,
            xb_proj_n_features=n_channels,
            qkv_bias=qkv_bias,
            proj_bias=qkv_bias,
            q_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def init_block_preprocess(self):
        """Customize the layer postprocessor that process the output of the blocks."""
        pass

    def init_block_postprocess(self):
        """Customize the layer postprocessor that process the output of the blocks."""
        pass

    def init_vq_mask(
        self, vq_mask: torch.tensor, patch_resolution: torch.tensor, **kwargs
    ):
        """Customize the visual and text mask based on the personal requirement.

        By default, the visual mask and query mask are combined as a whole.

        Args:
            visual_mask (torch.tensor): the visual mask with shape
                                    B, layer_Ph, layer_Pw
            query_mask (torch.tensor): the query mask with shape
                                    B, N

        Output:
            vq_mask (torch.tensor): a combined visual and query mask
                with shape, B, 1, 1, Ph * Pw + number_of_queries

        """
        pass

    def forward(
        self,
        tvq: torch.tensor,
        vq_mask: torch.tensor,
        patch_resolution: Tuple[int],
        **kwargs
    ):
        """Forward the input visual and query input.

        Args:
            tvq (torch.tensor): input visual and text feature,
                            Tensor with size (B, Ph*Pw+number_of_text_queries, C).

            vq_mask (torch.tensor): input masks of the visual and query
                        Shape (B, 1, 1, Ph * Pw + number_of_queries)

            patch_resolution (tuple[int]): the number information for patches along
                        the width and height. [Ph, Pw]


        """
        self.init_block_preprocess()

        # get the shape of the actual input
        # where L = Ph * Pw + num_queries
        B, L, layer_n_channels = tvq.shape
        # get the Ph and Pw
        [Ph, Pw] = patch_resolution

        # forward the first norm
        tvq = self.norm1(tvq)

        # splited the binded x containing patches and query tokens
        # tvq, B, L, layer_n_channels
        # tv, B, Ph, Pw, layer_n_channels
        # tq, B, num_queries, layer_n_channels
        tv, tq = tvq[:, : Ph * Pw, :], tvq[:, Ph * Pw :, :]
        tv = tv.view(B, Ph, Pw, layer_n_channels)

        tv_mask, tq_mask = vq_mask[:, :, :, : Ph * Pw], vq_mask[:, :, :, Ph * Pw :]
        self.init_vq_mask(vq_mask, patch_resolution)

        # forward the attentio module to boost tv and tq
        # tv, B, Ph, Pw, layer_n_channels
        # tq, B, num_queries, layer_n_channels
        tv, tq = self.attn_module(tv, tq, tv_mask, tq_mask**kwargs)

        self.init_block_postprocess()

        tv = tv.view(B, Ph * Pw, layer_n_channels)

        # the first residual connection
        tvq = tvq + self.drop_path(torch.cat([tv, tq], dim=1))

        # the seoncd residul connection for FFN
        tvq = tvq + self.drop_path(self.mlp(self.norm2(tvq)))

        return tvq

    def freeze_block(self):
        """Freeze the block."""
        for param in self.attn_module.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
