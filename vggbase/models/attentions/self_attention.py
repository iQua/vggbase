"""
The implementation of general self attention operation.

"""

from typing import Optional

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """The self attention module."""

    def __init__(
        self,
        in_n_features,
        n_heads,
        mapped_qkv_n_features: Optional[int] = None,
        proj_n_features: Optional[int] = None,
        qkv_bias=True,
        q_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """General self attention mechanism

        :param in_n_features: Number of features of the input.
        :param n_heads: Number of attention heads.
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param q_scale: Whether to perform a scale for the query. Default is `head_features**-0.5`.
        :param attn_drop: Dropout probability for attention matrix values. Default is `0.0`
        :param proj_drop: Dropout probability for projection values. Default is `0.0`

        """

        super().__init__()
        self.in_n_features = in_n_features
        self.n_heads = n_heads
        self.mapped_qkv_n_features = mapped_qkv_n_features
        self.proj_n_features = proj_n_features

        self.set_default_params(mapped_qkv_n_features, proj_n_features)

        self.head_features = mapped_qkv_n_features // n_heads

        self.scale = q_scale or self.head_features**-0.5

        self.qkv = nn.Linear(
            in_features=in_n_features,
            out_features=3 * mapped_qkv_n_features,
            bias=qkv_bias,
        )

        self.attn_dropout = nn.Dropout(attn_drop)

        self.proj = nn.Linear(self.mapped_qkv_n_features, self.proj_n_features)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.position_encoder = None
        self.init_position_encoder()

        # the attention outputs of the visual part
        self.attention_scores = None
        self.attention_probs = None
        self.attention_probs_drop = None

    def set_default_params(self, mapped_qkv_n_features, proj_n_features):
        """Set the default parameters."""

        if mapped_qkv_n_features is None:
            self.mapped_qkv_n_features = self.in_n_features

        if proj_n_features is None:
            self.proj_n_features = self.mapped_qkv_n_features

    def init_position_encoder(self):
        """Customize the position encoder."""
        pass

    def perform_position_encoding(self):
        """perform the position encoding."""
        pass

    def init_attention_end(self):
        """Customize the postprocessing after obtaining the attention."""
        pass

    def forward(self, input_x, x_mask, **kwargs):
        """Forward function.

        Args:
            input_x (torch.tensor): the input feature with shape
                        [B, H, W, C]
                        or
                        [B, N, C]
            x_mask (torch.tensor): the input mask for visual and query
                        with shape
                        [B, 1, 1, H * W]
                        or
                        [B, 1, 1, N
                        1: masked, 0: unmasked
        """
        # 1. process the input tensor
        input_dim = input_x.dim()
        if input_dim == 4:
            B, H, W, C = input_x.shape
            L = H * W
            input_x = input_x.view(B, L, C)

        if input_dim == 3:
            B, N, C = input_x.shape
            L = N
        # 2. forward to obtain q, k, v
        # x_qkv, [B, L, 3 * self.n_heads * self.head_features]
        x_qkv = self.qkv(input_x)

        # transpose for extracting q, k, v,
        # convert to
        # x_qkv, [3, B, self.n_heads, L, self.head_features]
        x_qkv = x_qkv.view(B, L, 3, self.n_heads, self.head_features).permute(
            2, 0, 3, 1, 4
        )

        # 3. extract the q, k, v
        #   x_q, [B, self.n_heads, L, self.head_features]
        #   x_k, [B, self.n_heads, L, self.head_features]
        #   x_v, [B, self.n_heads, L, self.head_features]
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]

        # 4. self-attention
        # x_q, [B, self.n_heads, L, self.head_features]
        x_q = x_q * self.scale

        # 5. add position encoding when possible
        self.perform_position_encoding()

        # get attention scores
        # transposed x_k, [B, self.n_heads, self.head_in_n_features L]
        # attention_scores, [B, self.n_heads, L, L]
        attention_scores = x_q @ x_k.transpose(-2, -1)
        # merge the mask to the scores
        # mask, [B, 1, 1, L] where L should equal to L2
        if x_mask is not None:
            attention_scores = attention_scores.masked_fill(x_mask == 1, -1e10)

        # postprocess the attention scores
        self.init_attention_end()

        self.attention_scores = attention_scores / math.sqrt(self.head_features)
        self.attention_probs = self.softmax(self.attention_scores)
        self.attention_probs_drop = self.attn_dropout(self.attention_probs)

        # x_v, [B, L, self.n_heads, self.head_features]
        x_v = (
            torch.matmul(self.attention_probs_drop, x_v)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # [B, L, self.n_heads * self.head_features]
        x_v = x_v.view(B, L, -1)
        # [B, L, self.n_heads * self.head_features]
        x_v = self.proj(x_v)
        x_v = self.proj_drop(x_v)

        # covert back to the input shape
        if input_dim == 4:
            # from x_v, [B, L, self.n_heads * self.head_features]
            # to x_v, [B, H, W, self.n_heads * self.head_features]
            x_v = x_v.view(B, H, W, -1)

        return x_v
