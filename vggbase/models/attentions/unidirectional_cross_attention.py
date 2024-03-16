"""
The implementation of general self attention operation.

The prototype of this attention mechanism is proposed and
utilized in the perceiver-io [1].

xs  ---> xs_v--------------
    ---> xs_k ---      |--> xq_v
                       |-> attn -
xq  -------------------> xq_q ---

where the 's' in 'xs' denotes the source of information to be
merged to create a new xq based on the xq's query.

the output xq_v is a new v created by merging the xs_v's
information based on the xq's query.


For the dimension, there are three things required to be noticed.
1- xv and xq can have different dimension.
    xv:  xv_n_features ,
    xq:  xq_n_features ,
2- The matmul performs on mapped xv_k and mapped xq_q, these two items
    should have the same dimension.
    i.e., mapped_qk_n_features
3- The mapped xv_v can have any dimension!
    i.e., mapped_v_n_features

This can be regarded as the information mechanism that merges the information
of xq to the information of xs.


Note:
 For many operations, we can also utilize the
    - einops
    - torch.einsum
 as what presented in perceiver-io [1].


[1]. https://github.com/krasserm/perceiver-io

"""

from typing import Optional

import math

import torch
import torch.nn as nn


# pylint: disable=invalid-name
class UniDirectionalCrossAttention(nn.Module):
    """unidirectional cross attention.

    See comment above."""

    def __init__(
        self,
        xs_n_features: int,
        xq_n_features: int,
        n_heads: int,
        mapped_qk_n_features: Optional[int] = None,
        mapped_v_n_features: Optional[int] = None,
        proj_n_features: Optional[int] = None,
        qkv_bias=True,
        proj_bias=True,
        q_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """Unidirectional cross attention mechanism.

        :param xs_n_features : #dimentions for the input xs.
        :param xq_n_features : #dimentions for the input xq.
        :param n_heads: Number of attention heads.
        :param mapped_qk_n_features : #dimentions for the mapped query of the input and xq
                              #dimentions for the mapped key of the input xs.
                            Default is `None`.
        :param mapped_v_n_features : #dimentions for the value for the input xs.
        :param proj_n_features : #dimentions for the finap projection.
        :param qkv_bias: Whether to use a bias term for query, key and value projections.
                        Default is `True`.
        :param proj_bias: Whether to use a bias term for the projection.
                        Default is `True`.
        :param q_scale: Whether to perform a scale for the query. Default is `head_features **-0.5`.
        :param attn_drop: Dropout probability for attention matrix values. Default is `0.0`
        :param proj_drop: Dropout probability for projection values. Default is `0.0`
        """

        super().__init__()
        # number of dimension for the input
        self.xs_n_features = xs_n_features
        self.xq_n_features = xq_n_features

        # number of dimension for the attention part
        self.n_heads = n_heads

        self.mapped_qk_n_features = mapped_qk_n_features
        self.mapped_v_n_features = mapped_v_n_features
        self.proj_n_features = proj_n_features

        # process the input features
        self.set_default_params(
            mapped_qk_n_features, mapped_v_n_features, proj_n_features
        )

        self.qk_head_features = self.mapped_qk_n_features // n_heads
        self.v_head_features = self.mapped_v_n_features // n_heads

        self.scale = q_scale or self.qk_head_features**-0.5

        # the qkv
        self.q_linear = nn.Linear(
            self.xq_n_features, self.mapped_qk_n_features, bias=qkv_bias
        )
        self.k_linear = nn.Linear(
            self.xs_n_features, self.mapped_qk_n_features, bias=qkv_bias
        )
        self.v_linear = nn.Linear(
            self.xs_n_features, self.mapped_v_n_features, bias=qkv_bias
        )

        self.attn_dropout = nn.Dropout(attn_drop)

        self.proj = nn.Linear(
            self.mapped_v_n_features, self.proj_n_features, bias=proj_bias
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        # the attention outputs
        # [B, self.n_heads, L2, L1]
        self.attention_scores = None
        self.attention_probs = None
        self.attention_probs_drop = None

    def set_default_params(
        self, mapped_qk_n_features, mapped_v_n_features, proj_n_features
    ):
        """Set the default parameters."""

        if mapped_qk_n_features is None:
            self.mapped_qk_n_features = self.xq_n_features

        if mapped_v_n_features is None:
            self.mapped_v_n_features = mapped_qk_n_features

        # from this setting, we can also witness that
        # the output tensor is a new created xq that merges
        # the information from xs but follows the structure of xq.
        if proj_n_features is None:
            self.proj_n_features = self.xq_n_features

        if mapped_qk_n_features % self.n_heads != 0:
            raise ValueError("num_qk_features  must be divisible by n_heads")

        if mapped_v_n_features % self.n_heads != 0:
            raise ValueError("num_v_features  must be divisible by n_heads")

    def init_position_encoding(self):
        """Customize the position encoding."""
        pass

    def init_attention_end(self):
        """Customize the postprocessing after obtaining the attention."""
        pass

    def forward(self, xs, xq, xs_mask=None, **kwargs):
        """Forward function.

        :param xs: An Array input of shape
            [B, H1, W1, C1] or [B, L1, C1] where C1 is the input channels
            (= `xs_n_features `)
        :param xq : Query input of shape
            [B, H2, W2, C2] or [B, L2, C2] where C2 is the input channels
            (= `xq_n_features `)
        :param xs_mask : Mask for the input xs. Its shape
            [B, 1, 1, L1] where 1: masked, 0: unmasked

        :return: attention result of shape
            [B, H2, W2, C] or [B, L2, C]
            where C the number of output channels (= `proj_n_features `)
        """
        # 1. process the input tensor
        xs_dim = xs.dim()
        xq_dim = xq.dim()
        if xs_dim == 4:
            B, H1, W1, C1 = xs.shape
            L1 = H1 * W1
            xs = xs.view(B, L1, C1)
        if xq_dim == 4:
            B, H2, W2, C2 = xq.shape
            L2 = H2 * W2
            xq = xq.view(B, L2, C2)
        if xs_dim == 3:
            B, N1, C1 = xs.shape
            L1 = N1
        if xq_dim == 3:
            B, N2, C2 = xq.shape
            L2 = N2

        # 1. forward to obtain q, k, v
        # xq_q, [B, L2, mapped_qk_n_features ]
        # xs_k, [B, L1, mapped_qk_n_features ]
        # xs_v, [B, L1, mapped_v_n_features ]
        #  where mapped_qk_n_features  = self.n_heads * self.qk_head_features
        #  where mapped_v_n_features  = self.n_heads * self.v_head_features
        xq_q = self.q_linear(xq)
        xs_k = self.k_linear(xs)
        xs_v = self.v_linear(xs)

        # 2. rearrange q, k, v
        # xq_q, [B, self.n_heads, L2, self.qk_head_features ]
        # xs_k, [B, self.n_heads, L1, self.qk_head_features ]
        # xs_v, [B, self.n_heads, L1, self.v_head_features ]
        xq_q = xq_q.view(B, L2, self.n_heads, self.qk_head_features).permute(0, 2, 1, 3)
        xs_k = xs_k.view(B, L1, self.n_heads, self.qk_head_features).permute(0, 2, 1, 3)
        xs_v = xs_v.view(B, L1, self.n_heads, self.v_head_features).permute(0, 2, 1, 3)

        # 4. cross-attention
        # xq_q, [B, self.n_heads, L2, self.qk_head_features ]
        xq_q = xq_q * self.scale

        # 5. add position encoding when possible
        self.init_position_encoding()

        # get attention scores
        # transposed xs_k, [B, self.n_heads, self.qk_head_features , L1]
        # attention_scores, [B, self.n_heads, L2, L1]
        attention_scores = xq_q @ xs_k.transpose(-2, -1)
        attn_max_neg = -torch.finfo(attention_scores.dtype).max

        # merge the mask to the scores
        # mask, [B, 1, 1, L1] where L should equal to L1
        if xs_mask is not None:
            attention_scores = attention_scores.masked_fill(xs_mask == 1, attn_max_neg)

        # postprocess the attention scores
        self.init_attention_end()

        # attention_scores, [B, self.n_heads, L2, L1]
        self.attention_scores = attention_scores / math.sqrt(self.head_features)
        self.attention_probs = self.softmax(self.attention_scores)
        self.attention_probs_drop = self.attn_dropout(self.attention_probs)

        # before mul, xq_v, [B, self.n_heads, L1, self.v_head_features ]
        # after mul, xq_v, [B, self.n_heads, L2, self.v_head_features ]
        # after permute, xq_v, [B, L2, self.n_heads, self.v_head_features ]
        xq_v = (
            torch.matmul(self.attention_probs_drop, xs_v)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # obtain x_v, [B, L2, mapped_v_n_features ]
        # where mapped_v_n_features  = self.n_heads * self.v_head_features
        xq_v = xq_v.view(B, L2, -1)

        # before proj, x_v, [B, L2, mapped_v_n_features ]
        # after proj, x_v, [B, L2, proj_n_features ]
        xq_v = self.proj(xq_v)
        xq_v = self.proj_drop(xq_v)

        # covert back to the input shape
        if xq_dim == 4:
            # obtain xq_v, [B, H2, W2, proj_n_features ]
            xq_v = xq_v.view(B, H2, W2, self.proj_n_features)

        return xq_v
