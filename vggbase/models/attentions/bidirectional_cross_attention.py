"""
The implementation of bi-directional cross attention.

This structure of this type of attention is applied in the work [1].

a2b attention part:
xa  ---> xa_v--------------
          ---> xa_k ---         |--> xb_v
                       |-> attn -
xb  ---> xb_q ---
---------------------------------------------
b2a attention part:
xb  ---> xb_v--------------
          ---> xb_k ---         |--> xa_v
                       |-> attn -
xa  ---> xa_q ---


where the a2b means that the information of a is merged into that
of q to create a new q based on the query of q. vice versa.

[1]. SelfDoc: Self-Supervised Document Representation Learning

"""

from typing import Optional

import torch.nn as nn

from vggbase.models.attentions.unidirectional_cross_attention import (
    UniDirectionalCrossAttention,
)


# pylint: disable=invalid-name
class BiDirectionalCrossAttention(nn.Module):
    """A bidirectional cross attention."""

    def __init__(
        self,
        xa_n_features: int,
        xb_n_features: int,
        n_heads: int,
        mapped_a2b_qk_n_features: Optional[int] = None,
        mapped_b2a_qk_n_features: Optional[int] = None,
        mapped_xa_v_n_features: Optional[int] = None,
        mapped_xb_v_n_features: Optional[int] = None,
        xa_proj_n_features: Optional[int] = None,
        xb_proj_n_features: Optional[int] = None,
        qkv_bias=True,
        proj_bias=True,
        q_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """Unidirectional cross attention mechanism.

        :param xa_n_features: #dimentions for the input xa.
        :param xb_n_features: #dimentions for the input xb.
        :param n_heads: Number of attention heads.
        :param mapped_a2b_qk_n_features: #dimentions for the mapped query of the input and xa
            #dimentions for the mapped key of the input xb. Default is `None`.
        :param mapped_b2a_qk_n_features: #dimentions for the mapped query of the input and xb
            #dimentions for the mapped key of the input xa. Default is `None`.
        :param mapped_xa_v_n_features: #dimentions for the value for the input xb.
        :param mapped_xb_v_n_features: #dimentions for the value for the input xa.
        :param xa_proj_n_features: #dimentions for the projection of xa.
        :param xb_proj_n_features: #dimentions for the projection of xb.
            Note that the
        :param qkv_bias: Whether to use a bias term for query, key and value projections.
                        Default is `True`.
        :param proj_bias: Whether to use a bias term for the projection.
                        Default is `True`.
        :param q_scale: Whether to perform a scale for the query. Default is `head_features**-0.5`.
        :param attn_drop: Dropout probability for attention matrix values. Default is `0.0`
        :param proj_drop: Dropout probability for projection values. Default is `0.0`
        """

        super().__init__()
        # number of dimension for the input
        self.xa_n_features = xa_n_features
        self.xb_n_features = xb_n_features

        # number of dimension for the attention part
        self.n_heads = n_heads

        ## a2b attention
        self.a2b_attn = UniDirectionalCrossAttention(
            xs_n_features=xa_n_features,
            xq_n_features=xb_n_features,
            n_heads=n_heads,
            mapped_qk_n_features=mapped_a2b_qk_n_features,
            mapped_v_n_features=mapped_xa_v_n_features,
            proj_n_features=xb_proj_n_features,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            q_scale=q_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        ## b2a attention
        self.b2a_attn = UniDirectionalCrossAttention(
            xs_n_features=xb_n_features,
            xq_n_features=xa_n_features,
            n_heads=n_heads,
            mapped_qk_n_features=mapped_b2a_qk_n_features,
            mapped_v_n_features=mapped_xb_v_n_features,
            proj_n_features=xa_proj_n_features,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            q_scale=q_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # the xa to xb attention outputs
        # [B, self.n_heads, L2, L1]
        self.a2b_attention_scores = None
        self.a2b_attention_probs = None
        self.a2b_attention_probs_drop = None

        # the xa to xb attention outputs
        # [B, self.n_heads, L1, L2]
        self.b2a_attention_scores = None
        self.b2a_attention_probs = None
        self.b2a_attention_probs_drop = None

    def init_position_encoding(self):
        """Customize the position encoding."""
        pass

    def init_attention_end(self):
        """Customize the postprocessing after obtaining the attention."""
        pass

    def forward(self, xa, xb, xa_mask=None, xb_mask=None, **kwargs):
        """Forward function.

        :param xa: An Array input of shape
            [B, H1, W1, C1] or [B, L1, C1] where C1 is the input channels
            (= `xa_n_features`)
        :param xb : Query input of shape
            [B, H2, W2, C2] or [B, L2, C2] where C2 is the input channels
            (= `xb_n_features`)
        :param xa_mask : Mask for the input xa. Its shape
            [B, 1, 1, L1] where 1: masked, 0: unmasked
        :param xa_mask : Mask for the input xb. Its shape
            [B, 1, 1, L2] where 1: masked, 0: unmasked

        :return:
            xa, attention result of shape
            [B, H2, W2, C] or [B, L2, C]
            where C the number of output channels (= `proj_n_features`)
            xb, attention result of shape
            [B, H2, W2, C] or [B, L2, C]
            where C the number of output channels (= `proj_n_features`)
        """

        # perform a2b attention
        xb = self.a2b_attn(xa, xb, xs_mask=xa_mask, **kwargs)
        # obtain attention
        self.a2b_attention_scores = self.a2b_attn.attention_scores
        self.a2b_attention_probs = self.a2b_attn.attention_probs
        self.a2b_attention_probs_drop = self.a2b_attn.attention_probs_drop

        # perform b2a attention
        xa = self.b2a_attn(xb, xa, xs_mask=xb_mask, **kwargs)
        self.b2a_attention_scores = self.b2a_attn.attention_scores
        self.b2a_attention_probs = self.b2a_attn.attention_probs
        self.b2a_attention_probs_drop = self.b2a_attn.attention_probs_drop

        return xa, xb
