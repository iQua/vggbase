"""
The language-guided visual grounding head for diffusion.

This part is referred to as language-guide RCNN block (LGRCNNBlock), 

``LGRCNNBlock`` performs:

0.  receiving `mapped_text` as inputs, denoted as xa
    receiving `bboxes_features` from previous ``VTCross`` as another
    input, denoted as xb

1.  operating ``RAM`` attention, which shows as follows

xa  ---> xa_v-----------------------------------|
    ---> xa_k ---|                              |
                 |                              |
                 ||->|                          |
                 |   |-->attn(b x [b, a])-->[xb_v,xa_v]---> xb_vv    
    ---> xb_k ---|   |                          |
xb  ---> xb_q -------|                          |
    ---> xb_v-----------------------------------|  
"""

import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from vggbase.boxes.bbox_coordinator import BBoxFasterRCNNCoordinater
from vggbase.models.mlps.general_mlps import build_mlp_from_config

from diffusionvg_generic import LGVGRCNNOutput


class FeedForwardWithResidual(nn.Module):
    """The FFN layer with residual after the LG Block."""

    def __init__(
        self,
        input_n_features: int,
        output_n_features: int,
        hidden_n_features: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        drop_path: float = 0.2,
    ):
        super().__init__()
        self.mlp = build_mlp_from_config(
            mlp_configs=dict(
                output_n_features=output_n_features,
                input_n_features=input_n_features,
                hidden_layers_n_features=[hidden_n_features],
                batch_norms=[None, None],
                activations=[activation, None],
                dropout_ratios=[dropout, dropout],
            )
        )
        self.norm = nn.LayerNorm(output_n_features)
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x):
        return self.norm(x + self.drop_path(self.mlp(x)))


class LGBlock(nn.Module):
    """The implementation of LG Block."""

    def __init__(
        self,
        input_xs_n_features: int,
        input_xq_n_features: int,
        n_heads: int,
        mapped_qkv_n_features: Optional[int] = None,
        proj_n_features: Optional[int] = None,
        qkv_bias=True,
        proj_bias=True,
        q_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        keep_attentions: bool = True,
    ):
        """Language-guided Module (i.e., LG attention)

        :param input_xs_n_features: #features for the input xs.
        :param input_xq_n_features: #features for the input xq.
        :param n_heads: Number of attention heads.
        :param mapped_qkv_n_features: #features for the mapped q, k, v for all inputs.
            Default is `None`.
        :param proj_n_features: #features for the projection for outputs of all inputs.
            Default is `None`.
        :param qkv_bias: Whether to use a bias term for query, key and value projections.
                        Default is `True`.
        :param proj_bias: Whether to use a bias term for the projection.
                        Default is `True`.
        :param q_scale: Whether to perform a scale for the query. Default is `head_features**-0.5`.
        :param attn_drop: Dropout probability for attention matrix values. Default is `0.0`
        :param proj_drop: Dropout probability for projection values. Default is `0.0`.
        """
        super().__init__()

        # number of features for the input
        self.ipt_xs_n_feas = input_xs_n_features
        self.ipt_xq_n_feas = input_xq_n_features

        # as the cross relation is learned, these two
        # inputs should have the same #features
        assert self.ipt_xs_n_feas == self.ipt_xq_n_feas

        # number of heads
        self.n_heads = n_heads

        self.mpd_qkv_n_feas = mapped_qkv_n_features
        self.proj_n_feas = proj_n_features

        # process the input channels
        self.set_default_features(mapped_qkv_n_features, proj_n_features)

        self.head_n_feas = self.mpd_qkv_n_feas // n_heads

        self.qkv_linear = nn.Linear(
            self.ipt_xq_n_feas, 3 * self.mpd_qkv_n_feas, bias=qkv_bias
        )

        self.scale = q_scale or self.head_n_feas**-0.5

        self.attn_dropout = nn.Dropout(attn_drop)

        self.proj = nn.Linear(self.mpd_qkv_n_feas, self.proj_n_feas, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        # the attention outputs
        # [B, self.n_heads, L1, L1]
        self.xs_attn_scores = None
        self.xs_attn_probs = None
        self.xs_attn_dprobs = None

        # [B, self.n_heads, L2, L12 + L2]
        # merge the information of input_xs and input_xq
        # to the input_xq to create a new value
        self.sq2q_cross_attn_scores = None
        self.sq2q_cross_attn_probs = None
        self.sq2q_cross_attn_dprobs = None

    def set_default_features(self, mapped_qkv_n_features, proj_n_features):
        """Set the channels."""

        if mapped_qkv_n_features is None:
            self.mpd_qkv_n_feas = self.ipt_xq_n_feas

        # from this setting, we can also witness that
        # the output tensor is a new created xq that merges
        # the information from xs but follows the structure of xq.
        if proj_n_features is None:
            self.proj_n_feas = self.ipt_xq_n_feas

        if self.mpd_qkv_n_feas % self.n_heads != 0:
            raise ValueError("num_qk_features must be divisible by n_heads")

    def extract_s2q_attention(self):
        """Extract xq to xs attentions."""
        # attn, [bs, self.n_heads, N, P + N]
        N, P_plus_N = self.sq2q_cross_attn_dprobs.shape[-2:]
        P = P_plus_N - N

        # extract [bs, self.n_heads, N, P]
        extracted_s2q_attn = self.sq2q_cross_attn_dprobs[:, :, :, :P]
        # extract [bs, N, P]
        return torch.mean(extracted_s2q_attn, dim=1)

    def forward(
        self,
        input_xs: torch.Tensor,
        input_xq: torch.Tensor,
        xs_mask: Optional[torch.Tensor] = None,
        xq_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward the LG attention.

        :param xs:  A `torch.FloatTensor` holding the features,
         such as the text query,
         of shape, [bs, P, D]
        :param xq: A `torch.FloatTensor` holding the features,
         such as bounding boxes,
         of shape, [bs, N, D]
        :xs_mask: A `torch.BoolTensor` holding the mask of `xs`,
         of shape, [bs, P]
        :xq_mask: A `torch.BoolTensor` holding the mask of `xs`,
         of shape, [bs, N].
        """
        bs, P = input_xs.shape[:2]
        N = input_xq.shape[1]

        # concat the three part into one to obtain
        # input_xsq, [bs, P + N, C]
        input_xsq = torch.cat([input_xq, input_xs], dim=1)

        # forward the qkv lieanr to obtain the combined q, k, v
        # full_qkv, [bs, P + N, 3 * self.mpd_qkv_n_feas]
        full_qkv = self.qkv_linear(input_xsq)

        # split the full_qkv to obtain,
        # xs_qkv, [bs, P, 3 * self.mpd_qkv_n_feas]
        # xq_qkv, [bs, N, 3 * self.mpd_qkv_n_feas]
        xs_qkv = full_qkv[:, :P, :]
        xq_qkv = full_qkv[:, P:, :]
        # extract q, k, v
        xs_qkv = xs_qkv.view(bs, P, 3, self.n_heads, self.head_n_feas)
        xq_qkv = xq_qkv.view(bs, N, 3, self.n_heads, self.head_n_feas)
        # [3, bs, self.n_heads, P, self.head_n_feas]
        xs_qkv = xs_qkv.permute(2, 0, 3, 1, 4)
        # [3, bs, self.n_heads, N, self.head_n_feas]
        xq_qkv = xq_qkv.permute(2, 0, 3, 1, 4)

        # [bs, self.n_heads, P or N, self.head_n_feas]
        xs_q, xq_q = xs_qkv[0], xq_qkv[0]
        xs_k, xq_k = xs_qkv[1], xq_qkv[1]
        xs_v, xq_v = xs_qkv[2], xq_qkv[2]

        # [bs, self.n_heads, P + N, self.head_n_feas]
        cross_xsq_k = torch.cat([xs_k, xq_k], dim=2)
        cross_xsq_v = torch.cat([xs_v, xq_v], dim=2)

        ## perform cross attention
        # obtain [bs, self.n_heads, N, P + N]
        cross_attn_scores = xq_q @ cross_xsq_k.transpose(-2, -1)
        # print("cross_attn_scores: ", cross_attn_scores)
        if xs_mask is not None:
            # assign mask xs_mask [bs, P]
            # pad to [bs, P+N]
            cross_mask = F.pad(xs_mask, (0, N), "constant", 0)
            # to [bs, 1, 1, P+N]
            cross_mask = cross_mask[:, None, None, :]
            cross_attn_scores = cross_attn_scores.masked_fill(
                cross_mask == 1, -float("Inf")
            )
        # print("cross_attn_scores: ", cross_attn_scores)
        self.sq2q_cross_attn_scores = cross_attn_scores / math.sqrt(self.head_n_feas)
        self.sq2q_cross_attn_probs = self.softmax(self.sq2q_cross_attn_scores)
        self.sq2q_cross_attn_dprobs = self.attn_dropout(self.sq2q_cross_attn_probs)

        # attn, [bs, self.n_heads, N, P + N]
        # cross_xsq_v, [bs, self.n_heads, P + N, self.head_n_feas]
        # output, [bs, self.n_heads, N, self.head_n_feas]
        # transpose, [bs, N, self.n_heads, self.head_n_feas]
        cross_xsq_v = (self.sq2q_cross_attn_dprobs @ cross_xsq_v).transpose(1, 2)

        ## perform self attention for text
        # attn, [bs, self.n_heads, P, P]
        self.xs_attn_scores = xs_q @ xs_k.transpose(-2, -1)
        if xs_mask is not None:
            # [bs, 1, 1, P]
            xs_mask = xs_mask[:, None, None, :]
            self.xs_attn_scores = self.xs_attn_scores.masked_fill(
                xs_mask == 1, -float("Inf")
            )

        self.xs_attn_scores = self.xs_attn_scores / math.sqrt(self.head_n_feas)
        self.xs_attn_probs = self.softmax(self.xs_attn_scores)
        self.xs_attn_dprobs = self.attn_dropout(self.xs_attn_probs)

        # [bs, P, self.n_heads, self.head_n_feas]
        xs_v = (self.xs_attn_dprobs @ xs_v).transpose(-2, -1)

        # [bs, P, self.n_heads * self.head_n_feas]
        xs_v = xs_v.contiguous().view(bs, P, -1)
        # [bs, N, self.n_heads * self.head_n_feas]
        cross_xsq_v = cross_xsq_v.contiguous().view(bs, N, -1)
        # to, [bs, P + N, self.n_heads, self.head_n_feas]
        input_xsq = torch.cat((xs_v, cross_xsq_v), dim=1)

        # projection of input_xsq to get
        # input_xsq, [B, P + N, self.n_heads * self.proj_n_features]
        input_xsq = self.proj(input_xsq)
        input_xsq = self.proj_drop(input_xsq)

        xs_v = input_xsq[:, :P, :]
        cross_xsq_v = input_xsq[:, P:, :]
        return xs_v, cross_xsq_v


class LGRCNN(nn.Module):
    """The multimodal head for the visual grounding."""

    def __init__(self, d_model, rcnn_config):
        super().__init__()

        # the core hidden feature size
        # along the whole model
        self.d_model = d_model

        # the configurations for
        # rcnn module
        self.rcnn_config = rcnn_config

        self.scale_shift_mapper_config = rcnn_config.scale_shift_mapper

        ## block time mlp
        self.scale_shift_mapper = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.scale_shift_mapper_config.in_features,
                self.scale_shift_mapper_config.out_features,
            ),
        )

        lgblock_config = rcnn_config.lgblock
        n_blocks = lgblock_config.n_blocks
        self.lg_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.lg_blocks.append(
                LGBlock(
                    input_xs_n_features=lgblock_config.input_text_feature,
                    input_xq_n_features=lgblock_config.input_visual_feature,
                    n_heads=lgblock_config.n_heads,
                    mapped_qkv_n_features=lgblock_config.mapped_qkv_features,
                    proj_n_features=lgblock_config.projection_features,
                    qkv_bias=True,
                    proj_bias=True,
                    q_scale=None,
                    attn_drop=lgblock_config.attention_dropout,
                    proj_drop=lgblock_config.projection_dropout,
                )
            )

        self.norm = nn.LayerNorm(lgblock_config.projection_features)

        self.ffn_with_res = FeedForwardWithResidual(
            input_n_features=lgblock_config.ffn.in_features,
            output_n_features=lgblock_config.ffn.out_features,
            hidden_n_features=lgblock_config.ffn.hidden_features,
            activation="gelu",
            dropout=lgblock_config.ffn.dropout,
            drop_path=lgblock_config.ffn.dropout_path,
        )

        # reg.
        box_reg_config = rcnn_config.bbox_regression
        n_reg = box_reg_config.regression_layers
        self.reg_module = nn.ModuleList([])
        for _ in range(n_reg):
            self.reg_module.append(
                nn.Linear(
                    box_reg_config.in_features, box_reg_config.out_features, False
                )
            )
            self.reg_module.append(nn.LayerNorm(box_reg_config.out_features))
            self.reg_module.append(nn.ReLU(inplace=True))

        # pred.
        coord_weights = self.rcnn_config.bbox_delta.coord_weights
        wh_scale_calmp = self.rcnn_config.bbox_delta.wh_scale_calmp
        self.bboxes_offsets = nn.Linear(box_reg_config.out_features, 4)
        self.bboxes_coordinater = BBoxFasterRCNNCoordinater(
            coord_weights, wh_scale_calmp
        )

    def forward(
        self,
        rois_features: Tensor,
        bboxes: Tensor,
        x_tquery: Tensor,
        tquery_mask: Tensor,
        time_steps: Tensor,
    ):
        """Forward the vg head.
        :param rois_features: A `torch.FloatTensor` holding the features
         of bounding boxes.
         of shape, [bs, N, roi_n_features]
        :param bboxes: A `torchFloatTensor` containing bounding boxes,
         of shape, [bs, n_boxes, 4].
         of format, [x1, y1, x2, y2]
        :param x_tquery: A `torch.FloatTensor` holding the query emebdding
         of shape, [batch_size, P, text_n_features],
         where `P` is #phrases.
        :param tquery_mask: A `torch.BoolTensor` holding the mask for the text
         query, where P is the max number of phrases in one batch
            [False: vaild parts, True: invaild parts]
         of shape, [batch_size, P]
        :param time_steps: A `torch.Tensor` containing the embedded time steps,
         of shape, [batch_size, d_model * time_emb_n_extension]

        """
        bs, n_bboxes = bboxes.shape[:2]
        n_phrases = x_tquery.shape[1]

        blks_attentions = []
        for block in self.lg_blocks:
            new_x_tquery, new_rois_features = block(
                input_xs=x_tquery, input_xq=rois_features, xs_mask=tquery_mask
            )
            x_trq_features = torch.cat([x_tquery, rois_features], dim=1)
            new_x_trq_features = torch.cat([new_x_tquery, new_rois_features], dim=1)
            x_trq_features = x_trq_features + new_x_trq_features
            x_trq_features = self.norm(x_trq_features)
            # FFN with residual
            x_trq_features = self.ffn_with_res(x_trq_features)
            x_tquery = x_trq_features[:, :n_phrases, :]
            rois_features = x_trq_features[:, n_phrases:, :]
            # record attentions
            # [bs, n_bboxes, n_phrases]
            blks_attentions.append(block.extract_s2q_attention())

        # stack blocks' attention
        # [bs, n_blks, n_bboxes, n_phrases]
        blks_attentions = torch.stack(blks_attentions, dim=1)

        # following the work
        # "Segmenter: Transformer for Semantic Segmentation"
        # to compute the similarity
        # rois_features, [bs, n_bboxs, D]
        # x_tquery, [bs, n_phrases, D]
        norm_rois_features = rois_features / rois_features.norm(dim=-1, keepdim=True)
        norm_x_tquery = x_tquery / x_tquery.norm(dim=-1, keepdim=True)
        # x_tquery -> transpose [bs, D, n_phrases]
        # matching_scores: [bs, n_bboxs, n_phrases]
        matching_scores = norm_rois_features @ norm_x_tquery.transpose(-2, -1)

        ## perform diffusion rescaling and shifting
        # convert time_steps
        # from [bs, D]
        # to [bs, n_bboxes, D] for scaling
        time_steps = time_steps[:, None, :]
        time_steps = torch.repeat_interleave(time_steps, n_bboxes, dim=1)

        # obtain the text-guided feature for each bbox
        # [bs, n_bboxes, D] for shifting
        bboxes_tquery = matching_scores @ x_tquery
        # [bs, n_bboxes, 2D]
        time_btext = torch.cat([time_steps, bboxes_tquery], dim=-1)
        # [bs, n_bboxes, 2D]
        scale_shift = self.scale_shift_mapper(time_btext)
        # [bs, n_bboxes, D]
        scale, shift = scale_shift.chunk(2, dim=2)

        # rois_features, [bs, n_bboxes, D]
        fc_features = rois_features * (scale + 1) + shift

        for reg_layer in self.reg_module:
            fc_features = reg_layer(fc_features)

        # bboxes_offsets, [bs, n_bboxes, 4]
        bboxes_offsets = self.bboxes_offsets(fc_features)

        # pred_bboxes, [bs * n_bboxes, 4]
        pred_bboxes = self.bboxes_coordinater.apply_offsets(
            bboxes=bboxes.view(-1, 4), offsets=bboxes_offsets.view(-1, 4)
        )

        return LGVGRCNNOutput(
            predicted_bboxes=pred_bboxes.view(bs, n_bboxes, -1),
            matching_scores=matching_scores,
            vg_attentions=blks_attentions,
            pre_rois_features=rois_features,
            pre_text_features=x_tquery,
        )
