"""
The implementation of the reconfigured attention module (i.e., RAM attention) in the work [1].

ViDT: An Efficient and Effective Fully Transformer-based Object Detector

For details, please access the Figure 3 of the paper.

"""

from typing import Optional

import math

import torch
import torch.nn as nn

from vggbase.models.regions import window_partition, window_reverse
from vggbase.models.position_encoding import relative_position_encoding
from vggbase.models.position_encoding import utils as position_encoding_utils


class ReconfiguredAttention(nn.Module):
    def __init__(
        self,
        input_xs_n_features: int,
        input_xq_n_features: int,
        window_size: int,
        n_heads: int,
        mapped_qkv_n_features: Optional[int] = None,
        proj_n_features: Optional[int] = None,
        qkv_bias=True,
        proj_bias=True,
        q_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mode="cross",
    ):
        """Reconfigured Attention Module (i.e., RAM attention)

        :param input_xs_n_features: #features for the input xs.
        :param input_xq_n_features: #features for the input xq.
        :param window_size: Number of window size.
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
        :param mode: The operation mode of the RAM attention. There are two options.
            One for `parallel` while one for `cross`
        """

        super().__init__()
        # the operation mode of the RAM
        self.mode = mode

        # number of features for the input
        self.input_xs_n_features = input_xs_n_features
        self.input_xq_n_features = input_xq_n_features

        # as mentioned in the RAM [1], all inputs utilizes
        # a shared layer to get the q, k, v
        # thus they should have the same channels
        assert input_xs_n_features == input_xq_n_features

        # number of window size
        self.window_size = window_size
        # resolution of window along the height and width
        self.window_resolution = [0, 0]
        self.wd_scale = self.window_size * self.window_size

        # number of heads
        self.n_heads = n_heads

        self.mapped_qkv_n_features = None
        self.proj_n_features = None

        # process the input channels
        self.set_default_features(mapped_qkv_n_features, proj_n_features)

        self.mapped_qkv_n_features = self.mapped_qkv_n_features
        self.head_features = self.mapped_qkv_n_features // n_heads

        self.qkv_linear = nn.Linear(
            self.input_xq_n_features, 3 * self.mapped_qkv_n_features, bias=qkv_bias
        )

        self.scale = q_scale or self.head_features**-0.5

        self.attn_dropout = nn.Dropout(attn_drop)

        self.proj = nn.Linear(
            self.mapped_qkv_n_features, self.proj_n_features, bias=proj_bias
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        # define the position encoder
        # by default, ram utilizes the relative position encoding
        self.rela_pos_encoder = None
        self.init_position_encoder()

        # the attention outputs
        # [B, self.n_heads, L1, L1]
        self.xs_attention_scores = None
        self.xs_attention_probs = None
        self.xs_attention_probs_drop = None

        # [B, self.n_heads, L2, L12 + L2]
        # merge the information of input_xs and input_xq
        # to the input_xq to create a new value
        self.sq2q_cross_attention_scores = None
        self.sq2q_cross_attention_probs = None
        self.sq2q_cross_attention_probs_drop = None

    def init_position_encoder(self):
        """Customize the position encoder."""
        self.rela_pos_encoder = relative_position_encoding.RelativePosition2DEncoder(
            enc_features=self.n_heads,
            enc_pos_numbers=self.window_size,
            is_custom_position_ids=False,
        )
        # obtain the encoding pool with shape: pool_capacity, enc_features
        # the encoding pool is actually the relative_position_bias_table
        #   thus, the capacity of encoding pool is:
        #   pool_capacity = 2*wh_s-1 * 2*ww_s-1
        self.rela_pos_encoder.build_position_encoding_pool()

        # as the relative_position_index only depends on the window size, thus
        #   we can compute it here directly as coords_bias_index
        # get pair-wise relative position index for each token inside the window
        inputs = torch.zeros((1, self.window_size[0], self.window_size[1]))
        coords_h, coords_w = position_encoding_utils.process_2D_inputs(
            inputs=inputs, is_mask_input=False, is_custom_position_ids=False
        )

        # obtain the coords bias index can be used to access the encodings
        #   from the encoding pool direclty.
        # coords_bias_index: batch_size, height * width, height * width
        #   where height = wh_s, width = ww_s
        coords_bias_index = (
            self.rela_pos_encoder.obtain_relative_position_bias_index_table(
                h_batch_position_ids=coords_h, w_batch_position_ids=coords_w
            )
        )

        # define self.coords_bias_index
        self.register_buffer("coords_bias_index", coords_bias_index)

    def obtain_position_encoding(self):
        """Obtain the position encoding."""
        # add relative pos bias
        # wh_s*ww_s,wh_s*ww_s,n_heads
        relative_position_bias = self.rela_pos_encoder.encoding_pool(
            self.coords_bias_index.view(-1)
        ).view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        # convert to n_heads, wh_s*ww_s, wh_s*ww_s
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # convert to 1, n_heads, wh_s*ww_s, wh_s*ww_s
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def init_attention_end(self):
        """Customize the postprocessing after obtaining the attention."""
        pass

    def set_default_features(self, mapped_qkv_n_features, proj_n_features):
        """Set the channels."""

        if mapped_qkv_n_features is None:
            self.mapped_qkv_n_features = self.input_xq_n_features

        # from this setting, we can also witness that
        # the output tensor is a new created xq that merges
        # the information from xs but follows the structure of xq.
        if proj_n_features is None:
            self.proj_n_features = self.input_xq_n_features

        if mapped_qkv_n_features % self.n_heads != 0:
            raise ValueError("num_qk_features must be divisible by n_heads")

    def forward(self, input_xs, input_xq, xs_mask=None, sq2q_mask=None, **kwargs):
        """Forward function.

        :param input_xs: A list containing two arrays.
            Two array should have the same dimension and #channels.
            [B, H11, W11, C] or [B, L11, C]
            [B, H12, W12, C] or [B, L12, C]
            where C is the input channels
            (= `input_xs_n_features`)
            The first array performs the window-based self-attention
            The second array performs the cross attention with the xq.
            When the RAM is not in the `cross` mode, there should not
            contain the second term.
        :param input_xq : Query input of shape
            [B, H2, W2, C] or [B, L2, C] where C is the input channels
            (= `input_xq_n_features`)
        :param xs_mask : Mask for the input xs. Its shape
            [B, 1, 1, L11] where 1: masked, 0: unmasked
        :param sq2q_mask : Mask for the input xq. Its shape

            In the 'cross' mode, the shape is
            [B, 1, 1, L12 + L2] where 1: masked, 0: unmasked
            In the 'parallel' mode, the shape is
            [B, 1, 1, L2] where 1: masked, 0: unmasked

        :return: attention result of shape
            [B, H2, W2, C] or [B, L2, C]
            where C the number of output channels (= `proj_n_features`)
        """
        # 1. process the input tensor
        xs1_dim = input_xs[0].dim()
        xq_dim = input_xq.dim()

        # prepare the dimension and sizes
        if xs1_dim == 4:
            B, H11, W11, C = input_xs[0].shape
            L11 = H11 * W11
        if xs1_dim == 3:
            B, N11, C = input_xs[0].shape
            L11 = N11
        if xq_dim == 4:
            B, H2, W2, C = input_xq.shape
            L2 = H2 * W2
            input_xq = input_xq.view(B, L2, C)
        if xq_dim == 3:
            B, N2, C = input_xq.shape
            L2 = N2
        input_xs[0] = input_xs[0].view(B, L11, C)

        # prepare the sizes for the cross attention part
        if self.mode is "cross":
            xs2_dim = input_xs[1].dim
            if xs2_dim == 4:
                B, H12, W12, C = input_xs[1].shape
                L12 = H12 * W12
            if xs2_dim == 3:
                B, N12, C = input_xs[1].shape
                L12 = N12
            input_xs[1] = input_xs[1].view(B, L12, C)
        else:
            L12 = 0

        # concat the three part into one to obtain
        # input_xsq, [B, L11 + L12 + L2, C]
        input_xsq = torch.cat([*input_xs, input_xq], dim=1)

        # forward the qkv lieanr to obtain the combined q, k, v
        # full_qkv, [B, L11 + L12 + L2, 3 * self.mapped_qkv_n_features]
        # of
        # full_qkv, [B, L11 + L2, 3 * self.mapped_qkv_n_features]
        full_qkv = self.qkv_linear(input_xsq)

        # split the full_qkv to obtain,
        # xs_qkv,       [B, L11, 3 * self.mapped_qkv_n_features]
        # xs_cross_qkv, [B, L12, 3 * self.mapped_qkv_n_features]
        # xq_qkv,       [B, L2, 3 * self.mapped_qkv_n_features]
        xs_qkv = full_qkv[:, :L11, :]
        if self.mode is "cross":
            xs_cross_qkv = full_qkv[:, L11 : L11 + L12, :]
        xq_qkv = full_qkv[:, L11 + L12 :, :]

        # 1. perform the window-based self attention for the input_xs
        # 1.1 get a window-based xs_qkv:
        # before partition,
        #   [B, H11, W11, 3 * self.mapped_qkv_n_features]
        # after partition,
        #   [per_sample_windows * B, window_size, window_size, 3 * self.mapped_qkv_n_features]
        # where per_sample_windows = window_resolution[0] * window_resolution[1]
        xs_qkv = xs_qkv.view(B, H11, W11, -1)
        xs_qkv, self.window_resolution = window_partition(xs_qkv, self.window_size)

        # get total number of windows for one batch of samples
        # i.e., B_windows = per_sample_windows * B
        # convert the xs_qkv to:
        # [3, B_windows, self.n_heads, self.wd_scale, head_features]
        B_windows = xs_qkv.shape[0]
        xs_qkv = xs_qkv.reshape(
            B_windows, self.wd_scale, 3, self.n_heads, self.head_features
        )
        xs_qkv = xs_qkv.permute(2, 0, 3, 1, 4)
        # obtain the q, k, v for the input xs
        # xs_q, [B_windows, self.n_heads, self.wd_scale, head_features]
        # xs_k, [B_windows, self.n_heads, self.wd_scale, head_features]
        # xs_v, [B_windows, self.n_heads, self.wd_scale, head_features]
        xs_q, xs_k, xs_v = xs_qkv[0], xs_qkv[1], xs_qkv[2]

        # xs_q
        xs_q = xs_q * self.scale

        # the @ here in the build operator of torch,
        #   @ is equivalent to matmul
        # the window_patch_attn shape:
        #  [B_windows, n_heads, self.wd_scale, self.wd_scale]
        xs_attention_scores = xs_q @ xs_k.transpose(-2, -1)
        # xs_pos_encoding, [1, n_heads, self.wd_scale, self.wd_scale]
        xs_pos_encoding = self.obtain_position_encoding()

        # adding bias to all windows
        xs_attention_scores = xs_attention_scores + xs_pos_encoding
        if xs_mask is not None:
            xs_attention_scores = xs_attention_scores.masked_fill(
                xs_mask == 1, -float("Inf")
            )

        self.xs_attention_scores = xs_attention_scores / math.sqrt(self.head_features)
        self.xs_attention_probs = self.softmax(self.xs_attention_scores)
        self.xs_attention_probs_drop = self.attn_dropout(self.xs_attention_probs)

        # before transpose, xs_v, [B_windows, n_heads, self.wd_scale, head_features]
        # to obtain xs_v, [B_windows, self.wd_scale, n_heads, head_features]
        xs_v = (self.xs_attention_probs_drop @ xs_v).transpose(1, 2)
        xs_v = xs_v.reshape(
            B_windows, self.window_size, self.window_size, self.mapped_qkv_n_features
        )

        # 2. perform the cross attention
        # extract the q, k, v for the input_xq
        # xq_qkv, [B, L2, 3, n_heads, head_features]
        xq_qkv = xq_qkv.view(B, -1, 3, self.n_heads, self.head_features)
        # xq_qkv, [3, B, n_heads, L2, head_features]
        xq_qkv = xq_qkv.permute(2, 0, 3, 1, 4)
        # xq_q, [B, n_heads, L2, head_features]
        # likewise for xq_k and xq_v
        xq_q, xq_k, xq_v = xq_qkv[0], xq_qkv[1], xq_qkv[2]

        # bind the xs to the xq for the cross attention operation
        if self.mode == "cross":
            # merge the k, v of input_xs to the k, v of input_xq
            # convert the shape to obtain
            # xs_cross_qkv, [B, H12, W12, 3, n_heads, head_features]
            xs_cross_qkv = xs_cross_qkv.view(
                B, H12, W12, 3, self.n_heads, self.head_features
            )
            # first get the only k and v parts
            # then permute the shape to be format:
            # xs_cross_kv, [2, B, n_heads, H12, W12, head_features]
            xs_cross_kv = (
                xs_cross_qkv[:, :, :, 1:, :, :].permute(3, 0, 4, 1, 2, 5).contiguous()
            )
            # xs_cross_kv, [2, B, n_heads, L12, head_features]
            xs_cross_kv = xs_cross_kv.view(2, B, self.n_heads, L12, -1)

            # extract "key and value" for the cross attention
            # xs_cross_k, [B, n_heads, L12, head_features]
            xs_cross_k, xs_cross_v = xs_cross_kv[0], xs_cross_kv[1]

            # bind key and value of input_xs's cross part and input_xq
            # to obtain
            # xq_k, [B, n_heads, L12 + L2, head_features]
            # xq_v, [B, n_heads, L12 + L2, head_features]
            xq_k = torch.cat([xs_cross_k, xq_k], dim=2)
            xq_v = torch.cat([xs_cross_v, xq_v], dim=2)

        #  [B, n_heads, L2, L12 + L2]
        sq2q_cross_attention_scores = xq_q @ xq_k.transpose(-2, -1)

        if sq2q_mask is not None:
            sq2q_cross_attention_scores = sq2q_cross_attention_scores.masked_fill(
                sq2q_mask == 1, -1e10
            )

        self.sq2q_cross_attention_scores = sq2q_cross_attention_scores / math.sqrt(
            self.head_features
        )
        self.sq2q_cross_attention_probs = self.softmax(self.sq2q_cross_attention_scores)
        self.sq2q_cross_attention_probs_drop = self.attn_dropout(
            self.sq2q_cross_attention_probs
        )

        # obtain the binded xsq
        # before,
        #   xq_v, [B, n_heads, L12 + L2, head_features]
        #   attn, [B, n_heads, L2, L12 + L2]
        # after @,
        #   [B, n_heads, L2, head_features]
        # after transpose,
        #   [B, L2, n_heads, head_features]
        xq_v = (self.sq2q_cross_attention_probs_drop @ xq_v).transpose(1, 2)
        # to xq_v, [B, L2, mapped_qkv_n_features]
        #   where mapped_qkv_n_features = n_heads * head_features
        xq_v = xq_v.reshape(B, -1, self.mapped_qkv_n_features)

        # perform the final projection
        # reverse the output of the window-based self attention
        # before, xs_v, [B_windows, self.wd_scale, mapped_qkv_n_features]
        # after, xs_v, [B, H11, W11, mapped_qkv_n_featuress]
        xs_v = window_reverse(xs_v, self.window_size, H11, W11)

        # combine the two
        # input_xs, [B, L11 + L2, mapped_qkv_n_featuress]
        input_xsq = torch.cat(
            [xs_v.view(B, H11 * W11, self.mapped_qkv_n_featuress), xq_v], dim=1
        )

        # projection of input_xsq to get
        # input_xsq, [B, L11 + L2, self.proj_n_features]
        input_xsq = self.proj(input_xsq)
        input_xsq = self.proj_drop(input_xsq)

        xs_v = input_xsq[:, :L11, :]
        if xs1_dim == 4:
            xs_v = xs_v.view(B, H11, W11, self.proj_n_features)

        xq_v = input_xsq[:, L11:, :]

        return xs_v, xq_v
