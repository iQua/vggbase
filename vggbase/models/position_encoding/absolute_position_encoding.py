"""
The absolute position encoding support two different methods,
    - BasicPositionEncoder
    - SinusoidalPosition1DEncoder
    - SinusoidalPosition2DEncoder

To verify the correcness of our position encoding part.
 We utilize the cross-validation of three sources, including:
    - https://github.com/wzlxjtu/PositionalEncoding2D
    - https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py
    - position embedding of swim transformer part
    - masked_sin_pos_encoding of the configurated attention-based swim transformer.

"""
import math

import torch

from .base import PositionEncoding

from .utils import process_1d_inputs, process_2D_inputs


class BasicPositionEncoder(PositionEncoding):
    """ The basic position encoder for the 1D input.

        Input:
            enc_position_number (int): the number of position
                required to be encoded.
                    For example, the maximum sequence length in nlp;
                                the maximum patches in the cv.
            enc_features (int): the encoding #features.

            is_custom_position_ids: whether the input ids are defined by
                the user.
                    If True, the value of the input tensor is the position idex.
                    Otherwise, the value of the input tensor makes no sense, thus
                        we need to genrate position idex for items.
    """

    def __init__(self,
                 enc_position_number,
                 enc_features,
                 is_custom_position_ids=False):
        super(BasicPositionEncoder, self).__init__()
        self.enc_position_number = enc_position_number
        self.enc_features = enc_features
        self.is_custom_position_ids = is_custom_position_ids

        self.encoding_pool_capacity = self.enc_position_number

    def forward(self, inputs):
        """ Forward the encoder to obtain the position encodings.

            Args:
                inputs (torch.tensor): a tensor with shape
                    batch_size, seq_length

            Output:
                encoded_positions: a tensor containing the position
                    embeddings with shape batch_size, seq_length, enc_features.
        """
        batch_position_ids = process_1d_inputs(
            inputs, is_custom_position_ids=self.is_custom_position_ids, device=inputs.device())

        # obtan encoded tensor:
        #   batch_size, seq_length, enc_features
        encoded_positions = self.encoding_pool(batch_position_ids)

        return encoded_positions


class SinusoidalPosition1DEncoder(PositionEncoding):
    """ The sin_pos_encoding that belongs to the absolute position embedding.
        The reason is described in https://zhuanlan.zhihu.com/p/105001610?utm_source=wechat_session.

        P E(pos,2i) = sin(pos/10000**(2i/dmodel))
        P E(pos,2i+1) = cos(pos/10000**(2i/dmodel))

        where pos is the position and i is the feature index.
        Inputs:
            embedding_features (int): the size of embedding features
            temperature: the temperature value
            scale: the normalization scale
    """

    def __init__(self,
                 enc_features,
                 temperature=10000.0,
                 scale=2 * math.pi,
                 is_custom_position_ids=False):
        super(SinusoidalPosition1DEncoder, self).__init__()
        # using a even enc features as
        #   half of the features is the sin coding.
        #   another half is the cos coding.
        #   They are staggered.
        #   Thus, the features looks like: [sin, cos, sin, cos, sin, cos, ...]
        if enc_features % 2 != 0:
            raise RuntimeError(
                "The input embedding_features has to be divisible by 2!")

        # compute the 1 / 10000**(2i/dmodel)
        # with enc_features 8, the correct inv_freq is
        #   tensor([1.0000, 0.1000, 0.0100, 0.0010])

        # shape of inv_freq is:
        #   enc_features / 2
        self.feature_indices = torch.arange(0,
                                            enc_features,
                                            2,
                                            dtype=torch.float32)
        self.inv_freq = 1.0 / (temperature
                               **(self.feature_indices / enc_features))

        # If you have parameters in your model, which should be saved and restored
        # in the state_dict, but not trained by the optimizer,
        # you should register them as buffers.
        self.register_buffer("inv_pos_sin_freq", self.inv_freq)
        self.is_custom_position_ids = is_custom_position_ids
        self.enc_features = enc_features
        self.scale = scale
        self.eps = 1e-6

    def forward(
        self,
        inputs,
    ):
        """ Generate the position encoding for the inputs.

            Args:
                inputs (torch.tensor): a tensor with shape
                        batch_size, seq_length

            Outputs:
                encoded_positions (torch.tensor): a tensor with shape
                    batch_size, seq_length, enc_features
        """
        device = inputs.device

        batch_position_ids = process_1d_inputs(
            inputs,
            is_custom_position_ids=self.is_custom_position_ids,
            device=device)

        # compute the pos/10000**(2i/dmodel), i.e., pos * self.inv_freq
        #   Here are two ways to compute pos_div_freq,
        #    whose shape is batch_size, seq_len, enc_features/2

        self.inv_freq = self.inv_freq.to(device)

        pos_div_freq = torch.einsum('bn,d->bnd', batch_position_ids,
                                    self.inv_freq)

        # or
        # pos_div_freq = batch_position_ids[:, :, None] * self.inv_freq

        # compute the encoding of even positions as sin
        #   even_encoded shape: batch_size, seq_length, enc_features / 2
        even_encoded = pos_div_freq.sin()
        # compute the encoding of even positions as cos
        #   odd_encoded shape: batch_size, seq_length, enc_features / 2
        odd_encoded = pos_div_freq.cos()

        # stack the encodes to shape:
        #  batch_size, seq_length, enc_features / 2, enc_features / 2
        encoded_positions = torch.stack((even_encoded, odd_encoded), dim=3)
        # then flatten it to make: They are staggered:
        #   i.e., sin, cos, sin, cos...
        # shape: batch_size, seq_length, enc_features
        encoded_positions = encoded_positions.flatten(2)
        encoded_positions = encoded_positions.to(device)
        return encoded_positions


class SinusoidalPosition2DEncoder(PositionEncoding):
    """ The sin_pos_encoding that belongs to the absolute position embedding.
        The reason is described in https://zhuanlan.zhihu.com/p/105001610?utm_source=wechat_session.

        P E(pos,2i) = sin(pos/10000**(2i/dmodel))
        P E(pos,2i+1) = cos(pos/10000**(2i/dmodel))

        where pos is the position and i is the feature index.
        Inputs:
            embedding_features (int): the size of embedded features
            temperature: the temperature value
            scale: the normalization scale
    """

    def __init__(self,
                 enc_features,
                 temperature=10000.0,
                 scale=2 * math.pi,
                 is_custom_position_ids=False,
                 is_apply_scale=False,
                 device=None):
        super(SinusoidalPosition2DEncoder, self).__init__()

        # using a even enc features as
        #   half of the features is the sin coding.
        #   another half is the cos coding.
        #   They are staggered.
        #   Thus, the features looks like: [sin, cos, sin, cos, sin, cos, ...]
        # enc_features / 4 here because
        #  - Each features use half of enc_features, we have 2 dims
        #  - Within half enc_features, half of it contains sin while the other half
        #       contains cos
        if enc_features % 4 != 0:
            raise RuntimeError(
                "The input embedding_features has to be divisible by 2!")

        # compute the 1 / 10000**(2i/dmodel)
        # with enc_features 8, the correct inv_freq is
        #   tensor([1.0000, 0.1000, 0.0100, 0.0010])
        # shape of inv_freq is:
        #   enc_features / 2
        self.feature_indices = torch.arange(0,
                                            enc_features,
                                            2,
                                            dtype=torch.float32,
                                            device=device)
        self.inv_freq = 1.0 / (temperature
                               **(self.feature_indices / enc_features))

        # If you have parameters in your model, which should be saved and restored
        # in the state_dict, but not trained by the optimizer,
        # you should register them as buffers.
        self.register_buffer("inv_pos_sin_freq", self.inv_freq)
        self.is_custom_position_ids = is_custom_position_ids
        self.enc_features = enc_features
        self.is_apply_scale = is_apply_scale
        self.scale = scale
        self.eps = 1e-6

    def forward(
            self,
            inputs,
            is_mask_input=False,  # whether the input tensor is a mask.
    ):
        """ Generate the position encoding for the inputs.

            Args:
                inputs (torch.tensor): a tensor with shape
                        If is_custom_position_ids is True:
                         the expected shape is batch_size, height, width, 2
                         The user defined coords is placed in 2 as [h, w].
                         Check the .utils.py for how to create the required
                          custom_position_ids.
                        else:
                            batch_size, height, width

                        If is_mask_input, the inputs is a tensor containing the
                         mask identifying the positions mask. Then, we need to
                         obtain the position encoding based on the mask.
                          the shape of inputs: batch_size, height, width

                is_mask_input (Boolean): whether the inputs presents the mask used
                    for the position encoding.
            Outputs:
                encoded_positions (torch.tensor): a tensor with shape
                    batch_size, height, width, enc_features
        """
        device = inputs.device
        inputs.int()

        # obtain the
        #   - h_batch_position_ids shape: batch_size, height, width
        #   - w_batch_position_ids shape: batch_size, height, width
        if is_mask_input:
            not_mask = ~inputs
            h_batch_position_ids = not_mask.cumsum(1, dtype=torch.float32)
            w_batch_position_ids = not_mask.cumsum(2, dtype=torch.float32)
        else:
            h_batch_position_ids, w_batch_position_ids = process_2D_inputs(
                inputs,
                is_mask_input,
                is_custom_position_ids=self.is_custom_position_ids,
                device=device)

        if self.is_apply_scale:
            # find the maximum value along the specific dim, i.e., height, width
            #  Finding along the height, -> batch_size, 1, width
            #  Finding along the width, -> batch_size, height, 1
            max_batch_h_pos = torch.amax(h_batch_position_ids,
                                         dim=1,
                                         keepdim=True)
            max_batch_w_pos = torch.amax(w_batch_position_ids,
                                         dim=2,
                                         keepdim=True)

            h_batch_position_ids = h_batch_position_ids / (
                max_batch_h_pos + self.eps) * self.scale
            w_batch_position_ids = w_batch_position_ids / (
                max_batch_w_pos + self.eps) * self.scale

        self.inv_freq = self.inv_freq.to(device)
        # compute the pos/10000**(2i/dmodel), i.e.,  pos * self.inv_freq
        #   Here are two ways to compute this,
        #   whose shape is batch_size, height, width, enc_features / 2
        # h_pos_div_freq = torch.einsum('bn,d->bnd', h_batch_position_ids,
        #                               self.inv_freq)
        # w_pos_div_freq = torch.einsum('bn,d->bnd', w_batch_position_ids,
        #                               self.inv_freq)
        # or
        h_pos_div_freq = h_batch_position_ids[:, :, :, None] * self.inv_freq
        w_pos_div_freq = w_batch_position_ids[:, :, :, None] * self.inv_freq

        # compute the encoding of even height positions as sin
        #   h_even_encoded shape: batch_size, height, width, enc_features / 4
        h_even_encoded = h_pos_div_freq[:, :, :, 0::2].sin()
        # compute the encoding of even height positions as cos
        #   odd_encoded shape: batch_size, height, width, enc_features / 4
        h_odd_encoded = h_pos_div_freq[:, :, :, 1::2].cos()

        # likewise for width
        w_even_encoded = w_pos_div_freq[:, :, :, 0::2].sin()
        w_odd_encoded = w_pos_div_freq[:, :, :, 1::2].cos()

        # stack the height pos encodes to shape:
        # batch_size, height, width, enc_features / 4, enc_features / 4
        h_encoded_positions = torch.stack((h_even_encoded, h_odd_encoded),
                                          dim=4)
        # then flatten it to make: They are staggered:
        #   i.e., sin, cos, sin, cos...
        # shape: batch_size, height, width, enc_features / 2
        h_encoded_positions = h_encoded_positions.flatten(3)

        # likewise for width
        w_encoded_positions = torch.stack((w_even_encoded, w_odd_encoded),
                                          dim=4)
        w_encoded_positions = w_encoded_positions.flatten(3)

        # concat these two and obtain the final encoded_positions:
        #   batch_size, height, width, enc_features
        encoded_positions = torch.cat(
            (h_encoded_positions, w_encoded_positions), dim=3)
        encoded_positions = encoded_positions.to(device)
        return encoded_positions


if __name__ == "__main__":
    # Test
    abs_pos_encoder = BasicPositionEncoder(enc_position_number=10,
                                           enc_features=5,
                                           is_custom_position_ids=True)
    abs_pos_encoder.build_position_encoding_pool()
    custom_pos_ids = torch.tensor([[2, 4, 6], [0, 3, 1]])
    custom_pos_enc = abs_pos_encoder(custom_pos_ids)
    print(custom_pos_enc)

    abs_pos_encoder2 = BasicPositionEncoder(enc_position_number=10,
                                            enc_features=5,
                                            is_custom_position_ids=False)
    abs_pos_encoder2.build_position_encoding_pool()
    custom_pos_ids = torch.tensor([[2, 4, 6], [0, 3, 1]])
    custom_pos_enc = abs_pos_encoder2(custom_pos_ids)
    print(custom_pos_enc)

    def positionalencoding1d(model_features, length):
        """
        :param model_features: #features of the model
        :param length: length of positions
        :return: length*model_features position matrix
        """
        if model_features % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd #features (got features={:d})".format(model_features))
        pe = torch.zeros(length, model_features)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, model_features, 2, dtype=torch.float) *
             -(math.log(10000.0) / model_features)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    positionalencoding1d(8, 3)

    # Test
    # output_dim = 8
    # indices = torch.arange(0, output_dim // 2, dtype=torch.float32)
    # indices = torch.pow(10000.0, -2 * indices / output_dim)

    # git_indices = torch.arange(0, output_dim, 2, dtype=torch.float32)
    # inv_freq = 1.0 / (10000.0**(git_indices / output_dim))

    # my_indices = torch.arange(0, output_dim, 2, dtype=torch.float32)
    # my_inv_freq = 1.0 / (10000.0**(my_indices / output_dim))

    # other_div_term = torch.exp(
    #     (torch.arange(0, output_dim, 2, dtype=torch.float32) *
    #      -(math.log(10000.0) / output_dim)))

    # print("from bert4: ", indices)
    # print("from git: ", inv_freq)
    # print("from my_inv_freq: ", my_inv_freq)
    # print("from other_div_term: ", other_div_term)
    # abs_pos_encoder = SinusoidalPosition1DEncoder(enc_features=8,
    #                                               is_custom_position_ids=True)
    # custom_pos_ids = torch.tensor([[2, 4, 6], [0, 3, 1]])
    # custom_pos_enc = abs_pos_encoder(custom_pos_ids)
    # print(custom_pos_enc)

    # abs_pos_encoder2 = SinusoidalPosition1DEncoder(
    #     enc_features=8, is_custom_position_ids=False)
    # custom_pos_ids = torch.tensor([[2, 4, 6], [0, 3, 1]])
    # custom_pos_enc = abs_pos_encoder2(custom_pos_ids)
    # print(custom_pos_enc)
