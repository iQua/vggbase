"""

Relative Position Encodings are a type of position embeddings for Transformer-based models
 that attempts to exploit pairwise, relative positional information.
 * Relative positional information is supplied to the model on two levels: values and keys.

Advantages:
 - no limitations to the number of tokens a model can process
 - generalize to sequences of unseen lengths, since theoretically the
  only information it encodes is the relative pairwise distance between two tokens.


The relative position encoding,
    - RelativePositionEncoder from the paper:
        https://arxiv.org/abs/1803.02155.
        They introduced a way of using pairwise distances as a way of
         creating positional encodings

Access a comprehensive relative position encoding in the paper:
    https://arxiv.org/pdf/2107.14222.pdf
    with github https://github.com/ICCV21/2D-RPE

To verify the correcness of our position encoding part.
 We utilize the cross-validation of three sources, including:
    - https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py
    - relative position embedding of swim transformer part


references:
 [1]. https://paperswithcode.com/method/relative-position-encodings
 [2]. https://medium.com/@_init_/how-self-attention-with-relative-\
     position-representations-works-28173b8c245a

Denoting relative shift number to be K.
 - Why we call it relative shift number:
    With current position as the baseline, we shift with number K to both left and right of this
    position.

Denoting number of positions (or seqence length) to be Q.

Note: one important thing about the relative position is the bias index.
    The 0 index of the bias is the largest bias in the left side (i.e., -K)  of the
     position i, i.e., i-K.
     position index : i-K, i-K+1, i-K+2, ..., i-1, i, i+1, ..., i+K-2, i+K-1, i+K
     bias index     :  0 ,   1  ,   2  , ..., K-1, K, K+1, ..., K+K-2, K+K-1, K+K
     making the bias table length to be 2*K+1.

     For example, when we have 5 postions (A-E) and the relative shift number
      is 3. For the position with index 4 and 0, its bias index are:
        positions       : A, B, C, D, E
        pos index       : 0, 1, 2, 3, 4
        For position 4:
        bias index      : 0, 0, 1, 2, 3
        For position 0:
        bias index      : 3, 4, 5, 6, 6
    Please access the figure in refer [2].

    In summart, with 0 as the starting index of bias, the bias index of the position
     itself is always K. Such as the example above,
      - For pos 4, bias index of pos 4 is 3 == K.
      - For pos 0, bias index of pos 0 is 3 == K.

    In general, we hope the considered relative positions cover the full sequence with length Q.
        Thus, the largest bias K is Q - 1, making the pos index Q - 1 can reach the starting
         index 0 of the sequence. Then, the bias length is 2 * (Q-1) + 1.

        Thus, we can also utilize the sequence length Q directly to compute the bias length:
            2 * (Q-1) + 1 = 2 * Q - 1.
"""

import torch
import torch.nn as nn

from .base import PositionEncoding

from .utils import process_1d_inputs, process_2D_inputs


def consistency_verify(relative_shift_numbers, position_numbers):
    """ Judge whether the two items satisfy the description above.
        That is to say, when we want to encode the relative position
         for the full positions, the relative_shift_numbers (if
         defined) satisfy the requirement:

            - the bias length: 2*K+1 = 2*Q-1
         or - K = Q - 1
        Args:
            relative_shift_numbers (int or list): K
            position_numbers (int or list): Q
    """

    def verify_number(relative_shift_number, position_number):
        if relative_shift_number != -1 \
            and position_number > 0:

            assert 2 * relative_shift_number + 1 \
                    == 2 * position_number - 1

    relative_shift_numbers = [relative_shift_numbers] if isinstance(
        relative_shift_numbers, int) else relative_shift_numbers
    position_numbers = [position_numbers] if isinstance(
        position_numbers, int) else position_numbers

    for idx in range(len(relative_shift_numbers)):
        relative_shift_number = relative_shift_numbers[idx]
        position_number = position_numbers[idx]
        verify_number(relative_shift_number, position_number)


class RelativePosition1DEncoder(PositionEncoding):
    """ The relative position encoder for the 1D input.

        Args:
            enc_relative_shift_number (int): the relative shift positions number
                to be encoded, i.e., the K in reference [2].
                Then, the relative bias number is 2 * enc_relative_shift_number + 1
                Default -1 for invaild, thus not setting this number.
            enc_seq_length (int): the seq_length to be encoded by the relative positions.
                It is denoted as Q mentioned in the description above.
                Then, the relative bias number is 2 * Q - 1.

            * Mandatory setting enc_relative_shift_number or enc_seq_length.

            enc_features (int): the encoding #features.
            is_custom_position_ids (boolean): whether the input ids are defined by
                the user.
                    If True, the value of the input tensor is the position idex.
                    Otherwise, the value of the input tensor makes no sense, thus
                        we need to genrate position idex for items.

        Note: why we define two params: enc_relative_shift_number & enc_seq_length
            Condition 1: The user only know the shift number (K) to be considered for
                the encoding of each position, thus setting enc_relative_shift_number.
            Condition 2: The user only know the number of positions Q (i.e., seq_length) to
                be encoded and expect the relative position can cover the full position,
                That is to say, the corresponding relative_shift_number K should
                be set up to be accessible from one end of the sequence to the other，
                i.e., K = Q - 1.

     """

    def __init__(self,
                 enc_features,
                 enc_relative_shift_number=-1,
                 enc_seq_length=0,
                 is_custom_position_ids=False):
        super(RelativePosition1DEncoder, self).__init__()

        self.enc_features = enc_features
        self.is_custom_position_ids = is_custom_position_ids

        # position_bias_number is a total number of biases for one postion.

        #   With the relative_shift_number (K), it should include the
        #       - left side bias
        #       - right side bias
        #       - itself.
        if enc_relative_shift_number != -1:
            self.position_bias_number = 2 * enc_relative_shift_number + 1
            self.enc_relative_shift_number = enc_relative_shift_number
        #   With the enc_seq_length (Q), the relative_shift_number K should
        #    be set up to be accessible from one end of the sequence to the other，
        #    i.e., K = Q - 1.
        elif enc_seq_length > 0:
            self.position_bias_number = 2 * enc_seq_length - 1
            self.enc_relative_shift_number = enc_seq_length - 1
        else:
            raise RuntimeError(
                "The enc_relative_shift_number or enc_seq_length \
                    is required to be defined,")

        # if both two items are defined
        consistency_verify(enc_relative_shift_number, enc_seq_length)

        self.encoding_pool_capacity = self.position_bias_number

    def obtain_relative_position_bias_index_table(self, batch_position_ids):
        """ Obtain the relative position bias index table.

            Args:
                batch_position_ids (torch.tensor): a tensor with shape
                    batch_size, seq_length, containing the ids of positions
                    for computing the relative bias

                positions_bias_index: (torch.tensor): a tensor
                    with shape batch_size, seq_length, seq_length
        """

        # copy the position ids to generate another base positon ids for
        #   the further relative positions computation
        base_batch_position_ids = torch.clone(batch_position_ids)
        # add newaxis in different dimension
        base_batch_position_ids = base_batch_position_ids[:, None, :]
        batch_position_ids = batch_position_ids[:, :, None]

        # compute the relative biases:
        #   batch_size, seq_len, seq_len
        relative_positions = base_batch_position_ids - batch_position_ids

        # for each matrix seq_len, seq_len in relative_positions[i, :, :]
        #   row index j denotes the position presented by j-th value (v_j) in the position_ids
        #   column index m denotes the position presented by m-th value (v_m) in the position_ids
        #   the value of each item in this matrix is the relative position between v_j and v_m,
        #    computed as v_j - v_m
        # thus the value of this matrix ranges from:
        #   -seq_len to +seq_len
        #  Note: the relative position between j with itself is 0.

        # However, in the relative encoding, we want to set index 0
        #   to bias v_j - enc_relative_shift_number. Thus, the bias index
        #   between j and itself is enc_relative_shift_number.

        # then shift the relative positions:
        shifted_relative_positions = relative_positions + self.enc_relative_shift_number

        #  (the current value only present the relative positions for positions, which
        #  cannot utilize by us to extract the encoding from the created encoding_pool as
        #  the index of encoding_pool ranges from 0 to self.position_bias_number.)

        # Therefore, in this stage, we are going to convert the value of shifted_relative_positions
        #   to that can be used in the encoding_pool.
        # In the 1D encoder here, we only need to limit the values of shifted_relative_positions,
        #   so that if the shifted values < 0, they are limitted to 0.
        #           if the shifted values >= position_bias_number (length of the encoding_pool),
        #           they are limitted to position_bias_number - 1.

        # We need to clipp the values to make element of shifted_relative_positions
        #   range from 0 to self.position_bias_number - 1
        #    where we minus 1 here because we need to obtain the index that
        #    starts from 0.
        # to check the correcness of positions_bias_index,
        # just set enc_relative_shift_number to 3, the the seq_length to 10.
        # the code compute a same matrix as the one in refer [2]
        positions_bias_index = torch.clamp(shifted_relative_positions,
                                           min=0,
                                           max=self.position_bias_number - 1)

        return positions_bias_index

    def forward(self, inputs):
        """ Generate the position encoding for the inputs.

            Args:
                inputs (torch.tensor): a tensor with shape
                        batch_size, seq_length

            Outputs:
                encoded_relative_positions (torch.tensor): a tensor with shape
                    batch_size, seq_length, seq_length, enc_features
        """
        batch_size, seq_len = inputs.shape

        # process the input to obtain the batch_position_ids
        #  with shape batch_size, seq_len.
        batch_position_ids = process_1d_inputs(
            inputs, is_custom_position_ids=self.is_custom_position_ids)

        # obtain the bias table with shape batch_size, seq_len, seq_len
        position_bias_index_table = self.obtain_relative_position_bias_index_table(
            batch_position_ids)

        # encode the relative position encoding with shape
        #   batch_size, seq_len, seq_len, enc_features
        encoded_relative_positions = self.encoding_pool(
            position_bias_index_table)

        return encoded_relative_positions


class RelativePosition2DEncoder(PositionEncoding):
    """ The relative position encoder for the 2D input.

        Args:
            enc_relative_shift_numbers (list): the relative shift positions number
                to be encoded. i.e., the K in reference [2].
                It contains the shift number along height and width direction, i.e.,
                 K_h and K_w, making a list input [K_h, K_w].

                Then, the relative bias number
                    - along height is 2 * K_h + 1
                    - along width is 2 * K_w + 1

            enc_pos_numbers (list): the position number to be encoded by the relative positions.
                It contains the number of positions along height and width direction, i.e.,
                 Q_h, Q_w.
                Then, the relative bias number is
                    - along height 2 * Q_h - 1
                    - along width 2 * Q_w - 1

            enc_features (int): the encoding #features.
            is_custom_position_ids (boolean): whether the input ids are defined by
                the user.
                    If True, the value of the input tensor is the position idex.
                    Otherwise, the value of the input tensor makes no sense, thus
                        we need to genrate position idex for items.
     """

    def __init__(self,
                 enc_features,
                 enc_relative_shift_numbers=[-1, -1],
                 enc_pos_numbers=[0, 0],
                 is_custom_position_ids=False):
        super(RelativePosition2DEncoder, self).__init__()
        #   With the relative_shift_number (K), it should include the
        #       - left side bias
        #       - right side bias
        #       - itself.
        if enc_relative_shift_numbers[0] != -1:
            self.enc_relative_h_shift_number = enc_relative_shift_numbers[0]
            self.enc_relative_w_shift_number = enc_relative_shift_numbers[1]
            self.h_bias_number = 2 * self.enc_relative_h_shift_number + 1
            self.w_bias_number = 2 * self.enc_relative_w_shift_number + 1

        #   With the enc_seq_length (Q), the relative_shift_number K should
        #    be set up to be accessible from one end of the sequence to the other，
        #    i.e., K = Q - 1.
        elif enc_pos_numbers[0] > 0:
            self.enc_h_pos_numbers = enc_pos_numbers[0]
            self.enc_w_pos_numbers = enc_pos_numbers[1]
            self.enc_relative_h_shift_number = self.enc_h_pos_numbers - 1
            self.enc_relative_w_shift_number = self.enc_w_pos_numbers - 1
            self.h_bias_number = 2 * self.enc_h_pos_numbers - 1
            self.w_bias_number = 2 * self.enc_w_pos_numbers - 1
        else:
            raise RuntimeError(
                "The enc_relative_shift_numbers or enc_pos_numbers \
                    is required to be defined,")

        # the reason why the length here is h_bias_number * w_bias_number
        #   is: we need to extend the relative bias along both the height and width directions
        #       to generate the encoding table.
        #   For example, for one position A with:
        #       - relative_h_shift_number: 2
        #       - relative_w_shift_number: 2
        #       the total bias along height of A is 2 * 2 + 1 - i.e., h_bias_number
        #       the total bias along width of A is 2 * 2 + 1 - i.e., w_bias_number
        #       Then, it looks like:
        #                     x
        #                     x
        #                 x x A x x
        #                     x
        #                     x
        #       Thus, finally the total 2D relative bias matrix is:
        #                 x x x x x
        #                 x x x x x
        #                 x x A x x
        #                 x x x x x
        #                 x x x x x
        #       There are 25 = (2 * 2 + 1) * (2 * 2 + 1)

        self.coord_bias_number = self.h_bias_number * self.w_bias_number
        self.is_custom_position_ids = is_custom_position_ids

        self.enc_features = enc_features

        consistency_verify(enc_relative_shift_numbers, enc_pos_numbers)

        self.encoding_pool_capacity = self.coord_bias_number

    def obtain_relative_position_bias_index_table(self, h_batch_position_ids,
                                                  w_batch_position_ids):
        """ Obtain the relative coords bias index table.

            Args:
                h_batch_position_ids (torch.tensor): a tensor with shape
                    batch_size, height, width, containing the ids of positions
                    along the height direction
                w_batch_position_ids (torch.tensor): a tensor with shape
                    batch_size, height, width, containing the ids of positions
                    along the width direction

                coords_bias_index: (torch.tensor): a tensor
                    with shape batch_size, height * width, height * width
                    the value in coords_bias_index[i, j, m] is the bias index
                    between the position j and m (note as we flatten the matrix
                    to one vector, each position actually corresponds to specific
                    coord). This bias index can access the encoding in the defined
                    encoding pool!!
        """
        # concat the h, w position ids to generate the coords with shape:
        #   batch_size, 2, height, widht
        batch_coords = torch.stack(
            tensors=[h_batch_position_ids, w_batch_position_ids], dim=1)

        # flatten the batch_coords to generate a list coords with shape:
        #   batch_size, 2, height * width
        # thus, batch_size, 0, height * width contains the position idx along
        #   the height direction.
        #       batch_size, 1, height * width contains the position idx along
        #   the width direction.
        batch_flatten_coords = torch.flatten(batch_coords,
                                             start_dim=2,
                                             end_dim=-1)

        # copy the coords to generate another base coords for
        #   the further relative coords computation
        base_batch_flatten_coords = torch.clone(batch_flatten_coords)
        # add newaxis in different dimension
        # base to :  batch_size, 2, 1, height * width
        base_batch_flatten_coords = base_batch_flatten_coords[:, :, None, :]
        # to batch_size, 2, height * width, 1
        batch_flatten_coords = batch_flatten_coords[:, :, :, None]

        # compute the relative coords biases:
        #   batch_size, 2, height * width, height * width
        # We use the broadcasting trick to achieve this.
        relative_coords = batch_flatten_coords - base_batch_flatten_coords

        # change shape to batch_size, height * width, height * width, 2
        # Then, for the changed tensor, I give a detailed description below:
        #   First, the mearnings of dimensions are:
        #    - relative_coords[i, :, :, :], relative bias matrix for the batch i
        #    - relative_coords[i, j, :, :], the relative bias matrix for j-th position
        #       of the batch i, Note: the h,w of each coords is flatted, thus
        #       the total number of positions for a 2D input is height * width. Then,
        #       for each position, such as j, we can compute its corresponding coord.
        #    - relative_coords[i, j, m, :], in batch i, the relative position between
        #       j-th position and m-position. As each position in this flatten dim
        #       corresponds to its coord, this is actually the relative bias between
        #       two coords.
        #    - relative_coords[i, j, m, 0], the relative bias along the height direction.
        #    - relative_coords[i, j, m, 1], the relative bias along the width direction.

        relative_coords = relative_coords.permute(0, 2, 3, 1).contiguous()

        # In this stage, we need to convert the shifted values to the
        #   index of encoding pool for further encoding based on index.

        # For two coords corresponding to positions j and m,
        #   Along the height, thus can be viewed as the 1D case,
        #       their relative position follows the description in 1D encoder part.
        #   Along the width, thus can be viewed as the 1D case,
        #       their relative position follows the description in 1D encoder part.

        # To make useful relative position index for further encoding, we
        # First need to shift the relative position in both height and width
        relative_coords[:, :, :, 0] += self.enc_relative_h_shift_number
        relative_coords[:, :, :, 1] += self.enc_relative_w_shift_number

        # Second, we clipped the values to make them range:
        #   0, self.h_bias_number that is the length of the
        #   bias in the height direction.
        #
        relative_coords[:, :, :, 0] = torch.clamp(relative_coords[:, :, :, 0],
                                                  min=0,
                                                  max=self.h_bias_number - 1)
        relative_coords[:, :, :, 1] = torch.clamp(relative_coords[:, :, :, 1],
                                                  min=0,
                                                  max=self.w_bias_number - 1)

        # Finally, different from the 1D case that stop here, the 2D case requires
        #   us to further process the relative coords to correspond to the index
        #   of the encoding_pool containing items with number of:
        #    coord_bias_number = h_bias_number * w_bias_number

        #   Then, the problem is pretty easy as it can be converted to:
        #    Support we have a matrix with shape P * Q. From matrix item 0, 0 to p, q,
        #     there contains how many items?
        #       Answer: p * Q + q

        # Therefore, we can convert the relative position along the height direction
        #   to the corresponding bias index in the encoding pool by:
        relative_coords[:, :, :, 0] *= self.w_bias_number
        coords_bias_index = relative_coords.sum(-1)

        return coords_bias_index

    def forward(self, inputs, is_mask_input=False):
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
                    for the position encoding. Default: False

            Outputs:
                encoded_relative_coords (torch.tensor): a tensor with shape
                    batch_size, height * width, height * width, enc_features
        """
        batch_size, height, width = inputs.shape

        # obtain the
        #   - h_batch_position_ids shape: batch_size, height, width
        #   - w_batch_position_ids shape: batch_size, height, width
        h_batch_position_ids, w_batch_position_ids = process_2D_inputs(
            inputs,
            is_mask_input=is_mask_input,
            is_custom_position_ids=self.is_custom_position_ids)

        # obtain the bias table with shape batch_size, height * width, height * width
        coords_bias_index_table = self.obtain_relative_position_bias_index_table(
            h_batch_position_ids, w_batch_position_ids)

        # encode the relative position encoding with shape
        #   batch_size, height * width, height * width, enc_features
        encoded_relative_coords = self.encoding_pool(coords_bias_index_table)

        return encoded_relative_coords


if __name__ == "__main__":
    rela_pos_encoder = RelativePosition1DEncoder(enc_features=6,
                                                 enc_relative_shift_number=3,
                                                 is_custom_position_ids=False)
    rela_pos_encoder.build_position_encoding_pool()
    rela_pos_encoder(inputs=torch.zeros((2, 10)))
