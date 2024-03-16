"""

Base class for the position encoding.

"""

from abc import abstractmethod
import torch
import torch.nn as nn


class PositionEncoding(nn.Module):
    """
    Position Encoding for 1D, 2D, and 3D datasets.
    """

    def __init__(self):
        super(PositionEncoding, self).__init__()
        # the encoding features
        self.enc_features = None
        # whether the input ids are defined by the user.
        #  If True, the value of the input tensor is the position idxs
        #   required to be encoded.
        #  Otherwise, the value of the input tensor makes no sense and
        #   only the shape of the input tensor guides the encoding part,
        #   such as defined the encoding pool length.
        self.is_custom_position_ids = None

        self.encoding_pool_capacity = None

    def build_position_encoding_pool(self,
                                     pretrained_enc_weights=False,
                                     init_name="xavier",
                                     mean=0,
                                     std=1):
        """ Build the position encoding pool to be used for
            input encoding. """

        # initial trainable encoding pool
        self.encoding_pool = nn.Embedding(
            num_embeddings=self.encoding_pool_capacity,
            embedding_features=self.enc_features,
            max_norm=True)
        if pretrained_enc_weights:
            self.encoding_pool.weight.data.copy_(pretrained_enc_weights)
        else:
            if init_name == "xavier":
                nn.init.xavier_normal_(self.encoding_pool.weight.data)
            if init_name == "trunc_normal":
                nn.init.trunc_normal_(self.encoding_pool.weight.data,
                                      mean=mean,
                                      std=std)
