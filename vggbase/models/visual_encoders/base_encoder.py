"""
The implementation of basic encoder.
"""

from typing import Type
import logging

from torch import nn

from vggbase.config import Config

BaseDatasets = {"imagenet-1k": "IMAGENET1K"}


class BaseRGBEncoder(nn.Module):
    """Base class for the visual (RGB) encoder."""

    def __init__(self, rgb_encoder_config: Type[Config]) -> None:
        super().__init__()
        self.encoder_name = rgb_encoder_config.name
        self.encoder_dataset = rgb_encoder_config.dataset
        self.encoder_weights_version = rgb_encoder_config.weights_version
        self.extract_layers = rgb_encoder_config.extract_layers
        self.trainable_layers = rgb_encoder_config.trainable_layers

        self.encoder_weights_name = self.get_weights_name(
            self.encoder_dataset, self.encoder_weights_version
        )

        # default to be an identity function
        self.encoder = nn.Identity()

        # by default, the rgb data will not be processed
        # thus maintaining 3 channels
        self.n_encoded_channels = 3

    def freeze_parameters(self):
        """Freeze parameters of the language embedder."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_weights_name(self, encoder_dataset, encoder_weights_version):
        """Initilize weights of the encoder."""
        return BaseDatasets[encoder_dataset] + "_" + encoder_weights_version

    def extract_weights(self, weights_pool):
        """Extract weights with the desired  weights_name"""

        encoder_weights = weights_pool[self.encoder_name]
        if hasattr(encoder_weights, self.encoder_weights_name):
            encoder_weights = getattr(encoder_weights, self.encoder_weights_name)
        else:
            logging.info("encoder (%s) with pre-trained weights (%s) is not existed.")

        return encoder_weights

    def forward(self, x_rgbs):
        """Forward the image to obtain feature map for subsequent learning.

        :param x_rgbs: A `torch.Tensor` with shape [batch_size, 3, H, W] holding
         one batch of image data.

        :return the combination of `trainable_layers` and `extra_blocks` in whihc
         `extra_blocks` is the last MaxPool by default.
         Therefore, the return dict contains: *trainable_layers, 'pool'
         where `pool` will always outputed.

        """
        return self.encoder(x_rgbs)
