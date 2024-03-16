"""
The encoder utilized to extract features from the image.

"""

import logging

from .resnet_encoder import ResNetEncoder
from .mobilenet_encoder import MobileNetEncoder
from .swim_transformer import SwimTransformerEncoder


encoders_pool = {
    "resnet": ResNetEncoder,
    "mobilenet": MobileNetEncoder,
    "swim_transformer": SwimTransformerEncoder,
}


def build_encoder(encoder_name: str, encoder_config: dict):
    """Get the desired rgb encoder."""
    logging.info("Setup the encoder with [%s] structure", encoder_name)
    return encoders_pool[encoder_name](encoder_config)
