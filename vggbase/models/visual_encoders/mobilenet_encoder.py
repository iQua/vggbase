"""
The encoder built upon the mobilenet to extract features from the image.

"""

import logging


from torchvision.models import MobileNet_V2_Weights
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

from torchvision.models.detection.backbone_utils import mobilenet_backbone

from vggbase.models.visual_encoders.base_encoder import BaseRGBEncoder

mobilenet_weights_pool = {
    "mobilenetv2": MobileNet_V2_Weights,
    "mobilenetv3_small": MobileNet_V3_Small_Weights,
    "mobilenetv3_large": MobileNet_V3_Large_Weights,
}


class MobileNetEncoder(BaseRGBEncoder):
    """The encoder for extracting features for Rois from one feature map."""

    def __init__(
        self,
        rgb_encoder_config,
    ):
        super().__init__(rgb_encoder_config)

        # load a pre-trained backbone and
        # freeze all layers
        # whether utilize the FPN
        self.backbone_fpn = rgb_encoder_config["is_fpn"]
        weights = self.extract_weights(weights_pool=mobilenet_weights_pool)

        self.encoder = mobilenet_backbone(
            backbone_name=self.encoder_name,
            fpn=self.backbone_fpn,
            weights=weights,
            trainable_layers=self.trainable_layers,
            returned_layers=self.extract_layers,
        )
        self.n_encoded_channels = self.encoder.out_channels
        logging.info(
            "Build the backbone (%s) with weights (%s) whose output #channels is %d",
            self.encoder_name,
            self.encoder_weights_name,
            self.encoder.out_channels,
        )
