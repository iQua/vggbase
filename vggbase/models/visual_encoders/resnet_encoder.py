"""
The ResNet encoder utilized to extract features from the image.

See models zoo on "https://pytorch.org/vision/stable/models.html".

For resnet, 
    conv2_x => layer1
    conv3_x => layer2
    conv4_x => layer3
    conv5_x => layer4
"""

import logging

from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from vggbase.models.visual_encoders.base_encoder import BaseRGBEncoder

resnet_weights_pool = {
    "resnet18": ResNet18_Weights,
    "resnet50": ResNet50_Weights,
}


class ResNetEncoder(BaseRGBEncoder):
    """The encoder for extracting features for Rois from one feature map."""

    def __init__(
        self,
        rgb_encoder_config,
    ):
        super().__init__(rgb_encoder_config)

        weights = self.extract_weights(weights_pool=resnet_weights_pool)
        # load a pre-trained backbone as the encoder
        # this encoder outputs
        #  a `OrderedDict` presenting as
        #  layer_idx: `troch.FloatTensor` while the final feature is
        #  pool: `troch.FloatTensor`.
        # thus its keys are: [*extract_layers, 'pool']
        self.encoder = resnet_fpn_backbone(
            backbone_name=self.encoder_name,
            weights=weights,
            trainable_layers=self.trainable_layers,
            returned_layers=self.extract_layers,
        )
        self.n_encoded_channels = self.encoder.out_channels

        logging.info(
            "Build the backbone %s whose output #channels is %d",
            self.encoder_name,
            self.encoder.out_channels,
        )
