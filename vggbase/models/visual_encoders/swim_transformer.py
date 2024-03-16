"""
The swim transformer is utilized to extract 
features from the image.

Swin transformer model on torchvision can support 
different input sizes, for example:

m = torchvision.models.swin_t(weights="DEFAULT")

x1 = torch.rand((1, 3, 224, 224))
x2 = torch.rand((1, 3, 150, 200))
x3 = torch.rand((1, 3, 123, 173))

print(m.features(x1).shape)  # torch.Size([1, 7, 7, 768])
print(m.features(x2).shape)  # torch.Size([1, 5, 7, 768])
print(m.features(x3).shape)  # torch.Size([1, 4, 6, 768])


The output of this encoder will be:
"1": for stage 1
"2": for stage 2
"3": for stage 3
"4": for stage 4

"""

from typing import Optional, List, Callable, Dict
import logging
from collections import OrderedDict

from torch import nn, Tensor
from torchvision.models import swin_transformer
from torchvision.models import swin_t, swin_b, swin_s
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models import (
    Swin_T_Weights,
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from vggbase.models.visual_encoders.base_encoder import BaseRGBEncoder


swim_transformer_models_pool = {
    "swin_t": swin_t,
    "swin_b": swin_b,
    "swin_s": swin_s,
    "swin_v2_t": swin_v2_t,
    "swin_v2_s": swin_v2_s,
    "swin_v2_b": swin_v2_b,
}


swim_trans_weights_pool = {
    "swin_t": Swin_T_Weights,
    "swin_b": Swin_B_Weights,
    "swin_s": Swin_S_Weights,
    "swin_v2_t": Swin_V2_T_Weights,
    "swin_v2_s": Swin_V2_S_Weights,
    "swin_v2_b": Swin_V2_B_Weights,
}


def stage_to_layers(stage_i):
    """map the stage number to layer."""
    return range(2 * (stage_i - 1), 2 * stage_i)


class SwimTransformerBackboneWithFPN(BackboneWithFPN):
    """An interface of BackboneWithFPN for swim transformer."""

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward the model.

        :return x: A `OrderedDict` in which keys are the
         stage's index.
         "1": for stage 1, [bs, out_channels, H1, W1]
         "2": for stage 2, [bs, out_channels, H2, W3]
         "3": for stage 3, [bs, out_channels, H3, W4]
         "4": for stage 4, [bs, out_channels, H4, W4]
        """
        # x will be a OrderedDict
        x = self.body(x)
        # convert
        # from [bs, H, W, C]
        # to [bs, C, H, W]
        for layer_name, out in x.items():
            x[layer_name] = out.permute(0, 3, 1, 2)
        x = self.fpn(x)
        return x


def swimtransformer_fpn_extractor(
    swim_trans_backbone: swin_transformer.SwinTransformer,
    trainable_stages: int,
    returned_stages: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    out_channels: int = 256,
) -> SwimTransformerBackboneWithFPN:
    """Extract features with swim transformers and FPN.

    The structure of swim transormers in torchvision is:
    0, 1, 2, 3, 4, 5, 6, 7
    in which `0` layer is patch_embed
        stage1:
    and `0` and `1` layers have the same out channels
        stage2:
        `2` and `3` layers have the same out channels
        stage3:
        `4` and `5` layers have the same out channels
        stage4:
        `6` and `7` layers have the same out channels
        each pais can be called `stage`
    we only utilize the first 4 stages when possible
    """

    # select layers that wont be frozen
    if trainable_stages < 0 or trainable_stages > 4:
        raise ValueError(
            f"Trainable stages should be in the range [0,4], got {trainable_stages}"
        )
    stages_to_train = ["4", "3", "2", "1"][:trainable_stages]
    if trainable_stages == 4:
        stages_to_train.append("bn1")

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    basic_stages = [1, 2, 3, 4]
    if returned_stages is None:
        returned_stages = basic_stages
    if not all([layer_i in basic_stages for layer_i in returned_stages]):
        raise ValueError(
            f"Each returned stage should be in [1, 2, 3, 4]. Got {returned_stages}"
        )
    return_layers = {
        str(stage_to_layers(k)[-1]): str(v + 1) for v, k in enumerate(returned_stages)
    }

    # get the layers for features
    feature_layers = swim_trans_backbone.features
    # obtain embedding dim from patch_embed
    embed_dim = feature_layers[0][0].out_channels

    for name, parameter in feature_layers.named_parameters():
        if all([not name.startswith(layer) for layer in stages_to_train]):
            parameter.requires_grad_(False)

    in_channels_list = [int(embed_dim * 2 ** (i - 1)) for i in returned_stages]

    return SwimTransformerBackboneWithFPN(
        feature_layers,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer,
    )


class SwimTransformerEncoder(BaseRGBEncoder):
    """The encoder for extracting features for Rois from one feature map."""

    def __init__(
        self,
        rgb_encoder_config,
    ):
        super().__init__(rgb_encoder_config)

        weights = self.extract_weights(weights_pool=swim_trans_weights_pool)
        # load a pre-trained backbone
        swim_trans = swim_transformer_models_pool[self.encoder_name](weights=weights)
        self.encoder = swimtransformer_fpn_extractor(
            swim_trans,
            trainable_stages=self.trainable_layers,
            returned_stages=self.extract_layers,
        )
        self.n_encoded_channels = self.encoder.out_channels

        logging.info(
            "Build the backbone %s whose output #channels is %d",
            self.encoder_name,
            self.encoder.out_channels,
        )
