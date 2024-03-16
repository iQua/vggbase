"""
The implementation of the executor for diffusion visual grounding.

"""

from diffusionvg_model import build_model

from vggbase.models import BasicVaQExecutor
from vggbase.models.model_generic import BaseVGModelInput
from vggbase.models.language.language_embedding import LanguageEmbeddingModule
from vggbase.models.visual_encoders import build_encoder


class DiffusionVGExecutor(BasicVaQExecutor):
    """This is a executor the grounding models."""

    def __init__(self, model_config: dict, **kwargs):
        super().__init__(model_config, **kwargs)

        # Get the number of features from the language module
        self.text_n_features = None
        # Get the number of channels from the rgb module
        self.rgb_n_channels = None

    def init_language_module(self, model_config: dict, **kwargs):
        """Initialize the RGB module."""

        language_module_config = model_config.language
        language_module = LanguageEmbeddingModule(language_module_config)
        # Set the language model untrainable
        language_module.freeze_parameters()

        self.text_n_features = language_module.n_encoded_features

        return language_module

    def init_rgb_module(self, model_config: dict, **kwargs):
        """Initialize the RGB module."""
        rgb_module_config = model_config.rgb
        rgb_module_type = rgb_module_config.module_type

        # Build the rgb encoder
        rgb_module = build_encoder(rgb_module_type, rgb_module_config)
        # Set the rgb model untrainable
        rgb_module.freeze_parameters()

        # Get the output channels
        self.rgb_n_channels = rgb_module.n_encoded_channels

        return rgb_module

    def init_grounding_module(self, model_config: dict, **kwargs):
        """On building the grounding module."""
        return build_model(
            model_config,
            text_n_features=self.text_n_features,
            rgb_n_channels=self.rgb_n_channels,
        )

    def resume_module(self, model_config, **kwargs):
        """Skip as nothing to resume."""
        pass

    def forward(self, inputs: BaseVGModelInput, **kwargs):
        """The forward step."""
        rgb_samples = inputs.rgb_samples
        text_samples = inputs.text_samples
        targets = inputs.targets

        # forward the language module
        x_text = self.language_module(text_samples.tensors, text_samples.mask)
        x_rgbs = self.rgb_module(rgb_samples.tensors.float())

        phase = "train" if "phase" not in kwargs else kwargs["phase"]

        if phase == "train":
            # forward the grounding module for training
            gt_vg_bboxes = [sample_target.vg_bboxes for sample_target in targets]

            return self.grounding_module(
                x_samples=gt_vg_bboxes,
                x_rgbs=x_rgbs,
                rgbs_hw=rgb_samples.unmaksed_hws,
                tquery=x_text,
                rgb_mask=rgb_samples.mask,
                tquery_mask=text_samples.mask_p,
                x_noises=None,
            )
        else:
            return self.grounding_module.diffusion_reverse_ddim_sampling(
                x_rgbs=x_rgbs,
                rgbs_hw=rgb_samples.unmaksed_hws,
                tquery=x_text,
                tquery_mask=text_samples.mask_p,
            )
