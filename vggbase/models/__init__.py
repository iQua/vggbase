"""
The interface to introduce the modules and models to 
the lighting package.
"""

from abc import abstractmethod

import torch.nn as nn

from vggbase.models.model_generic import BaseVGModelInput


class BasicVaQExecutor(nn.Module):
    """This is a executor the grounding models."""

    def __init__(self, model_config: dict, **kwargs):
        """Initializes the executor for the model."""
        super().__init__()

        # Initialize the Language, RGB, and Grounding modules
        self.language_module = self.init_language_module(model_config, **kwargs)
        self.rgb_module = self.init_rgb_module(model_config, **kwargs)
        self.grounding_module = self.init_grounding_module(model_config, **kwargs)

        # Load the pre-trained grounding module when possible
        self.resume_module(model_config, **kwargs)

    @abstractmethod
    def init_language_module(self, model_config: dict, **kwargs):
        """Initialize the RGB module."""
        raise NotImplementedError("A RGB module is required to be defined.")

    @abstractmethod
    def init_rgb_module(self, model_config: dict, **kwargs):
        """Initialize the RGB module."""
        raise NotImplementedError("A RGB module is required to be defined.")

    @abstractmethod
    def init_grounding_module(self, model_config: dict, **kwargs):
        """Initialize the visual grounding model."""
        raise NotImplementedError("A grounding module is required to be defined.")

    @abstractmethod
    def resume_module(self, model_config, **kwargs):
        """Load the pre-trained grounding module when possible."""
        raise NotImplementedError("A pretrained module is required to be defined.")

    @abstractmethod
    def forward(self, inputs: BaseVGModelInput, **kwargs):
        """Forward the model for inputs."""
        # text_samples = inputs.text_samples
        # rgb_samples = inputs.rgb_samples

        # # forward the language module
        # x_text, text_mask = self.language_module(text_samples.tensors)

        # # forward the rgb module
        # x_rgbs = self.rgb_module(rgb_samples.tensors)

        # # forward the grounding module

        # # forward the head module

        raise NotImplementedError("The forward function is required to be customized.")
