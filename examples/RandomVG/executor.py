"""
An executor to process the input samples to be forwarded to the model.
"""

from vg_model import RandomVG

from vggbase.models import BasicVaQExecutor
from vggbase.models.model_generic import BaseVGModelInput
from vggbase.datasets.data_generic import BaseVGCollatedSamples


class RandomVGExecutor(BasicVaQExecutor):
    """The executor for the random VG model."""

    def init_language_module(self, model_config: dict, **kwargs):
        return None

    def init_rgb_module(self, model_config: dict, **kwargs):
        return None

    def resume_module(self, model_config, **kwargs):
        return None

    def init_grounding_module(self, model_config: dict, **kwargs):
        """Initialize the visual grounding model."""

        n_proposals = model_config["grounding"]["n_proposals"]

        return RandomVG(n_proposals)

    def forward(self, inputs: BaseVGCollatedSamples, **kwargs):
        """Forward the model for inputs."""

        return self.grounding_module(BaseVGModelInput(**inputs))
