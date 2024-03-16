"""
An executor of the DirectVG model.
"""

from vg_model import DirectVG

from vggbase.models import BasicVaQExecutor
from vggbase.models.model_generic import BaseVGModelInput
from vggbase.datasets.data_generic import BaseVGCollatedSamples


class DirectVGExecutor(BasicVaQExecutor):
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

        return DirectVG(n_proposals)

    def forward(self, inputs: BaseVGCollatedSamples, **kwargs):
        """Forward the model for inputs."""

        return self.grounding_module(BaseVGModelInput(**inputs))
