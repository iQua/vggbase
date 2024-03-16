"""
The implementation of language-guided diffusion visual grounding method,
i.e., LG-DVG.
"""

from diffusionvg_executor import DiffusionVGExecutor

from vggbase import pipeline
from vggbase.config import Config


def _main():
    """A learning session with the random VG algorithm."""

    model_config = Config().model
    env_config = Config().items_to_dict(Config().environment._asdict())
    ## Prepare model
    vg_executor = DiffusionVGExecutor(model_config)
    vg_pipeline = pipeline.VGPipeline(vg_executor=vg_executor)
    vg_pipeline.setup()
    vg_pipeline.run(project_name="DiffusionVG", project_seed=env_config["seed"])


if __name__ == "__main__":
    _main()
