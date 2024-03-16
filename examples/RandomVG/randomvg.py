"""
A visual grounding approach with randomly detection.
"""

from executor import RandomVGExecutor
from pipeline import RandomVGPipeline

from vggbase.config import Config
from vggbase.learners.matcher import BaseMatcher


def _main():
    """A learning session with the random VG algorithm."""

    model_config = Config().items_to_dict(Config().model._asdict())
    env_config = Config().items_to_dict(Config().environment._asdict())
    eval_config = Config().items_to_dict(Config().evaluation._asdict())

    matcher = BaseMatcher(matcher_config=eval_config["matching"])
    ## Prepare model
    vg_executor = RandomVGExecutor(model_config)
    vg_pipeline = RandomVGPipeline(
        vg_executor=vg_executor,
        matcher=matcher,
        loss_criterion="",
        optimizer="",
        lr_scheduler="",
        recorder="",
    )
    vg_pipeline.setup()
    vg_pipeline.run(
        project_name=env_config["project_name"], project_seed=env_config["seed"]
    )


if __name__ == "__main__":
    _main()
