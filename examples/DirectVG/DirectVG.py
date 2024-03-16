"""
This is a direct way to perform the VG analysis by approaching the ground truth
gradually.

python examples/DirectVG/DirectVG.py -c examples/configs/F30KE/DirectVG.yml -b PROJECT
"""

from executor import DirectVGExecutor
from pipeline import DirectVGPipeline
from visualizer import DirectVGVisualizer

from vggbase.config import Config
from vggbase.learners.matcher import BaseMatcher


def _main():
    """A learning session with the random VG algorithm."""

    model_config = Config().items_to_dict(Config().model._asdict())
    env_config = Config().items_to_dict(Config().environment._asdict())
    eval_config = Config().items_to_dict(Config().evaluation._asdict())

    matcher = BaseMatcher(matcher_config=eval_config["matching"])

    ## Prepare model
    vg_executor = DirectVGExecutor(model_config)
    vg_pipeline = DirectVGPipeline(
        vg_executor=vg_executor,
        matcher=matcher,
        loss_criterion="",
        optimizer="",
        lr_scheduler="",
        recorder="",
        visualizer=DirectVGVisualizer(
            visualization_path=f"{Config().logging.visualization_path}/bases",
            is_create_new=True,
        ),
    )

    vg_pipeline.setup()
    vg_pipeline.run(
        project_name=env_config["project_name"], project_seed=env_config["seed"]
    )


if __name__ == "__main__":
    _main()
