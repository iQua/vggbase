"""
This implementation is for the unit test of visualization.

"""

import time
import datetime

import resource


from vggbase.datasets import VGDatasetBuilder
from vggbase.datasets.language import build_language_tokenizer
from vggbase.visualization import build_visualizer
from vggbase.utils.envs_utils import define_env
from vggbase.utils.logging_format import present_logging
from vggbase.config import Config


def main():
    """Main function to train a ViBertTransformer model for the VG task."""

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # obtain configs
    model_config = Config().model
    data_config = Config().data
    train_config = Config().train
    logging_config = Config().logging

    env_config = Config().environment
    env_config = Config().items_to_dict(env_config._asdict())

    # define the environment used for learning
    _, env_config = define_env(env_config=env_config)

    ####
    language_tokenizer = build_language_tokenizer(
        data_config=data_config, language_config=model_config.language_module
    )

    #################### Prepare the visualization ####################

    built_train_visualizer = build_visualizer(
        phase="train",
        visualization_dir=logging_config.visualizations_dir,
        is_unique_created=True,
    )

    built_test_visualizer = build_visualizer(
        phase="test",
        visualization_dir=logging_config.visualizations_dir,
        is_unique_created=True,
    )

    built_val_visualizer = build_visualizer(
        phase="val",
        visualization_dir=logging_config.visualizations_dir,
        is_unique_created=True,
    )

    #################### Prepare dataset ####################
    # build data module
    # for train
    vg_dataset_builder = VGDatasetBuilder(
        data_config=data_config,
        language_tokenizer=language_tokenizer,
        train_config=train_config,
    )

    vg_dataset_builder.prepare_data()

    vg_dataset_builder.setup(stage="train")
    trainset_loader = vg_dataset_builder.train_dataloader()

    # for validation
    vg_dataset_builder.setup(stage="val")
    valset_loader = vg_dataset_builder.val_dataloader()

    ######### Learning process
    present_logging("Training", level=1)

    start_time = time.time()

    for epoch in range(train_config.start_epoch, train_config.epochs):
        # iterate one epoch
        batch_idx = 0
        for samples in trainset_loader:
            built_train_visualizer.log_visuable_raw_info(
                collated_samples=samples,
                batch_idx=batch_idx,
            )
            batch_idx += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    present_logging(f"Training time {total_time_str}", level=4)


if __name__ == "__main__":
    main()
