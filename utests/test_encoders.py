"""
The whether the encoder can be built successfully.

"""

import time
import datetime

import resource


from vggbase.models.visual_encoders import build_encoder
from vggbase.datasets import VGDatasetBuilder
from vggbase.datasets.language import build_language_tokenizer
from vggbase.utils.envs_utils import define_env
from vggbase.utils.logging_format import present_logging
from vggbase.config import Config


def main():
    """Main function to test the encoders of VGBase."""

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # obtain configs
    model_config = Config().model
    data_config = Config().data
    train_config = Config().train

    env_config = Config().environment
    env_config = Config().items_to_dict(env_config._asdict())

    # define the environment used for learning
    _, env_config = define_env(env_config=env_config)

    ####
    language_tokenizer = build_language_tokenizer(
        data_config=data_config, language_config=model_config.language_module
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

    ######### Define encoder
    rgb_encoder = build_encoder(
        encoder_name=model_config.rgb_module_name,
        encoder_config=model_config.rgb_module,
    )
    print(rgb_encoder)

    ######### Learning process
    present_logging("Training", level=1)
    start_time = time.time()

    for epoch in range(train_config.start_epoch, train_config.epochs):
        # iterate one epoch
        for samples in trainset_loader:
            outputs = rgb_encoder(samples.rgb_samples.tensors)
            print([(k, v.shape) for k, v in outputs.items()])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    present_logging(f"Training time {total_time_str}", level=4)


if __name__ == "__main__":
    main()
