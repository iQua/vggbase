"""
A pipeline of VGGbase to show how to perform the text and image alignment 
and how to record experiments.

We should know that there are three types of image sizes:
- original image size: the size of the original image.
- preprocess image size: the size of the image after preprocessing.
- padded image size: the size of the image after padding to be organized
 into a batch.
"""

import os
import time

import lightning
import wandb
import torch
from torch import optim
from timm.scheduler import scheduler

from vggbase.datasets import VGDatasetBuilder
from vggbase.datasets.data_generic import BaseVGCollatedSamples
from vggbase.datasets.language.dynamic_tokenizer import LanguageDynamicTokenizer
from vggbase.models import BasicVaQExecutor
from vggbase.learners.matcher import BaseMatcher, HungarianMatcher
from vggbase.learners.loss_criterion import LossCriterion
from vggbase.learners.optimizers import OptimizerBuilder
from vggbase.learners.lr_schedulers import LRSchedulerBuilder
from vggbase.utils.recorder import BaseRecorder
from vggbase.visualization.visualizing import Visualizer

from vggbase.config import Config


class VGPipeline:
    """
    A base to perform the text and image alignment and record experiments.
    """

    def __init__(
        self,
        vg_executor: BasicVaQExecutor,
        dataset_builder: VGDatasetBuilder = None,
        tokenizer: LanguageDynamicTokenizer = None,
        matcher: BaseMatcher = None,
        loss_criterion: LossCriterion = None,
        optimizer: optim.Optimizer = None,
        lr_scheduler: scheduler.Scheduler = None,
        recorder: BaseRecorder = None,
        visualizer: Visualizer = None,
    ):
        # The main executor used to perform the visual grounding task.
        self.vg_executor = vg_executor
        # The builder to create and load datasets
        self.dataset_builder = dataset_builder
        # The tokenizer to process the string text into tokens.
        self.tokenizer = tokenizer
        # The matcher used to align the predicted bounding boxes with the
        # ground truth.
        self.matcher = matcher
        # The loss criterion used to calculate the loss for the matched
        # bounding boxes.
        self.loss_computer = loss_criterion

        # The optimizer
        self.optimizer = optimizer
        # The learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # The visualizer used to visualize the samples and the outputs.
        self.visualizer = visualizer

        # The recorder used to save the samples and the outputs.
        self.recorder = recorder

        # The dataloader
        self.trainset_loader = None
        self.valset_loader = None
        self.valset_loader = None

    def setup(self):
        """Setup the dataset, model, matcher, loss criterion and recorder."""
        # Get config
        model_config = Config().items_to_dict(Config().model._asdict())
        data_config = Config().items_to_dict(Config().data._asdict())
        eval_config = Config().items_to_dict(Config().evaluation._asdict())
        train_config = Config().items_to_dict(Config().trainer._asdict())
        if self.tokenizer is None:
            self.tokenizer = LanguageDynamicTokenizer(
                language_config=model_config["language"]
            )

        if self.dataset_builder is None:
            self.dataset_builder = VGDatasetBuilder(
                data_config=data_config,
                tokenizer=self.tokenizer,
            )

        self.dataset_builder.prepare_datasource()
        self.dataset_builder.set_collate_fn()

        if self.matcher is None:
            self.matcher = HungarianMatcher(matcher_config=eval_config["matching"])
        if self.loss_computer is None:
            self.loss_computer = LossCriterion(eval_config)

        if self.optimizer is None:
            self.optimizer = OptimizerBuilder().get(
                model_named_parameters=self.vg_executor.named_parameters(),
                trainer_config=train_config,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = LRSchedulerBuilder().get(
                optimizer=self.optimizer,
                trainer_config=train_config,
            )

        if self.visualizer is None:
            visual_path = Config().logging.visualization_path
            self.visualizer = Visualizer(
                visualization_path=f"{visual_path}/bases",
                is_create_new=True,
            )
        if self.recorder is None:
            # Be default, the recorder will save the records to
            # the result path
            result_path = Config().logging.result_path
            self.recorder = BaseRecorder(
                record_root=f"{result_path}/records",
                is_create_new=True,
            )

    def create_wb_config(self):
        """Create the wandb configuration."""
        train_config = Config().trainer
        config = dict(
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            learning_rate=train_config.learning_rate,
            dataset=Config().data.data_name,
            architecture="RandomVG",
        )
        return config

    def train_batch(
        self,
        fabric: lightning.Fabric,
        samples: BaseVGCollatedSamples,
        epoch_idx: int,
        batch_idx: int,
        vg_executor: BasicVaQExecutor,
        optimizer: optim.Optimizer,
        criterion: LossCriterion,
        save_location: str,
    ):
        """Train a batch of samples."""
        vg_executor.train()
        optimizer.zero_grad()

        cur = time.time()
        model_outputs = vg_executor(samples)
        end = time.time()

        match_outputs = self.matcher.forward(model_outputs, samples.targets)

        batch_loss = criterion(model_outputs, match_outputs, samples.targets)

        fabric.backward(batch_loss["loss"])
        optimizer.step()

        # Visualize the samples and outputs
        if batch_idx % Config().logging.tr_log_interval == 0:
            self.visualizer.visualize_collated_samples(
                collated_samples=samples, save_location=save_location
            )
            self.visualizer.visualize_model_outputs(
                collated_samples=samples,
                model_outputs=model_outputs,
                match_outputs=match_outputs,
                save_location=save_location,
            )

            # Save the models
            model_name = f"epoch{epoch_idx}-batch{batch_idx}.pth"
            model_folder = Config().logging.checkpoint_path
            torch.save(
                self.vg_executor.state_dict(),
                os.path.join(model_folder, model_name),
            )

            # Record the samples, outputs, and matching results
            self.recorder.save_batch_records(
                samples,
                model_outputs,
                match_outputs,
                statistics={"inference_time": end - cur},
                location=save_location,
            )

        return batch_loss

    def run(self, project_name: str = "vggbase", project_seed: int = 0):
        """Run the pipeline."""
        lightning.Fabric.seed_everything(seed=project_seed)
        train_config = Config().trainer

        fabric = lightning.Fabric(devices=1)
        fabric.launch()

        # Login to wandb
        wandb.login()
        wandb_config = self.create_wb_config()

        # Get trainset
        trainset_loader = self.dataset_builder.train_dataloader(
            batch_size=train_config.batch_size
        )
        trainset_loader = fabric.setup_dataloaders(trainset_loader)

        # Set the model
        self.vg_executor, self.optimizer = fabric.setup(
            self.vg_executor, self.optimizer
        )

        # Prepare the training
        # tell wandb to get started
        wb_run = wandb.init(project=project_name, config=wandb_config)

        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(
            self.vg_executor,
            self.loss_computer,
            log="all",
            log_freq=Config().logging.tr_log_interval,
        )

        for epoch_idx in range(train_config.start_epoch, train_config.epochs):
            epoch_save_name = f"train-epoch{epoch_idx}"
            for batch_idx, samples in enumerate(trainset_loader):
                save_folder_name = f"{epoch_save_name}/batch{batch_idx}"
                batch_loss = self.train_batch(
                    fabric=fabric,
                    samples=samples,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    vg_executor=self.vg_executor,
                    optimizer=self.optimizer,
                    criterion=self.loss_computer,
                    save_location=save_folder_name,
                )
                self.train_log(batch_loss, epoch_idx, batch_idx)
            self.lr_scheduler.step(epoch_idx + 1)

        wb_run.finish()

    def train_log(self, batch_loss: float, epoch_idx: int, batch_idx: int):
        """Log for the training process."""
        if batch_idx % Config().logging.tr_log_interval == 0:
            wandb.log({"loss": batch_loss, "epoch": epoch_idx, "batch": batch_idx})
