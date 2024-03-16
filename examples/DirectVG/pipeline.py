"""
A learning pipeline of the model

"""

from vggbase import pipeline

from vggbase.config import Config


class DirectVGPipeline(pipeline.VGPipeline):
    """A learning pipeline for the direct VG model."""

    def run(self, project_name: str = "direct-VGGbase", project_seed: int = 0):
        """Run the pipeline."""
        train_config = Config().trainer
        # Get trainset
        testset_loader = self.dataset_builder.test_dataloader(
            batch_size=train_config.batch_size
        )

        for epoch_idx in range(train_config.start_epoch, train_config.epochs):
            epoch_save_name = f"train-epoch{epoch_idx}"
            for batch_idx, samples in enumerate(testset_loader):
                save_folder_name = f"{epoch_save_name}/batch{batch_idx}"
                model_outputs = self.vg_executor(samples)

                # Match the bounding boxes with the ground truth
                # of shape, [batch_size, n_iterations+1, n_proposals]
                # where 1 here denotes the initial boxes
                match_outputs = self.matcher.forward(model_outputs, samples.targets)

                self.visualizer.visualize_collated_samples(
                    collated_samples=samples, save_location=save_folder_name
                )
                self.visualizer.visualize_model_outputs(
                    collated_samples=samples,
                    model_outputs=model_outputs,
                    match_outputs=match_outputs,
                    save_location=save_folder_name,
                )
