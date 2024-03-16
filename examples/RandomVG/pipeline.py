"""
A simple pipeline to perform the random visual grounding
"""

from vggbase import pipeline

from vggbase.config import Config


class RandomVGPipeline(pipeline.VGPipeline):
    """A learning pipeline for the random VG model."""

    def run(self, project_name: str = "Random-VGGbase", project_seed: int = 0):
        """Run the pipeline."""
        train_config = Config().trainer
        # Get trainset
        testset_loader = self.dataset_builder.test_dataloader(
            batch_size=train_config.batch_size
        )

        test_save_name = "test-try"
        for batch_idx, samples in enumerate(testset_loader):
            save_folder_name = f"{test_save_name}/batch{batch_idx}"

            model_outputs = self.vg_executor(samples)
            # Match the bounding boxes with the ground truth
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
