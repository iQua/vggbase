"""
Implementation of recorders for saving results to files.
"""

import os
import json
import glob

import torch
from torch.profiler import profile

from vggbase.datasets.data_generic import BaseVGCollatedSamples, BaseInputTarget
from vggbase.models.model_generic import BaseVGModelOutput
from vggbase.learners.learn_generic import BaseVGMatchOutput
from vggbase.visualization import utils

from vggbase.config import Config


class BaseRecorder:
    """
    A base recorder used to save the sample and the outputs.
    """

    def __init__(
        self,
        sample_filename: str = "samples",
        match_files: str = "matches",
        output_filename: str = "outputs",
        statistic_filename: str = "consumption_statistics",
        record_root: str = None,
        is_create_new: bool = True,
    ) -> None:
        self.sample_filename = sample_filename
        self.match_files = match_files
        self.output_filename = output_filename
        self.statistic_filename = statistic_filename

        self.record_root = os.getcwd() if record_root is None else record_root
        if is_create_new:
            self.record_root = utils.create_new_folder(self.record_root)
            os.makedirs(self.record_root)
        else:
            if not os.path.exists(self.record_root):
                raise FileNotFoundError(
                    "This directory is not existed, it should be created at first"
                )

        os.makedirs(self.record_root, exist_ok=True)

    def get_recorded_names(self):
        """Get the indexes of the records."""
        pattern = f"{self.record_root}/{self.output_filename}_*.json"

        # Use glob to find files matching the pattern
        exist_records = glob.glob(pattern)

        record_names = [
            record.split("_")[-1].split(".json")[0] for record in exist_records
        ]

        # Order the indexes
        record_names.sort()

        # We did not include the latest record as it may not be completed
        return record_names[:-1]

    def save_samples(self, samples: BaseVGCollatedSamples, path: str):
        """Save the 'idx' result to the disk."""

        # Save the image and the mask
        rgb_tensor = samples.rgb_samples.tensors.cpu()
        rgb_mask = samples.rgb_samples.mask.cpu()
        torch.save(rgb_tensor, f"{path}/collated_batch_images.pt")
        torch.save(rgb_mask, f"{path}/collated_batch_mask.pt")

        rgb_size = list(rgb_tensor.size())
        rgb_unmaksed_hws = samples.rgb_samples.unmaksed_hws
        text_samples = samples.text_samples
        text_tensors = text_samples.tensors.cpu().detach().numpy().tolist()
        text_masks = text_samples.mask.cpu().detach().numpy().tolist()
        text_unmaksed_hws = text_samples.unmaksed_hws

        # Convert the targets to json format
        targets = [BaseInputTarget.get_json(tgt) for tgt in samples.targets]

        data = {
            "RGB": {"size": rgb_size, "unmaksed_hws": rgb_unmaksed_hws},
            "Text": {
                "tensors": text_tensors,
                "mask": text_masks,
                "unmaksed_hws": text_unmaksed_hws,
            },
            "targets": targets,
        }

        save_path = f"{path}/{self.sample_filename}.json"
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def save_outputs(self, outputs: BaseVGModelOutput, path: str):
        """Save the outputs to the disk."""
        outputs = BaseVGModelOutput.get_json(outputs)
        save_path = f"{path}/{self.output_filename}.json"
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(outputs, file)

    def save_matches(self, matches: BaseVGMatchOutput, path: str):
        """Save the match to the disk."""
        matches = BaseVGMatchOutput.get_json(matches)
        save_path = f"{path}/{self.match_files}.json"
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(matches, file)

    def save_batch_records(
        self,
        samples: BaseVGCollatedSamples,
        model_outputs: BaseVGModelOutput,
        match_outputs: BaseVGMatchOutput,
        statistics: dict,
        location: str,
    ):
        """Save one record to the disk."""

        save_path = f"{self.record_root}/{location}"
        os.makedirs(save_path, exist_ok=True)
        self.save_samples(samples, save_path)
        self.save_outputs(model_outputs, save_path)
        self.save_matches(match_outputs, save_path)
        save_path = f"{save_path}/{self.statistic_filename}.json"
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(statistics, file)

    @staticmethod
    def save_profile(profiler: profile):
        """Save the profile to the disk."""
        result_path = Config().logging.result_path
        save_path = os.path.join(
            result_path, "profiles", f"global-step-{profiler.step_num}"
        )
        os.makedirs(save_path, exist_ok=True)

        profiler.export_chrome_trace(os.path.join(save_path, "profile-trace.json"))

        with open(
            os.path.join(save_path, "profile-summary.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(profiler.key_averages().table())
