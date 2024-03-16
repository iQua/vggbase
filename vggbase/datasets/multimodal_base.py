"""
Base class for multimodal datasets.
"""

from abc import abstractmethod
import json
import random
from collections import defaultdict

import torch
import skimage.io as io
import cv2

from vggbase.datasets.data_generic import DatasetCatalog, DatasetStatistics
from vggbase.boxes.bbox_generic import BaseSampleBBoxes
from vggbase.datasets.data_generic import BaseVGSample
from vggbase.datasets.base_structure import DataSourceStructure


class MultiModalDataSource(DataSourceStructure):
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """

    def compute_data_statistics(self, datacatalog_path):
        """Get the statistics of one dataset."""
        with open(datacatalog_path, "r", encoding="utf-8") as fp:
            datacatalog = json.load(fp)
        if "data_statistics" in datacatalog:
            return
        image_samples = datacatalog["image_samples"]
        n_samples = len(image_samples)
        num_phrases_per_image = defaultdict(int)
        num_bboxes_per_query = defaultdict(int)
        num_bboxes_per_phrase = defaultdict(int)
        for sample in image_samples:
            caption_annotations = sample["caption_annotations"]
            bbox_annotations = sample["bbox_annotations"]
            n_phrases = len(caption_annotations["caption_phrases"])
            num_phrases_per_image[n_phrases] += 1
            sample_bboxes = bbox_annotations["bboxes"]
            total_n_bboxes = 0
            for bboxes in sample_bboxes:
                n_bboxes = len(bboxes)
                num_bboxes_per_phrase[n_bboxes] += 1
                total_n_bboxes += n_bboxes
            num_bboxes_per_query[total_n_bboxes] += 1

        with open(datacatalog_path, "w", encoding="utf-8") as fp:
            json.dump(
                DatasetCatalog(
                    data_phase=datacatalog["data_phase"],
                    image_samples=datacatalog["image_samples"],
                    data_statistics=DatasetStatistics(
                        num_samples=n_samples,
                        num_phrases_per_image=num_phrases_per_image,
                        num_bboxes_per_query=num_bboxes_per_query,
                        num_bboxes_per_phrase=num_bboxes_per_phrase,
                    ),
                ),
                fp,
            )

    def compute_splits_data_statistics(self):
        """Compute the data statistics for splits."""
        for split_type in self.splits_info:
            datacatalog_path = self.get_split_datacatalog_path(split_type)
            self.compute_data_statistics(datacatalog_path)

    def num_modalities(self) -> int:
        """Number of modalities"""
        return len(self.supported_modalities)

    @abstractmethod
    def get_phase_dataset(self, phase, image_transform, text_transform):
        """Obtain the dataset with the modaltiy_sampler for the
        specific phase (train/test/val)"""
        raise NotImplementedError("Please Implement 'get_phase_dataset' method")

    @abstractmethod
    def get_train_set(self, image_transform, text_transform):
        """Obtain the train dataset with the modaltiy_sampler"""
        raise NotImplementedError("Please Implement 'get_train_set' method")

    @abstractmethod
    def get_test_set(self, image_transform, text_transform):
        """btain the test dataset with the modaltiy_sampler"""
        raise NotImplementedError("Please Implement 'get_test_set' method")


class MultiModalDataset(torch.utils.data.Dataset):
    """The base interface for the multimodal data"""

    def __init__(
        self,
        datacatalog_filepath,
        image_transform=None,
        text_transform=None,
    ):
        # the loaded datacatalog
        self.datacatalog = None
        with open(datacatalog_filepath, "r", encoding="utf-8") as fp:
            self.datacatalog = json.load(fp)
        # transforms for image and text if provided
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.supported_modalities = []
        self.image_samples_data = self.datacatalog["image_samples"]
        self.n_samples = self.datacatalog["data_statistics"]["num_samples"]
        self.samples_idx = list(range(0, self.n_samples))

        self.shuffle_whole_samples()

    def shuffle_whole_samples(self):
        """Shuffle the whole samples to introduce the randomness."""
        random.shuffle(self.samples_idx)

    def get_one_sample(self, sample_idx: int):
        """Get the sample from datacatalog."""
        shuffle_idx = self.samples_idx[sample_idx]
        image_sample = self.image_samples_data[shuffle_idx]
        image_id = image_sample["image_id"]
        image_file_path = image_sample["image_file_path"]
        image_hw = image_sample["image_hw"]
        caption_annos = image_sample["caption_annotations"]
        bbox_annos = image_sample["bbox_annotations"]
        image_data = io.imread(image_file_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return BaseVGSample(
            sample_id=image_id,
            image_data=image_data,
            caption=caption_annos["caption"],
            caption_phrases=caption_annos["caption_phrases"],
            caption_phrases_category=caption_annos["caption_phrases_category"],
            caption_phrases_category_id=caption_annos["caption_phrases_category_id"],
            caption_phrases_id=caption_annos["caption_phrases_id"],
            caption_phrases_bboxes=BaseSampleBBoxes(
                bboxes=bbox_annos["bboxes"],
                board_hw=image_hw,
                bbox_type=bbox_annos["bboxes_mode"],
                phrases_label=bbox_annos["bboxes_category_id"],
            ),
        )

    def perform_transform(self, vg_sample):
        """Perform the transform for the sample."""

        image_data = vg_sample.image_data
        base_bboxes = vg_sample.caption_phrases_bboxes
        caption_phrases = vg_sample.caption_phrases

        if self.image_transform is not None:
            transformed = self.image_transform(
                image=image_data,
                bboxes=base_bboxes.bboxes.tolist(),
                bboxes_labels=base_bboxes.bbox_ids,
            )

            # after transformation, the image has been converted
            #   to C, H, W
            image_data = transformed["image"]
            # the bounding boxes has been
            # 1. transformed to target format
            # 2. normalized based on the image
            transformed_bboxes = transformed["bboxes"]

            vg_sample.image_data = image_data
            vg_sample.caption_phrases_bboxes.bboxes = transformed_bboxes

            vg_sample.caption_phrases_bboxes.bbox_type = (
                self.image_transform.work_box_type
            )
            vg_sample.caption_phrases_bboxes.bboxes_board_size = list(
                image_data.shape[1:]
            )

        if self.text_transform is not None:
            vg_sample.caption_phrases = self.text_transform(caption_phrases)

        return vg_sample

    def __getitem__(self, sample_idx):
        """Get the sample for either training or testing given index."""

        raw_sample = self.get_one_sample(sample_idx)

        return self.perform_transform(raw_sample)

    def __len__(self):
        """obtain the number of samples."""
        return self.n_samples
