"""
Implementations of Visual Transformations.
"""

from typing import Any, List, Tuple

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2

from vggbase.boxes import bbox_convertion

augmentation_factory = {
    "default": [
        A.Sharpen(alpha=(0.0, 0.1)),
        A.Blur(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.HueSaturationValue((-10, 10)),
    ],
    "yolo5": [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_lower=75, p=0.0),
    ],
}


class VisualShapeConversion:
    """A series of conversion to adjust shapes of visual data."""

    def __init__(
        self,
        phase: str,
        conversion_config: dict,
    ):
        # Extract scales from the config
        scales = conversion_config["scales"] if "scales" in conversion_config else None
        # Extract target size from the config
        target_size = (
            conversion_config["target_size"]
            if "target_size" in conversion_config
            else ()
        )

        # Functions of shape Conversion to be applied
        # to the visual data
        self.functions = []
        self.functions = self.get_conversion(phase, scales, target_size)

    def get_conversion(self, phase: str, scales: List[int], target_size: Tuple[int]):
        """Define the conversion functions."""

        # Set the desired scales of visual data by default
        if scales is None:
            scales = [448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 640, 704]

        # Define default conversion functions for
        # different learning phases
        conversion_functions = []
        if phase in ["train", "fit"]:
            conversion_functions = [A.LongestMaxSize(max_size=scales)]

        if phase in ["val", "test", "testA", "testB"]:
            conversion_functions = []

        # When target size is specified, we need to pad the image
        if target_size:
            target_h, target_w = (
                target_size
                if isinstance(target_size, (list, tuple))
                else (target_size, target_size)
            )
            conversion_functions.extend(
                [
                    A.LongestMaxSize(max_size=max(target_h, target_w)),
                    A.PadIfNeeded(
                        min_height=target_h,
                        min_width=target_w,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                ]
            )

        return conversion_functions


class ViauslContentAugmentations:
    """A series of functions to augment the visual content."""

    def __init__(self, phase: str, augmentation_config: dict):
        # Functions of transformations to be applied
        # to the visual data
        self.functions = []

        # Extract mean and std for normalization
        self.norm_mean = (
            augmentation_config["norm_mean"]
            if "norm_mean" in augmentation_config
            else [0.485, 0.456, 0.406]
        )
        self.norm_std = (
            augmentation_config["norm_mean"]
            if "norm_std" in augmentation_config
            else [0.229, 0.224, 0.225]
        )

        # Only set augmentation functions for the training phase
        # but only for the test or val phases, such as
        # ['val', 'test', 'testA', 'testB']
        if phase in ["train", "fit"]:
            self.functions = self.get_augmentation(config=augmentation_config)

        # Obtain the mean and std of the denormalization
        # , which converts a normalized image to the original
        self.inv_normalize = self.obtain_denormalization()

    def get_augmentation(self, config: dict):
        """Define the default augment transform following the yolo4."""

        # Get the augmentation functions from the factory
        augment_style = (
            "default" if "augment_style" not in config else config["augment_style"]
        )
        augmentation_functions = augmentation_factory[augment_style]

        # Set normalization when needed
        # By defualt, True
        if "is_normalized" in config and config["is_normalized"]:
            augmentation_functions.append(A.Normalize(self.norm_mean, self.norm_std))

        return augmentation_functions

    def obtain_denormalization(self):
        """Obtain the denormalization function."""
        inv_normalize = A.Normalize(
            mean=[-m / s for m, s in zip(self.norm_mean, self.norm_std)],
            std=[1 / s for s in self.norm_std],
        )

        return inv_normalize


class VisualTransformations:
    """Define the visual transforme. It includes
    the visual augmentation and visual transformation."""

    def __init__(
        self,
        content_augmentations: ViauslContentAugmentations,
        shape_conversion: VisualShapeConversion,
        box_config: dict,
    ):
        # Define the augmentation and transformation
        self.augmentations = content_augmentations
        self.conversions = shape_conversion

        # Get the source and target box type
        self.source_box_type = box_config["source_type"]
        self.work_box_type = box_config["work_type"]

        # bounding boxes must be converted to
        # `pascal_voc`
        assert self.work_box_type == "pascal_voc"

        # Define the transformations in which the
        # 'label_fields' corresponds to the 'bboxes_labels'
        # of the generic dataset of vggbase
        augmentations = self.augmentations.functions
        conversions = self.conversions.functions
        self.functions = augmentations + conversions + [ToTensorV2()]

        self.transformations = A.Compose(
            self.functions,
            bbox_params=A.BboxParams(
                format=self.source_box_type, label_fields=["bboxes_labels"]
            ),
        )

    def __call__(
        self, image: np.ndarray, bboxes: list, bboxes_labels: list, **kwards: Any
    ):
        """
        Process the image and bboxes based on the defined content augmentations
        and shape conversions.
        """
        # Perform the transformations
        transformed_data = self.transformations(
            image=image, bboxes=bboxes, bboxes_labels=bboxes_labels, **kwards
        )
        image_data = transformed_data["image"]
        bboxes_data = transformed_data["bboxes"]

        # Get the shape of the image after the transformation
        _, h, w = image_data.shape

        # Convert the box type from
        # source_box_type to work_box_type
        # Make the function inputs
        # bboxes, [bs, N, 4]; board_hw, [bs, 2]
        bboxes_data = torch.tensor(bboxes_data)[None, ...]
        board_hw = torch.tensor([h, w])[None, ...]
        # The function returns the tensor of shape [bs, N, 4]
        # First remove bs to get [N, 4]
        # Then convert to list
        target_bboxes = bbox_convertion.convert_bbox_type(
            bboxes_data,
            source_type=self.source_box_type,
            target_type=self.work_box_type,
            board_hws=board_hw,
        )

        target_bboxes = target_bboxes.squeeze(0).tolist()
        transformed = {"image": image_data, "bboxes": target_bboxes}
        return transformed
