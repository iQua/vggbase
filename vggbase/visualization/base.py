"""
The base interface for the visualizer.

"""

import os
from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from vggbase.visualization import utils

plt.rcParams["savefig.bbox"] = "tight"


class BaseVisualizer(object):
    """The base visualizer.

    Args:
        visualization_path: A `str` path where to save visualization results.
        is_create_new: A `Boolean` indicating whether to create a new folder
         once the `visualization_path` exists.
    """

    def __init__(self, visualization_path, is_create_new=True):
        self.visualization_path = visualization_path
        if is_create_new:
            self.visualization_path = utils.create_new_folder(self.visualization_path)
            os.makedirs(self.visualization_path)
        else:
            if not os.path.exists(self.visualization_path):
                raise FileNotFoundError(
                    "This directory is not existed, it should be created at first"
                )

    def get_standard_images(self, batch_images: Union[list, torch.Tensor]):
        """
        Convert the images data to the standard, which is list containing
        multiple np.ndarray elements.

        :return A `list` of `np.ndarray` elements where each element
         contains the data with shape [H, W, C] of the specific image.
        """
        assert isinstance(batch_images, (list, torch.Tensor))

        if isinstance(batch_images, torch.Tensor):
            batch_size = batch_images.shape[0]
            batch_images = [batch_images[i] for i in range(batch_size)]

        def compound_process(image_data):
            image_data = self.convert_to_numpy(image_data)
            image_data = self.convert_image_to_channel_last(image_data)
            image_data = self.convert_image_to_visiable(image_data)
            return image_data

        return [compound_process(image_data) for image_data in batch_images]

    def get_standard_data(self, batch_data: Union[list, torch.Tensor]):
        """
        Convert the data to the standard, which is a list
        containing multiple np.ndarray elements.

        :return A `list` of `np.ndarray` elements where each element
         contains the bboxes with shape [N, 4] for the specific image.
        """
        assert isinstance(batch_data, (list, torch.Tensor))

        if isinstance(batch_data, torch.Tensor):
            batch_size = batch_data.shape[0]
            batch_data = [batch_data[i] for i in range(batch_size)]

        return [self.convert_to_numpy(data) for data in batch_data]

    def convert_to_numpy(self, tensor_data):
        """Convert the torch tensor to the numpy array."""
        tensor_data = (
            tensor_data
            if isinstance(tensor_data, np.ndarray)
            else (
                tensor_data.detach().cpu().numpy()
                if hasattr(tensor_data, "numpy")
                else np.array(tensor_data)
            )
        )

        return tensor_data

    def convert_image_to_channel_last(self, batch_images: np.ndarray):
        """
        Convert the image data to be the channel last for visualization.

        :param batch_images: a `np.ndarray` holding the images data
         of shape [c, h, w] or [bs, c, h, w]
        """
        num_dim = batch_images.ndim
        assert num_dim == 3 or num_dim == 4

        if batch_images.shape[-1] != 3:
            if num_dim == 3:
                batch_images = np.transpose(batch_images, (1, 2, 0))
            else:
                batch_images = np.transpose(batch_images, (0, 2, 3, 1))
        return batch_images

    def convert_image_to_visiable(self, image_data):
        """Convert the image data to the range of [0, 255] in each channel."""
        # must be a 3 channel image
        assert image_data.ndim == 3 or image_data.ndim == 4
        if np.mean(image_data) < 1:
            image_data = image_data * 255.0

        return image_data
