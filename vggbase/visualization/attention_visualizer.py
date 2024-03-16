"""
Implementations of the visualization for the attention maps.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from vggbase.visualization.base import BaseVisualizer


def draw_array(fig, ax, array):
    """Draw an array as an image."""
    im = ax.imshow(array)
    fig.colorbar(im)


class AttentionVisualizer(BaseVisualizer):
    """A Visualizer to draw attention maps"""

    def filter_attention(
        self, batch_attention_maps: np.ndarray, batch_attention_masks: np.ndarray
    ):
        """
        Filter out invaild attention map based on the masks.

        :param batch_attention_maps: The attention scores
         with shape, [batch_size, num_queries, H, W]
        :param batch_attention_masks (np.narray): The attention masks.
         with shape, [batch_size, 1, H, W]

        :return filtered_attention (list): each item is the
         filtered attention maps for the corresponding sample
         based on its attention mask.
        """
        batch_size = batch_attention_maps.shape[0]
        filtered_attention = list()
        # Visit each sample in the batch
        for batch_idx in range(batch_size):
            attention_maps = batch_attention_maps[batch_idx]
            attention_mask = batch_attention_masks[batch_idx][0, :, :]
            # Get the non-masked region
            attention_non_masked = attention_mask == 0
            non_masked_h = np.max(np.sum(attention_non_masked, axis=0))
            non_masked_w = np.max(np.sum(attention_non_masked, axis=1))
            # Extract the non-masked region from the attention map
            attention_maps = attention_maps[:, :non_masked_h, :non_masked_w]
            filtered_attention.append(attention_maps)

        return filtered_attention

    def plat_batch(
        self,
        batch_names: list,
        batch_images: List[np.ndarray],
        batch_attention_maps: List[np.ndarray],
        batch_phrases: List[List[str]],
        log_dir_name: str,
        visual_type: str,
    ):
        """Visual one attention map."""

        batch_size = len(batch_phrases)
        for batch_idx in range(batch_size):
            image_phrases = batch_phrases[batch_idx]
            image_name = str(batch_names[batch_idx])
            # num_padded_phrases, H, W
            image_attention_maps = batch_attention_maps[batch_idx]

            sample_attn_save_path = os.path.join(
                self.visualization_dir, image_name, log_dir_name
            )
            os.makedirs(sample_attn_save_path, exist_ok=True)

            num_maps = image_attention_maps.shape[0]
            vaild_num_maps = len(image_phrases)
            assert num_maps >= vaild_num_maps
            for attn_map_idx in range(vaild_num_maps):
                image_attn_map = image_attention_maps[attn_map_idx]

                image_ph = image_phrases[attn_map_idx][0]

                fig, ax = plt.subplots()
                if batch_images is None:
                    draw_array(fig, ax, image_attn_map)
                else:
                    image_data = batch_images[batch_idx]
                    draw_array(fig, ax, image_attn_map, image_data)

                visual_file_name = visual_type + "_" + image_ph
                save_file_png_path = os.path.join(
                    sample_attn_save_path, visual_file_name + ".png"
                )
                save_file_pdf_path = os.path.join(
                    sample_attn_save_path, visual_file_name + ".pdf"
                )

                plt.savefig(save_file_png_path, dpi=300, bbox_inches="tight")
                plt.savefig(save_file_pdf_path, bbox_inches="tight")
                plt.close()
                ax.clear()

    def log_attention(
        self,
        batch_names: list,
        batch_attention_maps: np.ndarray,
        batch_phrases: np.ndarray,
        batch_images: np.ndarray = None,
        batch_attention_masks: np.ndarray = None,
        log_name: str = "attentions",
    ):
        """Log the attention map as the one image purely."""
        if batch_images is not None:
            batch_images = self.convert_to_format(batch_images)
            batch_images = self.convert_image_to_visiable(batch_images)
            batch_images = self.convert_image_to_channel_last(batch_images)

        batch_attention_maps = self.convert_to_format(batch_attention_maps)

        # Plot the attention maps
        self.plat_batch(
            batch_names,
            batch_images,
            batch_attention_maps,
            batch_phrases,
            log_dir_name=log_name,
            visual_type="images",
        )

        # Plot the attention maps with masks
        if batch_attention_masks is not None:
            batch_attention_masks = self.convert_to_format(batch_attention_masks)
            batch_attention_maps = self.filter_attention(
                batch_attention_maps, batch_attention_masks
            )
            self.plat_batch(
                batch_names,
                batch_images,
                batch_attention_maps,
                batch_phrases,
                log_dir_name=log_name,
                visual_type="images_masked",
            )
