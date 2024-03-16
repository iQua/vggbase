"""
Implementation of the visualiztion for grounding results.
"""

from typing import Optional, List, Union, Dict
import os

import numpy as np
import cv2

from vggbase.visualization.settings import (
    cv_digital_colors_map,
    cv_color_setup,
)
from vggbase.visualization.utils import save_phrases, save_caption


def get_thickness(draw_config: dict, rectangle: Union[List[int], np.ndarray]):
    """Get the thickness"""
    [x_min, y_min, x_max, y_max] = rectangle
    return (
        draw_config["thickness"]
        if "thickness" in draw_config
        else 1 if x_max - x_min < 40 and y_max - y_min < 40 else 2
    )


def get_transparency(draw_config):
    """Get the transparency factor."""
    return (
        draw_config["transparency_factor"]
        if "transparency_factor" in draw_config
        else 0.2
    )


class GroundingVisualizer:
    """A Visualizer to draw the grounding boxs."""

    def __init__(self, visualization_path):
        self.visualization_path = visualization_path

    def log_batch_samples(
        self,
        batch_names: List[str],
        batch_images: List[np.ndarray],
        batch_phrases: List[List[str]],
        batch_captions: List[str],
        location: Optional[str] = None,
        log_type: str = "samples",
    ):
        """
        Log the samples data.

        :param batch_names: A batch of names for images.
         for example: ['7196226216', '2824227447', '7800436386']
        :param batch_images: A batch of RGB samples.
         of shape, [bs, H, W, C]
         of pixel range, [0, 255]
        :param batch_phrases: A batch of phrases for samples,
         each item holds phrases for corresponding sample.
        :param batch_captions: A batch of captions for samples,
         each item is a caption string.
        """

        save_path = self.visualization_path

        save_path = (
            os.path.join(save_path, location) if location is not None else save_path
        )
        os.makedirs(save_path, exist_ok=True)

        for img_idx, image_name in enumerate(batch_names):
            image_name = str(batch_names[img_idx])
            image_data = batch_images[img_idx]
            image_phrases = batch_phrases[img_idx]
            image_caption = batch_captions[img_idx]

            sample_save_path = os.path.join(save_path, image_name, log_type)
            os.makedirs(sample_save_path, exist_ok=True)

            cv2.imwrite(os.path.join(sample_save_path, image_name + ".jpg"), image_data)

            save_phrases(
                image_phrases,
                save_path=os.path.join(sample_save_path, image_name + "-phrases.txt"),
            )
            save_caption(
                image_caption,
                save_path=os.path.join(sample_save_path, image_name + "-caption.txt"),
            )

    def log_batch_boxes(
        self,
        batch_names: List[str],
        batch_images: List[np.ndarray],
        batch_boxes: List[np.ndarray],
        batch_box_ids: List[np.ndarray] = None,
        batch_box_annotations: List[List[str]] = None,
        location: Optional[str] = None,
        file_name: Optional[str] = None,
        log_type: str = "samples",
        draw_config: Optional[dict] = None,
    ):
        """
        Log bboxes corresponding to phrases on the image,

        :param batch_names: A batch of names for images.
         for example: batch_names:  ['7196226216', '2824227447', '7800436386']
        :param batch_images: A batch of RGB images,
         of shape, [C, H, W]
         of pixel range, [0, 255]
        :param batch_boxes: A batch of boxes for samples,
         of length == batch size, each item is ndarray [N, 4]
        :param batch_box_ids: A batch of box ids for samples,
         of length == batch size, each item is ndarray [N]
        :param batch_box_annotations: A batch of box annotations for samples,
         of length == batch size, each item is a str list with length [N]
        """
        save_path = self.visualization_path
        save_path = os.path.join(save_path, location) if location is not None else None

        os.makedirs(save_path, exist_ok=True)

        for img_idx, image_name in enumerate(batch_names):
            image_name = str(image_name)
            prd_image_data = batch_images[img_idx]
            image_boxes = batch_boxes[img_idx]

            box_ids = batch_box_ids[img_idx] if batch_box_ids is not None else None
            box_annos = (
                batch_box_annotations[img_idx]
                if batch_box_annotations is not None
                else None
            )
            sample_save_path = os.path.join(save_path, image_name, log_type)
            os.makedirs(sample_save_path, exist_ok=True)
            file_name = file_name if file_name is not None else image_name
            self.draw_boxes(
                draw_board=prd_image_data,
                boxes=image_boxes,
                box_ids=box_ids,
                box_annos=box_annos,
                colors_mapper=cv_digital_colors_map,
                cv_colors_setter=cv_color_setup,
                save_path=os.path.join(
                    os.path.join(sample_save_path, file_name + ".jpg")
                ),
                draw_config=draw_config,
            )

    def paste_text(
        self,
        rectangle_box: np.ndarray,
        text: str,
        draw_board: np.ndarray,
        draw_config: dict,
    ):
        """Paste the text on the rectangle."""
        text = str(text)
        if text == "":
            return draw_board
        bkg_color = (
            draw_config["box_color"] if "box_color" in draw_config else (125, 125, 125)
        )
        [x_min, y_min, _, _] = rectangle_box
        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x_min = int(x_min)
        text_y_min = int(y_min)
        text_x_max = int(text_x_min + text_size[0] + 2)
        text_y_max = int(text_y_min + text_size[1] + 2)

        draw_board = cv2.rectangle(
            draw_board,
            (text_x_min, text_y_min),
            (text_x_max, text_y_max),
            bkg_color,
            cv2.FILLED,
        )
        draw_board = cv2.putText(
            draw_board,
            text,
            (text_x_min, text_y_min + text_size[1]),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        return draw_board

    def draw_one_rectangle(
        self,
        rectangle_box: np.ndarray,
        draw_board: np.ndarray,
        draw_config: dict,
    ):
        """Draw on rectangle on the board."""
        [x_min, y_min, x_max, y_max] = rectangle_box
        box_cv_color = draw_config["box_color"]
        thickness = get_thickness(draw_board, rectangle_box)
        trans_alpha = get_transparency(draw_config)
        fill_box = draw_config["fill_box"] if "fill_box" in draw_config else False
        draw_blk = None
        if fill_box:
            draw_blk = draw_board.copy()
            draw_blk = cv2.rectangle(
                draw_blk, (x_min, y_min), (x_max, y_max), box_cv_color, cv2.FILLED
            )
            # making line overlays transparent rectangle over the image
            draw_board = cv2.addWeighted(
                draw_blk, trans_alpha, draw_board, 1 - trans_alpha, 0
            )

        # draw the line on the board
        draw_board = cv2.rectangle(
            draw_board,
            (x_min, y_min),
            (x_max, y_max),
            box_cv_color,
            thickness=thickness,
        )
        return draw_board

    def draw_rectangles(
        self,
        rectangle_boxes: np.ndarray,
        rectangle_annos: np.ndarray,
        image_data: np.ndarray,
        draw_config: dict,
    ):
        """
        Draw rectangles on the board.

        :param rectangle_box: A `np.ndarray` containing rectangles to
         be plot on the `image_data`,
         of shape, [N, 4]
         of format, xyxy, unnormalized
        :param rectangle_annos: A `np.ndarray` holding the string for
         each bounding box,
         of shape, [N, ]
         of format, str or ""
        :param image_data: A `np.ndarray` matrix holding the image data,
         of shape, [3, H, W]
         of format, unnormalized image, [0, 255]
        """

        image_board = image_data.copy()
        board_h, board_w = image_data.shape[:2]

        rectangle_boxes = rectangle_boxes.astype(int)

        rectangle_boxes[:, 0::2] = np.clip(
            rectangle_boxes[:, 0::2], a_min=0, a_max=board_w
        )
        rectangle_boxes[:, 1::2] = np.clip(
            rectangle_boxes[:, 1::2], a_min=0, a_max=board_h
        )

        for rect, rect_anno in zip(rectangle_boxes, rectangle_annos):
            image_board = self.draw_one_rectangle(
                rect,
                draw_board=image_board,
                draw_config=draw_config,
            )

            image_board = self.paste_text(
                rectangle_box=rect,
                text=rect_anno,
                draw_board=image_board,
                draw_config=draw_config,
            )

        return image_board

    def draw_boxes(
        self,
        draw_board: np.ndarray,
        boxes: np.ndarray,
        box_ids: np.ndarray,
        box_annos: np.ndarray,
        save_path: str,
        colors_mapper: Dict[int, str],
        cv_colors_setter: Dict[int, str],
        draw_config: Dict[str, float],
    ):
        """Visualize the boxes of the image.

        :param draw_board: RGB data, [H, W, C]
        :param boxes: Bounding boxes, [N, 4].
        :param box_ids: Bounding boxes id, [N, ].
        :param box_annos: Same format as `box_ids`.
        :param save_path: Same as function `draw_boxes`.
        :param save_path: Same as function `draw_boxes`.
        """
        n_boxes = len(boxes)

        if box_ids is None:
            box_ids = -1 * np.ones(n_boxes)
        if box_annos is None:
            box_annos = np.full((n_boxes), "", dtype=np.dtype("U10"))

        image_new = draw_board.copy()

        unique_ids = np.unique(box_ids)
        for select_id in unique_ids:
            selected_boxes = boxes[box_ids == select_id]
            selected_box_annos = box_annos[box_ids == select_id]
            box_color_name = colors_mapper[select_id]
            box_color_steup = cv_colors_setter(box_color_name)
            box_draw_dict = draw_config.copy()
            draw_config["box_color"] = box_color_steup
            if select_id == -1:
                box_draw_dict["fill_box"] = False
            image_new = self.draw_rectangles(
                selected_boxes,
                rectangle_annos=selected_box_annos,
                image_data=image_new,
                draw_config=draw_config,
            )

        cv2.imwrite(save_path, image_new)


if __name__ == "__main__":
    pass
