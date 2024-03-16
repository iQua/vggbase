"""
The implementation of utilities for refer-it-game datasets

"""

import os
import logging
import json
from typing import Callable

import cv2

from vggbase.datasets.data_generic import (
    DatasetCatalog,
    BaseImageSample,
    BaseBoxAnnotations,
    BaseCaptionAnnotations,
)


def operate_integration(
    phase,
    dataset_refer_op,
    split_samples_id,
    save_path,
):
    image_samples = []
    for refer_id in split_samples_id:
        ref = dataset_refer_op.loadRefs(refer_id)[0]
        image_id = ref["image_id"]

        image_name = str(image_id)
        image_path = dataset_refer_op.loadImgspath(image_id)[0]
        height, width = cv2.imread(image_path).shape[:2]
        caption_phrases_cate = dataset_refer_op.Cats[ref["category_id"]]
        caption_phrases_cate_id = ref["category_id"]

        sent_idx = 0
        for sent in ref["sentences"]:
            # a string,
            caption = sent["sent"]
            # convert the the list, thus be formal type
            image_id = image_name + "_" + str(sent_idx)

            caption_phrase = [caption]
            # [x, y, w, h]
            caption_phrase_bboxes = dataset_refer_op.getRefBox(ref["ref_id"])

            image_samples.append(
                BaseImageSample(
                    image_name=image_name,
                    image_file_path=image_path,
                    image_url=None,
                    image_hw=[height, width],
                    image_id=image_id,
                    bbox_annotations=BaseBoxAnnotations(
                        bboxes=[[caption_phrase_bboxes]],
                        bboxes_mode="coco",
                        bbox_ids=None,
                        bboxes_category=[[caption_phrases_cate]],
                        bboxes_category_id=[[caption_phrases_cate_id]],
                    ),
                    caption_annotations=BaseCaptionAnnotations(
                        caption=caption,
                        caption_phrases=caption_phrase,
                        caption_phrases_id=[str(caption_phrases_cate_id)],
                        caption_phrases_category=[[caption_phrases_cate]],
                        caption_phrases_category_id=[[caption_phrases_cate_id]],
                    ),
                )
            )
            sent_idx += 1

    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(
            DatasetCatalog(data_phase=phase, image_samples=image_samples),
            outfile,
            skipkeys=False,
        )


def integrate_data(
    splits_info,
    dataset_refer_op,
    metadata_filepath: str,
    datacatalog_filename_fn: Callable[[str], str],
):
    """Integrate the data into the format supported and required
    by VGGBase."""

    for split_type in splits_info:
        datacatalog_filepath = datacatalog_filename_fn(split_type)
        if os.path.exists(datacatalog_filepath):
            warn_info = f"Integrated {split_type}, but file already existed"
            logging.info(warn_info)
            continue

        split_filepath = splits_info[split_type]["split_file"]
        split_samples_id = None

        with open(split_filepath, "r", encoding="utf-8") as f:
            split_samples_id = json.load(f)
        operate_integration(
            phase=split_type,
            dataset_refer_op=dataset_refer_op,
            split_samples_id=split_samples_id,
            save_path=datacatalog_filepath,
        )
