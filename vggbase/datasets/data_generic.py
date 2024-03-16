"""
The generic terms for the multimodal datasets.
"""

from typing import List, Optional, Union, Dict
from dataclasses import dataclass

import numpy as np

from vggbase.boxes.bbox_generic import BaseSampleBBoxes, BaseVGBBoxes
from vggbase.utils.generic_components import FieldFrozenContainer, BaseVGList
from vggbase.utils.tensor_utils import BaseNestedTensor, DynamicMaskNestedTensor


@dataclass
class BaseBoxAnnotations(FieldFrozenContainer):
    """The basic bounding box (bbox) annotations contained in
    one image sample."""

    bboxes: Optional[List[List[int]]] = None
    bboxes_mode: Optional[str] = None
    bbox_ids: Optional[List[int]] = None
    bboxes_category: Optional[List[List[str]]] = None
    bboxes_category_id: Optional[List[List[int]]] = None


@dataclass
class BaseCaptionAnnotations(FieldFrozenContainer):
    """The basic annotations contained in one image sample."""

    caption: Optional[str] = None
    caption_phrases: Optional[List[str]] = None
    caption_phrases_id: Optional[List[int]] = None
    caption_phrases_category: Optional[List[List[str]]] = None
    caption_phrases_category_id: Optional[List[List[int]]] = None


@dataclass
class BaseImageSample(FieldFrozenContainer):
    """The basic common items contained in one image sample."""

    image_name: Optional[str] = None
    image_file_path: Optional[str] = None
    image_url: Optional[str] = None
    image_hw: Optional[List[int]] = None
    image_id: Optional[Union[str, int]] = None

    bbox_annotations: Optional[BaseBoxAnnotations] = None
    caption_annotations: Optional[BaseCaptionAnnotations] = None


@dataclass
class DatasetStatistics(FieldFrozenContainer):
    """The statistics of the dataset."""

    num_samples: int
    num_phrases_per_image: Optional[Dict[int, int]] = None
    num_bboxes_per_query: Optional[Dict[int, int]] = None
    num_bboxes_per_phrase: Optional[Dict[int, int]] = None


@dataclass
class DatasetMetaCatalog(FieldFrozenContainer):
    """The meta data information of one dataset.

    The design principle of this class derives from the
    `detectron2` package.

    It holds the meta information of one dataset

    Args:
        dataset_name: Holding the name of the dataset.
        phrase_category_mapper: Holding the mapper from phrase name
         to category id.

    len(phrase_classes) == len(phrase_classes_id)
    """

    dataset_name: str
    phrase_category_mapper: Optional[Dict[str, int]] = None


@dataclass
class DatasetCatalog(FieldFrozenContainer):
    """The samples catalog of one dataset."""

    data_phase: str
    image_samples: Optional[List[BaseImageSample]] = None
    data_statistics: Optional[DatasetStatistics] = None


@dataclass
class BaseVGSample(FieldFrozenContainer):
    """One basic sample of visual grounding task.

    Args:
        sample_id: A `str` presenting the image file id.
        image_data: A `np.ndarray` containing the loaded image
         data, of shape [H, W, C]
        caption: A `str` containing the text description of the
         image.
        caption_phrases: A `List` in which each item is a `str`
         showing the noun-phrase of the ``caption``.
        caption_phrases_category: A `List` in which each item is
         a 1-length list containing the categories of the corresponding
         phrase. Two examples,
          - [['people'], ['other']
          - [['vehicle', 'scene'], ['people']
        caption_phrases_category_id: A `List` in which each item is
         a 1-length list containing the category id.
         Same typle and shape as the ``caption_phrases_category``,
         For example, [[6], [2], [5]].
        caption_phrases_id: A `List` in which each item is
         a `str` containing the corresponding phrase id,
         For example, [['41678'], ['48912'], ['28912']]
        caption_phrases_bboxes: A `BaseSampleBBoxes` in which the ``bboxes``
         term is a `List` - each item is `List` containing bboxes for the
         corresponding phrase,
         See `BaseSampleBBoxes` above for details.
    """

    sample_id: Optional[str] = None
    image_data: Optional[np.ndarray] = None
    caption: Optional[str] = None
    caption_phrases: Optional[List[str]] = None
    caption_phrases_category: Optional[List[List[str]]] = None
    caption_phrases_category_id: Optional[List[List[int]]] = None
    caption_phrases_id: Optional[List[int]] = None
    caption_phrases_bboxes: Optional[BaseSampleBBoxes] = None

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # judge whether data samples satisfy the
        # defined structures
        if key == "image_data":
            assert isinstance(value, np.ndarray)
        if key == "caption":
            assert isinstance(value, str)
        if key == "caption_phrases":
            assert isinstance(value, list) and isinstance(value[0], str)
        if key == "caption_phrases_category":
            assert (
                isinstance(value, list)
                and isinstance(value[0], list)
                and isinstance(value[0][0], str)
            )
        if key == "caption_phrases_category_id":
            assert (
                isinstance(value, list)
                and isinstance(value[0], list)
                and isinstance(value[0][0], int)
            )
        if key == "caption_phrases_id":
            assert isinstance(value, list) and isinstance(value[0], str)
        if key == "caption_phrases_bboxes":
            assert isinstance(value, BaseSampleBBoxes)


@dataclass
class BaseInputTarget(FieldFrozenContainer):
    """The targets of the samples.

    Args:
        sample_id: A `str` holding the sample id.
        caption: A `str` holding the caption.
        caption_phrases: A `List` holding phrases of the caption.
        vg_bboxes: A `BaseVGBBoxes` holding bounding boxes
         as the target.
    """

    sample_id: Optional[str] = None
    caption: Optional[str] = None
    caption_phrases: Optional[List[str]] = None
    vg_bboxes: Optional[BaseVGBBoxes] = None


@dataclass
class BaseVGCollatedSamples(FieldFrozenContainer):
    """
    Base class for collated samples in VGGbase.

    Args:
        rgb_samples: A `BaseNestedTensor` holding rgbs
         of one batch of RGB samples,
         of shape, [bs, C, H, W]
        text_samples: A `DynamicMaskNestedTensor` holding
         text data processed by the tokenizer,
         of shape, [bs, P, L]
        targets: A `BaseVGList` holding the target
         of each sample,
         of shape, len(targets) == bs
    """

    rgb_samples: Optional[BaseNestedTensor] = None
    text_samples: Optional[DynamicMaskNestedTensor] = None
    targets: Optional[BaseVGList[BaseInputTarget]] = None
