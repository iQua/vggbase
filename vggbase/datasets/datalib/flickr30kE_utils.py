"""
Necessary functions for the Flickr30K Entities dataset.

Note.
The `tuple` type is not supported by JSON.

"""

from typing import Callable, List, Dict
import os
import json
import xml.etree.ElementTree as ET
import logging

import cv2

from vggbase.datasets.datalib import data_utils
from vggbase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseImageSample,
    BaseBoxAnnotations,
    BaseCaptionAnnotations,
)


def filter_bad_boxes(boxes_coor):
    """Filter the boxes with wrong coordinates"""
    filted_boxes = list()
    for box_coor in boxes_coor:
        [xmin, ymin, xmax, ymax] = box_coor
        if xmin < xmax and ymin < ymax:
            filted_boxes.append(box_coor)

    return filted_boxes


def get_sentence_data(parse_file_path):
    """Parses a sentence file from the Flickr30K Entities dataset

    Args:
        parse_file_path - full file path to the sentence file to parse
    Return:
        a list of dictionaries for each sentence with the following fields:
            sentence - the original sentence
            phrases - a list of dictionaries for each phrase with the
                    following fields:
                        phrase - the text of the annotated phrase
                        first_word_index - the position of the first word of
                                            the phrase in the sentence
                        phrase_id - an identifier for this phrase
                        phrase_type - a list of the coarse categories this phrase belongs to
    """
    with open(parse_file_path, "r", encoding="utf-8") as opened_file:
        sentences = opened_file.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")

                    phrase_id.append(parts[1][3:])

                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(
            first_word, phrases, phrase_id, phrase_type
        ):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(parse_file_path):
    """Parses the xml files in the Flickr30K Entities dataset
    Args:
        parse_file_path - full file path to the annotations file to parse
    Return:
        dictionary with the following fields:
            scene - list of identifiers which were annotated as
                    pertaining to the whole scene
            nobox - list of identifiers which were annotated as
                    not being visible in the image
            boxes - a dictionary where the fields are identifiers
                    and the values are its list of boxes in the [xmin ymin xmax ymax] format
    """
    tree = ET.parse(parse_file_path)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


def align_anno_sent(
    image_sents: List[dict], image_annos: dict, phrase_type_id_mapper: Dict[str, id]
):
    """Align the items in annotations and sentences

    :param image_sents: A list, in which rach itme is a dict that contains
     'sentence', 'phrases'
    :param image_annos: A dict containing 'boxes'

    :return aligned_items: A list, in which each itme is a dict that contains
     the sentence with corresponding phrases information, there should have several
     items because for one image, there are 5 sentences. Sometimes,
     some sentences are useless, making the number of items less than 5.
    """
    aligned_items = list()  # each item is a dict
    for sent_info in image_sents:
        img_sent = sent_info["sentence"]
        img_sent_phrases = list()
        img_sent_phrases_type = list()
        img_sent_phrases_type_id = list()
        img_sent_phrases_id = list()
        img_sent_phrases_boxes = list()
        for phrase_info_idx in range(len(sent_info["phrases"])):
            phrase_info = sent_info["phrases"][phrase_info_idx]

            phrase = phrase_info["phrase"]
            # a list, such as
            # ['people']
            # ['vehicel', 'scene']
            phrase_type = phrase_info["phrase_type"]

            phrase_type_id = [phrase_type_id_mapper[ph_type] for ph_type in phrase_type]

            phrase_id = phrase_info["phrase_id"]
            if phrase_id not in image_annos["boxes"].keys():
                continue

            phrase_boxes = image_annos["boxes"][phrase_id]  # a nested list
            filted_boxes = filter_bad_boxes(phrase_boxes)
            if not filted_boxes:
                continue

            img_sent_phrases.append(phrase)
            img_sent_phrases_type.append(phrase_type)
            img_sent_phrases_type_id.append(phrase_type_id)
            img_sent_phrases_id.append(phrase_id)
            img_sent_phrases_boxes.append(filted_boxes)

        if not img_sent_phrases:
            continue

        items = dict()
        # a string shows the sentence
        items["sentence"] = img_sent
        # a list that contains the phrases
        items["sentence_phrases"] = img_sent_phrases
        # a nested list that contains phrases type
        items["sentence_phrases_type"] = img_sent_phrases_type
        items["sentence_phrases_type_id"] = img_sent_phrases_type_id
        # a list that contains the phrases id
        items["sentence_phrases_id"] = img_sent_phrases_id
        # a nested list that contains boxes for each phrase
        items["sentence_phrases_boxes"] = img_sent_phrases_boxes

        aligned_items.append(items)

    return aligned_items


def operate_integration(
    phase,
    images_path,
    images_annotations_path,
    images_sentences_path,
    meta_catalog,
    save_path,
):
    """Obtain the integrated for images"""

    logging.info("Starting %s data integration for several minutes.", phase)
    phrase_type_id_mapper = meta_catalog.phrase_category_mapper

    image_samples = []
    for image_path_idx, image_path in enumerate(images_path):
        # data/Flickr30K_Entities/vggbase_metadata.json
        image_name = image_path.split("/")[-1].split(".")[0]
        image_sent_path = images_sentences_path[image_path_idx]
        image_anno_path = images_annotations_path[image_path_idx]

        image_sents = get_sentence_data(image_sent_path)
        image_annos = get_annotations(image_anno_path)
        aligned_items = align_anno_sent(image_sents, image_annos, phrase_type_id_mapper)

        if not aligned_items:
            continue
        for item_idx, item in enumerate(aligned_items):
            image_id = image_name + "_" + str(item_idx)
            height, width = cv2.imread(image_path).shape[:2]
            image_samples.append(
                BaseImageSample(
                    image_name=image_name,
                    image_file_path=image_path,
                    image_url=None,
                    image_hw=[height, width],
                    image_id=image_id,
                    bbox_annotations=BaseBoxAnnotations(
                        bboxes=item["sentence_phrases_boxes"],
                        bboxes_mode="pascal_voc",
                        bbox_ids=None,
                        bboxes_category=item["sentence_phrases_type"],
                        bboxes_category_id=item["sentence_phrases_type_id"],
                    ),
                    caption_annotations=BaseCaptionAnnotations(
                        caption=item["sentence"],
                        caption_phrases=item["sentence_phrases"],
                        caption_phrases_id=item["sentence_phrases_id"],
                        caption_phrases_category=item["sentence_phrases_type"],
                        caption_phrases_category_id=item["sentence_phrases_type_id"],
                    ),
                )
            )

    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(
            DatasetCatalog(data_phase=phase, image_samples=image_samples),
            outfile,
            skipkeys=False,
        )


def generate_phrase_type_ids(base_types_list, image_sents):
    for sent_info in image_sents:
        for phrase_info_idx in range(len(sent_info["phrases"])):
            phrase_info = sent_info["phrases"][phrase_info_idx]
            phrase_type = phrase_info["phrase_type"][0]
            if phrase_type not in base_types_list:
                base_types_list.append(phrase_type)
    return base_types_list


def obtain_phrase_type_ids_mapper(images_name, images_sentences_path):
    generated_phrase_types_list = list()
    for image_name_idx, image_name in enumerate(images_name):
        image_sent_path = images_sentences_path[image_name_idx]

        image_sents = get_sentence_data(image_sent_path)

        generated_phrase_types_list = generate_phrase_type_ids(
            generated_phrase_types_list, image_sents
        )

    generated_phrase_types_list.sort()
    phrase_types_id_mapper = {k: v for v, k in enumerate(generated_phrase_types_list)}

    return phrase_types_id_mapper


def integrate_data(
    data_info,
    splits_info,
    data_types,
    metadata_filepath: str,
    datacatalog_filename_fn: Callable[[str], str],
):
    """Integrate the data into the format supported and required
    by VGGBase.
    """

    def visit_split_data(split_type):
        split_data_types_samples_path = []
        split_info_file = splits_info[split_type]["split_file"]
        with open(split_info_file, "r", encoding="utf-8") as loaded_file:
            split_data_samples_id = [
                sample_id.split("\n")[0] for sample_id in loaded_file.readlines()
            ]
        for _, data_type in enumerate(data_types):
            data_of_type_format = data_info[data_type]["format"]
            data_of_type_path = data_info[data_type]["path"]

            split_data_type_samples_path = [
                os.path.join(data_of_type_path, sample_id + "." + data_of_type_format)
                for sample_id in split_data_samples_id
            ]

            split_data_types_samples_path.append(split_data_type_samples_path)

        return split_data_types_samples_path

    # obtain the phrase type id mapper
    meta_catalog = DatasetMetaCatalog(dataset_name="Flickr30K Entities")
    if os.path.exists(metadata_filepath):
        warn_info = f"Loading existed meta data from {metadata_filepath}"
        logging.info(warn_info)

        with open(metadata_filepath, "r", encoding="utf-8") as fp:
            loaded_meta_catalog = json.load(fp)

        meta_catalog["dataset_name"] = loaded_meta_catalog["dataset_name"]
        meta_catalog["phrase_category_mapper"] = loaded_meta_catalog[
            "phrase_category_mapper"
        ]

    else:
        train_datatypes_path = visit_split_data(split_type="train")
        phrase_type_id_mapper = obtain_phrase_type_ids_mapper(
            images_name=train_datatypes_path[0],
            images_sentences_path=train_datatypes_path[2],
        )

        # create meta data catalog
        meta_catalog["phrase_category_mapper"] = phrase_type_id_mapper
        logging.info("Saving metadata catalog to %s.", metadata_filepath)
        with open(metadata_filepath, "w", encoding="utf-8") as fp:
            json.dump(meta_catalog, fp, skipkeys=False)

    for split_type in splits_info:
        datacatalog_filepath = datacatalog_filename_fn(split_type)

        if os.path.exists(datacatalog_filepath):
            warn_info = f"Integrated {split_type}, but file already existed"
            logging.info(warn_info)
            continue

        split_data_types_samples_path = visit_split_data(split_type)

        operate_integration(
            phase=split_type,
            images_path=split_data_types_samples_path[0],
            images_annotations_path=split_data_types_samples_path[1],
            images_sentences_path=split_data_types_samples_path[2],
            meta_catalog=meta_catalog,
            save_path=datacatalog_filepath,
        )

        logging.info("Integration for %s Done!", split_type)


def create_splits_data(data_info, splits_info, data_types):
    """Create datasets for different splits."""
    # saveing the images and entities to the corresponding directory
    for split_type in list(splits_info.keys()):
        logging.info("Creating split %s data..........", split_type)
        # obtain the split data information
        # 0. getting the data
        split_info_file = splits_info[split_type]["split_file"]
        with open(split_info_file, "r", encoding="utf-8") as loaded_file:
            split_data_samples = [
                sample_id.split("\n")[0] for sample_id in loaded_file.readlines()
            ]
        splits_info[split_type]["num_samples"] = len(split_data_samples)

        # 1. create directory for the splited data if necessary
        for dt_type in data_types:
            split_dt_type_path = splits_info[split_type][dt_type]["path"]

            if not data_utils.is_data_exist(split_dt_type_path):
                os.makedirs(split_dt_type_path, exist_ok=True)
            else:
                logging.info("The path %s does exist", split_dt_type_path)
                continue

            raw_data_type_path = data_info[dt_type]["path"]
            raw_data_format = data_info[dt_type]["format"]
            split_samples_path = [
                os.path.join(raw_data_type_path, sample_id + raw_data_format)
                for sample_id in split_data_samples
            ]
            # 2. saving the splited data into the target file
            data_utils.copy_files(split_samples_path, split_dt_type_path)

    logging.info(" Done!")
