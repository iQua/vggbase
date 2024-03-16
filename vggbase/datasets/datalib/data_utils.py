"""
Useful tools for processing the data

"""
import logging
import shutil
import os
import re
import json

import numpy as np
import gdown
from torchvision.datasets.utils import download_url, extract_archive


def exist_file_in_dir(tg_file_name, search_dir, is_partial_name=True):
    """Judge whether the input file exists in the search_dir."""
    # the tg_file_name matches one file if it match part of the file name
    if is_partial_name:
        is_included_fuc = lambda src_f_name: tg_file_name in src_f_name
    else:
        is_included_fuc = lambda src_f_name: tg_file_name == src_f_name
    is_existed = any([is_included_fuc(f_name) for f_name in os.listdir(search_dir)])

    return is_existed


def is_data_exist(target_path):
    """Judeg whether the input file/dir existed and whether it contains useful data"""
    if not os.path.exists(target_path):
        logging.info("The path %s does not exist", target_path)
        return False

    # # remove all .DS_Store files
    # command = ['find', '.', '-name', '".DS_Store"', '-delete']
    # command = ' '.join(command)
    # #cmd = f"find . -name ".DS_Store" -delete"
    # subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    def get_size(folder):
        # get size
        size = 0
        for ele in os.scandir(folder):
            if not ele.name.startswith("."):
                size += os.path.getsize(ele)
            if size > 0:
                return size
        return size

    def is_contain_useful_file(target_dir):
        """Return True once reaching one useful file"""
        for _, _, files in os.walk(target_dir):
            for file in files:
                # whether a useful file
                if not file.startswith("."):
                    return True
        return False

    if os.path.isdir(target_path):
        if get_size(target_path) == 0 or not is_contain_useful_file(target_path):
            logging.info("The path %s does exist but contains no data", target_path)
            return False
        else:
            return True

    logging.info("The file %s does exist", target_path)
    return True


def dict_list2tuple(dict_obj):
    """Convert all list element in the dict to tuple"""
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            for inner_key, inner_v in value.items():
                if isinstance(inner_v, list):
                    # empty or None list, mainly for meta_keys
                    if not value or inner_v[0] is None:
                        dict_obj[key][inner_key] = ()
                    else:
                        dict_obj[key][inner_key] = tuple(inner_v)
        else:
            if isinstance(value, list):
                # empty or None list, mainly for meta_keys
                if not value or value[0] is None:
                    dict_obj[key] = ()
                else:
                    dict_obj[key] = tuple(value)
                for idx, item in enumerate(value):
                    item = value[idx]
                    if isinstance(item, dict):
                        value[idx] = dict_list2tuple(item)

    return dict_obj


def phrase_boxes_alignment(flatten_boxes, ori_phrases_boxes):
    """Align the phase and its corresponding boxes"""
    phrases_boxes = list()

    ori_pb_boxes_count = list()
    for ph_boxes in ori_phrases_boxes:
        ori_pb_boxes_count.append(len(ph_boxes))

    strat_point = 0
    for pb_boxes_num in ori_pb_boxes_count:
        sub_boxes = list()
        for i in range(strat_point, strat_point + pb_boxes_num):
            sub_boxes.append(flatten_boxes[i])

        strat_point += pb_boxes_num
        phrases_boxes.append(sub_boxes)

    pb_boxes_count = list()
    for ph_boxes in phrases_boxes:
        pb_boxes_count.append(len(ph_boxes))

    assert pb_boxes_count == ori_pb_boxes_count

    return phrases_boxes


def list_inorder(listed_files, flag_str):
    """ " List the files in order based on the file name"""
    filtered_listed_files = [fn for fn in listed_files if flag_str in fn]
    listed_files = sorted(filtered_listed_files, key=lambda x: x.strip().split(".")[0])
    return listed_files


def copy_files(src_files, dst_dir):
    """copy files from src to dst"""
    for file in src_files:
        shutil.copy(file, dst_dir)


def union_shuffled_lists(src_lists):
    """shuffle the lists"""
    for i in range(1, len(src_lists)):
        assert len(src_lists[i]) == len(src_lists[i - 1])
    processed = np.random.permutation(len(src_lists[0]))

    return [np.array(ele)[processed] for ele in src_lists]


def read_anno_file(anno_file_path):
    """Read the annotation file."""
    _, tail = os.path.split(anno_file_path)
    file_type = tail.split(".")[-1]

    if file_type == "json":
        with open(anno_file_path, "r", encoding="utf-8") as anno_file:
            annos_list = json.load(anno_file)
    else:
        with open(anno_file_path, "r", encoding="utf-8") as anno_file:
            annos_list = anno_file.readlines()

    return annos_list


def extract_compression_file_extension(filename):
    """Extracting the extension of the filename."""
    return re.search(r"(\.(?:zip|tar\.gz|tar))$", filename).group(1)


def remove_compression_file_extension(filename):
    """Removing the extension of the filename."""
    pattern = r"\.(zip|tar\.gz|tar)$"
    return re.sub(pattern, "", filename)


def is_compression_file(filename):
    """Whether this is a compression file."""
    pattern = r"\.(zip|tar\.gz|tar)$"
    return bool(re.search(pattern, filename))


def download_google_driver_data(
    download_file_id,
    extract_download_file_name,
    put_data_dir,
):
    """Download the data from google drvier."""
    download_data_file_name = extract_download_file_name + ".zip"
    download_data_path = os.path.join(put_data_dir, download_data_file_name)
    extract_data_path = os.path.join(put_data_dir, extract_download_file_name)
    if not is_data_exist(download_data_path) and not is_data_exist(extract_data_path):
        logging.info("Downloading the data to %s", download_data_path)

        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=t&id={download_file_id}",
            output=download_data_path,
        )

    if not is_data_exist(extract_data_path):
        logging.info("Extracting from %s", download_data_path)
        extract_archive(
            from_path=download_data_path,
            to_path=put_data_dir,
            remove_finished=False,
        )


def download_url_data(
    download_url_address,
    put_data_dir,
    extract_to_dir=None,
    obtained_file_name=None,
):
    """Download the raw data from one url address."""

    # Extract to the same dir as the download dir
    if extract_to_dir is None:
        extract_to_dir = put_data_dir

    download_file_name = os.path.basename(download_url_address)
    download_file_path = os.path.join(put_data_dir, download_file_name)

    download_extracted_file_name = remove_compression_file_extension(download_file_name)
    download_extracted_dir_path = os.path.join(
        extract_to_dir, download_extracted_file_name
    )
    obtained_file_path = download_extracted_dir_path
    if obtained_file_name is not None:
        obtained_file_path = os.path.join(extract_to_dir, obtained_file_name)

    if is_data_exist(obtained_file_path):
        return

    elif is_data_exist(download_extracted_dir_path):
        os.rename(download_extracted_dir_path, obtained_file_path)
        logging.info(
            "Renaming %s to %s dir.....",
            download_extracted_dir_path,
            obtained_file_path,
        )
    else:
        # Download the raw data if necessary
        if not is_data_exist(download_file_path):
            logging.info(
                "Downloading data %s to %s", download_file_name, download_file_path
            )
            download_url(
                url=download_url_address,
                root=put_data_dir,
                filename=download_file_name,
            )

        # Extract the data to the specific dir
        if is_compression_file(download_file_name):
            logging.info(
                "Extracting %s to %s dir.....",
                download_file_path,
                download_extracted_dir_path,
            )
            extract_archive(
                from_path=download_file_path,
                to_path=extract_to_dir,
                remove_finished=False,
            )

        if download_extracted_dir_path != obtained_file_path:
            os.rmdir(obtained_file_path)
            os.rename(download_extracted_dir_path, obtained_file_path)
            logging.info(
                "Renaming %s to %s dir.....",
                download_extracted_dir_path,
                obtained_file_path,
            )

    return obtained_file_path
