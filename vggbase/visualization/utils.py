"""
Useful tools for visualization.
"""

import os
import re


def save_phrases(phrases: list, save_path: str):
    """Save a list phrases of the image to the file."""

    with open(save_path, "w", encoding="utf-8") as file:
        for phrase in phrases:
            if isinstance(phrase, int):
                phrase = [str(phrase)]
            if isinstance(phrase, str):
                phrase = [phrase]
            s = " ".join(map(str, phrase))
            file.write(s + "\n")


def save_caption(caption: str, save_path: str):
    """Save a caption of the image to the file."""
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(caption)


def create_new_folder(save_dir: str):
    """
    Create a new folder in the root directory by adding a integer at the end of
    existing folder name.
    """
    if not os.path.exists(save_dir):
        return save_dir

    save_dir_name = os.path.basename(save_dir)
    parent_dir = os.path.dirname(save_dir)

    # Get names of all existing folder
    existed_folders = [
        dir_name
        for dir_name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, dir_name))
        and save_dir_name in dir_name
    ]
    # Extract the integers from the folder names
    folder_numbers = [
        int(re.search(r"\d+", dir_name.split("_")[-1]).group())
        for dir_name in existed_folders
        if re.search(r"\d+", dir_name.split("_")[-1])
    ]
    # Increment the largest integers by 1 to get
    # the new integers to create a new folder name
    sorted_numbers = sorted(folder_numbers)
    if sorted_numbers:
        new_number = sorted_numbers[-1] + 1
    else:
        new_number = "0"

    new_folder = os.path.join(parent_dir, save_dir_name + str(new_number))
    return new_folder
