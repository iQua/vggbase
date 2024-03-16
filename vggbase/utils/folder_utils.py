"""
Necessary utilities for folders.

"""
import os


def directory_contains_subfolder(directory_path: str, target_subfolder: str = None):
    """Judging whether a dir exists and a subfolder is contained.

    This is mainly for hugging face model downloding.
    """
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return False

    if target_subfolder is not None:
        # Get a list of subdirectories in the directory
        subdirectories_in_directory = [
            name
            for name in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, name))
        ]

        # Check if the directory is not empty
        if not subdirectories_in_directory:
            return False

        # Check if the target subfolder exists in the list of subdirectories
        if target_subfolder in subdirectories_in_directory:
            return True

    return False
