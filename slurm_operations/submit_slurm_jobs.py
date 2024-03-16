"""
Implementation of running multiples experiments with Slurm.
"""

import argparse
import logging
import glob
import os
from pathlib import Path

from slurm_utils import load_yml_config


def is_desired_file(key_word: str, file_name: str):
    """Whether the file name is the desired file defiend by key."""

    # if key_word is all, all files are desired.
    if key_word == "all":
        return True
    if key_word in file_name:
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="diffusion",
        help="scripts of which folder to run",
    )

    parser.add_argument(
        "-d",
        "--dirname",
        type=str,
        default="diffusion",
        help="scripts of which folder to run",
    )

    parser.add_argument(
        "-k", "--key", type=str, default="all", help="The key word of desired scripts."
    )

    args = parser.parse_args()
    config_path = args.config
    run_dir_name = args.dirname
    key = args.key

    loaded_config_data = load_yml_config(config_path)

    HOMEPATH = str(Path.home())
    # where to the scripts are stored
    scripts_dir = loaded_config_data["run"]["script_out_dir"]
    scripts_dir = scripts_dir.replace("~", HOMEPATH)

    experiment_script_files_path = glob.glob(
        os.path.join(scripts_dir, run_dir_name, "*.sh")
    )

    desired_files_path = [
        file_path
        for file_path in experiment_script_files_path
        if is_desired_file(key, file_path)
    ]
    for script_file_path in desired_files_path:
        logging.info("Running script: %s", script_file_path)
        os.system(f"sbatch {script_file_path}")
