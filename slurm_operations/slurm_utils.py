"""
Useful functions for slurm operations.
"""

import os

import yaml

from vggbase.config import Config, Loader


def load_yml_config(file_path: str) -> dict:
    """Load the configuration data from a yml file."""
    yaml.add_constructor("!include", Config.construct_include, Loader)
    yaml.add_constructor("!join", Config.construct_join, Loader)

    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as config_file:
            config = yaml.load(config_file, Loader)
    else:
        # if the configuration file does not exist, raise an error
        raise ValueError("A configuration file must be supplied.")

    return config


def get_server_scp_path(server_address: str, server_dir: str):
    """Obtain the server path."""

    return server_address + ":" + server_dir
