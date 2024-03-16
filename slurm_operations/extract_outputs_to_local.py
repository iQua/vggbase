"""
The implementation of sending the results obtained on the Sim server
 to the local computer.

The type of the experimental results should be provided by:
    - type

The type can be:
    - all, sending all logging/models/checkpoints/results to the local
    - slurm_loggings, sending logging only
    - loggings, sending logging dir of VGGbase
    - models, sending models only
    - checkpoints, sending checkpoints only
    - results, sending results only

For example:

    python extract_outputs_to_local.py -c slurm_config.yml

"""

import argparse
import os
import logging

logging.getLogger().setLevel(logging.INFO)

from slurm_utils import load_yml_config, get_server_scp_path


slurm_data_names = ["slurm_loggings"]
VGGbase_data_names = [
    "models",
    "checkpoints",
    "results",
    "loggings",
]


def obtain_data_path(config_data: dict):
    """Obtain the data path in response to the data type."""
    data_type = config_data["data"]["type"]

    server_address = config_data["server"]["address"]
    server_logging_dir = config_data["server"]["logging_dir"]
    server_vgb_data_dir = config_data["server"]["vgb_data_dir"]
    server_vgb_data_basename = config_data["server"]["vgb_data_basename"]

    local_logging_dir = config_data["local"]["logging_dir"]
    local_vgb_data_dir = config_data["local"]["vgb_data_dir"]
    local_data_basename = config_data["local"]["data_basename"]

    extract_server_paths = []
    to_local_paths = []

    data_names = slurm_data_names + VGGbase_data_names

    for dt_name in data_names:
        if dt_name == data_type or data_type == "all":
            if dt_name == "slurm_loggings":
                server_data_dir = server_logging_dir
                local_data_dir = local_logging_dir
            else:
                server_data_dir = os.path.join(
                    server_vgb_data_dir, server_vgb_data_basename
                )
                local_data_dir = local_vgb_data_dir

            server_scp_data_path = get_server_scp_path(server_address, server_data_dir)

            src_data_path = os.path.join(server_scp_data_path, dt_name)

            target_data_path = os.path.join(local_data_dir, local_data_basename)
            extract_server_paths.append(src_data_path)
            to_local_paths.append(target_data_path)

    return extract_server_paths, to_local_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./slurm_config.yml",
        help="the configuration file",
    )

    args = parser.parse_args()
    config_path = args.config

    loaded_config_data = load_yml_config(config_path)

    server_paths, local_paths = obtain_data_path(loaded_config_data)
    print(server_paths)
    for server_path, local_path in zip(server_paths, local_paths):
        logging.info("Extracting the %s from sim to local %s", server_path, local_path)
        os.makedirs(local_path, exist_ok=True)
        os.system(f"scp -r {server_path} {local_path}")
