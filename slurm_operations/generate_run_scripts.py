"""
Generate the running scripts for the Sim Server
based on the implemented method and configs

One Example:

#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=./slurm_loggings/{dataset_name}/{config_file_name}.out

For multi-GPUs:
#!/bin/bash
#SBATCH --time=25:00:00                 # 25 hours
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=3                      # total number of tasks across all nodes, i.e., nodes x gres
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3                    # request 3 GPUs per node
#SBATCH --mem=32G                       # 32GB RAM
#SBATCH --output=./slurm_loggings/{config_file_name}.out      # stdout/stderr file


Important:
    Before setting the resource requirement, please check the
exact resources on the server.

Dell Precision 7920 Tower Workstation,
    - 2 x Intel Xeon CPUs with 20 CPU cores each,
    - 1TB Intel NVMe PCIe SSD boot drive
    - 12TB data drive (two 6TB drives),
    - 256GB physical memory
    - 3 x NVIDIA RTX A4500 GPU
        20GB CUDA memory each.
#CPUs, lscpu


Just run:
    python generate_run_scripts.py


"""

import argparse
import os
import stat
import logging
from pathlib import Path
import glob

from slurm_utils import load_yml_config

logging.getLogger().setLevel(logging.INFO)


def search_file(folder_path, target_file):
    """Search the file under one foler."""
    for root, _, files in os.walk(folder_path):
        if target_file in files:
            return os.path.join(root, target_file).replace("~", str(Path.home()))
    return None


def extract_method_file(methods_root_dir, config_file_name):
    """Extract the Python file path of the corresponding method."""
    extracted_method_name = config_file_name.split("_")[0].split(".")[0]

    desired_file_name = extracted_method_name + ".py"

    method_python_file_path = search_file(methods_root_dir, desired_file_name)

    if method_python_file_path is None or not os.path.exists(method_python_file_path):
        logging.info(
            "Skipping as the method for %s is not existed!...\n", config_file_name
        )
        extracted_method_name = None
    return extracted_method_name, method_python_file_path


# the running policy hould be folloed
GPUs_setups = {
    1: {"n_cpus": 12, "memory": 72, "time": 24},
    2: {"n_cpus": 24, "memory": 144, "time": 24},
    3: {"n_cpus": 36, "memory": 216, "time": 12},
}


def generate_run_script(
    run_n_gpus: dict,
    desire_python: str,
    method_code_filepath: str,
    config_filepath: str,
    output_logging_path: str,
    data_out_path: str,
):
    """Generate the run script as  string."""
    run_mode = GPUs_setups[run_n_gpus]
    mode_gpus = n_gpus
    mode_time = run_mode["time"]
    mode_cpus = run_mode["n_cpus"]
    mode_memory = run_mode["memory"]

    header = "#!/bin/bash"
    time_line = f"#SBATCH --time={mode_time}:00:00"
    cpus_line = f"#SBATCH --cpus-per-task={mode_cpus}"
    gpu_line = f"#SBATCH --gres=gpu:{mode_gpus}"
    mem_line = f"#SBATCH --mem={mode_memory}G"

    output_line = f"#SBATCH --output={output_logging_path}.out"

    run_code_line = f"\n{desire_python} {method_code_filepath} -c {config_filepath} -b {data_out_path}"
    run_string = " \n".join(
        [header, time_line, cpus_line, gpu_line, mem_line, output_line, run_code_line]
    )

    return run_string


def save_run_script(script_content, script_file_save_path):
    """Save the run script content to the file."""
    logging.info("Saving the generated sbatch script to %s", script_file_save_path)

    if os.path.exists(script_file_save_path):
        logging.info("Already existed, skipping...\n")
    else:
        with open(script_file_save_path, mode="w", encoding="utf-8") as script_file:
            script_file.write(script_content)

        # add
        #   - stat.S_IRWXU: Read, write, and execute by owner 7
        #   - stat.S_IRGRP : Read by group
        #   - stat.S_IXGRP : Execute by group
        #   - stat.S_IROTH : Read by others
        #   - stat.S_IXOTH : Execute by others
        os.chmod(
            script_file_save_path,
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./slurm_config.yml",
        help="the configuration file for slurm",
    )

    parser.add_argument(
        "-g",
        "--nGPUs",
        type=int,
        default=1,
        help="number of gpus used for running.",
    )

    parser.add_argument(
        "-e",
        "--exclusive",
        type=str,
        default="base",
        help="keywords of methods that will not be considered.",
    )

    args = parser.parse_args()
    config_path = args.config
    n_gpus = args.nGPUs
    exclusive_keywords = args.exclusive.split(",")

    loaded_config_data = load_yml_config(config_path)

    HOMEPATH = str(Path.home())

    # target python used
    target_python_path = loaded_config_data["run"]["python_path"]
    target_python_path = target_python_path.replace("~", HOMEPATH)
    # number of GPUs used
    n_gpus = loaded_config_data["run"]["n_gpus"]
    # the path of all configuration files
    configs_files_dir = loaded_config_data["run"]["configs_dir"]
    configs_files_dir = configs_files_dir.replace("~", HOMEPATH)
    configs_files_path = glob.glob(os.path.join(configs_files_dir, "*/", "*.yml"))
    # the path of all implemented methods
    methods_dir = loaded_config_data["run"]["examples_dir"]
    # where to put the logging file of Slurm
    slurm_logging_dir = loaded_config_data["run"]["slurm_logging_dir"]
    slurm_logging_dir = slurm_logging_dir.replace("~", HOMEPATH)
    # where to store the generated script
    script_out_dir = loaded_config_data["run"]["script_out_dir"]
    script_out_dir = script_out_dir.replace("~", HOMEPATH)
    # where to store outputs of VGGbase
    vgb_data_dir = loaded_config_data["server"]["vgb_data_dir"]
    vgb_data_basename = loaded_config_data["server"]["vgb_data_basename"]
    vgb_data_dir = vgb_data_dir.replace("~", HOMEPATH)
    vgb_data_path = os.path.join(vgb_data_dir, vgb_data_basename)

    # create directories
    os.makedirs(slurm_logging_dir, exist_ok=True)
    os.makedirs(script_out_dir, exist_ok=True)

    for file_path in configs_files_path:
        # obtain the config file name
        file_name = os.path.basename(file_path)
        file_name_no_extension = file_name.split(".")[0]
        # obtain the code of the corresponding method
        method_name, method_code_file_path = extract_method_file(methods_dir, file_name)

        # filter invalid dir
        if (
            any([key_word in file_name for key_word in exclusive_keywords])
            or method_name is None
        ):
            continue

        config_dir_path = os.path.dirname(file_path)
        # obtain the dir name of the config file
        config_dir_name = os.path.basename(config_dir_path)

        # save script to the same subdir with the same dir name
        # of the config
        script_save_dir = os.path.join(script_out_dir, config_dir_name)
        script_save_file_path = os.path.join(
            script_save_dir, file_name_no_extension + ".sh"
        )
        vgdata_out_path = os.path.join(vgb_data_path, config_dir_name)
        sbatch_logging_dir = os.path.join(slurm_logging_dir, config_dir_name)

        sbatch_logging_path = os.path.join(sbatch_logging_dir, file_name_no_extension)
        os.makedirs(script_save_dir, exist_ok=True)
        os.makedirs(sbatch_logging_dir, exist_ok=True)

        save_run_script(
            generate_run_script(
                run_n_gpus=n_gpus,
                desire_python=target_python_path,
                method_code_filepath=method_code_file_path,
                config_filepath=file_path,
                output_logging_path=sbatch_logging_path,
                data_out_path=vgdata_out_path,
            ),
            script_save_file_path,
        )
