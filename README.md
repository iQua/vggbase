# Visual Grounding

A general platform to support the research on visual grounding --- visual and text alignment.

* [Introduction](#introduction)
* [Structures](#structures)
* [Working on this project](#working-on-this-project)


## Introduction

Visual grounding (VG) is a crucial task in the interdisciplinary subject of computer vision and natural language processing. With a focus on aligning semantically consistent phrase-region pairs from the given image and sentence, the general visual grounding problem can be extended to phrase localization and referring expression comprehension. The commonly used dataset for phrase localization is [Flickr30K Entities](https://bryanplummer.com/Flickr30kEntities/) while the corresponding datasets for referring expression comprehension are [ReferItGame](http://tamaraberg.com/referitgame/) and [Refcoco-based ones](https://github.com/lichengunc/refer).

This repository provides a fundamental structure, namely VGGBase, which achieves an easy-to-use, scalable, and flexible research platform to support the comprehensive research on the visual grounding (VG) task. Built upon the [Fabric](https://lightning.ai/docs/fabric/stable/), the features of VGGBase not only contribute to implementing an effective and efficient learning pipeline, as well as also boost the necessary components of this learning process. As a result, many different VG methods can be designed and reproduced based on the modules of the VGGBase.

## Structures

> Folder structure and functions for our project under the `vggbase` directory.

    .
    ├── configs                         # Configuration files
    ├── docs                            # Necessary documents
    ├── examples                        # The implemented VG methods
    ├── slurm_utils                     # Tools and materials to help run code on the Slurm platform
    ├── utests                          # Unit tests
    ├── models                          # Models and useful components
    ├── vggbase                          # The source code of VGGBase
    └──── boxes                         # Code for boxes
    └──── datasets                      # Code for multiple VG datasets
    └──── learners                      # Code for learning the VG models
    └──── evaluators                    # Code for evaluating the VG models
    └──── models                        # Code for building models
    └──── visualization                 # Code for visualizing the results
    └── README.md

## Working on this project
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg?style=flat-square) ![PyTorch](https://img.shields.io/badge/pytorch-1.13.1-%237732a8?style=flat-square)

Our current code works in a variety of modes, including cpu-based, single gpu-based, and multi-gpu-based learning. As the VGGBase is built upon the Fabric, the user can switch between these modes by setting the corresponding hyper-parameters.

### Installation

#### Setting up the Python environment

It is recommended that [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is used to manage Python packages. Before using our code, first install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), update your `conda` environment, and then create a new `conda` environment with Python 3.8 using the command:

```shell
conda update conda -y
conda create -n visualgrounding python=3.8
conda activate visualgrounding
```

where `visualgrounding` is the preferred name of your new environment.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

The CUDA version, used in the command above, can be obtained on Ubuntu Linux systems by using the command:

```shell
nvidia-smi
```

In macOS (without GPU support), the typical command would be:

```shell
conda install pytorch torchvision -c pytorch
```

### Setting up VGGBase

We will need to install the vggbase platform with required packages using `pip`:

```shell
pip install .
```

**Tip:** After the initial installation of the required Python packages, use the following command to upgrade all the installed packages at any time:

```shell
python upgrade_packages.py
```

If you use Visual Studio Code, it is possible to use `black` to reformat the code every time it is saved by adding the following settings to .`.vscode/settings.json`:

```
"python.formatting.provider": "black",
"editor.formatOnSave": true
```

In general, the following is the recommended starting point for `.vscode/settings.json`:

```
{
	"python.linting.enabled": true,
	"python.linting.pylintEnabled": black,
	"python.formatting.provider": "yapf",
	"editor.formatOnSave": true,
	"python.linting.pylintArgs": [
	    "--init-hook",
	    "import sys; sys.path.append('/absolute/path/to/project/home/directory')"
	],
	"workbench.editor.enablePreview": false
}
```

It goes without saying that `/absolute/path/to/project/home/directory` should be replaced with the actual path in the specific development environment.

**Tip:** When working in Visual Studio Code as your development environment, two of our colour theme favourites are called `Bluloco` (both of its light and dark variants) and `City Lights` (dark). They are both excellent and very thoughtfully designed. The `Python` extension is also required, which represents Microsoft's modern language server for Python.

### Running the workloads

The implemented visual grounding methods and other learning algorithms are placed under the folder `examples/`, which contains `configs/` holding configuration files for the corresponding implementation. 

To start a learning workload, the configuration file and the learning method should be well-prepared following the VGGBase's requirement.

0. The user needs to set the configuration file based on their own needs. For example, the users are desired to change the `project_dir` under the `environment` and `data_path` under the `data` based on their requirements.

```shell
environment:
    project_dir: /project/user

data:
    data_path: /data/user
```

1. The user must choose which method to run and prepare the dataset. 

As there are three datasets, including _ReferItGame_, _RefCOCO_, and _Flickr30K Entities_, supported by VGGBase, the corresponding data should be prepared before any operations.

To run the _ReferItGame_ and _RefCOCO_-related datasets, the COCO dataset should be downloaded using URLs presented in the configuration file `./configs/coco.yml`. 

To run the _Flickr30K Entities_ dataset, the user does not need to download the data as the program will download it automatically.

After downloading, the dataset name and where raw data is stored should be set in the configuration following Step 0.


2. To run the method, the user is expected to follow the generalized command below.

```shell
python examples/{method_name}.py -c {configuration file path} -b {project path}
```

Let us take a quick example of running the _DiffusionVG_ method.

```shell
python examples/DiffusionVG/DiffusionVG.py -c examples/configs/F30KE/DiffusionVG_resnet18_phrasebert_proposal150.yml -b PROJECT
```

Another great example is running diffusion models. The _cifar10-64_ dataset should be downloaded and set in the configuration file. Then, the user can run this method by using the command:
```shell
python examples/VisualDiffusion/visualDiffusion.py -c examples/configs/CIFAR64/visualDiffusion_cifar64.yml
```


This project uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `examples/configs/` directory.
