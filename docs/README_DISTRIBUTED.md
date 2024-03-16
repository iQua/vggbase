# Distributed training preparation
This document is expected to give a brift description of distributed training on the PyTorch. The major content consists of 1. basic principles; 2. basic structures; 3; useful materials. Specifically, the Slurm workload manager will be discussed for smoothly running the project on the server.

![Python](https://img.shields.io/badge/python-3.5%20%7C%203.7-blue.svg?style=flat-square)
![PyTorch](https://img.shields.io/badge/pytorch-1.11.0-%237732a8?style=flat-square)


#### Summary

* [Introduction](#introduction)
* [Principles](#principles)
* [Structures](#structures)
* [Materials](#materials)
* [Slurm](#slurm)
* [Slurm and Pytorch](#slurm-and-pytorch)


## Introduction

With the increase in data volume, data modality number, and model complexity in the machine learning domain, the training time for one model is prone to reaching multiple days. This makes the study of exploring how to train a model efficiently a hotspot of community work. Among all [solutions](https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/) for this objective, distributed training is the most intuitive and powerful tool as the training process can be split into multiple tasks that are performed synchronously on multiple computing resources, such as multiple GPUs. For instance, it is generally more than three times faster after transferring a single-GPU training to multiple-GPU training. Noticing such a trend, the  PyTorch library includes *torch.distributed* package, which helps users scale out the training procedure to eliminate the learning time significantly. It is necessary to figure out that PyTorch is not motivated by this trend to provide the package. In contrast, the support for distributed training behaves as the fundamental tool for any professional platform.

This document aims to provide a snapshot of the basic principles and terminologies used by current distributed training tools, how PyTorch implements those tools, and finally, what can be used in Slurm to perfectly achieve distributed training in multiple nodes.

## Principles

Distributed computing refers to the way of writing a program that makes use several distinct components connected over network. Typically, large scale computation is achieved by such an arrangement of computers capable of handling high density numeric computations in parallel. In distributed computing terminology, these computers are often referred to as nodes and a collection of such nodes form a cluster over the network. These nodes are usually connected via Ethernet, but other high-bandwidth networks are also used to take full advantage of the distributed architecture.

Here are mainly three types of realistic environments for distributed training. The most general case is the **single node single GPU training**. Then, the **single-node multi-GPU training** is what some small labs support. The most powerful one is the **multi-node multi-GPU training**, as more computing resources can be utilized to train the model in parallel.

Without losing generality, our description is dominated by the multi-node multi-GPU case, in which there is a total N number of nodes while each node contains a G number of GPUs.

Following this setting, the ***world size*** is the total number of application processes running at one time across all the nodes. Then, the number of processes running on each node is referred to as the ***local world size*** L. Therefore, each application process is assigned two IDs: a ***local rank*** in [0, L-1] and a ***global rank*** in [0, N-1] that is a unique identifier for each node. While there are quite a few ways to map processes to nodes, a good rule of thumb is to have one process span a single GPU, i.e. L==G. This enables the distributed application to have as many parallel reader streams as there are GPUs and, in practice, provides a good balance between I/O and computational costs. We assume that the application follows this heuristic in the rest of this tutorial. Under this setting, world size equals the number of N * G.  Under this setting, world size equals the number of N * G. Then, one node should work as the master node to coordinate every process across nodes. To support this, the variable ***master address*** that is the hostname of the master-worker node should be set to be shared on every node. Then, the ***master port*** for the master-worker node communicates on is also set and shared among nodes.



## Structures
There are two popular ways of parallelizing Deep learning models:

* Model parallelism: refers to a model being logically split into several parts (i.e., some layers in one part and some in other), then placing them on different hardware/devices. Although placing the parts on different devices does have benefits in terms of execution time (asynchronous processing of data), it is usually employed to avoid memory constraints. Models with very large number of parameters, which are difficult fit into a single system due to high memory footprint, benefits from this type of strategy. In summary, the high-level idea of model parallel is to place different sub-networks of a model onto different devices, and implement the ``forward`` method accordingly to move intermediate outputs across devices. As only part of a model operates on any individual device, a set of devices can collectively serve a larger model.

* Data parallelism: on the other hand, refers to processing multiple pieces (technically batches) of data through multiple replicas of the same network located on different hardware/devices. Unlike model parallelism, each replica may be an entire network and not just a part of it. This strategy, as you might have guessed, can scale up well with increasing amount of data. But, as the entire network has to reside on a single device, it cannot help models with high memory footprints. In summary, the given input is split across the GPUs by chunking in the batch dimension. 

Pytorch supports this by providing:

- torch.nn.DataParallel: uses a single process multi-threading method to train the same model on different GPUs. It keeps the main process on one GPU and runs a different thread on other GPUs. Since multi-threading in python suffers from GIL(Global Interpreter Lock) issues, this restricts fully parallelized distributed training setup.

- torch.nn.Parallel.DistributedDataParallel: uses multi-processing to spawn separate processes on separate GPUs and leverage the full parallelism across GPUs.

AI Platform Training supports the following backends for distributed PyTorch training:

- gloo: recommended for CPU training jobs
- nccl: recommended for GPU training jobs


## Materials

Access the [tutorial](https://cloud.google.com/ai-platform/training/docs/distributed-pytorch) for configuring distributed training for PyTorch

Access the [bolg](https://towardsdatascience.com/distributed-neural-network-training-in-pytorch-5e766e2a9e62) to figure out when and how
to uese torch.nn.DataParallel and torch.nn.Parallel.DistributedDataParallel

Access the Pytorch [documentation](https://pytorch.org/docs/stable/distributed.html) for details


## Slurm
There are generally multiple (even thousands of) nodes in the network in the built distributed computing environment, making organizing and executing a user's job hard-to-implement work. Therefore, there is a high requirement for a high-level tool to facilitate the job implementation while hidden detailed low-level computing logic for the user. 

[Slurm](https://slurm.schedmd.com/overview.html) is a system for managing and scheduling Linux clusters. It is open-source, fault-tolerant, and scalable, suitable for clusters of various sizes. When Slurm is implemented, it can perform these tasks:

1. Assign a user to a compute node. The access provided can be exclusive, with resources being limited to an individual user, or non-exclusive, with the resources, shared among multiple users.
2. Provide the framework for launching and monitoring jobs on assigned nodes. The jobs are typically managed in parallel, running on multiple nodes.
3. Manage the pending job queue, determining which job is next in line to be assigned to the node.

Slurm monitors resources and jobs through a centralized manager)and can use a backup manager in case of failure. Each node has a slurmd (a daemon), which waits for jobs, executes them, and returns their status via fault-tolerant communication.

Users can initiate, manage and terminate jobs using the following commands:

- srun      — launches jobs
- scancel   — cancels jobs
- sinfo     — for system status
- squeue    — for status of pending job
- sacct     — for completed or running jobs
- sview     — graphic status report showing network topology
- scontrol  — cluster configuration and monitoring tool
- sacctmgr  — database administrative tool
- sbatch    – submit a batch script

There are mainly three types of jobs supported by the Slurm: 

1. Batch Jobs        The [sbatch](https://slurm.schedmd.com/sbatch.html) command is used to submit a batch script to Slurm. 
    - It is designed to reject the job at submission time if there are requests or constraints that Slurm cannot fulfill as specified.

2. Interactive Jobs  The [salloc](https://slurm.schedmd.com/salloc.html) command is used to submit an interactive job to Slurm. 
    - An interactive job is a job that returns a command line prompt (instead of running a script) when the job runs. 

3.  Xterm Jobs        The sxterm command is used to submit an xterm job to Slurm.
    - An xterm job is a job that launches an xterm window when the job runs.

**For more details, please access the [Slurm User Manual](https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-user-manual)!**


### Anatomy of a Batch Job
A batch job requests computing resources and specifies the application(s) to launch on those resources along with any input data/options and output directives. The user submits the job, usually in the form of a batch job script, to the batch scheduler.
The batch job script is composed of four main components:

- Interpreter. The interpreter used to execute the script
- Submission option. “#” directives that convey default submission options.
- Variables. The setting of environment and/or script variables (if necessary)
- Application. The application(s) to execute along with its input arguments and options.

The term "script" is used throughout this subsection to mean an executable file that the user creates and submits to the job scheduler to run on a node or collection of nodes. The script will include a list of SLURM directives (or commands) to tell the job scheduler what to do. The job flags are used with `SBATCH` command.  The syntax for the SLURM directive in a script is  `#SBATCH <flag>`.  Some of the flags are used with the srun and salloc commands, as well as the fisbatch wrapper script for interactive jobs.

One direct example is:
```
#!/bin/bash                     # Interpreter
#SBATCH --time=25:00:00         # <Submission option> <Variables>
#SBATCH --cpus-per-task=4       # 
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=main_train.out

python main_train.py            # Application
```


Instead of listing [all options](https://slurm.schedmd.com/pdfs/summary.pdf), here is the summary of some important flags:


Resource        |   Flag Syntax     |   Description     |   Notes
|:--------:|:--------------:|:-----------:|:-----:|
<sub>email address</sub>         | <sub>--mail-user</sub>       | <sub>Mail address to contact job owner</sub>  | <sub>expected but not necessary</sub>
<sub>email notification</sub>    | <sub>--mail-type</sub>       | <sub>When to notify a job owner: none, all, begin, end, fail, requeue, array_tasks</sub> | <sub>expected but not necessary</sub>
<sub>job name</sub>              | <sub>--job-name</sub>        | <sub>Name of job</sub> | <sub>default is the JobID</sub>
<sub></sub>            | <sub></sub>     | <sub></sub>  | <sub></sub>
<sub>time</sub>                  | <sub>--time</sub>            | <sub>Expected runtime of the job</sub> | <sub>default is 72 hours</sub>
<sub>partition</sub>                  | <sub>--partition</sub>            | <sub>Request a specific partition for the resource allocation</sub> | <sub>default partition selected by slurm controller, it is sim</sub>
<sub>nodes</sub>                 | <sub>--nodes</sub>           | <sub>Number of compute nodes for the job</sub> | <sub>default is 1</sub>
<sub>memory</sub>                | <sub>--mem</sub>             | <sub>Memory (MB) limit per compute node for the job</sub> | <sub>default limit is 3000MB per core</sub>
<sub>cpus/cores</sub>            | <sub>--ntasks-per-node</sub> | <sub>Specify how many tasks will run on each allocated node</sub> | <sub>default is 1</sub>
<sub>task cpus</sub>             | <sub>--cpus-per-task</sub>   | <sub>Number of CPUs per task (threads)</sub> | <sub>default 1</sub>
<sub>cpu memory</sub>            | <sub>--mem-per-cpu</sub>     | <sub>Per core memory limit</sub> | <sub>default limit is 3000MB per core</sub>
<sub>gpus</sub>                  | <sub>--gpus-per-node</sub>   | <sub>Number of gpus per node</sub> | <sub>default None</sub>
<sub>task gpus</sub>             | <sub>--gpus-per-task</sub>   | <sub>Specify the number of GPUs required for the job on each task</sub> | <sub>default None</sub>
<sub>gpu memory</sub>            | <sub>--mem-per-gpu</sub>     | <sub>Minimum memory required per allocated GPU</sub>  | <sub>default None</sub>
<sub></sub>            | <sub></sub>     | <sub></sub>  | <sub></sub>
<sub>permission</sub>            | <sub>--exclusive</sub>     | <sub>Use the compute node(s) exclusively, i.e. do not share nodes with other jobs.</sub>  | <sub>default None</sub>
<sub></sub>            | <sub></sub>     | <sub></sub>  | <sub></sub>
<sub>output file</sub>            | <sub>--output</sub>     | <sub>Redirect standard output to to file</sub>  | <sub>default is the JobID</sub>
<sub>error output file</sub>            | <sub>--error</sub>     | <sub>Redirect standard error to file</sub>  | <sub>default is the JobID</sub>
<sub>job dir</sub>            | <sub>--chdir</sub>     | <sub>Set the current working directory. </sub>  | <sub>default is the current working directory</sub>
<sub>tese mode</sub>            | <sub>--test-only</sub>     | <sub>Validate the batch script and return the estimated start time</sub>  | <sub>default is the JobID</sub>

Note:
- The `--mem`, `--mem-per-cpu` and `--mem-per-gpu` options are mutually exclusive.
- `--cpus-per-task` is Used for shared memory jobs that run locally on a single compute node
- `--exclusive` is dangerous to be used. CAUTION: Only use this option if you are an experienced user, and you really understand the implications of this feature. If used improperly, the use of this option can lead to a massive waste of computational resources
- In `--test-only`, the submission validates the batch script and return the estimated start time considering the current cluster state, job queue, and other job arguments. No job is actually submitted.

### Job States

The basic job states are these:

- Pending/PD - the job is in the queue, waiting to be scheduled
- Held - the job was submitted, but was put in the held state (ineligible to run)
- Running/R - the job has been granted an allocation.  If it’s a batch job, the batch script has been run
- Complete/CG - the job has completed successfully
- Timeout - the job was terminated for running longer than its wall clock limit
- Preempted - the running job was terminated to reassign its resources to a higher QoS job
- Failed - the job terminated with a non-zero status
- Node Fail - the job terminated after a compute node reported a problem

The queue reasons are:
- Resources - ob is waiting for compute nodes to become available
- Priority - Jobs with higher priority are waiting for compute nodes
- ReqNodeNotAvail - The compute nodes requested by the job are not available


## Slurm and Pytorch

When submitting the job containing code written in PyTorch to the SLURM, the distributed mode is configured by the `sbatch` script, making the detailed hyper-settings unknown to the PyTorch distributed part. There is a need to extract configurations from SLURM and then assign variables to those in distributed computing functions of the Pytorch. Here are the correspondences between variables in SLURM and variables in the PyTorch.

SLURM variables  | PyTorch variables | Notes
------------- | ------------- | -------------
<sub>SLURM_NODELIST</sub>  | <sub>node_list</sub>         | <sub>nodes list for learning</sub>
<sub>SLURM_STEPS_GPUS</sub>  | <sub>gpu_ids</sub>         | <sub>running gpus' ids</sub>
<sub>SLURM_LOCALID </sub> | <sub>gpu_id</sub>             | <sub>gpu local rank i.e., [0, G-1] where G is number of gpus obtained by torch.cuda.device_count() ass one gpu per task</sub>
<sub>SLURM_PROCID</sub>, <sub>--gpus-per-</sub>  | <sub>dist_rank</sub>           | <sub>gpu global rank  i.e., [0, N-1]</sub>
<sub>SLURM_NTASKS</sub>  | <sub>world_size</sub>          | <sub>number of individual tasks launched, i.e., N * G</sub>
<sub>MASTER_ADDR</sub>  | <sub>MASTER_ADDR</sub>         | <sub>used to communicate between nodes</sub>
<sub>MASTER_PORT</sub>  | <sub>MASTER_PORT</sub>         | <sub>used to communicate between nodes</sub>


* [Segmenter](https://github.com/rstrudel/segmenter)

