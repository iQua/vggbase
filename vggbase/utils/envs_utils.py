"""
Useful function for environments.

For the distributed learning mode, there are
two classicial modes, including DP and DDP.
where DP denotes nn.DataParallel
while DDP denotes nn.parallel.DistributedDataParallel

The tutorial for DP is:
https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/data_parallel_tutorial.py


For multiple nodes where each node can consist of multiple GPU devices.
    - N: the number of nodes on which the application is running
    - G: the number of GPUs per node

world_size (W): a group containing all the processes for the distributed training.
    Thus, this the number of processes for the training.
    As generally each GPU corresponds to one process,  world size is
    usually the number of GPUs you are using for distributed training.

    where W = N x G

rank: is the unique ID given to a process, so that other processes know
    how to identify a particular process.
    It contains:
    - local_rank in  [0, L-1], the rank of the process on the local machine
    - global_rank in [0, W-1], the rank of the process in the whole world

init_process_group: is used by each subprocess in distributed training.
    So it only accepts a single rank


Basic assumption: One GPU corresponds to one process

Considering one example in the context of multi-node training

To illustrate that, let's say there are 2 nodes (machines) with 3 GPU each,
    you will have a total of 4 processes (p1…p4).

world_size: 2 x 3 = 6

            |           Node1          |           Node2          |
____________|  p11   |   p12  |   p13  |  p21   |   p22  |   p23  |
local_rank  |   0    |    1   |    2   |   0    |    1   |    2   |
gloabl_rank |   0    |    1   |    2   |   3    |    4   |    5   |
device id   |   0    |    1   |    2   |   3    |    4   |    5   |


For SLURM_PROCID, this is slurm platfrom in which the job can be submitted by
    sbatch.
    - SLURM_NODELIST: number of nodes can be used.

"""
