"""
The useful tools for models.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clones(module, n_repeat):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_repeat)])
