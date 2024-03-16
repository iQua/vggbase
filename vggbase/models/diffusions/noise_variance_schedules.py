"""
The implementations of noise variance schedules.

In math, the noise variance schedule is denoted as `beta`.
"""

import math
from typing import Union, List

import torch


def cosine_beta_schedule(chain_steps: int, cosine_s: float):
    """
    Cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = chain_steps + 1
    steps_idx = torch.linspace(0, chain_steps, steps, dtype=torch.float64)
    alphas_cumprod = (
        torch.cos(
            ((steps_idx / chain_steps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5
        )
        ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(chain_steps: int, schedule_range: List[float]):
    """
    Linear schedule
    as proposed in the original paper
    """
    start_value, end_value = schedule_range
    assert start_value < end_value

    return torch.linspace(start_value, end_value, chain_steps)


def sigmoid_beta_schedule(
    chain_steps, start: int = -3, end: int = 3, tau: int = 1, clamp_min: float = 1e-5
):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = chain_steps + 1
    t = torch.linspace(0, chain_steps, steps, dtype=torch.float64) / chain_steps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get(scheduler: str, chain_steps: int, **kwargs: Union[str, dict]):
    """Get the scheduler."""
    registered_schedulers = {
        "cosine_scheduler": cosine_beta_schedule,
        "linear_scheduler": linear_beta_schedule,
    }
    if scheduler not in registered_schedulers:
        raise Exception(f"Scheduler {scheduler} does not exist")

    if "cosine_scheduler" == scheduler:
        if "cosine_s" not in kwargs:
            kwargs["cosine_s"] = 0.008

    if "linear_scheduler" == scheduler:
        assert "schedule_range" in kwargs

    return registered_schedulers[scheduler](chain_steps, **kwargs)
