from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SnapshotResult:
    m: int
    t_star: float
    T_grid: np.ndarray
    p_uncond_mean: np.ndarray
    p_cond_mean: np.ndarray
    p_uncond_draws: np.ndarray  # (M_accept, nT)
    p_cond_draws: np.ndarray  # (M_accept, nT)
    draws_array: np.ndarray  # (M_accept, 5) [R0,k,r,alpha,theta]
    infection_times_2d: List[np.ndarray]  # per-trajectory infection times (days since start_date)
    n_obs: int
    next_T: Optional[float] = None
    stopped_pairs: Optional[List[Sequence[Tuple[float, float]]]] = None
