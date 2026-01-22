import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 100
    mu_eff: float = 0.1
    aspect_ratio: float = 1.0
    min_nv: int = 20
    mass: float = 10.0
    n_dynamics_steps: int = 10_000_000
    phi_target: float = 0.9
    e_int: float = 1.0
    dt: float = 1e-2
    delta_phi: float = 1e-2
    target_temperature: float = 1e-3

default_config = Config()