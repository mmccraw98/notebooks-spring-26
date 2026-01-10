import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 100
    n_dynamics_steps: int = 1_000_000
    phi_target: float = 0.9
    e_int: float = 1.0
    dt: float = 1e-2
    delta_phi: float = 1e-2
    target_temperatures: np.ndarray = field(
        default_factory=lambda: np.logspace(-5, -2, 10)
    )

default_config = Config()