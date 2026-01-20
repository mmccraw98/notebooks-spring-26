import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 100
    n_dynamics_steps: int = 10_000_000
    phi_target: float = 0.9
    e_int: float = 1.0
    dt: np.ndarray = field(
        default_factory=lambda: np.ones(10) * 1e-2
    )
    delta_phi: float = 1e-2
    target_temperatures: np.ndarray = field(
        default_factory=lambda: np.logspace(-5, -2, 10)
    )

default_config = Config()

dt_base = 1e-2
T_min = 1e-5
T_max = 1e-3
n_systems = 3
T = np.logspace(np.log10(T_min), np.log10(T_max), n_systems)
dt = dt_base * np.sqrt(T_min / T)

config2 = Config(target_temperatures=T, dt=dt)