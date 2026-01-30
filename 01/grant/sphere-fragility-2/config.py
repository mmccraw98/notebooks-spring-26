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
    delta_phi: float = 1e-3
    target_temperatures: np.ndarray = field(
        default_factory=lambda: np.logspace(-5, -2, 10)
    )
    save_stride: int = 500
    force_model_type: str = 'spring'

dt_base = 1e-2

T_min = 1e-5
T_max = 1e-3
n_systems = 3
T = np.logspace(np.log10(T_min), np.log10(T_max), n_systems)
dt = dt_base * np.sqrt(T_min / T)

config2d = Config(target_temperatures=T, dt=dt)

config2d_2 = Config(target_temperatures=np.array([1e-2, 5e-2]), dt=np.array([1e-2, 3.33e-2]), n_dynamics_steps=1_000_000, phi=0.7, save_stride=50)


T_min_wca = 1e-5
T_max_wca = 1e0
n_systems_wca = 5
T_wca = np.logspace(np.log10(T_min_wca), np.log10(T_max_wca), n_systems_wca)
dt_wca = dt_base * np.sqrt(T_min_wca / T_wca)
config2d_wca = Config(target_temperatures=T_wca, dt=dt_wca, force_model_type='wca')

config3d = Config(target_temperatures=T, dt=dt, dim=3, phi=0.4)