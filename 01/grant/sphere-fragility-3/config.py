from dataclasses import dataclass, field

@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 1000
    mass: float = 1.0
    n_dynamics_steps: int = 100_000
    phi_target: float = 0.9
    e_int: float = 1.0
    dt: float = 1e-2
    can_rotate: bool = False
    subtract_drift: bool = True
    delta_phi: float = 5e-2
    target_temperature: float = 1e-5
    min_save_decade: int = 1000
    force_model: str = "spring"  # wca_shifted
    reset_save_decade: int = 10_000

config_2d_wca_1 = Config(force_model='wca_shifted', target_temperature=1e-5, dt=6e-3)
config_2d_wca_2 = Config(force_model='wca_shifted', target_temperature=1e-3, min_save_decade=100, dt=6e-3)
config_2d_wca_3 = Config(force_model='wca_shifted', target_temperature=1e-1, min_save_decade=10, dt=6e-3)

config_2d_1 = Config(force_model='spring', target_temperature=1e-5)
config_2d_2 = Config(force_model='spring', target_temperature=1e-3, min_save_decade=100)
config_2d_3 = Config(force_model='spring', target_temperature=1e-1, min_save_decade=10)
