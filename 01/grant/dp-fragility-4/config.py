from dataclasses import dataclass, field

@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 1000
    min_nv: int = 20
    mu_eff: float = 0.1
    aspect_ratio: float = 1.0
    el: float = 1e1
    eb: float = 1e0
    ec: float = 1e3
    em: float = None
    mass: float = 1.0
    n_dynamics_steps: int = 100_000
    phi_target: float = 0.9
    e_int: float = 1.0
    dt: float = 1e-3
    can_rotate: bool = False
    subtract_drift: bool = True
    delta_phi: float = 1e-2
    target_temperature: float = 1e-5
    min_save_decade: int = 1_000
    reset_save_decade: int = 100_000

config2d_floppy = Config(eb=None)
config2d_soft = Config(eb=1e-4)
config2d_med = Config(eb=1e-2)
config2d_hard = Config(eb=1e0)
