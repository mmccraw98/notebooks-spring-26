from dataclasses import dataclass, field

@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 10
    nv: int = 20
    mu_eff: float = 0.1
    mass: float = 10.0
    n_dynamics_steps: int = 10_000_000
    phi_target: float = 0.9
    el: float = 1e1
    eb: float = 1e0
    ec: float = 1e3
    em: float = 1e0
    e_int: float = 1.0
    dt: float = 1e-2
    delta_phi: float = 1e-3
    target_temperature: float = 1e-6
    save_stride: int = 500

config2d_soft = Config(eb=1e-4, em=None)
config2d_med = Config(eb=1e-2, em=None)
config2d_hard = Config(eb=1e0, em=None)