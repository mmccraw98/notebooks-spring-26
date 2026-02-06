from dataclasses import dataclass, field

@dataclass
class Config:
    phi: float = 0.67
    dim: int = 2
    N: int = 100
    nv: int = 20
    mass: float = 1.0
    n_dynamics_steps: int = 1_000_000
    phi_target: float = 0.87
    e_int: float = 1.0
    dt: float = 1e-2
    can_rotate: bool = True
    subtract_drift: bool = True
    delta_phi: float = 1e-3
    target_temperature: float = 1e-5
    save_stride: int = 1000

config = Config()