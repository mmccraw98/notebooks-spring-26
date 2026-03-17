from dataclasses import dataclass, field

@dataclass
class Config:
    phi: float = 0.6
    dim: int = 2
    N: int = 1000
    min_nv: int = 5
    mu_eff: float = 0.1
    aspect_ratio: float = 1.0
    mass: float = 1.0
    n_dynamics_steps: int = 1_000_000
    pe_target: float = 1e-2
    e_int: float = 1.0
    dt: float = 1e-2
    can_rotate: bool = True
    subtract_drift: bool = True
    delta_phi: float = 1e-2
    target_temperature: float = 1e-5
    min_save_decade: int = 100
    reset_save_decade: int = 100_000

# config_001_10 = Config(mu_eff=0.01, aspect_ratio=1.0)
config_010_10 = Config(mu_eff=0.1, aspect_ratio=1.0)
config_050_10 = Config(mu_eff=0.5, aspect_ratio=1.0)
config_100_10 = Config(mu_eff=1.0, aspect_ratio=1.0)

# config_001_12 = Config(mu_eff=0.01, aspect_ratio=1.2)
config_010_12 = Config(mu_eff=0.1, aspect_ratio=1.2)
config_050_12 = Config(mu_eff=0.5, aspect_ratio=1.2)
config_100_12 = Config(mu_eff=1.0, aspect_ratio=1.2)

# config_001_15 = Config(mu_eff=0.01, aspect_ratio=1.5)
config_010_15 = Config(mu_eff=0.1, aspect_ratio=1.5)
config_050_15 = Config(mu_eff=0.5, aspect_ratio=1.5)
config_100_15 = Config(mu_eff=1.0, aspect_ratio=1.5)

# config_001_20 = Config(mu_eff=0.01, aspect_ratio=2.0)
config_010_20 = Config(mu_eff=0.1, aspect_ratio=2.0)
config_050_20 = Config(mu_eff=0.5, aspect_ratio=2.0)
config_100_20 = Config(mu_eff=1.0, aspect_ratio=2.0)
