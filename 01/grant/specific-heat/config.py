import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:
    phi: float = 0.4
    dim: int = 2
    N: int = 1000
    min_nv: int = 20
    e_int: float = 1.0
    dt: float = 1.0
    delta_phi: float = 1e-2
    n_duplicates: int = 10

default_config = Config()