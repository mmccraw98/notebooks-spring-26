import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from jaxdem_scripts.ga_utils import create_bidisperse_ga_dps_2d
from jaxdem_scripts.compression_loop_dp import run_1, JobConfig as JCFG

e_b = 1e-1
e_m = 1e0  # length
e_c = 1e3  # area
tau_s = 10

N_dps = 10
mu_eff = 0.5
min_nv = 8
phi = 0.8
aspect_ratio = 1.0
dp_mass = 1.0
dt = 1e-3
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-4

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=100_000,
    min_save_decade=1000,
)

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-plastic-fragilitiy/version-1/'

state, system = create_bidisperse_ga_dps_2d(
    N_dps,
    mu_eff,
    min_nv,
    phi,
    aspect_ratio,
    dp_mass,
    dt,
    e_int,
    e_m,
    e_c,
    e_b,
    e_l=None,
    e_gamma=None,
    tau_s=tau_s,
)

state = jd.utils.thermal.set_temperature(
    state,
    temp,
    can_rotate,
    subtract_drift,
    np.random.randint(0, 1e9)
)

phi = jd.utils.packingUtils.compute_packing_fraction(state, system)

stride = cfg.min_save_decade
n = cfg.n_steps // stride
save_strides = jnp.full(n, stride)

def save_fn(st, sy):
    # perm = jnp.empty_like(st.unique_id)
    # perm = perm.at[st.unique_id].set(jnp.arange(st.unique_id.shape[0], dtype=st.unique_id.dtype))
    return dict(
        step_count=sy.step_count,
        #
        pos=st.pos,
        rad=st.rad,
        pid=st.bonded_id,
        # pos=st.pos[perm],
        # vel=st.vel[perm],
        pe=jd.utils.thermal.compute_potential_energy(st, sy),
        ke=jd.utils.thermal.compute_translational_kinetic_energy(st),
    )

rollout_kwargs = dict(strides=save_strides, save_fn=save_fn)

state, system, logged = jd.System.trajectory_rollout(
    state, system, **rollout_kwargs,
)

from anim_utils import animate
animate(logged['pos'], logged['rad'], logged['pid'], system.domain.box_size, 'test.gif')
# print(logged['ke'])
# print(logged['pe'])

# import matplotlib.pyplot as plt
# plt.plot(logged['pe'])
# plt.plot(logged['ke'])
# plt.plot(logged['ke'] + logged['pe'])
# plt.savefig('energies.png')

# while phi < 1.0:
#     state, system, mean_pe, _ = run_1(
#         state,
#         system,
#         data_root,
#         cfg,
#         save_strides=save_strides,
#         compress=True,
#         save_fn=None,
#         save_all=False
#     )
#     state, system = system.step(state, system, n=500_000)
#     state, system, mean_pe, _ = run_1(
#         state,
#         system,
#         data_root,
#         cfg,
#         save_strides=save_strides,
#         compress=False,
#         save_fn=None,
#         save_all=False
#     )
#     phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
