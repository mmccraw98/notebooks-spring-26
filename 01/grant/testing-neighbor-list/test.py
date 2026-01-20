import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    state = jd.utils.h5.load('example-state.h5')
    system = jd.utils.h5.load('example-system.h5')





    # CREATE CELL LIST
    # min_rad = jnp.min(state.rad)
    max_rad = jnp.max(state.rad)
    # max_rad = 0.7
    # alpha = max_rad / min_rad

    cell_size = 5.0 * max_rad
    # cell_size = 2.0 * max_rad
    # if alpha < 2.5:
    #     cell_size = 2 * max_rad
    # else:
    #     cell_size = max_rad / 2

    search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
    search_range = jnp.maximum(1, search_range)
    search_range = jnp.array(search_range, dtype=int)

    # this is the stencil in a single axis
    # [-search_range, -search_range - 1, ... -1, 0, 1, ..., search_range - 1, search_range]
    r = jnp.arange(-search_range, search_range + 1, dtype=int)
    # this is the stencil in the full space [stencil_ids_x, stencil_ids_y, stencil_ids_z]
    # where stencil_ids is [r, r, r] for d-dimensions
    mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
    # merge the stencil ids to be [[stencil_id_x_0, stencil_id_y_0, stencil_id_z_0], ...]
    # it has shape ((search_range * 2 + 1) ** dim, dim)
    # THIS is basically the stencil - it stores relative cell coordinates
    neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)


    # GET SPATIAL PARTITION
    iota = jax.lax.iota(int, state.N)  # basically np.arange(state.N, dtype=int)
    # count number of cells in each dimension
    grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
    # calculate grid strides (for calculating cell ids)
    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )
    # calculate particle cell coordinates [c_x, c_y, c_z]
    p_cell_coords = jnp.floor((state.pos - system.domain.anchor) / cell_size).astype(int)
    # wrap indices
    p_cell_coords -= grid_dims * jnp.floor(p_cell_coords / grid_dims).astype(int)
    # calculate cell id (hash) from the product of the cell coordinates and the grid strides [c_id_0, c_id_1, ... c_id_N]
    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)
    # sort the cell ids (hashes)
    # pass the cell hash and the iota (sphere IDs)
    # mutually sorts the cell hash and the sphere IDs
    # the sorted sphere IDs (perm) is then used to reorder the sphere data

    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)
    # reorder the cell coordinates
    p_cell_coords = p_cell_coords[perm]
    # identify coordinates of neighboring cells for each particle
    # apply the cell stencil (neighbor_mask) to all particle cell coordinates
    # it is shaped (N, stencil dims, dim)
    neighbor_cell_coords = p_cell_coords[:, None, :] + neighbor_mask
    # wrap coordinates for periodic domains
    neighbor_cell_coords -= grid_dims * jnp.floor(
        neighbor_cell_coords / grid_dims
    ).astype(int)
    # transform the neighbor list from cell coordinate space (N, stencil dims, dim) to cell id space (N, stencil dims)
    neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)



    # my cell list creation
    # do not sort the particle data, just use the sorted ids
    sorted_particle_ids = iota[perm]
    sorted_clump_ids = state.ID[perm]
    sorted_pos = state.pos[perm]
    # cutoff_sq = (2.0) ** 2
    cutoff_sq = 2 * cell_size ** 2

    print(cell_size)
    max_neighbors = 100

    neighbor_list = jnp.full(shape=(state.N, max_neighbors), fill_value=-1, dtype=int)
    
    i = 100
    pid_i = sorted_particle_ids[i]
    pos_i = sorted_pos[i]
    ID_i = sorted_clump_ids[i]
    neighbor_count = 0
    for cell_id in neighbor_cell_hashes[i]:
        # find where the block for cell_id begins
        start_idx = jnp.searchsorted(p_cell_hash, cell_id, side="left", method="scan_unrolled")
        # loop over the list of cell ids (p_cell_hash) until the cell id changes (we leave the block)
        k = start_idx
        while (p_cell_hash[k] == cell_id) * (k < state.N):
            dr = system.domain.displacement(pos_i, sorted_pos[k], system)
            d_sq = jnp.sum(dr ** 2)
            if (d_sq <= cutoff_sq) * (ID_i != sorted_clump_ids[k]):
                neighbor_list = neighbor_list.at[pid_i, neighbor_count].set(sorted_particle_ids[k])  # use the unsorted ids for both
                neighbor_count += 1
            k += 1
    
    print(neighbor_list[pid_i])


    from matplotlib.patches import Circle
    plt.xlim(0, system.domain.box_size[0])
    plt.ylim(0, system.domain.box_size[1])
    plt.gca().set_aspect('equal')
    plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), jnp.sqrt(cutoff_sq), facecolor='k', alpha=0.2))
    for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
        plt.gca().add_artist(Circle(p, r))
    plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), state.rad[pid_i], facecolor='k'))
    for i in neighbor_list[pid_i]:
        if i == -1:
            continue
        plt.gca().add_artist(Circle(jnp.mod(state.pos[i], system.domain.box_size), state.rad[i], facecolor='r'))
    plt.savefig('test.png')


    exit()


    # CREATE CELL LIST
    collider = system.collider
    iota = jax.lax.iota(int, state.N)  # basically np.arange(state.N, dtype=int)
    # cutoff_sq = (10 * jnp.max(state.rad)) ** 2
    cutoff_sq = (2.0) ** 2
    max_neighbors = 100
    # GET SPATIAL PARTITION
    perm, p_cell_coords, p_cell_hash, neighbor_cell_coords
    p_neighbor_hashes = neighbor_cell_hashes

    # apply the cell sorting
    state = jax.tree.map(lambda x: x[perm], state)
    sorted_particle_ids = iota[perm]

    pos = state.pos

    def per_particle(idx, pos_i, stencil):
        def stencil_body(i, carry):
            global_c, n_list, overflow = carry
            target_cell_hash = stencil[i]
            start_idx = jnp.searchsorted(p_cell_hash, target_cell_hash, side="left", method="scan_unrolled")

            def cond_fun(val):
                k, _, _, _ = val
                return (k < state.N) * (p_cell_hash[k] == target_cell_hash)

            def body_fun(val):
                k, c, nl, ovr = val
                dr = system.domain.displacement(pos_i, pos[k], system)
                d_sq = jnp.sum(dr**2, axis=-1)
                valid = (state.ID[k] != state.ID[idx]) * (d_sq <= cutoff_sq)
                nl = nl.at[c].set(k)
                return k + 1, c + valid, nl, ovr + c > max_neighbors

            _, global_c, n_list, overflow = jax.lax.while_loop(
                cond_fun, body_fun, (start_idx, global_c, n_list, overflow)
            )
            return global_c, n_list, overflow > 0

        init_carry = (0, jnp.full((max_neighbors,), -1, dtype=int), False)
        final_c, final_n_list, final_ovr = jax.lax.fori_loop(
            0, stencil.shape[0], stencil_body, init_carry
        )
        return final_n_list, final_ovr

    # particle_id = 0
    # idx = iota[particle_id]  # particle id after sorting
    # pos_i = pos[particle_id]  # particle pos
    # stencil = p_neighbor_hashes[particle_id]  # cell ids in the stencil

    # global_c = 0
    # n_list = jnp.full((max_neighbors,), -1, dtype=int)
    # overflow = False
    # for i in range(0, stencil.shape[0]):  # loop over the cell ids
    #     target_cell_hash = stencil[i]  # current cell id

    #     # find the location of the first cell id in the list of cell ids for all particles
    #     start_idx = jnp.searchsorted(p_cell_hash, target_cell_hash, side="left", method="scan_unrolled")
        
    #     # loop over all the following particles within the same cell id
    #     # starting at start_idx and ending when the cell id changes
    #     print(i, '-'*10)
    #     k = start_idx
    #     c = global_c
    #     nl = n_list
    #     ovr = overflow
    #     while (k < state.N) * (p_cell_hash[k] == target_cell_hash):
    #         dr = system.domain.displacement(pos_i, pos[k], system)
    #         d_sq = jnp.sum(dr**2, axis=-1)
    #         valid = (state.ID[k] != state.ID[idx]) * (d_sq <= cutoff_sq)
    #         nl = nl.at[c].set(k)
    #         print(d_sq)

    #         k += 1
    #         c += valid
    #         # nl = nl
    #         ovr += c > max_neighbors

    # print(p_cell_hash.shape, target_cell_hash)
    # print(nl)
    



    # # build the neighbor list
    # neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, p_neighbor_hashes)
    # state, system, neighbor_list, jnp.any(overflows)
    # sorted_state = state
    # sorted_nl_indices = neighbor_list

    # # map the neighbor list indices back to particles
    # # sorted_nl_indices[i, j] = k -> sorted_state.ID[k]
    # # print(sorted_state.ID)
    # # print(sorted_nl_indices)

    # print(neighbor_list)

    # from matplotlib.patches import Circle
    # plt.xlim(0, system.domain.box_size[0])
    # plt.ylim(0, system.domain.box_size[1])
    # plt.gca().set_aspect('equal')
    # for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
    #     plt.gca().add_artist(Circle(p, r))
    # pid = 124
    # plt.gca().add_artist(Circle(jnp.mod(state.pos[pid], system.domain.box_size), state.rad[pid], facecolor='k'))
    # for i in neighbor_list[pid]:
    #     if i == -1:
    #         continue
    #     plt.gca().add_artist(Circle(jnp.mod(state.pos[i], system.domain.box_size), state.rad[i], facecolor='r'))
    # plt.savefig('test.png')

    # print(jnp.argmin(jnp.linalg.norm(jnp.mod(state.pos, system.domain.box_size) - system.domain.box_size / 2, axis=-1)))

    # print(neighbor_list[pid])