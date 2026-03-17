from anim_utils import animate
import numpy as np

if __name__ == "__main__":
    data = np.load('anim-data.npz')
    pos = data['pos']
    rad = data['rad']
    cid = data['cid']
    box_size = data['box_size']
    animate(
        pos,
        rad,
        cid,
        box_size,
        'compress.gif',
        frames=120,
    )
