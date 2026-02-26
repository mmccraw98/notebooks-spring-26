import numpy as np
import matplotlib.pyplot as plt

data = np.load('virials.npz')
bv = data['bonded_virial']
cv = data['contact_virial']
v = bv + cv
t = []
corr_bv, corr_cv, corr_v = [], [], []
for i in range(1, bv.shape[0] - 1):
    t.append(i)
    corr_bv.append(np.mean(bv[i:] * bv[:-i], axis=0))
    corr_cv.append(np.mean(cv[i:] * cv[:-i], axis=0))
    corr_v.append(np.mean(v[i:] * v[:-i], axis=0))

t = np.array(t)
corr_bv = np.array(corr_bv)
corr_cv = np.array(corr_cv)
corr_v = np.array(corr_v)

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for i, (c, name) in enumerate(zip([corr_bv, corr_cv, corr_v], ['Bonded', 'Contact', 'Total'])):
    # ax[i].plot(t, c[:, 0, 0])
    ax[i].plot(t, c[:, 0, 1])
    ax[i].set_title(name)
for a in ax:
    a.set_xscale('log')
plt.savefig('plot.png')
plt.close()