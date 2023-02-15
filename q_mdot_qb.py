import os
import matplotlib.pyplot as plt
from parallelrun import ParallelProcessor as P
import numpy as np
path = '/home/kaho/kepshot_run/starshot_paper/'

_allp = os.listdir(path)
q = []
mdot = []
qb = []
for i in _allp:
    print(i)
    p = P.load(path+i)
    q.extend(p.Q)
    qb.extend(p.Qb)
    if len(np.unique(p.mdot)) == 2:
        submdot = [None] * (len(p.results))
        submdot[::2] = p.mdot[:len(p.mdot)//2]
        submdot[1::2] = p.mdot[len(p.mdot)//2:]
        mdot.extend(submdot)
    elif len(np.unique(p.mdot)) > 2 and len(np.unique(p.Q)) > 1:
        submdot = [None] * (len(p.results))
        submdot[:len(p.mdot)//2] = p.mdot[::2]
        submdot[len(p.mdot)//2:] = p.mdot[1::2]
        mdot.extend(submdot)
    else:
        mdot.extend(p.mdot)
q = np.round(q,3)
mdot = np.round(mdot,3)
qb = np.round(qb,3)
_index = np.where((q>=5.4)&(q<=5.9))
q = q[_index]
mdot = mdot[_index]
qb = qb[_index]

# drop values over 3 decimal points
_drop = np.arange(0.005, 0.1, 0.01).round(3)
_index2 = ~np.isin(mdot, _drop)
q = q[_index2]
mdot = mdot[_index2]
qb = qb[_index2]
# drop repeat runs
_stack = np.vstack((q, mdot))
_, _ind = np.unique(_stack, axis=1, return_index=True)
if len(_ind) != len(q):
    _ind.sort()
    q = list(q[_ind])
    mdot = list(mdot[_ind])
    qb = list(qb[_ind])

qb2 = [x for _, x in sorted(zip(q, qb), key=lambda pair: pair[0])] # extra argument is for not s
sortmdot = [x for _, x in sorted(zip(q, mdot), key=lambda pair: pair[0])]
sortqb= [x for _, x in sorted(zip(sortmdot, qb2), key=lambda pair: pair[0])]

a = np.unique(q)
b = np.unique(mdot)
qqb = np.array(sortqb).reshape(len(b), len(a))

fig, ax = plt.subplots()
pcm = plt.pcolor(a, b, qqb)

ax.set_xlabel('Surface luminosity / $\dot{m}$ (MeV/u)', fontsize=15)
ax.set_ylabel('$\dot{m}_{\mathrm{Edd}}$', fontsize=15)
ax.tick_params(labelsize=13)
cbar = plt.colorbar(pcm)
cbar.set_label('Base luminosity / $\dot{m}$ (MeV/u)', fontsize=15, labelpad=10)
# cbar.update_ticks(size=15)
plt.tight_layout()