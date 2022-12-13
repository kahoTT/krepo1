import os
import matplotlib.pyplot as plt
from parallelrun import ParallelProcessor as P
import numpy as np
path = '/home/kaho/kepshot_run/starshot_paper2/'

_allp = os.listdir(path)
_allp.sort()
q = []
mdot = []
qb = []
for i in _allp:
    p = P.load(path+i)
    q.extend(p.Q)
    mdot.extend(p.mdot)
    qb.extend(p.Qb)

# drop repeat runs
_stack = np.vstack(q, mdot)
_, _ind = np.unique(_stack, axis=1, return_index=True)
sortind = _ind.sort()
q = q[sortind]
mdot = mdot[sortind]
qb = qb[sortind]

qb2 = [qb2 for _,qb2 in sorted(zip(q, qb))]
sortqb = [sortqb for _,sortqb in sorted(zip(mdot, qb2))]
sortq = q.sort()
sortmdot = mdot.sort()

a = np.unique(sortq)
b = np.unique(sortmdot)
qqb = np.array(sortqb).reshape(len(b), len(a), order='F')

fig, ax = plt.subplots()
pcm = plt.pcolor(a, b, qqb)

ax.set_xlabel('Surface luminosity / $\dot{m}$ (MeV/u)', fontsize=15)
ax.set_ylabel('$\dot{m}_{\mathrm{Edd}}$', fontsize=15)
ax.tick_params(labelsize=13)
cbar = plt.colorbar(pcm)
cbar.set_label('Base luminosity / $\dot{m}$ (MeV/u)', rotation=270, fontsize=15, labelpad=20)
# cbar.update_ticks(size=15)
plt.tight_layout()