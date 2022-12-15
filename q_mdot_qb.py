import os
import matplotlib.pyplot as plt
from parallelrun import ParallelProcessor as P
import numpy as np
path = '/home/kaho/kepshot_run/starshot_paper/'

_allp = os.listdir(path)
q = []
mdot = []
qb = []
for i in _allp[:3]:
    p = P.load(path+i)
    q.extend(p.Q)
    qb.extend(p.Qb)
    if len(np.unique(p.mdot)) > 1:
        submdot = [None] * (len(p.results))
        submdot[::2] = p.mdot[:len(p.mdot)//2]
        submdot[1::2] = p.mdot[len(p.mdot)//2:]
        mdot.extend(submdot)
    else:
        mdot.extend(p.mdot)

# drop repeat runs
_stack = np.vstack((q, mdot))
_, _ind = np.unique(_stack, axis=1, return_index=True)
if len(_ind) != len(q):
    sortind = _ind.sort()
    q = list(np.array(q)[sortind])
    mdot = list(np.array(mdot)[sortind])
    qb = list(np.array(qb)[sortind])

qb2 = [qb2 for _,qb2 in sorted(zip(q, qb))]
sortmdot = [sortmdot for _,sortmdot in sorted(zip(q, mdot))]
sortqb = [sortqb for _,sortqb in sorted(zip(sortmdot, qb2))]

a = np.unique(q)
b = np.unique(mdot)
qqb = np.array(sortqb).reshape(len(b), len(a))

fig, ax = plt.subplots()
pcm = plt.pcolor(a, b, qqb)

ax.set_xlabel('Surface luminosity / $\dot{m}$ (MeV/u)', fontsize=15)
ax.set_ylabel('$\dot{m}_{\mathrm{Edd}}$', fontsize=15)
ax.tick_params(labelsize=13)
cbar = plt.colorbar(pcm)
cbar.set_label('Base luminosity / $\dot{m}$ (MeV/u)', rotation=270, fontsize=15, labelpad=20)
# cbar.update_ticks(size=15)
plt.tight_layout()