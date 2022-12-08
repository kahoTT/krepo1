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
# mdot = [mdot for _,mdot in sorted(zip(q, mdot))]
# qb = [qb for _,qb in sorted(zip(q, qb))]
# q = [q for _,q in sorted(zip(mdot, q))]
# qb = [qb for _,qb in sorted(zip(mdot, qb))]

a = np.unique(q)
b = np.unique(mdot)
qqb = np.array(qb).reshape(len(b), len(a), order='F')

fig, ax = plt.subplots()
pcm = ax.pcolor(a, b, qqb)
fig.colorbar(pcm, ax=ax)
