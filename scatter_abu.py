import os
import matplotlib.pyplot as plt
from parallelrun import ParallelProcessor as P
import numpy as np


class c12(object):
    def __init__(self):
        path = '/home/kaho/kepshot_carbon/'
        _allp = os.listdir(path)
        qb = []
        mdot = []
        c12 = []
        for i in _allp:
            p = P.load(path+i)
            qb.extend(p.Qb)
            mdot.extend(p.mdot)
            c12.extend(p.abub[6,12])
        plt.scatter(mdot, qb, s=np.ones(len(c12))*30, c=c12)
        cb = plt.colorbar()
        cb.set_label(label='Carbon mass fraction', fontsize = 15)
        plt.ylabel('Base luminosity / $\dot{m}$ (MeV/u)', fontsize=15)
        plt.xlabel('$\dot{M}_{\mathrm{Edd}}$', fontsize=15)
        plt.ylim(0.5, 0.7)
        plt.xlim(0.07, 0.105)
        plt.tick_params(labelsize=13)
        cb.update_ticks(labelsize=13)
        plt.show()
	