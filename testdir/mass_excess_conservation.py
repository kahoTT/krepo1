import os
from starshot.base import Base as S
import numpy as np
from physconst import *
from matplotlib import pyplot as plt

s = []
for i in np.sort(os.listdir('/home/kaho/kepshot_run/alex_code/')):
    s.append(S.load('/home/kaho/kepshot_run/alex_code/'+i))

b = s[0].abub
me = s[0].B.mass_excess(b.ions)
mex1 = np.sum(b.abu(molfrac=True) * me, axis=1)
mex1[1]
lqb = []
lme = []
lnuc = []
lnu = []
lie = []
lpdv = []
lpo = []
nu0 = np.sum(s[0].snu[1:-1]*s[0].xm[1:-1])/s[0].mdot *AMU/MEV
nuc0 = np.sum(s[0].sn[1:-1]*s[0].xm[1:-1])/s[0].mdot *AMU/MEV
pdv0 = s[0].pn[1]/s[0].dn[1]/s[0].mdot *AMU/MEV
po0 = GRAV*(s[0].zm[1]/s[0].rn[1] - s[0].zm[-2]/s[0].rn[-2])*AMU/MEV
print(po0)

for i in s[1:]:
    b = i.abub
    me = i.B.mass_excess(b.ions)
    mex = np.sum(b.abu(molfrac=True) * me, axis=1)
    mex[1]
    lqb.append(i.Qb-s[0].Qb)
    lme.append(mex[1]-mex1[1])
    lie.append(i.en[1] * AMU /MEV - s[0].en[1] * AMU/MEV)
    nu = np.sum(i.snu[1:-1]*i.xm[1:-1])/i.mdot * AMU/MEV
    lnu.append(nu-nu0)
    nuc = np.sum(i.sn[1:-1]*i.xm[1:-1])/i.mdot * AMU/MEV
    lnuc.append(nuc-nuc0)
    pdv = i.pn[1]/i.dn[1]/i.mdot *AMU/MEV
    lpdv.append(pdv-pdv0)
    po = GRAV*(i.zm[1]/i.rn[1] - i.zm[-2]/i.rn[-2])*AMU/MEV
    lpo.append(po-po0)

def plot_d():
    l_label = [None, None, None, None, None, None]
    for i in range(len(lie)):
        if i == (len(lie) - 1):
            l_label = ['M.E.', 'Nuclear', 'Internal', 'Nuetrino', 'Mech', 'Grav']
        plt.plot(lqb[i],lme[i],'r.', label=l_label[0])
        plt.plot(lqb[i],lnuc[i],'b.', label=l_label[1])
        plt.plot(lqb[i],lie[i],'g.', label=l_label[2])
        plt.plot(lqb[i],lnu[i],'k.', label=l_label[3])
        plt.plot(lqb[i],lpdv[i],'y.', label=l_label[4])
        plt.plot(lqb[i],lpo[i],'c.', label=l_label[5])
        _sum = lme[i]+lnuc[i]+lie[i]+lnu[i]+lpdv[i]+lpo[i]
        plt.plot(lqb[i], _sum, 'k*')
    plt.xlabel('$\Delta Q_b, \mathrm{MeV/u}$')
    plt.ylabel('$\Delta, \mathrm{MeV/u}$')
    plt.show()
    plt.legend()