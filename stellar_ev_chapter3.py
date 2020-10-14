from matplotlib import pylab as plt
import numpy as np

class dentem():
    def plot(ax=None, fig=None):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        a = 7.56e-15 # constant a for radiation pressure
        R = 83144626.1815324 # gas constant
        mu = 1 / 2 # mean molecular weight for ions and e^-
        mu_e = 1 # mean molecular weight for e^-
        P_c = 6.62607015e-27 # planck constant
        m_e = 9.1093897e-28 # electron mass
        m_u = 1.6605390671738466e-24 # atomic mass unit
        C = 29979245800
        K_NR = P_c**2 * (3 / np.pi)**(2/3) / 20 / m_e / m_u**(5/3) # non relativistic 
        K_ER = P_c * 3e10 * (3 / np.pi)**(1/3) / 8 / m_u**(4/3) # extremely relativistic
        d = np.linspace(0, 1e10 , 100000)
        T_ri = np.cbrt(3 * R * d / mu / a)  
        T_ie = K_NR * mu / R * d**(2/3)
        d_vl = m_u * 8 * np.pi / 3 * (m_e * C  / P_c)**3
        ax.plot(d, T_ri)
        ax.plot(d, T_ie)
#        _ymax =  K_NR * mu / R * d_vl**(2/3)
        ax.axvline(x=d_vl, color='k')
        ax.set_ylim(1e3 , 1e9)
        ax.set_xlim(1e4 , 1e10)
        plt.show()
