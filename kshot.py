from physconst import ARAD, RK, GRAV, SB, CLIGHT, MEV, NA
from isotope import ion as I, ufunc_A, ufunc_Z
from heat.net import Net3aC12 as Net
from heat.eos import eos as Eos
from heat.kappa import kappa as Kappa
from abuset import AbuSet
from heat.numeric import *
import numpy as np
from functools import partial
from heat.kappa import kappa as TabKappa_
from matplotlib import pyplot as plt

class TabKappa(object):
    def __init__(self, *args, **kwars):
        self.kappa = TabKappa_(*args, **kwars)
    def __call__(self, *args, **kwargs):
        ki, kitx, kidx = self.kappa.__call__(*args, **kwargs)
        ki2 = 2 * ki
        return ki, kitx * ki2, kidx * ki2

class NetWork(Net):
    def __init__(self, abu, *args, **kwargs):
        self.abu = abu
        ppn = np.array([abu.Y(I.he4), 0, 0])
        super().__init__(ppn, *args, **kwargs)

    def mu(self):
        mui = np.sum(self.ppn * (1 + ufunc_Z(self.ions)))
        mui += np.sum([a / i.mu for i,a in self.abu if i not in self.ions])
        return 1 / mui

    def sdot(self, t, d, dt):
        result = super().epsdt(t, d, dt, update=True)
        return result

class SimpleEos(object):
    def __init__(self , net):
        self.mu = net.mu()

    def __call__(self , T , rho):
        mu = self.mu
        p = ARAD / 3 * T**4  + RK * T * rho / mu
        u = 3 / 2* RK *T /mu + ARAD*T**4/rho
        pt = 4/3*ARAD*T**3 + RK*rho/mu
        pd = RK*T/mu
        ut = 1.5 * RK / mu + 4*ARAD*T**3/rho
        ud = -4*ARAD*T**4/rho**2
        return p,u,pt,pd,ut,ud    

class SimpleKappa(object):
    def __init__(self, abu):
        self.abu = abu
    def __call__(self, T, rho, dT = True, drho=True):
        k = 0.4 * self.abu.Ye()
        result = [1/k]
        if dT:
            kt = 0.
            result.append(kt)
        if drho:
            kd = 0.
            result.append(kd)
        if len(result) == 0:
            return result[0]
        return result

class SimpleNet(object):
    def __init__(self, abu, *args, **kwargs):
        kappa = kwargs.pop('kappa', 'abc')
        if kappa == 'simple':
            self._kappai = SimpleKappa(abu) 
        else:
            self._kappai = partial(TabKappa(), T4=False) 
        eos = kwargs.pop('eos', 'table')
        if eos == 'simple':
            self.eos = SimpleEos(self._net) 
        else:
            self.eos = partial(Eos(), T4=False) 
        self._net = NetWork(abu, *args, **kwargs)
        self.sdot = self._net.sdot 


class Shot(object):

#    Shooting code.
#    We use same grid variable naming scheme as in Kepler
#      0      1         2              jm
#    .....|--------|--------|--//--|--------|........
#         r0       r1       r2              r[jm]
#            t[1]     t[2]           t[jm]      t[jm+1]
#            d[1]     d[2]           d[jm]	d[jm+1]
#            p[1]     p[2]           p[jm]      p[jm+1]
#            e[1]     e[2]           e[jm]
#         xl0                               xl[jm]
#         g0                                g[jm]
#    t[jm+1], d[jm+1], p[jm] - surface
#    in loops we us:
#      0 - current zone
#      1 - zone above
#      2 - 2 aones above
#      m - zone below.
#      m2 - 2 zones below
#    specifically, zone j has index 0 and lower boundary values
#    at index m and upper boundary values at indices 0
#    --//--|-------|-------|-------|-------|--//--
#              m       0       1       2
#         m2       m       0       1       2
    def __init__(self, L=7e35, R=1e6, M=2.8e33, mdot=5e17, # Mdot must be in unit g/s
            abu = None, 
            amode = None,
            xms=1e13, xmsf=1.2,
            net='',
            eos=''
                 ): 
        if abu is None:
            abu = dict(he4=0.99, n14=0.009, fe56=0.001)
        abu = AbuSet(abu)
        if net == 'simple':
            net = SimpleNet(abu)
            self.net = net
            eos = net.eos
            sdot = net.sdot
            kappa = net._kappai
        else:
            print(f'please define a network')
        #breakpoint ()   
        #break 

        T = qqrt(L / (4 * np.pi * R**2 * SB)) 
        g = GRAV*M/R**2
        d0 = 1
        while True:
            p0,u0,_,p0bd0,u0bt0,u0bd0 = eos(T , d0)
            ki0,_,ki0bd0 = kappa(T , d0)
            f = p0 - g / 1.5 * ki0
            if np.abs(f) < 1e-12 * p0:
                break
            df = p0bd0 - g / 1.5 * ki0bd0           # ki0bd0 has to be divided by 
#            df = p0bd0 - g / 1.5 * ki0bd0 * 2 * ki0  # original derivative of ki  
            dd0 = f/df
            d0n = d0 - dd0  
            d0 = np.minimum(GOLDEN * d0, np.maximum(d0 / GOLDEN, d0n))  # 1.61 , d0 is the boundary density 
### 1st zone        
        p_surf = p0
        t_surf = T
        #dm = d0 * 4 * np.pi * (R**3 - (R-dr)**3) / 3  # solve directly , dm : change of mass
        t0 = T 
        r0 = R
        xm1 = 4 * np.pi * r0**2   * p0  / g  # surface mass
        xl0 = L
        z0 = M
        xm0 = xms
        g0 = GRAV*z0/r0**2
        p1 = p_surf + 0.5 * xm0 * g0 / (4 * np.pi * r0**2) 
        u1 = u0
        d1 = d0
        ppn0 = net._net.ppn.copy()
        while True:
            p0,u0,p0bt0,p0bd0,u0bt0,u0bd0 = eos(t0 , d0)   # what is b? 
            ki0,ki0bt0,ki0bd0 = kappa(t0 , d0) 

            rm = np.cbrt(r0**3 - 3 * xm0 / (4 * np.pi * d0)) # the density of the first half zone is the surface density and unchanged
            dr0 = r0 - rm # rm : curretly r at bottom
            ac = 4 * np.pi * r0**2 * ARAD * CLIGHT / 3
            acdr0 = ac * ki0 / (d0 * 0.5 * dr0) # 0.5 comes from going to a half of the zone
            l0 = (t0**4 - t_surf**4) * acdr0
            
            f0 = p0 - p1
            h0 = l0 - xl0

            b = np.array([f0,h0]) # by Alex
            b1 = np.array([p1,xl0])

            if np.abs(f0/p1) < 1e-12 and np.abs(h0/xl0) < 1e-12:
                break

            dr0bd0 = - xm0 / (rm**2 * 4 * np.pi * d0**2)
            f0bt0 = p0bt0
            f0bd0 = p0bd0 

            h0bt0 = l0 * ki0bt0 / ki0 + acdr0 * 4 * t0 ** 3
            h0bd0 = l0 * (ki0bd0 / ki0 - 1 / d0 - dr0bd0 / dr0)

#            h0bt0 = l0 * 2 * ki0bt0  + acdr0 * 4 * t0 ** 3 # original kappa
#            h0bd0 = l0 * (2 * ki0bd0 - 1 / d0 - dr0bd0 / dr0) # original kappa

            A = np.array([[f0bt0, f0bd0],[h0bt0, h0bd0]]) # by Alex
            c = np.linalg.solve(A,b) # by Alex
            v = np.array([t0, d0]) # by Alex
            t0, d0 = v - c # by Alex
# goes to center of the zone
# P,T defined by center except the surface, 
# 'kappa' function return ki and 2x its logarithmic derivatives

# third step: include energy generation, we have t0 d0 p0 at the zone center
# for pdv : using the boundary pressure
        p1  = p_surf
        dt0 = xm0 / mdot
        s0  = net.sdot(t0, d0, dt0)
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
#        z0  = M - xm1 
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
        print(f'first zone , tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
        print(f'next zone mass={xm0*1.2:12.5e}')

# l0 is actually depends on current u0 which is computed in iteration
# 1st zone mass is modified by not affecting all the variables fro which the original zone mass is used
        sv1 = sv0 = 0

#        k = np.log10(1 - (M - xm_surf)/xms + xmsf * (M - xm_surf) / xms) / np.log10(xmsf) -1
        k   = 2000 
        tn  = np.ndarray(k)
        dn  = np.ndarray(k)
        xm  = np.ndarray(k)
        pn  = np.ndarray(k)
        sv  = np.ndarray(k)
        xln = np.ndarray(k)
        sn  = np.ndarray(k)
        smn  = np.ndarray(k)
        scn = np.ndarray(k)
        rn  = np.ndarray(k)
        dln = np.ndarray(k) 
        xlnsv  = np.ndarray(k)
        abu = np.ndarray(k, dtype=np.object)
        yy  = np.ndarray(k) 

        tn[0]  = t_surf
        dn[0]  = d1
        xm[0]  = xm1
        pn[0]  = p1
        sv[0]  = 0
        xln[0] = xln[1] = xl0
        dln[0] = 0
        sn[0]  = 0
        smn[0]  = 0
        scn[0] = 0
        rn[0]  = np.inf
        xlnsv[0]  = 0
        abu[0] = ppn0
        yy[0] = 0

        tn[1]  = t0
        dn[1]  = d0
        xm[1]  = xm0
        pn[1]  = p0
        rn[1]  = R
        sn[1]  = s0
        smn[1]  = s0* xm0
        xlnsv[1]  = 0
        abu[1] = net._net.ppn.copy()
        yy[1] = xm1 / (4 * np.pi * R**2)

# starting from the second zone
        for j in range(1 , k , 1):
            xm2 = xm1
            xm1 = xm0
            xm0 = xm1 * xmsf
            dt0 = xm0 / mdot
            r1  = r0
            r0  = rm
            z1  = z0  # mass for computing gravity
            z0  = z1 - xm1
            g1  = g0
            g0  = GRAV * z0 / r0**2
    
            ki1 = ki0
            p2  = p1
            p1  = p0
            t1  = t0
            d2  = d1
            d1  = d0
            u2  = u1
            u1  = u0    
            xl1 = xl0
            sv2 = sv1
            sv1 = sv0
            s1  = s0

            p   = p1 +  0.5 * (xm0 + xm1) * g0 / (4 * np.pi * r0**2) 
            dmx1 = 2 * mdot / (xm1 + xm2)
            dmx0 = 2 * mdot / (xm0 + xm1)
            dt0 = xm0 / mdot

            if amode == 1:
                pdv1 = 2 / (1 / p2 + 1 / p1) * (1 / d2 - 1 / d1)
                du1 = u2 - u1
                dL1 = (du1 + pdv1) * dmx1
            else:
                du1 = u2 - u1
                dL1 = du1 * dmx1

#            pdv1 = 0.5 * (p2 +  p1) * (1 / d2 - 1 / d1)
            ac = (4 * np.pi * r0**2)**2 * ARAD * CLIGHT / (3 * (xm0 + xm1))  # use xm0 , xm1
            jj = 1
            while True:
                jj += 1
                p0,u0,p0bt0,p0bd0,u0bt0,u0bd0 = eos(t0 , d0)  
                ki0,ki0bt0,ki0bd0 = kappa(t0 , d0) 

                du0    = u1 - u0
                du0bt0 = - u0bt0
                du0bd0 = - u0bd0

                if amode == 1:
                    pdv0    = 2 / (1 / p1 + 1 / p0) * (1 / d1 - 1 / d0)
                    pdv0bt0 = pdv0 / (1 / p1 + 1 / p0) * p0bt0 / p0**2
                    pdv0bd0 = pdv0 / (1 / p1 + 1 / p0) * p0bd0 / p0**2 + 2 / (1 / p1 + 1 / p0) * (1 / d0**2)
                    dL0  = (du0 + pdv0) * dmx0
                    dxl0bt0 = - 0.5 * (du0bt0 + pdv0bt0) * dmx0 * xm1
                    dxl0bd0 = - 0.5 * (du0bd0 + pdv0bd0) * dmx0 * xm1
                    sv1  = 0.5 * (dL1 + dL0)
                else:
                    pdv = (p2 + p1) / (d2 + d1) - (p1 + p0) / (d1 + d0) # divided by two on the Numerator and denominator
                    pdv0bt0 = -  p0bt0 / (d1 + d0)
                    pdv0bd0 = (p1 + p0 - (d1 + d0) * p0bd0) / (d1 + d0)**2
                    dphi = g0 * r0 - g1 * r1
                    dL0 = du0 * dmx0
                    dw = (pdv + dphi) *  mdot / xm1
                    dxl0bt0 = - 0.5 * du0bt0 * dmx0 * xm1 - pdv0bt0 * mdot
                    dxl0bd0 = - 0.5 * du0bd0 * dmx0 * xm1 - pdv0bd0 * mdot
                    sv1  = 0.5 * (dL1 + dL0) + dw
#                pdv0    = 0.5 * (p1 + p0) * (1 / d1 - 1 / d0)
#                pdv0bt0 = 0.5 * p0bt0 * (1 / d1 - 1 / d0)
#                pdv0bd0 = 0.5 * p0bd0 * (1 / d1 - 1 / d0) + 0.5 * (p1 + p0) / d0**2 

                dL   = (sv1 + s1) * xm1
                xl0  = xl1 - dL

                acdr0 = ac * (ki0 + ki1)
                l0 = (t0**4 - t1**4) * acdr0
                f0 = p0 - p
                h0 = l0 - xl0
    
                b = np.array([f0,h0])
                b1 = np.array([p,xl0])
    
                if np.abs(f0/p) < 1e-12 and np.abs(h0/xl1) < 1e-12:
                    break
                print(f'Iteration {jj}={f0/p , h0/xl1}')
                f0bt0 = p0bt0
                f0bd0 = p0bd0 
    
                h0bt0 = (t0**4 - t1**4) * ac * ki0bt0 + acdr0 * 4 * t0 ** 3 - dxl0bt0
                h0bd0 = (t0**4 - t1**4) * ac * ki0bd0 - dxl0bd0

#                h0bt0 = (t0**4 - t1**4) * ac * 2 * ki0bt0 * ki0  + acdr0 * 4 * t0 ** 3 - dxl0bt0
#                h0bd0 = (t0**4 - t1**4) * ac * 2 * ki0bd0 * ki0 - dxl0bd0
    
                A = np.array([[f0bt0, f0bd0],[h0bt0, h0bd0]])
                c = np.linalg.solve(A,b)
                v = np.array([t0, d0])
                t0, d0 = v - c


            s0  = net.sdot(t0, d0, dt0)
            rm  = np.cbrt(r0**3 - 3 * xm0 / (4 * np.pi * d0))
            yy0 = xm1 / (4 * np.pi * r0**2)

            tn[j+1]  = t0
            dn[j+1]  = d0
            xm[j+1]  = xm0
            pn[j+1]  = p0
            xln[j+1] = xl0
            sv[j] = sv1
            dln[j] = dL * xm0
            sn[j+1]  = s0
            smn[j+1]  = s0 * xm0
            rn[j+1]  = r0
            xlnsv[j+1] = sv1 * xm1
            abu[j+1] = net._net.ppn.copy()
            yy[j+1] = yy[j] + yy0
       
            print(f'zone {j+1}, tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
            print(f'current zone mass={xm0:12.5e}, next zone mass={xm0*xmsf:12.5e}')

            if d0 > 5e11 or t0 > 5e9:
                break

# phoney
        rn[j+2] = rm
        sv[j+1] = sv1**2 / sv2
        xlnsv[j+2] = sv[j+1] * xm0
        xl0 = xl0 - (s0 + sv[j+1]) * xm0
# phoney
        tn[j+2] = np.nan
        dn[j+2] = np.nan
        pn[j+2] = np.nan
        sv[j+2] = np.nan
        xm[j+2] = M
        xln[j+2] = xl0
        sn[j+2] = np.nan
        smn[j+2] = np.nan
        abu[j+2] = np.array([0,0,0])

        tn      = tn[:j+3][::-1]
        dn      = dn[:j+3][::-1]
        pn      = pn[:j+3][::-1]
        sv      = sv[:j+3][::-1]
        xm      = xm[:j+3][::-1]
        xln     = xln[:j+3][::-1]
        rn      = rn[:j+3][::-1]
        sn      = sn[:j+3][::-1]
        smn      = smn[:j+3][::-1]
        xlnn = np.append(smn[1:], 0) 
        xlnsv      = xlnsv[:j+3][::-1]
        abu      = abu[:j+3][::-1]
        yy      = yy[:j+3][::-1]

        y = np.cumsum((xm[1:] / (4 * np.pi * rn[:-1]**2))[::-1])[::-1]
        y_m = np.zeros(j+3)
        y_m[1:-1] = 0.5 * (y[:-1] + y[1:])
        y_m[0] = y[0]
        y_m[-1] = y[-1]

        self.pn  = pn
        self.tn  = tn
        self.dn  = dn
        self.sv  = sv
        self.xm  = xm
        self.xln = xln
        self.dln = dln
        self.rn  = rn
        self.sn  = sn
        self.mdot  = mdot

        self.smn  = smn
        self.y   = np.append(y,0)
        self.y_m = y_m
        self.xlnsv = xlnsv
        self.xlnn = xlnn
        self.abu = abu
        self.ppn = np.array([a for a in abu])
        self.yy  = yy 

    def plot_l(self, escale=None):
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        if escale=='MeV':
            scale = MEV * self.mdot * NA
            ax.set_ylabel('Specific flux ($\mathrm{MeV\,nucleons}^{-1}\,\mathrm{s}^{-1}$)')
        else:
            scale = 1
            ax.set_ylabel('Luminosity ($\mathrm{erg\,s}^{-1}$)')

        ax.set_xscale('log')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        xlnn = np.cumsum(self.xlnn[ir])[ir]
        xlnsv = np.cumsum(self.xlnsv[ir])[ir]
        xlsum = self.xln + xlnn + xlnsv

        ax.plot(self.y_m[i1], self.xln[i1] / scale, label= '$L_{\mathrm{m}}$')
        ax.plot(self.y_m[i1], xlnn[i1] / scale, label = '$L_{\mathrm{nuc}}$')
        ax.plot(self.y_m[i1], xlnsv[i1] / scale, label = '$L_{\mathrm{grav}}$')
        ax.plot(self.y_m[i1], xlsum[i1] / scale, '--', label='sum')
        ax.legend(loc='best')
        plt.show()
       
    def plot_l2(self):
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        
        scale = MEV * self.mdot * NA
        ax.set_xscale('log')
        ax.set_ylabel('Specific flux ($\mathrm{MeV\,nucleons}^{-1}\,\mathrm{s}^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        xlnn = np.cumsum(self.xlnn[ir])[ir]
        xlnsv = np.cumsum(self.xlnsv[ir])[ir]
        xlsum = self.xln + xlnn + xlnsv

        ax.plot(self.y_m[i1], self.xln[i1] / scale, label= '$L_{\mathrm{m}}$')
        ax.plot(self.y_m[i1], xlnn[i1] / scale, label = '$L_{\mathrm{nuc}}$')
        ax.plot(self.y_m[i1], xlnsv[i1] / scale, label = '$L_{\mathrm{grav}}$')
        ax.plot(self.y_m[i1], xlsum[i1] / scale, '--', label='sum')
        ax.legend(loc='best')
        plt.show()
       
    def plot_td(self):
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Temperature ($\mathrm{K}$), Density ($\mathrm{g\,cm}^{-3}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        ax.plot(self.y_m[i1], self.tn[i1], label= '$\mathrm{T}$')
        ax.plot(self.y_m[i1], self.dn[i1], label= '$\\rho$')
        ax.legend(loc='best')
        plt.show()

    def plot_abu(self):
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_ylabel('Mass fraction')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        for j,i in enumerate(self.net._net.ions):
            ax.plot(self.y_m[i1], self.ppn[i1, j] * i.A, label=i.mpl)
        ax.legend(loc='best')
        plt.show()

    def plot_s(self):
        i1 = slice(1, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Specific energy generation rate ($\mathrm{erg\,g}^{-1}\mathrm{s}^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        ax.plot(self.y_m[i1], self.sn[i1], label= 'Nuclear')
        ax.plot(self.y_m[i1], self.sv[i1], label= 'Gravothermol')

        ax.legend(loc='best')
        smax = np.maximum(np.max(self.sn[i1]), np.max(self.sv[i1])) * 2
        smin = smax * 1e-18
        ax.set_ylim(smin, smax)

        plt.show()
