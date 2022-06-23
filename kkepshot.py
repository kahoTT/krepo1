from logging import raiseExceptions
from physconst import ARAD, RK, GRAV, SB, CLIGHT, MEV, NA, SIGT, AMU 
from isotope import ion as I, ufunc_A, ufunc_Z, ufunc_idx, ufunc_ion_from_idx, ioncacheza, ufunc_idx_ZA
from heat.net import Net3aC12 as Net
from heat.eos import eos as Eos
from heat.kappa import kappa as Kappa
from abuset import AbuSet, AbuDump
from heat.numeric import *
import numpy as np
from functools import partial
from heat.kappa import kappa as TabKappa_
from matplotlib import pyplot as plt
from starshot.kepnet import KepNet
from heat.numeric import sqrt, cbrt, qqrt, GOLDEN
import matplotlib.colors as colors
from serialising import Serialising
from ioncolor import IonColor
from utils import index1d
import time

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


class Shot(Serialising):

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
#      1 - one zone above
#      2 - two zones above
#      m - one zone below.
#      m2 - two zones below
#    specifically, zone j has index 0 and lower boundary values
#    at index m and upper boundary values at indices 0
#    --//--|-------|-------|-------|-------|--//--
#              m       0       1       2
#         m2       m       0       1       2
    def __init__(self, L=7e35, R=1e6, M=2.8e33, mdot=5e17, # default mdot = 1 Eddington accretion rate
            abu = None, 
            amode = 1,
            xms=1e13, xmsf=1.2,
            net='',
            eos='',
            kepler = 'restart', # module | process | restart 
            yfloorx = 1.e-3,
            safenet = True,
            eosmode = None, # static | burn | adapt , is NOT adaptive step size !!!
            kaptab = 4,
            dtcp  = None,
            scale = 1,
            accuracy = 1.e-10,
            accept = 1.e-8,
            Q = None,
            ymax = 1e12,
            endnet = True,
            silent = None,
            burn = True,
            par = None,
            track_abu = True,
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
            if eosmode is None:
                eosmode = 'adapt' # default
            if par is None:
                par = dict()
            else:
                par = dict(par)
            if burn == False:
                par.setdefault('tnucmin', 1.e99)
            net = KepNet( # default
                abu,
                kepler=kepler,
                eosmode=eosmode,
                safenet=safenet,
                kaptab=kaptab,
                yfloorx=yfloorx,
                dtcp=dtcp,
                scale=scale,
                **par,
                )
            self.net = net
            eos = net.eos
            sdot = net.sdot

        xledd = 4 * np.pi * CLIGHT * GRAV * M * AMU * abu.mue() / SIGT
        xaccedd = xledd * R / (GRAV * M)
        if mdot < 10:
            mdot = xaccedd * mdot
        self.mdot = mdot / xaccedd
        if L < 10:
            L = L * xledd
        if Q is not None:
            L = mdot * Q * NA * MEV
        else:
            Q = L / (mdot * NA * MEV)
        self.Q = Q
        self.L = L
        self.Ledd = L/xledd
        print(f'[SHOT] Mdot = {mdot:12g} g/s ({mdot/xaccedd:12g} Edd.)')
        print(f'[SHOT] L    = {L:12g} erg/s ({L/xledd:12g} Edd.)')

# surface zone
        T = qqrt(L / (4 * np.pi * R**2 * SB)) 
        g = GRAV*M/R**2
        d0 = 1.
        dt0 = 1.
        jj = 0
        while True:
            jj += 1
            p0, u0, _, p0bd0, _, _, ki0, _, ki0bd0, _ = eos(T, d0, dt0)
            h = p0 - g / 1.5 * ki0 # ki = 1/kappa
            print(f'[SHOT] Iteration {jj} {h/p0}')
            if np.abs(h) < 1e-12 * p0:
               break
            dh = p0bd0 - g / 1.5 * ki0bd0     
            dd0 = h/dh
            d0n = d0 - dd0  
            d0 = np.minimum(GOLDEN * d0, np.maximum(d0 / GOLDEN, d0n))  # 1.61 , d0 is the boundary density 
        print(f'[SHOT] surface zone , tn={T:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc')

### 1st zone ###        
        p_surf = p0
        t_surf = T
        #dm = d0 * 4 * np.pi * (R**3 - (R-dr)**3) / 3  # solve directly , dm : change of mass
        t0 = T 
        r0 = R
        xm1 = 4 * np.pi * r0**2   * p0  / g  # surface mass
        xl0 = L
        z0 = M
        xm0 = xms
        g0 = GRAV * z0 / r0**2
        p = p_surf + 0.5 * xm0 * g0 / (4 * np.pi * r0**2) 
        u1 = u0
        d1 = d0
        ki1 = ki0 
        dt0 = xm0 / mdot
        ppn0 = net.abu()
        jj = 0
        ri = 1
        fmin = 1
        while True:
            jj += 1
            p0, u0, p0bt0, p0bd0, _, _, ki0, ki0bt0, ki0bd0, dxmax  = eos(t0, d0, dt0)
            rm = np.cbrt(r0**3 - 3 * xm0 / (4 * np.pi * d0)) 
#            dr0 = r0 - rm # rm : curretly r at zone bottom
            rmc = np.cbrt(r0**3 - 3 * xm0 / (8 * np.pi * d0)) 
            drc0 = r0 - rmc # goes to center of zone
            ac = 16 * np.pi * r0**2 * ARAD * CLIGHT / 3
            acdr0 = ac * ki0 / (d0 * drc0) 
            l0 = (t0**4 - t_surf**4) * acdr0
            
            h0 = p0 - p
            f0 = l0 - xl0

            b = np.array([h0,f0]) # by Alex
            b1 = np.array([p,xl0])
            dvr = b / b1

            print(f'[SHOT] Iteration {jj} {h0/p, f0/xl0}')
            if np.max(np.abs(dvr)) < 1e-12:
                break

            if jj >= 5:
                if np.max(np.abs(dvr)) < accuracy:
                    break

            drc0bd0 = - xm0 / (rmc**2 * 8 * np.pi * d0**2) # need to change
            h0bt0 = p0bt0
            h0bd0 = p0bd0 

            f0bt0 = l0 * ki0bt0 / ki0 + acdr0 * 4 * t0 ** 3
            f0bd0 = l0 * (ki0bd0 / ki0 - 1 / d0 - drc0bd0 / drc0)

            A = np.array([[h0bt0, h0bd0],[f0bt0, f0bd0]]) 
            c = np.linalg.solve(A,b) 
            v = np.array([t0, d0]) 
            dfr = c / v
            dfrm = np.max(np.abs(dfr))
            if dfrm > GOLDEN - 1:
                ri = fmin / (dfrm * GOLDEN)
#            if ri != fmin:
#                print(f'[SHOT] {ri} reduction for the correction of temperature and density')
            ri = 1
            t0, d0 = v - c * ri
# goes to center of the zone
# P,T defined by center except the surface, 
# 'kappa' function return ki and 2x its logarithmic derivatives

# third step: include energy generation, we have t0 d0 p0 at the zone center
# for pdv : using the boundary pressure
        p1  = p_surf
        s0, snu0, dxmax = sdot(t0, d0, dt0)
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
#        z0  = M - xm1 
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
        print(f'[SHOT] first zone , tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
        print(f'[SHOT] next zone mass={xm0*1.2:12.5e}')

# l0 is actually depends on current u0 which is computed in iteration
# 1st zone mass is modified by not affecting all the variables fro which the original zone mass is used
        sv1 = sv0 = 0

#        k = np.log10(1 - (M - xm_surf)/xms + xmsf * (M - xm_surf) / xms) / np.log10(xmsf) -1
        k   = 100000 
        tn  = np.ndarray(k)
        dn  = np.ndarray(k)
        ki  = np.ndarray(k)
        xm  = np.ndarray(k)
        pn  = np.ndarray(k)
        zm  = np.ndarray(k)
        sv  = np.ndarray(k)
        un = np.ndarray(k)
        en = np.ndarray(k)
        mec = np.ndarray(k)
        phi = np.ndarray(k)
        xln = np.ndarray(k)
        sn  = np.ndarray(k)
        snun  = np.ndarray(k)
        smn  = np.ndarray(k)
        smnun  = np.ndarray(k)
        scn = np.ndarray(k)
        rn  = np.ndarray(k)
        dln = np.ndarray(k) 
        xlnsv  = np.ndarray(k)
        xlnint  = np.ndarray(k)
        xlnmec  = np.ndarray(k)
        xlnphi  = np.ndarray(k)
        if track_abu:
            abu = np.ndarray(k, dtype=np.object)
        abulen = np.ndarray(k)
        mue = np.ndarray(k)
        max_mass_no = np.ndarray(k)
        gn  = np.ndarray(k) 
        y  = np.ndarray(k)

        tn[0]  = t_surf
        dn[0]  = d1
        ki[0]  = 1/ki1
        xm[0]  = xm1
        pn[0]  = p1
        zm[0]  = M
        sv[0]  = 0
        un[0]  = 0
        en[0]  = u1
        mec[0] = 0
        phi[0] = 0
        xln[0] = xln[1] = xl0
        dln[0] = 0
        sn[0]  = 0
        snun[0]  = 0
        smn[0]  = 0
        smnun[0]  = 0
        scn[0] = 0
        rn[0]  = np.inf
        xlnsv[0]  = 0
        xlnint[0] = 0
        xlnmec[0] = 0
        if track_abu:
            abu[0] = ppn0
        abulen[0] = len(abu[0])
        mue[0] = abu[0].mue()
        max_mass_no[0] = np.max(ufunc_A(abu[0].iso))
        y[0] = 0
        gn[0] = g

        tn[1]  = t0
        dn[1]  = d0
        ki[1]  = 1/ki0
        xm[1]  = xm0
        pn[1]  = p0
        zm[1]  = z0
        en[1]  = u0
        sn[1]  = s0
        snun[1]  = snu0
        smn[1]  = s0 * xm0
        smnun[1]  = snu0 * xm0
        xlnsv[1]  = 0
        xlnint[1] = 0
        xlnmec[1] = 0
        if track_abu:
            abu[1] = net.abu()
        abulen[1] = len(abu[1])
        mue[1] = abu[1].mue()
        max_mass_no[1] = np.max(ufunc_A(abu[1].iso))
        rn[1]  = R
        rn[2]  = rm
        y[1] = xm1 / (4 * np.pi * R**2)             
        y[2] = y[1] +  xm0 / (4 * np.pi * R**2)
        gn[1] = g0

# starting from the SECOND ZONE
        second_last_step = None
        for j in range(1 , k , 1): 
            xm2 = xm1
            xm1 = xm0
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
            snu1  = snu0
    ### adaptive network set in to change the xmsf ###
            xmaf = 1
            while True:
                restart = None
                if second_last_step is True:
                    xm0 = xm0_2last
                else:
                    xm0 = xm1 * xmsf * xmaf # a for adaptive; f for factor
                p   = p1 +  0.5 * (xm0 + xm1) * g0 / (4 * np.pi * r0**2) 
                dmx1 = 2 * mdot / (xm1 + xm2)
                dmx0 = 2 * mdot / (xm0 + xm1)
                dt0 = xm0 / mdot
                du1 = u2 - u1
    
                if amode == 1: # harmonic mean, the accurate one. Currently just amode == 1 is used
                    pdv1 = 2 / (1 / p2 + 1 / p1) * (1 / d2 - 1 / d1)
                    dL1 = (du1 + pdv1) * dmx1
    
                elif amode == 2: # geometric mean
                    pdv1 = np.sqrt(p1*p2) * (1 / d2 - 1 / d1)
                    dL1 = (du1 + pdv1) * dmx1

                elif amode == 3: # Arithmetic mean
                    pdv1 = 0.5 * (p1 + p0) * (1 / d2 - 1 / d1)
                    dL1 = (du1 + pdv1) * dmx1

                else:
                    raise AttributeError(f'Need to define a way to manage pressure between zones')
      
                jj = 0
                fmin = 1
                ac = (4 * np.pi * r0**2)**2 * ARAD * CLIGHT / (3 * (xm0 + xm1))  # use xm0 , xm1
    ### main loop ###
                while True:
                    jj += 1
# elements are updated in the following step
                    p0, u0, p0bt0, p0bd0, u0bt0, u0bd0, ki0, ki0bt0, ki0bd0, dxmax = eos(t0 , d0, dt0)  
    
                    du0    = u1 - u0
                    du0bt0 = - u0bt0
                    du0bd0 = - u0bd0
    
                    if amode == 1: # Harmonic mean for boundary pressure
                        pdv0    = 2 / (1 / p1 + 1 / p0) * (1 / d1 - 1 / d0)
                        pdv0bt0 = pdv0 / (1 / p1 + 1 / p0) * p0bt0 / p0**2
                        pdv0bd0 = pdv0 / (1 / p1 + 1 / p0) * p0bd0 / p0**2 + 2 / (1 / p1 + 1 / p0) * (1 / d0**2)
                        dL0  = (du0 + pdv0) * dmx0
                        dxl0bt0 = - 0.5 * (du0bt0 + pdv0bt0) * dmx0 * xm1
                        dxl0bd0 = - 0.5 * (du0bd0 + pdv0bd0) * dmx0 * xm1
                        sv1  = 0.5 * (dL1 + dL0) # 0.5 should be better to put on line 411
    
                    elif amode == 2: # geometic mean
                        pdv0 = np.sqrt(p0 * p1) * (1 / d1 - 1 / d0)
                        pdv0bt0 = 0.5 * pdv0 * p0bt0 / p0
                        pdv0bd0 = 0.5 * pdv0 * p0bd0 / p0 + np.sqrt(p0 * p1) / d0**2
                        dL0  = (du0 + pdv0) * dmx0
                        dxl0bt0 = - 0.5 * (du0bt0 + pdv0bt0) * dmx0 * xm1
                        dxl0bd0 = - 0.5 * (du0bd0 + pdv0bd0) * dmx0 * xm1
                        sv1  = 0.5 * (dL1 + dL0) 

                    elif amode == 3: # Arithmetic mean
                        pdv0 = 0.5 * (p0 + p1) * (1 / d1 - 1 / d0)
                        pdv0bt0 = 0.5 * p0bt0 * (1 / d1 - 1 / d0)
                        pdv0bd0 = 0.5 * p0bd0 * (1 / d1 - 1 / d0) + 0.5 * (p0 + p1) / d0**2
                        dL0  = (du0 + pdv0) * dmx0
                        dxl0bt0 = - 0.5 * (du0bt0 + pdv0bt0) * dmx0 * xm1
                        dxl0bd0 = - 0.5 * (du0bd0 + pdv0bd0) * dmx0 * xm1
                        sv1  = 0.5 * (dL1 + dL0) 

                    elif amode == 4: # Use integral
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
    
                    dL   = (sv1 + s1) * xm1 # move outside the amode conditions, maybe better to put them back
                    xl0  = xl1 - dL # move outside the amode conditions, maybe better to put them back
    
                    acdr0 = ac * (ki0 + ki1) # the boundary of opacity
                    l0 = (t0**4 - t1**4) * acdr0
                    h0 = p0 - p
                    f0 = l0 - xl0
        
                    b = np.array([h0,f0])
                    b1 = np.array([p,xl0])
                    dvr = b / b1

# for finding t0 and d0
    ### check whether change of abundance is too large

                    if jj <= 5:
                        if np.max(np.abs(dvr)) < accuracy:
                            if dxmax > 1:
                                break                                    
                            else:
                                print(f'[SHOT] Time step reduced as it is too large dxmax = {dxmax}')
                                xmaf *= (GOLDEN - 1)  
                                restart = True
                                break
                            
                    elif jj > 5: 
                        if np.max(np.abs(dvr)) < accept:
                            if dxmax > 1:
                                break                                    
                            else:
                                print(f'[SHOT] Time step reduced as it is too large dxmax = {dxmax}')
                                xmaf *= (GOLDEN - 1)  
                                restart = True
                                break
                        else:
                            xmaf *= (GOLDEN - 1)
                            restart = True
                            break

                    print(f'[SHOT] Iteration {jj}: dvr = [{dvr[0], dvr[1]}]: dxmax = {dxmax}')
                    h0bt0 = p0bt0
                    h0bd0 = p0bd0 
        
                    f0bt0 = (t0**4 - t1**4) * ac * ki0bt0 + acdr0 * 4 * t0 ** 3 - dxl0bt0
                    f0bd0 = (t0**4 - t1**4) * ac * ki0bd0 - dxl0bd0
        
                    A = np.array([[h0bt0, h0bd0],[f0bt0, f0bd0]])
                    c = np.linalg.solve(A,b)
                    v = np.array([t0, d0])
                    dfr = c / v
                    dfrm = np.max(np.abs(dfr))
                    print(f'[SHOT] dfrm = {dfrm}')
                    if dfrm > GOLDEN - 1:
                        ri = fmin / (dfrm * GOLDEN)
#                    if ri != fmin:
#                        print(f'[SHOT] {ri} reduction for the correction of temperature and density')
                    ri = 1
                    t0, d0 = v - c * ri

# If it seems the dxmax is unlikely > 1 after iterations
                    if jj > 1 and dxmax < 0.1 and np.max(np.abs(dvr)) < 1e-3:
                        print(f'[SHOT] Time step reduced as it is too large dxmax = {dxmax}')
                        xmaf *= (GOLDEN - 1)  
                        restart = True
                        break

                if restart == True: # for adaptive network
                    continue

                rm  = np.cbrt(r0**3 - 3 * xm0 / (4 * np.pi * d0))
                ym = xm0 / (4 * np.pi * r0**2)
                y[j+2] = y[j+1] + ym
                if y[j+2] > ymax and second_last_step is None:
                    second_last_step = True
                    print(f'[SHOT]------------------\n[SHOT] penultimate zone\n[SHOT]------------------')
                    xm0_2last = (ymax - y[j+1]) * (4 * np.pi * r0**2) 
                    k = j + 1
                    continue # for index j

                break # break for loop j finished
            s0, snu0, dxmax  = sdot(t0, d0, dt0)
            print(f'[SHOT] dxmax = {dxmax}')


            tn[j+1]  = t0
            dn[j+1]  = d0
            ki[j+1]  = 1/ki0
            xm[j+1]  = xm0
            pn[j+1]  = p0
            zm[j+1]  = z0
            xln[j+1] = xl0
            sv[j] = sv1
            un[j] = 0.5 * (du0 * dmx0 + du1 * dmx1)
            en[j+1] = u0
            mec[j]= 0.5 * (pdv0 * dmx0 + pdv1 * dmx1)
            phi[j] = (g0 * r0 - g1 * r1) * dmx0
            dln[j] = dL * xm0
            sn[j+1]  = s0
            snun[j+1]  = snu0
            smn[j+1]  = s0 * xm0
            smnun[j+1]  = snu0 * xm0
            xlnsv[j+1] = sv1 * xm1
            xlnint[j+1] = un[j] * xm1
            xlnmec[j+1] = mec[j] * xm1
            xlnphi[j+1] = phi[j] * xm1
            if track_abu:
                abu[j+1] = net.abu()
            abulen[j+1] = len(abu[j+1])
            mue[j+1] = abu[j+1].mue()
            max_mass_no[j+1] = np.max(ufunc_A(abu[j+1].iso))
            rn[j+2]  = rm
            gn[j+1]  = g0

            print('-----------------------------------------------------------------------')
            print(f'[SHOT] zone {j+1}, tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
            print(f'current zone mass={xm0:12.5e}, next zone mass={xm0*xmsf:12.5e}')
            print('-----------------------------------------------------------------------')

            if j == k:
                break # for index j

            if ymax is None:
                if (d0 > 5e11 or t0 > 5e10):
                    break # for index j

# bottom heat
        Qb = xl0 / (MEV * mdot * NA)
        Lb = xl0
        self.Qb = Qb
        self.Lb = Lb
        print(f'Base heat flux Qb={Qb} MeV/nucleon or Lb={Lb:12.5e} erg/s')

# phoney
        sv[j+1] = sv1**2 / sv2
        un[j+1] = un[j]
        mec[j+1] = mec[j]
        xlnsv[j+2] = sv[j+1] * xm0
        xlnint[j+2] = un[j+1] * xm0
        xlnmec[j+2] = mec[j+1] * xm0
        xlnphi[j+2] = phi[j+1] * xm0
        xl0 = xl0 - (s0 + sv[j+1]) * xm0
#

        tn[j+2] = np.nan
        dn[j+2] = np.nan
        ki[j+2] = np.nan
        pn[j+2] = np.nan
        zm[j+2] = z0 - xm0
        gn[j+2] = GRAV * zm[j+2] / rn[j+2]**2
        sv[j+2] = np.nan
        un[j+2] = np.nan
        mec[j+2] = np.nan
        phi[j+2] = np.nan
        xm[j+2] = zm[j+2]
        xln[j+2] = xl0
        en[j+2] = en[j+1]
        sn[j+2] = np.nan
        snun[j+2] = np.nan
        smn[j+2] = np.nan
        smnun[j+2] = np.nan
        if track_abu:
            abu[j+2] = AbuSet(dict())
        abulen[j+2] = len(abu[j+2])
        mue[j+2] = np.nan
        max_mass_no[j+2] = 0

        tn      = tn[:j+3][::-1]
        dn      = dn[:j+3][::-1]
        ki      = ki[:j+3][::-1]
        pn      = pn[:j+3][::-1]
        sv      = sv[:j+3][::-1]
        un      = un[:j+3][::-1]
        en      = en[:j+3][::-1]
        xm      = xm[:j+3][::-1]
        xln     = xln[:j+3][::-1]
        rn      = rn[:j+3][::-1]
        gn      = gn[:j+3][::-1]
        sn      = sn[:j+3][::-1]
        snun      = snun[:j+3][::-1]
        smn      = smn[:j+3][::-1]
        smnun      = smnun[:j+3][::-1]
        xlnn = np.append(smn[1:], 0) 
        xlnun = np.append(smnun[1:], 0) 
        xlnsv      = xlnsv[:j+3][::-1]
        xlnint      = xlnint[:j+3][::-1]
        xlnmec      = xlnmec[:j+3][::-1]
        xlnphi      = xlnphi[:j+3][::-1]
        if track_abu:
            abu      = abu[:j+3][::-1]
        abulen   = abulen[:j+3][::-1]
        mue = mue[:j+3][::-1]
        max_mass_no = max_mass_no[:j+3][::-1]
        y      = y[:j+3][::-1]

        y_m = np.zeros(j+3)
        y_m[1:-1] = 0.5 * (y[:-2] + y[1:-1])
        y_m[0] = np.nan
        y_m[-1] = y[-2]

        self.pn  = pn
        self.zm  = zm
        self.tn  = tn
        self.dn  = dn
        self.ki  = ki
        self.sv  = sv
        self.un  = un
        self.en  = en
        self.xm  = xm
        self.xln = xln
        self.dln = dln
        self.rn  = rn
        self.gn  = gn
        self.sn  = sn
        self.snun  = snun
        self.mdot  = mdot

        self.smn  = smn
        self.y  = y
        self.y_m = y_m
        self.xlnsv = xlnsv
        self.xlnint = xlnint
        self.xlnmec = xlnmec
        self.xlnphi = xlnphi
        self.xlnn = xlnn
        self.xlnun = xlnun
        if track_abu:
            self.abu = abu
        self.abulen = abulen
        self.mue = mue
        self.max_mass_no = max_mass_no

# mapping ions
        if track_abu:
            # map all ions
            print(f'[{self.__class__.__name__}] Mapping ions....')
            ions = set()
            ionsa = list()
            for a in abu:
                idx = ufunc_idx(a.iso)
                ions |= set(idx)
                ionsa.append(idx)
            ions = np.array(sorted(ions))
            nions = len(ions)
            abub = np.zeros((nions, j + 3))
            for i, a in enumerate(abu):
                ii = index1d(ionsa[i], ions)
                abub[ii,i] = a.abu
            # ions = ufunc_ion_from_idx(ions)
            ions = ioncacheza(*ufunc_idx_ZA(ions))
            self.abub = AbuDump(
                abub,
                ions,
                molfrac = False,
                bottom = 1,
                top = j + 3,
                xm = self.xm,
                zm = self.zm,
                )

        self.maxions = self.abulen.argmax()
        self.da = ufunc_idx(self.abu[self.maxions].iso)
        self.pabu = np.ndarray(len(self.abu)-1) # the array stars from the second element, skipping the phoney value
        
        if endnet:
            net.done()

    def plot_l(self, escale=None):
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        if escale is None:
            scale = MEV * self.mdot * NA
            ax.set_ylabel('Specific flux ($\mathrm{MeV/u}$)')
        else:
            scale = 1
            ax.set_ylabel('Luminosity ($\mathrm{erg\,s}^{-1}$)')

        ax.set_xscale('log')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')
#        ax.set_ylim(-.3e35, 2e36)

        xlnn = np.cumsum(self.xlnn[ir])[ir]
        xlnun = np.cumsum(self.xlnun[ir])[ir]
        xlnsv = np.cumsum(self.xlnsv[ir])[ir]
        xlsum = self.xln + xlnn + xlnsv

        ax.plot(self.y_m[i1], self.xln[i1] / scale, label= '$L_{\mathrm{m}}$')
        ax.plot(self.y_m[i1], (xlnn[i1] + xlnun[i1]) / scale, label = '$L_{\mathrm{nuc}}$')
        ax.plot(self.y_m[i1], xlnsv[i1] / scale, label = '$L_{\mathrm{grav}}$')
        ax.plot(self.y_m[i1], xlnun[i1] / scale, color='#BFBFBF', ls='--', label = r'$L_{\nu}$')
        ax.plot(self.y_m[i1], xlsum[i1] / scale, ':', label='sum')
        ax.legend(loc='best')
        plt.show()
       
    def plot_l2(self):
        i1 = slice(1, None)
        i0 = slice(None, 1)
        i2 = slice(1, -1)
        i3 = slice(2, None)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        
        scale = 1 / (MEV * self.mdot * NA)
        ax.set_xscale('log')
        ax.set_ylabel('Specific flux ($\mathrm{MeV\,u}^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        xlne = (self.en[i2] - self.en[-2]) * self.mdot 
        ax.plot(self.y_m[i2], xlne * scale, label='Int')

        xlnpdv = (self.pn[i2] / self.dn[i2] - self.pn[-2] / self.dn[-2]) * self.mdot 
        ax.plot(self.y_m[i2], xlnpdv * scale, label='Mech')

        xlngrav = (self.gn[i2] * self.rn[i2] - self.gn[-2] * self.rn[-2]) * self.mdot
        ax.plot(self.y[i2], xlngrav * scale, label='Grav')

        xlnsv = np.cumsum(self.xlnsv[ir])[ir][i2]
        ax.plot(self.y[i2], xlnsv * scale, label='Adv')

        sum  = xlne + xlnpdv
        sum2 = xlngrav - xlnsv

        ax.plot(self.y_m[i2], sum * scale, label='Enthalpy', ls='--')
        ax.plot(self.y[i2], sum2 * scale, label='grav + advection', ls='-.')


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

    def plot_abu(self, lim = 1e-3, A = ''):
        i1 = slice(1, None)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Mass fraction')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')
        ax.set_ylim(1.e-3, 1.5)
        c = IonColor()

        for i,a in self.abub:
            if A:
                if i.A > A:
                    break
            am = np.max(a[i1])
            if am > lim:
                ax.plot(self.y_m[i1], a[i1], label=i.mpl, color=c(i))
                maxabu = np.argmax(a[i1])
                ax.text(
                    self.y_m[i1][maxabu], a[i1][maxabu], i.mpl, color=c(i),
                    ha='center', va='center', clip_on=True)

    def plot_s(self):
        i1 = slice(1, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Specific energy generation rate ($\mathrm{erg\,g}^{-1}\mathrm{s}^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        l = ax.plot(self.y_m[i1], self.sn[i1] + self.snun[i1], label= 'Nuclear')
        ax.plot(self.y_m[i1], -(self.sn[i1] + self.snun[i1]), color=l[0].get_color(), ls=':')
#        ax.plot(self.y_m[i1], self.sv[i1], label= 'Gravothermol')
#        ax.plot(self.y_m[i1], self.sv[i1], 'r.', label= 'Gravothermol')
        ax.plot(self.y_m[i1], self.snun[i1],'--' ,color='#BFBFBF' ,label= 'Neutrino loss')

        ax.legend(loc='best')
        smax = np.maximum(np.max(self.sn[i1]), np.max(self.sv[i1])) * 2
        smin = smax * 1e-18
        ax.set_ylim(smin, smax)

        plt.show()

    def plot_map(self, start=0, end=None): # start from the 2nd most bottom zone
        i1 = slice(1, None)
        i0 = slice(None, -1)
        ir = slice(None, None, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
#        ax.set_yscale('log')
#        ax.set_ylabel('Mass fraction')
        ax.set_ylabel('Mass number')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')

        max_mass_no = self.max_mass_no[int(start+1):end].max()
        y = np.r_[1:max_mass_no+1]
        
        x = self.y_m[int(start+1):end]
        zz = self.abu[int(start+1):end]

        for j in range(0, len(x), 1):
            z1 = np.zeros(len(y))
            for ii in y:
                i = int(ii)
                _int = np.where(i == ufunc_A(zz[j].iso))
                if _int[0].size == 0:
                    z1[i-1] = 0
                else:
                    z1[i-1] = sum(zz[j].abu[_int])
            if j == 0:
                z = z1
            else:
                z = np.vstack((z,z1))
        pcm = ax.pcolor(x, y, z.T, cmap = 'Reds', norm=colors.LogNorm(vmin = 1e-10, vmax = max(map(max, z.T)))) 
        fig.colorbar(pcm, ax=ax, extend='max')
        
    def plot_mue(self):
        i1 = slice(1, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
#        ax.set_yscale('log')
#        ax.set_ylabel('Opacity ($cm^2\,g^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')
        
        ax.plot(self.y_m[i1], self.mue[i1], label='$\mu_e$')
        ax.legend(loc='best')

    def plot_ki(self):
        i1 = slice(1, -1)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Opacity ($\mathrm{cm}^2\,\mathrm{g}^{-1}$)')
        ax.set_xlabel('Column depth ($\mathrm{g\,cm}^{-2}$)')
        
        ax.plot(self.y_m[i1], self.ki[i1], label='$\kappa$')
        ax.legend(loc='best')