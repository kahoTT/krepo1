from physconst import ARAD, RK, GRAV, SB, CLIGHT
from isotope import ion as I, ufunc_A, ufunc_Z
from heat.net import Net3aC12 as Net
from heat.eos import eos as Eos
from heat.kappa import kappa as Kappa
from abuset import AbuSet
from heat.numeric import *
import numpy as np
from functools import partial

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

class NetWork(Net):
    """
    modified network that takes into account inert trace elements
    """
    def __init__(self, abu, *args, **kwargs):  #first abu , create abu for NetWork, **kwargs pass ketworads
        self.abu = abu
        ppn = np.array([abu.Y(I.he4), 0, 0])
        super().__init__(ppn, *args, **kwargs)
    def mu(self):
        """
        compute mu from combination of network abundaces and total abundances
        """
        mui = np.sum(self.ppn * (1 + ufunc_Z(self.ions)))
        mui += np.sum([a / i.mu for i,a in self.abu if i not in self.ions])
        return 1 / mui

#class SimpleEos(object):
#    def __init__(self , net):
#        self.mu = net.mu() # network is a class, 
#
#    def __call__(self , T , rho):
#        mu = self.mu
#        p = ARAD / 3 * T**4  + RK * T * rho / mu # P(T,rho) = Rgas*T*rho/mu ,  P_rad = 1/3*a*T**4 
#        u = 3 / 2* RK *T /mu + ARAD*T**4/rho  #  a*T**4 is raidation energy per unit volume
#        pt = 4/3*ARAD*T**3 + RK*rho/mu
#        pd = RK*T/mu
#        ut = 1.5 * RK / mu + 4*ARAD*T**3/rho # no 3/2 but 1.5
#        ud = -4*ARAD*T**4/rho**2
#        return p,u,pt,pd,ut,ud    

#class SimpleKappa(object): # I guess my kappa can't resolve the two partial terms
#    def __init__(self, abu):
#        self.abu = abu
#    def __call__(self, T, rho, dT = True, drho=True):
#        k = 0.4 * self.abu.Ye()
#        result = [1/k]
#        if dT:
#            kt = 0.
#            result.append(kt)
#        if drho:
#            kd = 0.
#            result.append(kd)
#        if len(result) == 0:
#            return result[0]
#        return result

### note ###
# 2/3 optical depth >>> efective Temp. Radius
# HOW TO SOLVE , DON'T KNOW RHO , root to find zero, 
# g / (1.5 * kappa(T, rho)) == P(T, rho)
# ki0bd0 = (d(1/kappa)/(d rho))/(2 * kappa) , ki = 1/kappa

class Shot(object):
    def __init__(self, L=7e35, R=1e6, M=2.8e33, mdot=5e17, # Mdot must be in unit g/s
            abu = dict(he4=0.99, n14=0.009, fe56=0.001), 
            xms=1e13, xmsf=1.2): 
        abu = AbuSet(abu)
        net = NetWork(abu)
        kappa = partial(Kappa(), T4=False)  # leave all other varables open
        eos = partial(Eos(), T4=False) 
        T = qqrt(L / (4 * np.pi * R**2 * SB)) 
        g = GRAV*M/R**2
        d0 = 1
        # first guess value for density,>>> P - g / (...) = 0, 
        # call eos function to have rho
        while True:
            p0,u0,_,p0bd0,u0bt0,u0bd0 = eos(T , d0) # we need only 2, look up the SimpleEos
            ki0,_,ki0bd0 = kappa(T , d0)                 # we will use heat kappa , look at call, 3 results
            f = p0 - g / 1.5 * ki0 # 2/3 optical depth
            if np.abs(f) < 1e-12 * p0:
                break
            df = p0bd0 - g / 1.5 * ki0bd0 * 2 * ki0          # ki0bd0 has to be divided by 
            dd0 = f/df
            d0n = d0 - dd0  
            d0 = np.minimum(GOLDEN * d0, np.maximum(d0 / GOLDEN, d0n))  # 1.61 , d0 is the boundary density 
### 1st zone        

        p_surf = p0
        t_surf = T
        #dm = d0 * 4 * np.pi * (R**3 - (R-dr)**3) / 3  # solve directly , dm : change of mass
        t0 = T 
        r0 = R
        xm1 = 4 * np.pi * r0**2   * ki0 * 2 / 3 # surface mass
        xl0 = L
        z0 = M
        xm0 = xms
        g0 = GRAV*z0/r0**2
        p1 = p_surf + 0.5 * xm0 * g0 / (4 * np.pi * r0**2) 
        u1 = u0
        d1 = d0
#        u1bt1 = u0bt0
#        u1bd1 = u0bd0
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

            h0bt0 = l0 * 2 * ki0bt0 + acdr0 * 4 * t0 ** 3
            h0bd0 = l0 * (2 * ki0bd0 - 1 / d0 - dr0bd0 / dr0)

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
        s0  = net.epsdt(t0, d0, dt0, update=True)
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
        z0  = M - xm1 
#@&&!Y*@&^$*@&#(**&!(*#&! may have problem!!!!(*&#*@$*($^
        print(f'first zone , tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
        print(f'next zone mass={xm0*1.2:12.5e}')

# l0 is actually depends on current u0 which is computed in iteration
# 1st zone mass is modified by not affecting all the variables fro which the original zone mass is used

#        k = np.log10(1 - (M - xm_surf)/xms + xmsf * (M - xm_surf) / xms) / np.log10(xmsf) -1
        nmax = 2**11 
        tn  = np.ndarray(nmax+1)
        dn  = np.ndarray(nmax+1)
        xm  = np.ndarray(nmax+1)
        pn  = np.ndarray(nmax+1)

        xln = np.ndarray(nmax)
        sn  = np.ndarray(nmax)
        rn  = np.ndarray(nmax)
#        zn  = np.ndarray(num_run) 
#        gn  = np.ndarray(num_run) 

        tn[0]  = t_surf
        dn[0]  = d1
        xm[0]  = xm1
        pn[0]  = p1
        xln[0] = xl0
        sn[0]  = s0

        tn[1]  = t0
        dn[1]  = d0
        xm[1]  = xm0
        pn[1]  = p0
        rn[1]  = R

# starting from the second zone
        for j in range(nmax):
            if d0 > 5e11 or t0 > 5e9:
                break
            xm2 = xm1
            xm1 = xm0
            xm0 = xm1 * xmsf
            dt0 = xm0 / mdot
            r1  = r0
            r0  = rm
            z1  = z0  # mass for computing gravity
            z0  = z1 - xms
            g1  = g0
            g0  = GRAV * z0 / r0**2
    
            ki1 = ki0
            p2  = p1
            p1  = p0
            t1  = t0
            d2  = d1
            d1  = d0
            u1  = u0    
            xl1 = xl0
            s1  = s0
            p   = p1 +  xm0 * g0 / (4 * np.pi * r0**2) 
            dmx = 2 * mdot / (xm0 + xm1)
            dt0 = xm0 / mdot
            pdv1 = p2 * (d1 - d2) / d2**2
            while True:
                p0,u0,p0bt0,p0bd0,u0bt0,u0bd0 = eos(t0 , d0)  
                ki0,ki0bt0,ki0bd0 = kappa(t0 , d0) 
    
                pdv0 = p1 * (d0 - d1) / d1**2
                du   = u0 - u1
                sdot = du * dmx
                dL   = (sdot + (pdv1 - pdv0) / dt0) * d1  # mass from zone 1 to zone 2, so * d1
                xl0  = xl1 - dL

                rm = np.cbrt(r0**3 - 3 * xm0 / (4 * np.pi * d0))
                dr0 = r0 - rm
                ac = 4 * np.pi * r0**2 * ARAD * CLIGHT / 3
                acdr0 = ac * ki0 / (d0 * 0.5 * dr0)
                l0 = (t0**4 - t1**4) * acdr0
    
                f0 = p0 - p
                h0 = l0 - xl0
    
                b = np.array([f0,h0])
                b1 = np.array([p,xl0])
    
                if np.abs(f0/p) < 1e-12 and np.abs(h0/xl0) < 1e-12:
                    break
    
                dr0bd0 = - xm0 / (rm**2 * 4 * np.pi * d0**2)
                f0bt0 = p0bt0
                f0bd0 = p0bd0 
    
                h0bt0 = l0 * 2 * ki0bt0 + acdr0 * 4 * t0 ** 3
                h0bd0 = l0 * (2 * ki0bd0 - 1 / d0 - dr0bd0 / dr0)
    
                A = np.array([[f0bt0, f0bd0],[h0bt0, h0bd0]])
                c = np.linalg.solve(A,b)
                v = np.array([t0, d0])
                t0, d0 = v - c

            dt0 = xm0 / mdot
            s0  = net.epsdt(t0, d0, dt0, update=True)
            dL  = s0 * xm0

            tn[j+1]  = t0
            dn[j+1]  = d0
            xm[j+1]  = xm0
            pn[j+1]  = p0
            xln[j] = xl0
            sn[j]  = s0
            rn[j]  = r0

            print(f'zone {j}, tn={t0:12.5e} K, dn={d0:12.5e} g/cc, P={p0:12.5e} erg/cc, sn={s0:12.5e} erg/g/s, xln={xl0:12.5e} erg/s')
            print(f'current zone mass={xm0:12.5e}, next zone mass={xm0*xmsf:12.5e}')

        self.xm  = xm[:j+1]
        self.xln = xln[:j]
        self.rn  = rn[:j]

        self.y = np.cumsum((self.xm[1:] / (4 * np.pi * self.rn**2)))

#        self.y_m = np.zeros(jm + 2)
#        self.y_m[1:-1] = 0.5 * (self.y[:-2] + self.y[1:-1])
#        self.y_m[ 0] = self.y[0]
#        self.y_m[-1] = self.y[-2]

