from astropy.time.utils import twoval_to_longdouble
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt 
import stingray
from mc_sim import simLC
import time
import sys
import minbar
b = minbar.Bursts()
o = minbar.Observations()

# Normalized light curves and fill spaces with zeors
# The normalization is applied to the whole lightcurve, even wavelet spectrum could be done seperately
# An examples from 1636 is obsid('60032-05-02-00') and 1606 obsid('10072-05-01-00') and EXO 0748 obsid('90039-01-03-05')

# This class is to fill the gap data with mean value
class fill(object):
    def __init__(self, t=None, y=None, dt=None):
        if dt is None:
            dt = t[1] - t[0]
        mean = sum(y) / len(y)
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
            tc = np.array([])
            for i in ag[1:]:
                ta = np.arange(t[i] + dt, t[i+1], dt)
                tc = np.concatenate([tc, ta])
            yc = np.ones(len(tc)) * mean
            tc = np.concatenate([t, tc])
            yc = np.concatenate([y, yc])
            y_c = np.array([x for _,x in sorted(zip(tc, yc))])
            t_c = np.sort(tc)
            self.tc = t_c
            self.yc = y_c
        else:
            self.tc = t
            self.yc = y


class sim(simLC):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        # see if there any large data gaps. If so, have to simulate them one by one
        if dt is None:
            dt = t[1] - t[0]
        res = [(sub2 - sub1 > 100) for sub1, sub2 in zip(t[:-1], t[1:])] # the value should be based on something? For example, < 100, the data point
        if np.any(res) == True:
            l_ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
            slices = np.concatenate(([slice(a0+1, a1+1) for a0, a1 in zip(l_ag[:-1], l_ag[1:])], [slice(l_ag[-1]+1, None)]), axis=0)
            self.slices = slices
        else:
            slices = ((),) # have problems for different observations
        lct = np.array([])
        lcy = np.array([])
        for s in slices:
            _t = t[s]
            _y = y[s]
            super().__init__(_t, _y, dt, input_counts, norm)
            lct = np.concatenate((lct, _t), axis=0)
            lcy = np.concatenate((lcy, self.counts), axis=0)
#            self.plot_spec() plot all spectra
        self.lct = lct
        self.lcy = lcy
            
class wavelet_spec(object):
    def __init__(self, y, f, sigma, dt, powera):
        mother = pycwt.Morlet(sigma)
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.wavelet.cwt(y, dt, wavelet=mother, freqs=f)
        power = (np.abs(wave)) ** 2
        Liu_power = power / scales[:, None]
        if powera is None:
            _pow = power
        elif powera == 'Liu':
            _pow = Liu_power
        self.power = _pow
        self.coi = coi


class analysis(object):
    def __init__(self, t=None, y=None, filename=None, dt=None, obsid=None, kepler=None, f1=4e-3, f2=15e-3, nf=200, test=500):
#read lc
        start_time = time.time()
        if t is not None and y is not None:
            pass
        elif filename:
            if kepler == True:
                import lcdata
                lc = lcdata.load(filename)
                t = lc.time
                y = lc.xlum
            else:
                self.lc = fits.open(filename)
                t1 = self.lc[1].data['TIME']
                t = t1 - t1[0]
                y = self.lc[1].data['RATE']
        elif obsid:
            b.clear()
            ifb = b.obsid(obsid)
            o.clear()
            o.obsid(obsid)
            obs = minbar.Observation(o[o['entry']]) 
            _path = obs.instr.lightcurve(obsid)
            self.lc = fits.open(obs.get_path()+'/'+_path)
            t1 = self.lc[1].data['TIME']
            y = self.lc[1].data['RATE']
            t = t1 - t1[0]
        else:
            raise AttributeError(f'give me a light curve')

        if np.any(np.isnan(y)) == True:
            print('data cleaning: arrays contain nan data')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]
            print('data cleaning: nan data are clean')
        else:
            print('data cleaning: No nan data')

# dealing with bursts
        if len(np.where(ifb == obsid)) == 0:
            print('data cleaning: No bursts in this observation')
            tnb = t
            ynb = y
            ltnb = 1
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bursttime = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bst = bursttime - 5
            bet = bst + obs.bursts['dur'] * 3 # scaling the time of the duration
            barray = list()
            a1 = None
            for i in range(len(b.get('bnum'))):
                a = list(abs(t-bst[i])).index(min(abs(t - bst[i])))
                _a = list(abs(t-bet[i])).index(min(abs(t - bet[i])))
#                a = np.where(t == min(t, key = lambda x:abs(x - bst[i])))[0][0]
#                _a = np.where(t == min(t, key = lambda x:abs(x - bet[i])))[0][0]
                barray.extend(np.r_[a:_a])
                if i == 0: 
                    if a != 0: # for the case of starting in the middle of a burst
                        tnb = t[:a],
                        ynb = y[:a],
                    else:
                        tnb = ()
                        ynb = ()
                else:
                    tnb += t[a1:a],
                    ynb += y[a1:a],
                a1 = _a + 1
            if _a == len(t) - 1: # for the case of ending in the middle of a burst
                pass
            else:
                tnb += t[a1:],
                ynb += y[a1:],
            self.tb = t[barray]
            self.yb = y[barray]
            self.bursttime = bursttime
            ltnb = len(tnb)
        self.tnb = tnb
        self.ynb = ynb
        self.t = t
        self.y = y
        if o['instr'][0] == 'XPj':
            dt = self.dt = 0.125
        else:
            dt = self.dt = t[1] - t[0]

        f = np.linspace(f1, f2, nf)
        self.f = f

        maxp = ()
        for i2 in range(ltnb): # tnb is a tuple
            if ltnb == 1:
                i2 = slice(None)
            maxplist = list()
            print(i2)
            for i3 in range(test):
                testtime = time.time() - start_time
#                print(f'{testtime}')
                s = sim(t=tnb[i2], y=ynb[i2], dt=dt)
                testtime2 = time.time() - start_time
#                print(f'{testtime2}')
                _f = fill(s.lct, s.lcy, dt=dt)
#                plt.plot(_f.tc, _f.yc, alpha=0.6)
#                plt.show()
                ws = wavelet_spec(y=(_f.yc-_f.yc.mean()), f=f, sigma=10, dt=dt, powera=None)
                norm_pow = 2*ws.power*len(_f.yc)/sum(_f.yc)*dt
                for i4 in range(len(ws.power[0])):
                    _int = np.where(f < 1/ws.coi[i4])
                    norm_pow[:,i4][_int] = np.nan
                maxplist.append(np.nanmax(norm_pow))
    #            plt.contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis)
    #           plt.colorbar()
    #            plt.fill(np.concatenate([_f.tc[:1], _f.tc, _f.tc[-1:]]),
    #                     np.concatenate([[f1], 1/ws.coi, [f1]]), 'k', alpha=0.3, hatch='x')
    #            plt.ylim(f1, f2)
    #            plt.plot(_f.tc, _f.yc, 'b')
            maxp += maxplist,
            coiarray = ws.coi,
        self.maxp = maxp
        self.coi = coiarray
        self.finish_time = time.time() - start_time
        print(f'Finish time = {self.finish_time}')

#    def plot_hist(self):


# Plot without burst
    def plot_nob(self):
        for s in range(len(self.tnb)):
            plt.plot(self.tnb[s],self.ynb[s])
        plt.ylabel('Counts/s')
        plt.xlabel('Time (s)')
        plt.show()

 
# plot only burst
    def plot_b(self):
        plt.plot(self.tb, self.yb, 'rx')
        plt.ylabel('Count/s')
        plt.xlabel('Time (s)')
        plt.show()
 
    def plot_lc(self,
                astart = None,
                aend = None,
                tstart = None,
                tend = None
               ):
        ii = slice(astart, aend)
        plt.plot(self.t[ii], self.y[ii],'.')
        plt.show()

    def plot_spec(
            self,
            astart = None,
            aend = None,
            sigma = 10,
            power = None, 
#            tstart = None,
#            tend = None
            ):
    
        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax
        f = self.f 
        t = self.tnb
        y = self.ynb
        dt = self.dt
        ta = self.t
        ya = self.y

        for i in range(len(t)):
            _f = fill(t[i], y[i], dt=dt)
            ws = wavelet_spec(y=(_f.yc-_f.yc.mean()), f=f, sigma=10, dt=dt, powera=None)
            norm_pow = 2*ws.power*len(_f.yc)/sum(_f.yc)*dt
            sig = np.ones(norm_pow.shape) * 17.649476428307167
            sig = norm_pow / sig
            for i1 in range(len(ws.power[0])):
                _int = np.where(f < 1/ws.coi[i1])
                norm_pow[:,i1][_int] = np.nan
            ax[0].plot(_f.tc, _f.yc)
            cm = ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis)
#            ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis)
#            ax[1].contour(_f.tc, f, sig, [-99,1], colors='k')

        ax[0].set_ylabel('Count/s')
        ax[1].set_ylabel('Frequency Hz')
        fig.subplots_adjust(hspace=0.05)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
#        fig.colorbar(cm)
        self.rpow = norm_pow

    def plot_maxp(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

#        bins = int(len(self.maxp) * 0.02)  
        ax.axes.hist(self.maxp, density=True, label='simulation')        


