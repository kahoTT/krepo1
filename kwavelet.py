import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt as wavelet
import stingray
from mc_sim import simLC

# Normalized light curves and fill spaces with zeors
# The normalization is applied to the whole lightcurve, even wavelet spectrum could be done seperately
# An example from 1636 is obsid('60032-05-02-00')

# This class is to fill the gap data with mean value
class fill(object):
    def __call__(self, t=None, y=None, dt=None):
        if dt is None:
            dt = t[1] - t[0]
        mean = sum(y) / len(y)
        std = y.std()
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            print('data cleaning: Gaps between data')
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
        else:
            print('data cleaning: No gaps between data')
        tc = np.array([])
        for i in ag[1:]:
            ta = np.arange(t[i] + dt, t[i+1], dt)
            tc = np.concatenate([tc, ta])
        yc = np.ones(len(tc)) * mean
        tc = np.concatenate([t, tc])
        yc = np.concatenate([y, yc])
        y_c = np.array([x for _,x in sorted(zip(tc, yc))])
        t_c = np.sort(tc)
        print('Data gaps filled with mean value')
        return t_c, y_c.T, mean, std

class sim(simLC):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        # see if there any large data gaps. If so, have to simulate them one by one
        res = [(sub2 - sub1 > 100) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            l_ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
            slices = np.concatenate(([slice(a0+1, a1+1) for a0, a1 in zip(l_ag[:-1], l_ag[1:])], [slice(l_ag[-1]+1, None)]), axis=0)
            for s in slices:
                _t = t[s]
                _y = y[s]
                super().__init__(_t, _y, dt, input_counts, norm)
                plt.plot(self.fre, self.spec_power)
                plt.yscale('log')
                plt.xscale('log')
#        else:

class Cleaning(object): 
    def __init__(self, telescope=None, t=None, y=None, f=None, dt=None, ag=None):
        p = np.polyfit(t, y, 3) # fit a 1-degree polynomial function
        y_notrend = y - np.polyval(p, t)
        std = y_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        y_norm = y_notrend / std  # Normalized dataset
        alpha, _, _ = wavelet.ar1(y) # Model red noise
        
# for future development; split the array according to the size of the dead time
#        res = [(sub2 - sub1 > (1 / f.min()) * 4) for sub1, sub2 in zip(t[ag][:-1], t[ag][1:])]
#        if np.any(res) == True:
#            l_ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
#            slices = np.concatenate(([slice(a0+1, a1+1) for a0, a1 in zip(l_ag[:-1], l_ag[1:])], [slice(l_ag[-1]+1, None)]), axis=0)
#        else:
#            slices = 'null'
#        for s in slices:
            
        tc = np.array([])
        for i in ag[1:]:
            ta = np.arange(t[i] + dt, t[i+1], dt)
            tc = np.concatenate([tc, ta])
        yc = np.zeros(len(tc))
        tc = np.concatenate([t, tc])
        yc = np.concatenate([y_norm, yc])
        y_c = [x for _,x in sorted(zip(tc, yc))]
        t_c = np.sort(tc)
        self.tc = t_c       
        self.yc = y_c

class Analysis(object):
    def __call__(self, t, y, f, sigma, dt, ag):
        mother = wavelet.Morlet(sigma)
        slices = 'n'
        powera = Liu_powera = None
        for s in slices:
            if s == 'n':
                wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, dt, wavelet=mother,freqs=f)
            else:
                pass
#                wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y[s], dt, wavelet=mother,freqs=f)
            power = (np.abs(wave)) ** 2
            Liu_power = power / scales[:, None]
            if powera is None:
                powera = power
            else:
                powera = np.hstack((powera, power))
            if Liu_powera is None:
                Liu_powera = Liu_power
            else:
                Liu_powera = np.hstack((Liu_powera, Liu_power))
        return  powera, scales, Liu_powera, coi

class read_lc(object):
    def __init__(self, t=None, y=None, filename=None, dt=None, obsid=None, kepler=None):
#read lc
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
            import minbar
            b = minbar.Bursts()
            o = minbar.Observations()
            b.obsid(obsid)
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
        if len(b.get('bnum')) == 0:
            print('data cleaning:no bursts on this observation')
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bursttime = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bst = bursttime - 5
            bet = bst + obs.bursts['dur'] * 3 # two time of the duration
            barray = list()
            a1 = None
            for i in range(len(b.get('bnum'))):
                a = np.where(t == min(t, key = lambda x:abs(x - bst[i])))[0][0]
                _a = np.where(t == min(t, key = lambda x:abs(x - bet[i])))[0][0]
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
            self.tnb = tnb
            self.ynb = ynb
        self.t = t
        self.y = y
        if o['instr'][0] == 'XPj':
            self.dt = 0.125
        else:
            self.dt = t[1] - t[0]
        self.bursttime = bursttime

# Plot without burst
    def plot_nob(self):
        for s in range(len(self.tnb)):
            plt.plot(self.tnb[s],self.ynb[s])
        plt.ylabel('Count/s')
        plt.xlabel('Time (s)')
        plt.show()

 
# plot only burst
    def plot_b(self):
        plt.plot(self.tb, self.yb, 'rx')
        plt.ylabel('Count/s')
        plt.xlabel('Time (s)')
        plt.show()
 
    def plot_c(
               self,
               f1 = 2e-3, 
               f2 = 13e-3,
               nf = 200,
               ):
        f = np.linspace(f1, f2, nf)
        self.f = f
        c = Cleaning(None, self.tnb, self.ynb, self.f, self.dt, self.ag)
        self.tc = c.tc
        self.yc = c.yc
        plt.plot(c.tc, c.yc)
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

    def plot(
            self,
            astart = None,
            aend = None,
            sigma = 15,
            power = None, 
            f1 = 2e-3, 
            f2 = 13e-3,
            nf = 200,
#            tstart = None,
#            tend = None
            ):
    
        ii = slice(astart, aend)
        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax

        f = np.linspace(f1, f2, nf)
        self.f = f
        c = Cleaning(None, self.tnb[ii], self.ynb[ii], self.f, self.dt, self.ag)
        An = Analysis()
        p, s, lp, coi = An(c.tc, c.yc, self.f, sigma, self.dt, self.ag)
        self.coi=coi       

#        ax[0].clear()
        ax[0].plot(self.t[ii], self.y[ii])
        ax[0].set_ylabel('Count/s')
        fig.subplots_adjust(hspace=0.05)
#        ax[0].set_xticks([])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
#        if power == 'normal':
#            ax[1].contourf(self.t[ii], self.f, p, cmap=plt.cm.viridis)
#        else:
#            ax[1].contourf(self.t[ii], self.f, lp, cmap=plt.cm.viridis)
        if power == 'normal':
            ax[1].contourf(c.tc, f, p, cmap=plt.cm.viridis)
        else:
            ax[1].contourf(c.tc, f, lp, cmap=plt.cm.viridis)
#        ax[1].fill(np.concatenate([self.t[:1], self.t, self.t[-1:]]),
#                   np.concatenate([[f1], 1/self.coi, [f1]]), 'k', alpha=0.3, hatch='x')
        ax[1].set_ylim(f1,f2)
