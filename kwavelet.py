from astropy.time.utils import twoval_to_longdouble
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt 
import stingray
from mc_sim import simLC
import time
import minbar
b = minbar.Bursts()
o = minbar.Observations()

# Normalized light curves and fill spaces with zeors
# The normalization is applied to the whole lightcurve, even wavelet spectrum could be done seperately
# An examples from 1636 is obsid('60032-05-02-00') and 1608 obsid('10072-05-01-00') and EXO 0748 obsid('90039-01-03-05')
# 1323 obsid('96405-01-02-01')

# This class is to fill the gap data with mean value, will change to fill with the fitted polynomail vales
class fill(object):
    def __init__(self, t=None, y=None, dt=None, plot=False):
        if dt is None:
            dt = t[1] - t[0]
        p = np.polyfit(t, y, 3)
        if plot == True:
            plt.plot(t,y)
            plt.plot(t, np.polyval(p, t))
        dat_notrend = y - np.polyval(p, t)
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
            tc = np.array([])
            for i in ag[1:]:
                ta = np.arange(t[i] + dt, t[i+1], dt)
                tc = np.concatenate([tc, ta])
            yc = np.zeros(len(tc))
            tc = np.concatenate([t, tc])
            yc = np.concatenate([dat_notrend, yc])
            y_c = np.array([x for _,x in sorted(zip(tc, yc))])
            t_c = np.sort(tc)
            self.tc = t_c
            self.yc = y_c
        else:
            self.tc = t
            self.yc = dat_notrend

class sim(simLC): # Main purposeof this class is to divide lightcurve into different sections and being put to another simulation module
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        # see if there any large data gaps. If so, have to simulate them one by one
        if dt is None:
            dt = t[1] - t[0]
        res = [(sub2 - sub1 > 400) for sub1, sub2 in zip(t[:-1], t[1:])] # the value should be based on something? For example, < 100, the data point
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
    def __init__(self, t=None, y=None, burst=False, filename=None, dt=None, obsid=None, kepler=None, f1=4e-3, f2=15e-3, nf=200, test=500):
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
            b.clear()
            b.obsid(obsid)
            ifb = b.get('obsid')
            o.clear()
            o.obsid(obsid)
            name = o.get('name')[0]
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

        if any(y < 0):
            bglc = fits.open(obs.get_path()+'/bkg_0.125s.lc')
            bg = bglc[1].data['RATE']
            y = y + bg[_int]
            self.bg = bg

        if np.any(np.isnan(y)) == True:
            print('data cleaning: arrays contain nan data after background correction')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]

# dealing with bursts
        if len(np.where(ifb == obsid)[0]) == 0:
            print('data cleaning: No bursts in this observation')
            tnb = t
            ynb = y
            ltnb = 1
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bursttime = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bst = bursttime - 5
            bet = bst + obs.bursts['dur'] * 4 # scaling the time of the duration
            barray = list()
            nbarray = list()
            a1 = None
            for i in range(len(b.get('bnum'))):
                a = list(abs(t-bst[i])).index(min(abs(t - bst[i])))
                _a = list(abs(t-bet[i])).index(min(abs(t - bet[i])))
                barray.extend(np.r_[a:_a])
                if i == 0: 
                    if a != 0: # for the case of starting in the middle of a burst
                        nbarray.extend(np.r_[a1:a])
                        tnb = t[:a],
                        ynb = y[:a],
                    else:
                        tnb = ()
                        ynb = ()
                else:
                    nbarray.extend(np.r_[a1:a])
                    tnb += t[a1:a],
                    ynb += y[a1:a],
                a1 = _a + 1
            if _a == len(t) - 1: # for the case of ending in the middle of a burst
                pass
            else:
                nbarray.extend(np.r_[a1:len(t)])
                tnb += t[a1:],
                ynb += y[a1:],
            self.tb = t[barray]
            self.yb = y[barray]
            if burst == False: # lightcurve not divided the number of bursts
                tnb = t[nbarray]
                ynb = y[nbarray]
                ltnb = 1
            else:
                ltnb = len(tnb)
            self.bursttime = bursttime
        self.obsid = obsid
        self.t = t
        self.y = y
        self.tnb = tnb
        self.ynb = ynb
        self.burst = burst
        self.ltnb = ltnb
        self.name = name
        if o['instr'][0] == 'XPj':
            dt = self.dt = 0.125
        else:
            dt = self.dt = t[1] - t[0]
        f = np.linspace(f1, f2, nf)
        self.f = f

        if test == 0:
            pass
        else:
            p = ()
            for i2 in range(ltnb): # tnb is a tuple
                if ltnb == 1:
                    i2 = slice(None)
                plist = list()
                start_time = time.time()
                for i3 in range(test):
                    testtime = time.time() - start_time
                    s = sim(t=tnb[i2], y=ynb[i2], dt=dt) # Simulation class
                    _f = fill(s.lct, s.lcy, dt=dt) # fill class
                    ystd = s.lcy.std()
                    ws = wavelet_spec(y=(_f.yc / ystd), f=f, sigma=10, dt=dt, powera=None)
                    # Normalisation of power. Ideally use leahy power
#                    norm_pow = 2*ws.power*len(_f.yc)/sum(_f.yc)*dt
                    if len(f) == 1: 
                        norm_pow = ws.power[0] # dealing with extra [] for 1D f array  
                        _int = np.where(f < 1/ws.coi)
                        norm_pow[_int] = np.nan
                    else:
                        norm_pow = ws.power  
                        for i4 in range(len(norm_pow[0])):
                            _int = np.where(f < 1/ws.coi[i4])
                            norm_pow[:,i4][_int] = np.nan
#                    plist.append(norm_pow)
        #            plt.contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis)
        #           plt.colorbar()
        #            plt.fill(np.concatenate([_f.tc[:1], _f.tc, _f.tc[-1:]]),
        #                     np.concatenate([[f1], 1/ws.coi, [f1]]), 'k', alpha=0.3, hatch='x')
        #            plt.ylim(f1, f2)
        #            plt.plot(_f.tc, _f.yc, 'b')
                p += plist,
                coiarray = ws.coi,
            if ltnb == 1:
                self.p = p[0]
            else:   
                self.p = p
            self.coi = coiarray
#            self.finish_time = time.time() - start_time
#            print(f'Finish time = {self.finish_time}')



# Plot without burst
    def plot_nob(self): # put self arguments
        if self.ltnb > 1 and self.burst == True:
            for s in range(len(self.tnb)):
                plt.plot(self.tnb[s],self.ynb[s])
        else:
            plt.plot(self.tnb, self.ynb)
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

        vmin = 0
        vmax = 0
        for i in range(self.ltnb): 
            if self.ltnb == 1:
                i = slice(None)
            _f = fill(t[i], y[i], dt=dt)
            ystd = y[i].std()
            ws = wavelet_spec(y=(_f.yc / ystd), f=f, sigma=10, dt=dt, powera=None)
#            norm_pow = 2*ws.power*len(_f.yc)/sum(_f.yc)*dt
            norm_pow = ws.power
            if vmax >= np.max(norm_pow):
                pass
            else:
                vmax = np.max(norm_pow)
### Significancy
#            sig = np.ones(norm_pow.shape) * 17.649476428307167
#            sig = norm_pow / sig
            for i1 in range(len(ws.power[0])):
                _int = np.where(f < 1/ws.coi[i1])
                norm_pow[:,i1][_int] = np.nan
            ax[0].plot(_f.tc, _f.yc)
#            cm = ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis, vmin=vmin, vmax=(int(vmax)+1))
#            ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis)
#            ax[1].contour(_f.tc, f, sig, [-99,1], colors='k')

        ax[0].set_ylabel('Count/s')
        ax[1].set_ylabel('Frequency Hz')
        fig.subplots_adjust(hspace=0.05)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
        fig.suptitle(f'{self.name} obsid: {self.obsid}')
#        fig.colorbar(cm, ax=ax)
### norm_pow may need to modity, as this only has the elements for the last loop
        self.rpow = norm_pow

    def plot_maxp(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

#        bins = int(len(self.maxp) * 0.02)  
        ax.axes.hist(self.maxp, density=True, label='simulation')        
        fig.suptitle(f'{self.name} obsid: {self.obsid}')


