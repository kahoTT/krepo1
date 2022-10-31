from re import L
from astropy.time.utils import twoval_to_longdouble
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt 
import mc_sim
import time
import minbar
import stingray
minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'

# Normalized light curves and fill spaces with zeors
# The normalization is applied to the whole lightcurve, even wavelet spectrum could be done seperately
# Examples from 1636 is obsid('60032-05-02-00') and 1608 obsid('10072-05-01-00') and EXO 0748 obsid('90039-01-03-05')
# 1323 obsid('96405-01-02-01')

# This class is to fill the gap data with mean value, will change to fill with the fitted polynomail vales
class fill(object):
    def __init__(self, t=None, y=None, dt=None, plot=False):
        if dt is None:
            dt = t[1] - t[0]
        p = np.polyfit(t, y, 2) # value is the d.o.f
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

class sim(mc_sim.simLC): # Main purpose of this class is to divide lightcurve into different sections and being put to another simulation module
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        # see if there any large data gaps. If so, have to simulate them one by one
        if dt is None:
            dt = t[1] - t[0]
        super().__init__(t, y, dt, input_counts, norm)
        self.lct = t
        self.lcy = self.counts
            
class wavelet_spec(object):
    def __init__(self, y, f, sigma, dt, powera=None):
        mother = pycwt.Morlet(sigma)
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.wavelet.cwt(y, dt, wavelet=mother, freqs=f)
        power = (np.abs(wave)) ** 2
        fft_power = np.abs(fft) ** 2
        Liu_power = power / scales[:, None]
        if powera is None:
            _pow = power
        elif powera == 'Liu':
            _pow = Liu_power
        self.power = _pow
        self.fftfreqs = fftfreqs
        self.fft_power = fft_power
        self.coi = coi

def Slice(t, gap=400):
    res = [(sub2 - sub1 > gap) for sub1, sub2 in zip(t[:-1], t[1:])] # the value should be based on something? 
    if np.any(res) == True:
        l_ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
        slices = [slice(a0+1, a1+1) for a0, a1 in zip(l_ag[:-1], l_ag[1:])]
        slices.append(slice(l_ag[-1]+1, None))
    else:
        slices = [slice(None)] # have problems for different observations
    return slices 


class analysis(object):
    def __init__(self, t=None, y=None, filename=None, dt=None, obsid=None, name=None, kepler=None, f=None, f1=4e-3, f2=15e-3, nf=200, sims=1, sigma=6):
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
            b = minbar.Bursts()
            o = minbar.Observations()
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
            if o['instr'][0] == 'XPj':
                dt = 0.125
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
        if not obsid:
            tnb = t
            ynb = y
            name = None
        elif (len(np.where(ifb == obsid)[0]) == 0):
            print('data cleaning: No bursts in this observation')
            tnb = t
            ynb = y
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bursttime = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bst = bursttime - 5
            bet = bst + obs.bursts['dur'] * 4 # scaling the time of the duration
            barray = []
            nbarray = []
            a1 = None
            for i in range(len(b.get('bnum'))): # extract burst data and non-burst data
                a = list(abs(t-bst[i])).index(min(abs(t - bst[i])))
                _a = list(abs(t-bet[i])).index(min(abs(t - bet[i])))
                barray.extend(np.r_[a:_a])
                if i == 0: 
                    if a != 0: 
                        nbarray.extend(np.r_[a1:a])
                    else: # for the case of starting in the middle of a burst
                        pass
                else:
                    nbarray.extend(np.r_[a1:a])
                a1 = _a + 1
            if _a == len(t) - 1: # for the case of ending in the middle of a burst
                pass
            else:
                nbarray.extend(np.r_[a1:len(t)])
            self.tb = t[barray]
            self.yb = y[barray]
            tnb = t[nbarray]
            ynb = y[nbarray]
            self.bursttime = bursttime

# divide the light curve  

        tnb_s = []
        ynb_s = []
        for j in Slice(tnb):
            tnb_s.append(tnb[j])
            ynb_s.append(ynb[j])
        self.tnb_s = tnb_s
        self.ynb_s = ynb_s
        ltnb = len(tnb_s)
        self.ltnb = ltnb

        self.obsid = obsid
        self.t = t
        self.y = y
        self.tnb = tnb
        self.ynb = ynb
        self.name = name
        if not dt:
            dt = t[1] - t[0]
        self.dt = dt
        if f is None:
            f = np.linspace(f1, f2, nf)
        else:
            f = np.array([f])
        self.f = f

        # Simulation, the number of sims means the number of simulations
        tc = []
        _powall = []
        lsigma3 = []
        for i2 in range(ltnb): # tnb is a list 
            realf = fill(tnb_s[i2], ynb_s[i2], dt=dt)
            tc.append(realf.tc)
            rystd = realf.yc.std()
            # power spectrum for real data to for normalising the synthetic ones
            rws = wavelet_spec(y=(realf.yc/rystd), f=f, sigma=sigma, dt=dt, powera='Liu')
            realresult, realmodel = mc_sim.PowFit(f=rws.fftfreqs, y=rws.fft_power, f2=f)
            rpower = rws.power * realresult.x[2] / realmodel[:, np.newaxis]  # dealing with extra [] for 1D f array  
            for i5 in range(len(rpower[0])):
                _int = np.where(f < 1/rws.coi[i5])
                rpower[:,i5][_int] = np.nan
            if sims == 0:
                _powall.append(rpower)
            else:
                for i3 in range(sims):
#                    testtime = time.time() - start_time
                    s = sim(t=tnb_s[i2], y=ynb_s[i2], dt=dt) # Simulation class
                    _f = fill(s.lct, s.lcy, dt=dt) # fill class
                    ystd = _f.yc.std()
                    ws = wavelet_spec(y=(_f.yc / ystd), f=f, sigma=sigma, dt=dt, powera=None)

                    # Case when using single frequency
                    if len(f) == 1: 
                        norm_pow = ws.power[0] * realresult.x[2] / realmodel # dealing with extra [] for 1D f array  
                        _int = np.where(f > 1/ws.coi)
                        synp = norm_pow[_int]
                    else:
                        norm_pow = ws.power * realresult.x[2] / realmodel[:, np.newaxis]  
                        for i4 in range(len(norm_pow[0])):
                            _int = np.where(f < 1/ws.coi[i4])
                            norm_pow[:,i4][_int] = np.nan
                    if i3 == 0:
                        synp = norm_pow
                    else:
                        synp = np.concatenate((synp, norm_pow), axis=1)
                synpall = synp.reshape(1, synp.size)[0]
                _int2 = np.isnan(synpall)
                synpall = np.sort(synpall[~_int2])
                sigma3 = synpall[int(len(synpall) * 0.9973)]
                _pow = rpower / sigma3
                _powall.append(_pow)
                lsigma3.append(sigma3)
        #           plt.colorbar()
        #            plt.fill(np.concatenate([_f.tc[:1], _f.tc, _f.tc[-1:]]),
        #                     np.concatenate([[f1], 1/ws.coi, [f1]]), 'k', alpha=0.3, hatch='x')
        #            plt.ylim(f1, f2)
        #            plt.plot(_f.tc, _f.yc, 'b')
                # coiarray = ws.coi,
            # self.coi = coiarray
            self.rws = rws
            self.p = _powall
            self.sigma = lsigma3
            self.tc = tc
        self.finish_time = time.time() - start_time
        print(f'Finish time = {self.finish_time}')

# Plot without burst
    def plot_nob(self): # put self arguments
        plt.plot(self.tnb, self.ynb)
        if self.ltnb > 1:
            for s in range(self.ltnb):
                plt.plot(self.tnb_s[s],self.ynb_s[s], alpha=0.5)
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

    def plot_wspec(
            self,
            astart = None,
            aend = None,
            tstart = None,
            tend = None
            ):
    
        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax
        f = self.f 
        t = self.tnb_s
        y = self.ynb_s
        tnb = self.tnb
        ynb = self.ynb
        dt = self.dt
        ta = self.t
        ya = self.y
        p = self.p
        tc = self.tc
        sigma = self.sigma

        ax[0].plot(tnb, ynb)
        for i in range(self.ltnb): 
            ax[1].contourf(tc[i], f, p[i], cmap=plt.cm.viridis)
            ax[1].contour(tc[i], f, p[i], 1, colors='k')


        # vmin = 0
        # vmax = 0
        # if vmax >= np.max(norm_pow):
        #     pass
        # else:
        #     vmax = np.max(norm_pow)
        # cm = ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis, vmin=vmin, vmax=(int(vmax)+1))
        # ax[1].contourf(tc, f, p, cmap=plt.cm.viridis)

        ax[0].set_ylabel('Count/s')
        ax[1].set_ylabel('Frequency Hz')
        fig.subplots_adjust(hspace=0.05)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
        fig.suptitle(f'{self.name} obsid: {self.obsid}')
#        fig.colorbar(cm, ax=ax)
### norm_pow may need to modity, as this only has the elements for the last loop

# plot fft spectrum
    def plot_spec(self, sigma=10):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        f = self.f 
        t = self.tnb
        y = self.ynb
        dt = self.dt

        for i in range(self.ltnb): 
            if self.ltnb == 1:
                i = slice(None)
            _f = fill(t[i], y[i], dt=dt)
            ystd = y[i].std()
            ws = wavelet_spec(y=(_f.yc / ystd**2), f=f, sigma=10, dt=dt, powera='Liu')

        ax.plot(ws.fftfreqs, ws.fft_power)
        plt.yscale('log')
        plt.xscale('log')