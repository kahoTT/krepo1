import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt as wavelet


class Analysis(object):
    def __init__(self):
        pass

    def __call__(self, t, y, f, sigma, dt, ag):
        mother = wavelet.Morlet(sigma)
        p = np.polyfit(t, y, 3) # fit a 1-degree polynomial function
        y_notrend = y - np.polyval(p, t)
        std = y_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        y_norm = y_notrend / std  # Normalized dataset
#        alpha, _, _ = wavelet.ar1(y) 
        slices = np.concatenate(([slice(a0+1, a1+1) for a0, a1 in zip(ag[:-1], ag[1:])], [slice(ag[-1]+1, None)]), axis=0)
        powera = Liu_powera = None
        for s in slices:
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y_norm[s], dt,wavelet=mother,freqs=f)
            power = (np.abs(wave)) ** 2
            Liu_power = power / scales[:, None]
            print(power.shape)
            if powera is None:
                powera = power
            else:
                powera = np.hstack((powera, power))
            if Liu_powera is None:
                Liu_powera = Liu_power
            else:
                Liu_powera = np.hstack((Liu_powera, Liu_power))
        return  powera, scales, Liu_powera, coi

class Wave(object):
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
# reading lc
#        break ###

        if np.any(np.isnan(y)) == True:
            print('arrays contain nan data')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]
            print('nan data are clean')
        else:
            print('No nan data')

# dealing with bursts
        if len(b.get('bnum')) == 0:
            print('no bursts on this observation')
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bst = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bet = bst + obs.bursts['dur']
#            bet = bst + 100
            barray = list()
            for i in range(len(b.get('bnum'))):
                a = np.where(t == min(t, key = lambda x:abs(x - bst[i])))[0]
                b = np.where(t == min(t, key = lambda x:abs(x - bet[i])))[0]
                barray.extend(np.r_[a:b])
            tb = t[barray]
            yb = y[barray]
            t = np.delete(t, barray)
            y = np.delete(y, barray)
# dealing with bursts

        if dt is None:
            dt = t[1]-t[0]
        else:
            pass
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            print('Gaps between data')
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
        else:
            print('No gaps between data')
        self.t = t
        self.y = y
        self.dt = dt
#        self.gnum = i+1
        self.ag = ag
        self.tb = tb
        self.yb = yb

#    def sig(arg):
#        y, alpha, self.analysis = arg 
#        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,significance_level=0.99,wavelet=mother)

    def plot_nob(self):
        slices = np.concatenate(([slice(a0+1, a1+1) for a0, a1 in zip(self.ag[:-1], self.ag[1:])], [slice(self.ag[-1]+1, None)]), axis=0)
        for s in slices:
            plt.plot(self.t[s],self.y[s])
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
 
    def plot_b(self):
        plt.plot(self.tb, self.yb, 'rx')
        plt.ylabel('Count/s')
        plt.xlabel('Time (s)')
        plt.show()
 

    def plot(
            self,
            astart = None,
            aend = None,
            f1 = 2e-3,
            f2 = 13e-3,
            nf = 200,
            sigma = 15,
            power = None 
#            tstart = None,
#            tend = None
            ):
    
        ii = slice(astart, aend)
        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax

        f = np.linspace(f1, f2, nf)
        self.f = f

        self.an = Analysis()
        p, s, lp, coi = self.an(self.t[ii], self.y[ii], self.f, sigma, self.dt, self.ag)
        self.coi=coi       

        ax[0].clear()
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
            ax[1].contourf(self.t[ii], self.f, p, cmap=plt.cm.viridis)
        else:
            ax[1].contourf(self.t[ii], self.f, lp, cmap=plt.cm.viridis)
#        ax[1].fill(np.concatenate([self.t[:1], self.t, self.t[-1:]]),
#                   np.concatenate([[f1], 1/self.coi, [f1]]), 'k', alpha=0.3, hatch='x')
        ax[1].set_ylim(f1,f2)
