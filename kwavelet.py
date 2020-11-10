import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt as wavelet


class Analysis(object):
    def __init__(self):
        pass

    def __call__(self, t, y, f, sigma, dt):
        t0 = t[0]
        mother = wavelet.Morlet(sigma)
        p = np.polyfit(t - t0, y, 1) # fit a 1-degree polynomial function
        y_notrend = y - np.polyval(p, t - t0)
        std = y_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        y_norm = y_notrend / std  # Normalized dataset
#        alpha, _, _ = wavelet.ar1(y) 
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y_norm, dt,wavelet=mother,freqs=f)
        power = (np.abs(wave)) ** 2
        Liu_power = power / scales[:, None]
        return power , scales, Liu_power, coi
        self.dt = dt

class Wave(object):
    def __init__(self, t=None, y=None, filename=None, dt=None):
        if t is not None and y is not None:
            pass
        else:
            if filename:
                self.lc = fits.open(filename)
                t = self.lc[1].data['TIME']
                y = self.lc[1].data['RATE']
            else:
                raise AttributeError(f'give me a light curve')
        if np.any(np.isnan(y)) == True:
            print('arrays contain nan data')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]
            print('nan data are clean')
        else:
            print('No nan data')
        if dt is None:
            dt = t[1]-t[0]
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]
        if np.any(res) == True:
            print('Gaps between data')
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
#            for i in range(0,(len(ag)-1)):
#                globals()['t%s' % i] = t[(ag[i]+1):(ag[i+1]+1)]
#                globals()['y%s' % i] = y[(ag[i]+1):(ag[i+1]+1)]
#            globals()['t%s' % str(i+1)] = t[ag[-1]:]
#            globals()['y%s' % str(i+1)] = y[ag[-1]:]
        else:
            print('No gaps between data')
        self.t = t
        self.y = y
        self.dt = dt
#        self.gnum = i+1
        self.ag = ag

#    def sig(arg):
#        y, alpha, self.analysis = arg 
#        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,significance_level=0.99,wavelet=mother)

    def test_lc(self):
        slices = [slice(a0+1, a1+1) for a0, a1 in zip(self.ag[:-1], self.ag[1:])]
        for s in slices:
            plt.plot(self.t[s],self.y[s])
        plt.show()

    def plot_lc(self,
                astart = None,
                aend = None,
                tstart = None,
                tend = None
               ):
        ii = slice(astart, aend)
        plt.plot(self.t[ii], self.y[ii])
        plt.show()
 

    def plot(
            self,
            astart = None,
            aend = None,
            f1 = 2e-3,
            f2 = 12e-3,
            nf = 200,
            sigma = 6,
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
        p, s, lp, coi = self.an(self.t[ii], self.y[ii], self.f, sigma, self.dt)
        self.coi=coi       

        ax[0].clear()
        ax[0].plot(self.t[ii], self.y[ii])
        ax[0].set_ylabel('Count/s')
        fig.subplots_adjust(hspace=0.05)
#        ax[0].set_xticks([])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
        if power == 'normal':
            ax[1].contourf(self.t[ii], self.f, p, cmap=plt.cm.viridis)
        else:
            ax[1].contourf(self.t[ii], self.f, lp, cmap=plt.cm.viridis)
#        ax[1].fill(np.concatenate([self.t[:1], self.t, self.t[-1:]]),
#                   np.concatenate([[f1], 1/self.coi, [f1]]), 'k', alpha=0.3, hatch='x')
        ax[1].set_ylim(f1,f2)
