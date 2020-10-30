import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt as wavelet
import matplotlib.pylab

class Analysis(object):
    def __init__(self):
        pass

    def __call__(self, t, y, f, sigma, dt):
        if dt is None:
            dt = t[1]-t[0]
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
        return power , scales, Liu_power

class Wave(object):
    def __init__(self, t=None, y=None, filename=None):
        if t is not None and y is not None:
            pass
        else:
            if filename:
                self.lc = fits.open(filename)
                t = self.lc[1].data['TIME']
                y = self.lc[1].data['RATE']
            else:
                raise AttributeError(f'give me a light curve')
        self.t = t
        self.y = y


#    def sig(arg):
#        y, alpha, self.analysis = arg 
#        signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,significance_level=0.99,wavelet=mother)
     

    def plot(
            self,
            astart = None,
            aend = None,
            dt = None,
            f1 = 2e-3,
            f2 = 12e-2,
            nf = 200,
            sigma = 6,
            power = None 
#            tstart = None,
#            tend = None
            ):
    
        ii = slice(astart, aend)
        fig = plt.gcf()
        ax = fig.subplots(2) 
        self.fig = fig
        self.ax = ax

        f = np.linspace(f1, f2, nf)
        self.f = f

        self.an = Analysis()
        p, s, lp = self.an(self.t[ii], self.y[ii], self.f, sigma, dt)

        ax[0].clear()
        ax[0].plot(self.t[ii], self.y[ii])
        ax[0].set_ylabel('Count/s')
        fig.subplots_adjust(hspace=0.05)
        ax[0].set_xticks([])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
        if power == 'normal':
            ax[1].contourf(self.t[ii], self.f, p, cmap=plt.cm.viridis)
        else:
            ax[1].contourf(self.t[ii], self.f, lp, cmap=plt.cm.viridis)
