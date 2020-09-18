test
from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pylab as plt

from astropy.timeseries import LombScargle

import scales
import lcdata

class Spectral(object):
    def __init__(self, filename):
        self.lc = lcdata.load(filename)
        self.fig = None
        self.ax = None

    @staticmethod
    def lsf(args):
        t, y, f = args
        ls = LombScargle(t, y).power(f, normalization='standard')
        width = len(t)
        power = ls * 0.5 * (width - 1)
        time = np.average(t)
        return power, time

    def plot(
            self,
            fig = None,
            ax = None,
            f1 = 2**-13,
            f2 = 2**-3,
            nf = 2**10,
            step = 2**7,
            width = 2**11,
            tstart = None,
            tend = None,
            logf = True,
            logp = True,
            time_window = False,
            parallel = True,
            ):
        """
        time_window:
           select whether to use time (True) or #points (False) for step and width
        parallel:
           use process pool (True) or serial mode (False)
        """

        t = self.lc.time
        y = self.lc.xlum

        if tstart is None:
            tstart = t[0]
            istart = 0
        else:
            istart = np.searchsorted(t, tstart) + 1

        if tend is None:
            if time_window:
                tend = t[-1] - width
                iend = np.searchsorted(t, tend)
            else:
                iend = len(t) - width
        else:
            if time_window:
                iend = np.searchsorted(t, tend - width)
            else:
                iend = np.searchsorted(t, tend) - width

        if ax is None:
            if fig is None:
                fig = plt.figure()
                ax = fig.add_subplot()
            else:
                fig.clear()
                ax = fig.add_subplot()
        else:
            fig  = ax.figure
        self.fig = fig
        self.ax = ax

        if logf:
            f = f1 * np.exp(np.arange(nf) * (np.log(f2) - np.log(f1)) / (nf - 1))
        else:
            f = np.arange(nf) * (f2 - f1) / (nf - 1) + f1
        nf = len(f)

        if time_window:
            sections = [
                slice(
                    np.searchsorted(t, tx) + 1,
                    np.searchsorted(t, tx + width) + 1,
                    )
                for tx in np.arange(tstart, tend, step)
                ]
        else:
            sections = [slice(j, j + width) for j in range(istart, iend, step)]
        ns = len(sections)
        power = np.ndarray((ns , nf))
        time = np.ndarray(ns)

        if parallel:
            pool = Pool()
            tasks = [(t[s], y[s], f) for s in sections]
            results = pool.imap(self.lsf, tasks)
            for i, (xpower, xtime) in enumerate(results):
                power[i] = xpower
                time[i] = xtime
            pool.close()
        else:
            for i,s in enumerate(sections):
                ls = LombScargle(t[s], y[s]).power(f, normalization='standard')
                power[i] = ls * 0.5 * (width - 1)
                time[i] = np.average(t[s])
        fmax = f[np.argmax(power, axis=1)]

        fp = np.ndarray(nf + 1)
        fp[1:-1] = 0.5 * (f[1:] + f[:-1])
        fp[ 0] = 0.5 * (3 * f[0] - f[1])
        fp[-1] = 0.5 * (3 * f[-1] - f[-2])

        tp = np.ndarray(ns + 1)
        tp[1:-1] = 0.5 * (time[1:] + time[:-1])
        tp[ 0] = 0.5 * (3 * time[0] - time[1])
        tp[-1] = 0.5 * (3 * time[-1] - time[-2])

        if logf:
            ax.set_ylabel('log frequency (Hz)')
            fp  = np.log10(fp)
            fmax = np.log10(fmax)
        else:
            ax.set_ylabel('frequency (Hz)')

        if logp:
            power = np.log10(power)
            cb_label = 'log significance'
        else:
            cb_label = 'significance'
        img = ax.pcolorfast(tp, fp, power.T, cmap='Reds')
        fig.colorbar(img, label = cb_label)

        ax.plot(time, fmax)

        ax.set_xscale('timescale')
