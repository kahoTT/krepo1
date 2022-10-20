from functools import partial
import numpy as np
import stingray
from stingray.simulator import simulator
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def PowFit(f, y, f2=None, guess=None, rebin_log=True, exclude=True):
    nan = np.isnan(y)
    notnan = ~nan
    f = f[notnan]
    y = y[notnan]
    if exclude == True:  
        _ind = np.where((f <= 5e-3) | (f >= 15e-3))
        f = f[_ind]
        y = y[_ind]
    if guess is None: 
        _ind2 = np.where(f >= 2e-2)
        guess = y[_ind2].mean()
    x0 = np.array([3, -1, guess])
    if rebin_log == True:
        rf, rebinp, _, _ = stingray.rebin_data_log(f, y, 0.05)
        rebinf = (rf[1:]+rf[:-1])/2
        nan2 = np.isnan(rebinp)
        notnan2 = ~nan2
        rebinf = rebinf[notnan2]
        rebinp = rebinp[notnan2]
        result = least_squares(partial(G, rebinf, rebinp), x0)
    else:
        result = least_squares(partial(G, f, y), x0)
    if f2 is None:
        f2 = f
    model = F(f2, *result.x)
    return result, model

class simLC(object):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None', red_noise=1, model='n', gen = True):
        self.norm = norm
        if dt is None:
            dt = t[1] - t[0]       
        # fill light curve with mean value 
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]  
        if np.any(res) == True:
            ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
            tc = np.array([])
            for i in ag[1:]:
                ta = np.arange(t[i] + dt, t[i+1], dt)
                tc = np.concatenate([tc, ta])
            yc = np.ones(len(tc)) * y.mean()
            tc = np.concatenate([t, tc])
            yc = np.concatenate([y, yc])
            y_c = np.array([x for _,x in sorted(zip(tc, yc))])
            t_c = np.sort(tc)
        else:
            t_c = t
            y_c = y
        lc = stingray.Lightcurve(t_c-t_c[0], y_c, input_counts=input_counts, dt = dt, skip_checks=False)
        spec = stingray.Powerspectrum(lc, norm=norm)   
        spec.power = abs(spec.power)
        logspec = spec.rebin_log(0.05) # have an impact on having a flat or inclined spectrum 
        result, omodel = PowFit(f=logspec.freq, y=logspec.power,
                                f2=spec.freq, rebin_log=False) 

        # check if data has gap and make correction 

        if np.any(res) == True:
            n_of_data = int((t[-1] - t[0]) / dt + 1)
            factor = n_of_data / (len(t)) 
            result.x[2] = result.x[2]*factor
        else:
            n_of_data = len(t)
        lmodel = F(spec.freq, *result.x)

        if gen == True:
            sim = simulator.Simulator(N=n_of_data, mean=y.mean(), dt=dt, rms=y.std()/y.mean(), red_noise=red_noise) 
            if model == 'o':
                lc = sim.simulate(omodel)
            elif model == 'n':
                lc = sim.simulate(lmodel)
            if np.any(res) == True:
                _intin = np.isin(lc.time, (t-t[0]))
            else:
                _intin = ()
            self.time = lc.time[_intin]
            self.counts = lc.counts[_intin]
        self.result = result

# define function within class        
#        result = least_squares(self.g, x0)
#        lmodel = self.f(*result.x)

        self.freq = spec.freq
        self.pow = spec.power
        self.omodel = omodel
        self.lmodel = lmodel
        self.spec = spec

    def plot_spec(self):
        fig, ax = plt.subplots()
        ax.plot(self.freq, self.pow , ds='steps-mid')
        ax.plot(self.freq, self.omodel, label='Original spec')
        ax.plot(self.freq, self.lmodel, label='Power boosted spec')
        plt.xscale('log')
        plt.yscale('log')
        if self.norm == 'None':
            plt.ylabel('Abs power')
        else:
            plt.ylabel(self.norm + ' power')
        plt.xlabel('Frequency (Hz)')
        ax.legend(loc='best')
        plt.show()

    def plot_lc(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.counts)
        plt.ylabel('Cts/s')
        plt.xlabel('time s') 
        plt.show()

    def f(self, A, B, C): # the down side of this method is that the input frequency array is fix. need another code to fix it
       return A * self.logfre ** (B) + C

    def g(self, args):
       return (np.log(self.f(*args)) - np.log(self.logpow)) 

def F(x, A, B, C): # B is the alpha, the slope of power spectrum in log space
    return A * x** (B) + C

def G(x, y, args):
    return (np.log(F(x, *args)) - np.log(y)) 