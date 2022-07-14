from functools import partial
import numpy as np
import stingray
from stingray.simulator import simulator
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class simLC(object):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None', exclude=True, red_noise=1):
        self.norm = norm
        if dt is None:
            dt = t[1] - t[0]       
        lc = stingray.Lightcurve(t, y, input_counts=input_counts, dt = dt, skip_checks=True)
        spec = stingray.Powerspectrum(lc, norm=norm)   
        spec.power = abs(spec.power)
        logspec = spec.rebin_log(0.01) # have an impact on having a flat or inclined spectrum 
        _ind2 = np.where(logspec.freq >= 2e-2)
        logpow1 = logspec.power[_ind2]
        if exclude == True:  
            _ind = np.where((logspec.freq <= 5e-3) | (logspec.freq >= 15e-3))
            logfre = logspec.freq[_ind]
            logpow = logspec.power[_ind]
        else:
            logfre = logspec.freq
            logpow = logspec.power
        nan = np.isnan(logpow1)
        notnan = ~nan
        guess_horizontal = logpow1[notnan].mean()
        x0 = np.array([3, -2, guess_horizontal])
        nan2 = np.isnan(logpow)
        notnan2 = ~nan2
        self.logfre = logfre[notnan2]
        self.logpow = logpow[notnan2]
        result = least_squares(partial(G, self.logfre, self.logpow), x0)
        omodel = F(spec.freq, *result.x)
        # check if data has gap and make correction
        res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]  
        if np.any(res) == True:
            n_of_data = int((t[-1] - t[0]) / dt + 1)
            factor = n_of_data / (len(t)) 
        else:
            n_of_data = len(t)
        result.x[2] = result.x[2]*factor
        #
        lmodel = F(spec.freq, *result.x)

# make the lightcurve with the not data gaps
        sim = simulator.Simulator(N=n_of_data, mean=y.mean(), dt=dt, rms=y.std()/y.mean(), red_noise=red_noise) 
        lc = sim.simulate(lmodel)
        self.time = lc.time
        self.counts = lc.counts
        self.result = result

# define function within class        
#        result = least_squares(self.g, x0)
#        lmodel = self.f(*result.x)

        self.fre = spec.freq
        self.pow = spec.power
        self.omodel = omodel
        self.lmodel = lmodel

    def plot_spec(self):
        fig, ax = plt.subplots()
        ax.plot(self.fre, self.pow , ds='steps-mid')
        ax.plot(self.fre, self.omodel)
        ax.plot(self.fre, self.lmodel)
        plt.xscale('log')
        plt.yscale('log')
        if self.norm == 'None':
            plt.ylabel('Abs power')
        else:
            plt.ylabel(self.norm + ' power')
        plt.xlabel('Frequency (Hz)')
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