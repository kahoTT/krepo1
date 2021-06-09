from functools import partial
import numpy as np
import stingray
from stingray.simulator import simulator
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class simLC(object):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        self.norm = norm
        if dt is None:
            dt = t[1] - t[0]       
        lc = stingray.Lightcurve(t, y, input_counts=input_counts)
        spec = stingray.Powerspectrum(lc, norm=norm)   
        spec.power = abs(spec.power)
        logspec = spec.rebin_log(0.001)  
        _ind = np.where((logspec.freq <= 4e-3) | (logspec.freq >= 15e-3))
        _ind2 = np.where(logspec.freq >= 1e-2)
        self.logfre = logspec.freq[_ind]
        self.logpow = logspec.power[_ind]
        guess_horizontal = logspec.power[_ind2].mean()
        x0 = np.array([3, -2, guess_horizontal])
        result = least_squares(partial(G, self.logfre, self.logpow), x0)
        lmodel = F(spec.freq, *result.x)
        sim = simulator.Simulator(N=len(t), mean=y.mean(), dt=dt, rms=y.std()/y.mean()) #!!! may be skip the check !!!
        lc = sim.simulate(lmodel)
        self.time = t 
        self.counts = lc.counts

# define function within class        
#        result = least_squares(self.g, x0)
#        lmodel = self.f(*result.x)

        self.fre = spec.freq
        self.pow = spec.power
        self.lmodel = lmodel

    def plot_spec(self):
        fig, ax = plt.subplots()
        ax.plot(self.fre, self.pow , ds='steps-mid')
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

def F(x, A, B, C):
    return A * x** (B) + C

def G(x, y, args):
    return (np.log(F(x, *args)) - np.log(y)) 