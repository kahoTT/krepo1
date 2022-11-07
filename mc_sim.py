from functools import partial
import numpy as np
import stingray
from stingray.simulator import simulator
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def Powfit(logfreq=None, f=None, y=None, wf=None, guess=None, rebin_log=True, exclude=True, factor=None):
    """Lightcurve may contain gaps, we use n_model to try to take it into account"""
    nan = np.isnan(y)
    notnan = ~nan
    logfreq = logfreq[notnan]
    y = y[notnan]
    if exclude == True:  
        _ind = np.where((logfreq <= 5e-3) | (logfreq >= 15e-3))
        logfreq = logfreq[_ind]
        y = y[_ind]
    if guess is None: 
        _ind2 = np.where(logfreq >= 2e-2)
        guess = y[_ind2].mean()
    x0 = np.array([3, -1, guess])
    if rebin_log == True:
        rf, rebinp, _, _ = stingray.rebin_data_log(logfreq, y, 0.05)
        rebinf = (rf[1:]+rf[:-1])/2
        nan2 = np.isnan(rebinp)
        notnan2 = ~nan2
        rebinf = rebinf[notnan2]
        rebinp = rebinp[notnan2]
        result = least_squares(partial(G, rebinf, rebinp), x0)
    else:
        result = least_squares(partial(G, logfreq, y), x0)
     
    if wf is None:
        wf = f
    if result.x[0] * result.x[1] > 0:
        o_model = np.ones(len(f)) * guess
        n_model = o_model * factor
        norm_f = None
    else:
        o_model = F(f, *result.x)
        n_result = result
        n_result.x[2] = n_result.x[2] * factor
        n_model = F(f, *n_result.x)
        norm_f = F(wf, *result.x)
    return result, o_model, n_model, norm_f

def Fillpoint(t=None, y=None, dt=None):
    """fill missing data of light curve with mean value."""
    if dt is None:
        dt = t[1] - t[0]       
    res = [(sub2 - sub1 > dt) for sub1, sub2 in zip(t[:-1], t[1:])]  
    ares = np.any(res)
    if ares == True:
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
        n_of_data = int((t[-1] - t[0]) / dt + 1)
        factor = n_of_data / (len(t)) 
    else:
        t_c = t
        y_c = y
        n_of_data = len(t)
        factor = 1
    return t_c, y_c, n_of_data, factor, dt, ares 

def Genspec(t=None, y=None, input_counts=False, norm='leahy', dt=None):
    lc = stingray.Lightcurve(t-t[0], y, input_counts=input_counts, dt = dt, skip_checks=False)
    spec = stingray.Powerspectrum(lc, norm=norm)   
    spec.power = abs(spec.power)
    logspec = spec.rebin_log(0.05) # have an impact on having a flat or inclined spectrum 
    return spec, logspec

def simlc(res=None, t=None, y=None, dt=None, N=None, red_noise=1, o_model=None, n_model=None, model='n'):
    sim = simulator.Simulator(N=N, mean=y.mean(), dt=dt, rms=y.std()/y.mean(), red_noise=red_noise) 
    if model == 'o':
        lc = sim.simulate(o_model)
    elif model == 'n':
        lc = sim.simulate(n_model)
    if np.any(res) == True:
        _intin = np.isin(lc.time, (t-t[0]))
    else:
        _intin = ()
    time = lc.time[_intin]
    counts = lc.counts[_intin]
    return time, counts

class RealLc(object):
    def __init__(self, t=None, y=None):
        self.t = t
        self.y = y
    def __call__(self, wf=None, dt=None, input_counts=False, norm='leahy', red_noise=1, model='n'):
        tc, yc, N, factor, dt, res = Fillpoint(self.t, self.y, dt)
        spec, logspec = Genspec(t=tc, y=yc, dt=dt, norm=norm, input_counts=input_counts)
        result, o_model, n_model, norm_f = Powfit(logfreq=logspec.freq, f=spec.freq, y=logspec.power, wf=wf, rebin_log=False, factor=factor)
        # time, counts = simlc(res=res, t=self.t, y=self.y, dt=dt, N=N, o_model=o_model, n_model=n_model, red_noise=red_noise)
        return o_model, n_model, norm_f, logspec

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
        result, omodel = Powfit(logfreq=logspec.freq, y=logspec.power,
                                f=spec.freq, rebin_log=False) 

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

def F(x, A, B, C): 
    # B is the alpha, the slope of power spectrum in log space
    return A * x** (B) + C

def G(x, y, args):
    return (np.log(F(x, *args)) - np.log(y)) 