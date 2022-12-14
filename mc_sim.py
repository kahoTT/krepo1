from functools import partial
import numpy as np
import stingray
from stingray.simulator import simulator
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def Powfit(freq=None, f=None, y=None, wf=None, guess=None, rebin_log=False, exclude=True, factor=None):
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
    x0 = np.array([3, -2, guess])
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

    o_model = F(f, *result.x)
    if o_model[0] < guess:
        # best fit model does not have decreasing trend over frequenies
        o_model = np.ones(len(f)) * guess
        n_model = o_model * factor
        norm_f = None
    else:
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

def simlc(ares=None, t=None, y=None, dt=None, N=None, red_noise=1, o_model=None, n_model=None, model='n'):
    sim = simulator.Simulator(N=N, mean=y.mean(), dt=dt, rms=y.std()/y.mean(), red_noise=red_noise) 
    if model == 'o':
        lc = sim.simulate(o_model)
    elif model == 'n':
        lc = sim.simulate(n_model)
    if ares == True:
        _intin = np.isin(lc.time, (t-t[0]))
    else:
        _intin = ()
    time = lc.time[_intin]
    counts = lc.counts[_intin]
    return time, counts

class RealLc(object):
    def __init__(self, t=None, y=None, wf=None, dt=None, input_counts=False, norm='leahy', red_noise=1, model='n', exclude=True):
        self.t = t
        self.y = y
        tc, yc, N, factor, dt, res = Fillpoint(t, y, dt)
        spec, logspec = Genspec(t=tc, y=yc, dt=dt, norm=norm, input_counts=input_counts)
        result, o_model, n_model, norm_f = Powfit(logfreq=logspec.freq, f=spec.freq, y=logspec.power, wf=wf, rebin_log=False, exclude=exclude, factor=factor)
        time, counts = simlc(ares=res, t=self.t, y=self.y, dt=dt, N=N, o_model=o_model, n_model=n_model, red_noise=red_noise)
        self.result = result
        self.o_model = o_model
        self.n_model = n_model
        self.norm_f = norm_f
        self.time = time
        self.counts = counts
        self.spec = spec
        self.logspec = logspec

    def plot_spec(self):
        fig, ax = plt.subplots()
        ax.plot(self.logspec.freq, self.logspec.power , ds='steps-mid')
        ax.plot(self.spec.freq, self.o_model, label='Original spec')
        ax.plot(self.spec.freq, self.n_model, label='Power boosted spec')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Leahy Power')
        plt.xlabel('Frequency (Hz)')
        ax.legend(loc='best')
        plt.show()

def F(x, A, B, C): 
    # B is the alpha, the slope of power spectrum in log space
    return A * x** (B) + C

def G(x, y, args):
    return (F(x, *args) - y) 

def Horizontalfit(freq, power):
    aver = sum(power) / len(power)
    model = np.ones(len(freq)) * aver
    return aver, model

def Horizontalfit_log(freq, power):
    aver = np.exp(sum(np.log(power)) / len(power))
    model = np.ones(len(freq)) * aver
    return aver, model