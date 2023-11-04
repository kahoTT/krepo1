import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares, curve_fit
from scipy import stats
from functools import partial

""" specify your path of the file """
path = 'Downloads/AE Capital Quantitative Project v1.4/quantTest_data.csv'
day = 86400

""" function to open the file """
def opencsv(path):
    file = open(path, "r")
    _l = []
    for line in file:
        line = line.replace('\n', '') # get rid on line break command \n at the end of each line
        _l.append(line.split(',')) # separate column by ,
    file.close()
    a = np.array(_l).astype(float) # change the list to array and covert str to float
    return a

""" Clean the NaN """
def clean(t, y):
    _int = np.where(np.isnan(y) == False)
    t = t[_int]
    y = y[_int]
    return t, y

def logdata(p, t=None):
    logp = np.log(p[1:]/p[0:-1])
    if t is not None:
        t = t[1:]
    return logp, t

def lineardata(logp, p, logt, t):
    _int = np.isin(t, logt[:-1])
    _int_shift = np.append(_int[1:], True)      # rotate left for 1 step
    return np.exp(logp) * p[_int_shift]

def gaussian(bins, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
    # return A * np.exp( - (bins - mu)**2 / (2 * sigma**2))

def laplace(bins, mu, b):
    return 1/(2*b) * np.exp( - abs(bins - mu) / b)

def check(t, dt=None):
    """Check if the resolution is consistent."""
    if dt is None:
        dt = t[1] - t[0]       
    res = [((sub2 - sub1) < dt * 1.05) for sub1, sub2 in zip(t[:-1], t[1:])]  
    ares = np.any(res)
    return ares

def mw(t, t_current, wsize=None):
    """
    returns: indexes of the selected window size.
    """
    if wsize is None:
        wsize = 10
    _int  = np.where((0 < (t_current - t)) & ((t_current - t) <= wsize))
    return _int

def findstart(t, wsize):
    """Find the index for start of prediction."""
    return  np.argmin(abs(t - wsize)) + 1

def f(x, y, args):
    return (gaussian(x, *args) - y)

def least_squre_fit(y, bins=100):
    bar, x1, _ = plt.hist(y, bins=bins)
    guess_height = max(bar)
    guess_mean = y.mean()
    guess_std = y.std()
    x = (x1[1:] + x1[0:-1])/2
    guess = np.array([guess_height, guess_mean, guess_std])
    return least_squares(partial(f, x, bar), x0=guess)

def chi_square_stat(C, O):
    chi_squared = np.sum( (C-O)**2  / C)
    degrees_of_freedom = len(O) - 3
    return stats.chi2.sf(chi_squared, degrees_of_freedom)

def fit_dist(logp, bins=50,
              method='laplace',  # norm | laplace
              plot=False,
              ):
    """Plot distribution with best fit curve."""
    bar, x = np.histogram(logp, bins=bins, density=True)
    xbin = 0.5 * (x[1:] + x[:-1])
    if method == 'norm':
        guess_mean = logp.mean()
        guess_std = logp.std()
        param_optimised, param_covariance_matrix = curve_fit(gaussian, xbin, bar,p0=[guess_mean, guess_std],maxfev=5000)
    elif method == 'laplace':
        guess_mean = logp.mean()
        guess_b = logp.std() / np.sqrt(2)
        param_optimised, param_covariance_matrix = curve_fit(laplace, xbin, bar, p0=[guess_mean, guess_b],maxfev=5000)

    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(xbin, bar)
        if method == 'norm':
            ax.plot(xbin, gaussian(xbin, *param_optimised))
        elif method == 'laplace':
            ax.plot(xbin, laplace(xbin, *param_optimised))

    return param_optimised, param_covariance_matrix

class PredictCurve(object):
    """Predict prices using moving average approach
    
    """
    def __init__(self, path=path, wsize=10):
        a = opencsv(path)
        rawt = a[:, 0]
        t0 = rawt[0]
        t = rawt - t0
        if not check(t=t):
            raise AttributeError('Time resolution is not consistent')
        else:
            dt = t[1] - t[0]
        p1 = a[:, 1]
        p2 = a[:, 2]
        logp1, lt = logdata(p=p1, t=t)
        logp2, _ = logdata(p=p2)
        clt1, clp1 = clean(lt, logp1)
        clt2, clp2 = clean(lt, logp2)

        starttime = time.time()
	# First time series
        list_p1 = []
        list_upper = []
        list_lower = []
        _startind1 = findstart(t=clt1, wsize=wsize)
        predlogt1 = np.append(clt1[_startind1:], (clt1[-1]+dt))
        for i in predlogt1:
            _int1 = mw(t=clt1, t_current=i, wsize=wsize)
            pred1 = clp1[_int1].mean()
            predstd = clp1[_int1].std()
            upperlim = pred1 + 2 * predstd
            lowerlim = pred1 - 2 * predstd
            list_p1.append(pred1)
            list_upper.append(upperlim)
            list_lower.append(lowerlim)
        predlogp1 = np.array(list_p1)
        predlogu1 = np.array(list_upper)
        predlogl1 = np.array(list_lower)
        predp1 = lineardata(predlogp1, p1, predlogt1, t)
        predu1 = lineardata(predlogp1, p1, predlogt1, t)
        predl1 = lineardata(predlogp1, p1, predlogt1, t)

        # # Second time series
        # list_p2 = []
        # _startind2 = findstart(t=clt2, wsize=wsize)
        # predlogt2 = np.append(clt2[_startind2:], (clt2[-1]+dt))
        # for j in clt2:
        #     _int2 = mw(t=clt2, t_current=j, wsize=wsize)
        #     pred2 = np.mean(clp2[_int2])
        #     list_p2.append(pred2)
        # predlogp2 = np.array(list_p2)
        endtime = time.time()
            
        self.a = a
        self.p1 = p1
        self.p2 = p2
        self.clt1 = clt1
        self.clp1 = clp1
        self.predlogt1 = predlogt1
        self.predlogp1 = predlogp1
        self.predp1 = predp1
        self.predu1 = predu1
        self.predl1 = predl1

        self.clt2 = clt2
        self.clp2 = clp2
        # self.predlogt2 = predlogt2
        # self.predlogp2 = predlogp2
        self.runtime = endtime - starttime


# def rebin(t, p, binsize):