import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares, curve_fit
from scipy import stats
from functools import partial

""" specify your path of the file """
path = 'Downloads/AE Capital Quantitative Project v1.4/quantTest_data.csv'
day = 86400

def opencsv(path):
    """Return an array containing both time-series."""
    file = open(path, "r")
    _l = []
    for line in file:
        line = line.replace('\n', '') # get rid on line break command \n at the end of each line
        _l.append(line.split(',')) # separate column by ,
    file.close()
    a = np.array(_l).astype(float) # change the list to array and covert str to float
    return a

def clean(t, y):
    """Clean all NaNs."""
    _ind = np.where(np.isnan(y) == False)
    t = t[_ind]
    y = y[_ind]
    return t, y

def logdata(p, t=None):
    """Log return of price."""
    logp = np.log(p[1:]/p[0:-1])
    if t is not None:
        t = t[1:]
    return logp, t

def lineardata(logp, p, logt, t):
    """Return r to linear space, currently not being used"""
    _ind = np.isin(t, logt[:-1])
    _ind_shift = np.append(_ind[1:], True)      # rotate left for 1 step
    return np.exp(logp) * p[_ind_shift]

def gaussian(bins, A, mu, sigma):
    """Gaussain model."""
    return A * np.exp( - (bins - mu)**2 / (2 * sigma**2))

def laplace(bins, A, mu, b):
    """Laplace model."""
    return A * np.exp( - abs(bins - mu) / b)

def check(t, dt=None):
    """Check if time resolution is consistent."""
    if dt is None:
        dt = t[1] - t[0]       
    res = [((sub2 - sub1) > dt * 1.05) for sub1, sub2 in zip(t[:-1], t[1:])]  
    ares = np.any(res)
    return ares

def window(t, t_current, wsize=None):
    """
    returns: indexes of data within the past window with size=wsize, excluding the current timestamp.
    """
    if wsize is None:
        wsize = 365
    _ind  = np.where((0 < (t_current - t)) & ((t_current - t) <= wsize))
    return _ind

def findstart(t, wsize):
    """Find the index for start of analysis."""
    return  np.argmin(abs(t - wsize)) + 1

def statsproperty(t, r, t_current, wsize, bins=100, method='laplace'):
    """Return the statistical property of the past data (exclude the current data) with window size = wsize."""
    _ind1 = window(t=t, t_current=t_current, wsize=wsize)
    params, _, _ = fit_dist(r=r[_ind1], bins=bins, method=method)
    return params

def find_consecutive(t, dt, datalength=3):
    """Record the timestamps where the previous data with length=datalength has 
    consecutively the same time resolution (dt) in between.
    """
    ind_list = []
    for i,j in enumerate(t[datalength-1:]):
        if not check(t[i:i+datalength], dt):
            ind_list.append(j)
    return ind_list

def fit_dist(r, bins=100,
              method='laplace',  # norm | laplace
              plot=False,
              color='k',  # k | g
              ):
    """Find the best fit curve for the distribution.

       Model is either normal or laplace distribution
    """
    bar, x = np.histogram(r, bins=bins)
    xbin = 0.5 * (x[1:] + x[:-1])
    if method == 'norm':
        guess_A = max(bar)
        guess_mean = r.mean()
        guess_std = r.std()
        param_optimised, param_covariance_matrix = curve_fit(gaussian, xbin, bar,p0=[guess_A, guess_mean, guess_std],maxfev=5000)
    elif method == 'laplace':
        guess_A = max(bar)
        guess_mean = r.mean()
        guess_b = r.std() / np.sqrt(2)
        param_optimised, param_covariance_matrix = curve_fit(laplace, xbin, bar, p0=[guess_A, guess_mean, guess_b],maxfev=5000)

    if plot == True:  # plotting
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        fig, ax = plt.subplots()
        ax.bar(xbin, bar, width=(xbin[1]-xbin[0]), color=color)
        plot_xbin = np.linspace(xbin[0], xbin[-1], 10000)
        if method == 'norm':
            ax.plot(plot_xbin, gaussian(plot_xbin, *param_optimised), color='r', label='Gaussian')
        elif method == 'laplace':
            ax.plot(plot_xbin, laplace(plot_xbin, *param_optimised), color='b', label='Laplace')
        ax.set_ylabel('Frequency', fontsize=15)
        ax.set_xlabel('r', fontsize=15)
        ax.legend()
        ax.set_xlim(-0.0005, 0.0005)
        plt.tight_layout()

    return param_optimised, param_covariance_matrix, xbin

def correlation(t1, y1, t2, y2, plot=False):
    """Find the correlation coefficient of the variables."""
    _ind1 = np.isin(t1, t2)
    cy1 = y1[_ind1]
    _ind2 = np.isin(t2, t1)
    cy2 = y2[_ind2]
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(cy1, cy2, '.')
    return np.corrcoef(cy1, cy2)[0][1]

def chi_square_stat(C, O):
    """Chi square test for modeling fitting, not being used at the moment."""
    chi_squared = np.sum( (C-O)**2  / C)
    degrees_of_freedom = len(O) - 3
    return stats.chi2.sf(chi_squared, degrees_of_freedom)

class PredictData(object):
    """Predict ups and downs of price.
    
    Assess the probability of log returns based on the distribution of their past 'wsize' data, 
    the default window size is 365 days.
    
    Record the data that is less likely to occur based on the distribution,
    i.e, data occur at both edges of the distribution.

    Predict the next move is a rise if consecutively large drops occur, or vice versa.
    """
    def __init__(self, path=path, wsize=365, bins=100, method='laplace', analysis1=False, analysis2=False):
        a = opencsv(path)
        rawt = a[:, 0]
        t0 = rawt[0]
        t = rawt - t0  # time stamps
        if check(t=t):
            raise AttributeError('Time resolution is not consistent')
        else:
            dt = t[1] - t[0]
        p1 = a[:, 1]  # price of time series 1
        p2 = a[:, 2]  # price of time series 2
        r1, lt = logdata(p=p1, t=t)  # log return price of time series 1
        r2, _ = logdata(p=p2)  # log return of time series 2
        clt1, clr1 = clean(lt, r1)
        clt2, clr2 = clean(lt, r2)
        self.t = t
        self.dt = dt
        self.p1 = p1
        self.clt1 = clt1
        self.clr1 = clr1
        self.p2 = p2
        self.clt2 = clt2
        self.clr2 = clr2

        starttime = time.time()
        if analysis1 == True:
            # First time series
            list_t1in = []  # record data with consecutively large increases
            list_r1in = []
            list_t1de = []  # record timestamps with consecutively large drops
            list_r1de = []
            mess1 = []  # record fails of modeling
            _startind1 = findstart(t=clt1, wsize=wsize)
            t1_array = clt1[_startind1:]
            r1_array = clr1[_startind1:]
            for i,j in zip(t1_array, r1_array):
                try:
                    params = statsproperty(t=clt1, t_current=i, r=clr1, wsize=wsize, bins=60, method=method)
                    mean = params[1]
                    if method == 'laplace':
                        std = np.sqrt(2*params[2]**2)
                    elif method == 'norm':
                        std = params[2]
                    if j > mean + 4 * std:  # record data offset the mean with 4 sigma
                        list_t1in.append(i)
                        list_r1in.append(j)
                    elif j < mean - 4 * std:  # record data offset the mean with 4 sigma
                        list_t1de.append(i)
                        list_r1de.append(j)
                except:
                    mess1.append(i)
            _indin = np.isin(t, list_t1in)  # find out indexes where price increase is high
            increase_t1 = t[_indin]
            increase_p1 = p1[_indin]
            _indde = np.isin(t, list_t1de)  # find out indexes where price decrease is high
            decrease_t1 = t[_indde]
            decrease_p1 = p1[_indde]
            # Find predictions, 'in' means increase, 'de' means decrease
            _predict_ind_in1 = find_consecutive(increase_t1, dt=dt, datalength=3)
            predict_in_t1 = t[np.isin(t, _predict_ind_in1)]
            predict_in_p1 = p1[np.isin(t, _predict_ind_in1)]
            _predict_ind_de1 = find_consecutive(decrease_t1, dt=dt, datalength=3)
            predict_de_t1 = t[np.isin(t, _predict_ind_de1)]
            predict_de_p1 = p1[np.isin(t, _predict_ind_de1)]

            self.t1in = list_t1in
            self.r1in = list_r1in
            self.t1de= list_t1de
            self.r1de = list_r1de
            self.increase_t1 = increase_t1
            self.increase_p1 = increase_p1
            self.decrease_t1 = decrease_t1
            self.decrease_p1 = decrease_p1
            self.predict_in_t1 = predict_in_t1
            self.predict_in_p1 = predict_in_p1
            self.predict_de_t1 = predict_de_t1
            self.predict_de_p1 = predict_de_p1
            self.mess1 = mess1

        if analysis2 == True:
            # Second time series, basically the same as the method used in time-series 1,
            # just different variables. Can wrapped up as a function instead
            list_t2in = []
            list_r2in = []
            list_t2de = []
            list_r2de = []
            mess2 = []  # record fails of modeling
            _startind2 = findstart(t=clt2, wsize=wsize)
            t2_array = clt2[_startind2:]
            r2_array = clr2[_startind2:]
            for k,l in zip(t2_array, r2_array):
                try:
                    params = statsproperty(t=clt2, t_current=k, r=clr2, wsize=wsize, bins=200, method=method)
                    mean = params[1]
                    if method == 'laplace':
                        std = np.sqrt(2*params[2]**2)
                    elif method == 'norm':
                        std = params[2]
                    if j > mean + 4 * std:
                        list_t2in.append(k)
                        list_r2in.append(l)
                    elif j < mean - 4 * std:
                        list_t2de.append(k)
                        list_r2de.append(l)
                except:
                    mess2.append(k)
            _indin = np.isin(t, list_t2in)  # find out indexes in linear space where price increase is high
            increase_t2 = t[_indin]
            increase_p2 = p2[_indin]
            _indde = np.isin(t, list_t2de)  # find out indexes in linear space where price decrease is high
            decrease_t2 = t[_indde]
            decrease_p2 = p2[_indde]
            # Find predictions, 'in' means increase, 'de' means decrease
            _predict_ind_in2 = find_consecutive(increase_t2, dt=dt, datalength=3)
            predict_in_t2 = t[np.isin(t, _predict_ind_in2)]
            predict_in_p2 = p2[np.isin(t, _predict_ind_in2)]
            _predict_ind_de2 = find_consecutive(decrease_t2, dt=dt, datalength=3)
            predict_de_t2 = t[np.isin(t, _predict_ind_de2)]
            predict_de_p2 = p2[np.isin(t, _predict_ind_de2)]

            self.t2in = list_t2in
            self.r2in = list_r2in
            self.t2de= list_t2de
            self.r2de = list_r2de
            self.increase_t2 = increase_t2
            self.increase_p2 = increase_p2
            self.decrease_t2 = decrease_t2
            self.decrease_p2 = decrease_p2
            self.predict_in_t2 = predict_in_t2
            self.predict_in_p2 = predict_in_p2
            self.predict_de_t2 = predict_de_t2
            self.predict_de_p2 = predict_de_p2
            self.mess2 = mess2

        endtime = time.time()
        self.runtime = endtime - starttime

    def plot_time_series(self):
        """Plot time-series in linear space."""
        t = self.t
        p1 = self.p1
        p2 = self.p2

        fig, ax = plt.subplots(2, sharex=True)
        fig.subplots_adjust(hspace=0.05)
        ax[0].plot(t, p1, label='t1', color='k')
        ax[0].set_ylabel('Price')
        ax[0].legend()
        ax[1].plot(t, p2, label='t2', color='g')
        ax[1].set_ylabel('Price')
        ax[1].set_xlabel('Days Since 2008 Jan 1')
        ax[1].legend()
        
    def plot_log_return(self):
        """Plot time-series in log return space."""
        clt1 = self.clt1
        clr1 = self.clr1
        clt2 = self.clt2
        clr2 = self.clr2

        fig, ax = plt.subplots(2, sharex=True)
        fig.subplots_adjust(hspace=0.05)
        ax[0].plot(clt1, clr1, label='t1', color='k')
        ax[0].set_ylabel('r')
        ax[0].legend()
        ax[1].plot(clt2, clr2, label='t2', color='g')
        ax[1].set_ylabel('r')
        ax[1].set_xlabel('Days Since 2008 Jan 1')
        ax[1].legend()

    def correlation_analysis(self, wsize=365):
        """Analyse the correlation of variables."""
        t = self.t
        t1, p1 = clean(t, self.p1)
        t2, p2 = clean(t, self.p2)
        clr1 = self.clr1
        clr2 = self.clr2
        clt1 = self.clt1
        clt2 = self.clt2

        corr_p = []
        corr_r = []
        loops = int((t[-1] - t[0]) / wsize)
        for i in range(loops):  # loops go through data in each year
            _ind1 = np.where( (t1 >= wsize*i) & (t1 < wsize*(i+1)) )
            _ind2 = np.where( (t2 >= wsize*i) & (t2 < wsize*(i+1)) )
            _ind3 = np.where( (clt1 >= wsize*i) & (clt1 < wsize*(i+1)) )
            _ind4 = np.where( (clt2 >= wsize*i) & (clt2 < wsize*(i+1)) )
            corr_p.append(correlation(t1[_ind1], p1[_ind1], t2[_ind2], p2[_ind2], plot=True))
            corr_r.append(correlation(clt1[_ind3], clr1[_ind3], clt2[_ind4], clr2[_ind4]))
        
        self.corr_p = corr_p
        self.corr_r = corr_r


        



