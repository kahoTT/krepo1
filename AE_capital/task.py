import csv
import numpy as np

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
def clean(t,y):
    _int = np.where(np.isnan(y) == False)
    t = t[_int]
    y = y[_int]
    return t, y

def logdata(p, t=None):
    logp = np.log(p[1:]/p[0:-1])
    if t is not None:
        t = t[1:]
    return logp, t

def analysis():
    a = opencsv(path=path)
    t = a[:, 0]
    p1 = a[:, 1]
    p2 = a[:, 2]
    logp1, lt = logdata(p=p1, t=t)
    logp2, _ = logdata(p=p2)
    return lt, logp1, logp2

def guassian(bins, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))

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
    _int  = np.where((0 < (t - t_current)) & ((t - t_current) < wsize))
    return _int

def predict(p):
    _sum1 = sum(p)
    _sum2 = sum(p[1:])
    pred = _sum1 - _sum2
    return pred

class PredictCurve(object):
    """Predict prices using moving average approach
    
    """
    def __init__(self, path=path, wsize=10):
        a = open(path)
        t = a[:, 0]
        if check(t=t):
            raise AttributeError('Time resolution is not consistent')
        p1 = a[:, 1]
        p2 = a[:, 2]
        logp1, lt = logdata(p=p1, t=t)
        logp2, _ = logdata(p=p2)
        clt1, clp1 = clean(lt, logp1)
        clt2, clp2 = clean(lt, logp2)
        for i in (t-t[0]):
            if i + wsize > t[-1]:	# loop stops if the window covers inexsit time
                break
            

        self.a = a
        self.p1 = p1
        self.p2 = p2
        self.clt1 = clt1
        self.clp1 = clp1
        self.clt1 = clt1
        self.clp2 = clp2


# def rebin(t, p, binsize):