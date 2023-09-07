import os
import numpy as np
import stingray
from astropy.io import fits
from kwavelet import detrend, wavelet_spec, analysis
from mc_sim import Fillpoint, Genspec
import matplotlib.pyplot as plt
import minbar
import glob
from pycwt.mothers import Morlet
from matplotlib.ticker import MaxNLocator
from serialising import Serialising as S
import matplotlib.patches as mpatches

path  = "/b/kaho/XTEJ1701-462"
path2 = "/b/kaho/fits/XTEJ1701-462"
obsid1 = "92405-01-03-06"
obsid2 = "92405-01-09-09"
obsid3 = "92405-01-29-06"
obsid4 = "92405-01-63-08"
f1 = 3.5e-3
f2 = 4.8e-3
f3 = 5.6e-3
f4 = 3.5e-3

def read(obsid=None):
    lp =  os.listdir(path+"/"+obsid)
    lct = np.array([])
    lcy = np.array([])
    for i in lp:
        lc = fits.open(path+"/"+obsid+"/"+i)
        t = lc[1].data['TIME']
        y = lc[1].data['RATE']
        lct = np.concatenate((lct, t*86400), axis=0)
        lcy = np.concatenate((lcy, y), axis=0)
    return lct, lcy, (lct[1]-lct[0])

def readmulti(obsids=None, reduction=True):
    mlct = np.array([])
    mlcy = np.array([])
    for i in obsids:
        lct, lcy, dt = read(obsid=i)
        mlct = np.concatenate((mlct, lct), axis=0)
        if reduction == True:
            dy,_ = detrend(t=lct, y=lcy, dt=dt)
            dy_norm = dy / dy.std()
            mlcy = np.concatenate((mlcy, dy_norm), axis=0)
        else:
            mlcy = np.concatenate((mlcy, lcy), axis=0)
        t_c, y_c, N, _, _, _ = Fillpoint(t=(mlct-mlct[0]), y=mlcy, dt=dt)
    return t_c, y_c, dt, N
        
class Ws(object):
    def __init__(self, obsids, f1=1e-3, f2=15e-3, nf=800, sigma=10):
        f = np.linspace(f1, f2, nf)
        t_c, y_c, dt, N = readmulti(obsids=obsids)
        w = wavelet_spec(y=y_c, f=f, sigma=sigma, dt=dt)
        self.t_c = t_c
        self.y_c = y_c
        self.f = f
        self.power = w.power
        self.fftfreqs = w.fftfreqs
        self.fftpower = w.fft_power
        self.coi = w.coi 

    def plot(self):
        power  = self.power
        fftfreqs = self.fftfreqs
        fftpower = self.fftpower
        coi = self.coi 
        f = self.f
        for i5 in range(len(power[0])):
            _int = np.where(f < 1/coi[i5])
            power[:,i5][_int] = np.nan
        t_c = self.t_c
        y_c = self.y_c
        f = self.f

        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax

        ax[0].plot(t_c, y_c)
        ax[1].contourf(t_c, f, power, cmap=plt.cm.viridis)
        ax[0].set_ylabel('Counts/s')
        fig.subplots_adjust(hspace=0.05)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')

def plot_pulsefrac(obsid, f0, bnum=10):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    lc = fits.open(glob.glob(path2+"/"+obsid+"/stdprod/*_n2a.lc.gz")[0])
    # lc = fits.open(glob.glob(path2+"/"+obsid+"/stdprod/*_n1.lc.gz")[0])
    t = lc[1].data['TIME']
    y = lc[1].data['RATE']
    e = lc[1].data['ERROR']
    dt = t[1] - t[0]
    _int = np.where(np.isnan(y) == False)
    t = t[_int]
    y = y[_int]
    Ncfrac = (f0*(t - t[0])-.4)%1
    _dur = 1 / (f0 * bnum)
    _bin = 1 /bnum
    pbin = np.zeros(bnum)
    ybi = np.zeros(bnum)
    ebi = np.zeros(bnum)
    for i in range(bnum):
        _index = np.where(((i*_bin)<=Ncfrac)&(Ncfrac<((i+1)*_bin)))
        ybi[i] = sum(y[_index]) / len(y[_index])
        ebi[i] = np.sqrt(sum(e[_index]**2)) / len(e[_index])
        pbin[i] = (i*_bin + (i+1)*_bin)/2
    mean = ybi.mean()
    meanerr = np.sqrt(sum(ebi**2))/len(ybi)
    rms = np.sqrt(sum((ybi - mean)**2)/(bnum-1)) / mean
    pbin = np.concatenate((pbin, pbin + 1), axis=0)
    ybi = np.concatenate((ybi, ybi), axis=0)
    ebi = np.concatenate((ebi, ebi), axis=0)
    serr = np.sqrt( sum((ybi-mean)**2*ebi**2)/(len(ybi)-1)/sum((ybi-mean)**2)   )
    _rmserr = np.sqrt( 1/mean**2*serr**2 + rms/mean**2*meanerr**2  ) 
    fig, ax = plt.subplots()
    ax.errorbar(pbin, ybi, ebi, fmt='k', drawstyle='steps-mid', capsize=4)
    ax.set_ylabel('Counts/s', fontsize=15)
    ax.set_xlabel('Phase', fontsize=15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    plt.tight_layout()
    return rms, _rmserr

def plot_wavelet(f0=6):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()
    m = Morlet(f0)
    t = np.arange(-3, 3, 0.01)
    ax.plot(t, m.psi(t), 'k', label='Re($\psi$)')
    ax.plot(t, m.psi(t).imag, 'k--', label='Im($\psi$)')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='5'))
    ax.set_xlabel('t', fontsize=15)
    ax.set_ylabel('$\psi$(t)', fontsize=15)
    ax.legend()
    plt.tight_layout()

def plot_whole_lc(o=None, b=None):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.figure(figsize=(10,6))
    if o == None:
        o = minbar.Observations()
    if b == None:
        b = minbar.Bursts()
    o.instr('PCA')
    o.name_like('XTE J1701-462')
    b.name_like('XTE J1701-462')
    btime = b['time'][1:4]
    allo = o.get_records()
    MJD_s = 53754
    l_qpos = [105, 162, 388, 780]
    qpo_id = ['03-06', '09-09', '29-06', '63-08']
    _int = np.where( ((allo['tstart']-MJD_s) < 500) & (allo['flux'] > 0.5) )[0]
    _int2 = np.where((allo['tstart']-MJD_s) > 500)[0]
    _int3 = np.concatenate((_int, _int2), axis=0)
    s = S.load('/home/kaho/mhz_QPOs_search_in_minbar/J1701-462/all_detections.gz')
    MJDs = s['tstart']
    lum = 10.20232346
    plt.plot((allo['tstart'][_int3][:-3]-MJD_s), allo['flux'][_int3][:-3], 'k.')
    # plt.errorbar((allo['tstart'][:-3]-MJD_s), allo['count'][:-3],  yerr=allo['e_count'][:-3], fmt='k.')
    for j,i in enumerate(l_qpos):
        plt.annotate(qpo_id[j], xy=(allo['tstart'][i]-MJD_s, max(allo['flux']*.55)), xycoords='data',
            xytext=(allo['tstart'][i]-MJD_s, max(allo['flux']*.65)), textcoords='data',
            va='bottom', ha='center',
            arrowprops=dict(arrowstyle='->', color='r'), size=10, color='r')
    for k in MJDs:
        plt.annotate("", xy=((k-MJD_s), max(allo['flux']*.75)), xycoords='data',
                     xytext=(k-MJD_s, max(allo['flux']*.85)), textcoords='data',
            va='bottom', ha='center', arrowprops=dict(arrowstyle='->', color='k'), size=10)
    for j in btime:
        plt.axvline(j-MJD_s,  c='k', ls='--')
    plt.axhline(lum, c='r', ls='--')
    plt.xlabel(f"Days since 2006 January 19 (MJD {MJD_s})", fontsize=15)
    plt.ylabel('Mean flux $10^{−9}\;\mathrm{erg}\;\mathrm{cm}^{−2}\;\mathrm{s}^{-1}$', fontsize=15)
    plt.tight_layout()

def spec(obsid = None, t=None, y=None):
    if obsid:
        lc = fits.open(glob.glob(path2+"/"+obsid+"/stdprod/*_s2a.lc.gz")[0])
        # lc = fits.open(glob.glob(path2+"/"+obsid+"/stdprod/*_n2a.lc.gz")[0])
        # lc = fits.open(glob.glob(path2+"/"+obsid+"/stdprod/*_n1.lc.gz")[0])
        t1 = lc[1].data['TIME']
        y = lc[1].data['RATE']
    else:
        t1=t
        y=y
    if np.any(np.isnan(y)) == True:
        _int = np.where(np.isnan(y) == False)
        t1 = t1[_int]
        y = y[_int]
    t = t1 - t1[0]
    dt = t1[1] - t1[0]
    yd, p = detrend(t=t, y=y, plot=False)
    raw_p = abs(np.fft.fft(yd*dt))**2
    ny_fre = len(raw_p)//2
    norm_p = (raw_p*2/sum(y*dt))[1:ny_fre]
    re = 1 / (dt*len(y))
    f = np.arange(1, len(norm_p)+1)*re

    ym = yd + y.mean()
    if np.any(ym < 0):
        spec=[None]
    else:
        spec = Genspec(t=t, y=ym)

    return f, norm_p, spec[0], t, y

def plot_combine_spec(obsid = [obsid1, obsid4]):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    col = ["k","r"]
    fig, ax = plt.subplots()
    for i,j in zip(obsid, col):
        f, norm_p, _ = spec(obsid=i)
        ax.plot(f*1e3, norm_p, j, drawstyle='steps-mid', label=i, color=j)
    ax.set_ylabel('Leahy Power', fontsize=15)
    ax.set_xlabel('frequency (mHz)', fontsize=15)
    ax.legend(fontsize=15)
    plt.tight_layout()

def luminosity(obsid = obsid1, o=None):
    b = minbar.Bursts()
    b.name_like('J1701-462')
    o.clear()
    if o == None:
        o = minbar.Observations()
    o.obsid(obsid)
    flux = o.get('flux')
    e_flux = o.get('e_flux')
    b.create_distance_correction()
    dist = b['distcor'][0]
    e_dist = b['distcore'][0]
    lum = (flux*(b['distcor'][0])).to('erg s-1')
    e_lum = (e_flux*(b['distcor'][0])).to('erg s-1')
    # e_lum = lum.base * np.sqrt(((e_flux/flux)**2).base + ((e_dist/dist)**2).base)
    return lum, e_lum

def joint_lc(obsids=None, o=None):
    if o == None:
        o = minbar.Observations()
    o.clear()
    for i,j in enumerate(obsids):
        lc = fits.open(glob.glob(path2+"/"+j+"/stdprod/*_n2a.lc.gz")[0])
        t1 = lc[1].data['TIME']
        y1 = lc[1].data['RATE']
        print(j, t1[0])
        if i == 0:
            ts = t1[0]
            t = t1 - ts
            y = y1
        else:
            t = np.concatenate((t, t1-ts), axis=0)
            y = np.concatenate((y, y1), axis=0)

    return t, y

def plot_wspec(obsid=''):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    obsid1 = "92405-01-03-06"
    obsid2 = "92405-01-09-09"
    obsid3 = "92405-01-29-06"
    obsid4 = "92405-01-63-08"  
    if obsid == obsid1:
        s = S.load('/home/kaho/mhz_QPOs_search_in_minbar/J1701-462/92405-01-03-06_sigma=10_all.gz')
    elif obsid == obsid2:
        s = S.load('/home/kaho/mhz_QPOs_search_in_minbar/J1701-462/92405-01-09-09_sigma=10.gz')
    elif obsid == obsid3:
        s = S.load('/home/kaho/mhz_QPOs_search_in_minbar/J1701-462/92405-01-29-06_sigma=10.gz')
    elif obsid == obsid4:
        s = S.load('/home/kaho/mhz_QPOs_search_in_minbar/J1701-462/92405-01-63-08_sigma=10.gz')
    s.plot_wspec()
    s.fig.set_size_inches(6.4, 5)
    s.fig.suptitle('')
    s.ax[0].get_lines()[0].set_color("black")
    s.ax[1].set_ylim(None,7e-3)
    # s.ax[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    s.ax[1].set_yticklabels((s.ax[1].get_yticks()*1000).astype(int))
    # s.ax[1].yaxis.set_minor_locator(MaxNLocator(nbins=4))
    s.ax[1].set_ylabel('Frequency (mHz)', fontsize=15)
    s.ax[1].xaxis.get_label().set_fontsize(15)
    s.ax[0].yaxis.get_label().set_fontsize(15)
    s.fig.tight_layout()