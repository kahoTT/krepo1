from re import L
from astropy.time.utils import twoval_to_longdouble
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pycwt 
import mc_sim
import time as T
import minbar
import stingray
import glob
minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'

# Normalized light curves and fill spaces with zeors
# The normalization is applied to the whole lightcurve, even wavelet spectrum could be done seperately
# Examples from 1636 is obsid('60032-05-02-00') and 1608 obsid('10072-05-01-00') and EXO 0748 obsid('90039-01-03-05')
# 1323 obsid('96405-01-02-01') but with negative values
# 1608 obsid('10072-05-01-00') Revnivtsev et al. 2000
# 4U 1608-52 obsid: 10094-01-01-00 Revnivtsev et al. 2000
# Aql X-1 "20098-03-08-00" Revnivtsev et al. 2000

# This class is to fill the gap data with mean value, will change to fill with the fitted polynomail vales
def detrend(t=None, y=None, dt=None, plot=False, dof=2):
    if dt is None:
        dt = t[1] - t[0]
    p = np.polyfit(t, y, dof) # value is the d.o.f
    if plot == True:
        plt.plot(t,y)
        plt.plot(t, np.polyval(p, t))
        plt.show()
    dat_notrend = y - np.polyval(p, t)
    return dat_notrend, p

# class sim(mc_sim.simLC): # Main purpose of this class is to divide lightcurve into different sections and being put to another simulation module
#     def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
#         # see if there any large data gaps. If so, have to simulate them one by one
#         if dt is None:
#             dt = t[1] - t[0]
#         super().__init__(t, y, dt, input_counts, norm)
#         self.lct = t
#         self.lcy = self.counts
            
class wavelet_spec(object):
    def __init__(self, y, f, sigma, dt, powera=None):
        mother = pycwt.Morlet(sigma)
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.wavelet.cwt(y, dt, wavelet=mother, freqs=f)
        power = (np.abs(wave)) ** 2
        fft_power = np.abs(fft) ** 2
        Liu_power = power / scales[:, None]
        if powera is None:
            _pow = power
        elif powera == 'Liu':
            _pow = Liu_power
        self.power = _pow
        self.fftfreqs = fftfreqs
        self.fft_power = fft_power
        self.coi = coi

def Slice(t, gap=400):
    res = [(sub2 - sub1 > gap) for sub1, sub2 in zip(t[:-1], t[1:])] # the value should be based on something? 
    if np.any(res) == True:
        l_ag = np.concatenate(([-1], (np.where(res))[0]), axis=0)
        slices = [slice(a0+1, a1+1) for a0, a1 in zip(l_ag[:-1], l_ag[1:])]
        slices.append(slice(l_ag[-1]+1, None))
    else:
        slices = [slice(None)] # have problems for different observations
    return slices 


class analysis(object):
    """
    argument:_re is minbar table
    """
    number_obs = 1e5
    total_sims = int(1 / (1-.9973) * number_obs) + 1
    def __init__(self, t=None, y=None, filename=None, dt=None, obsid=None, name=None, kepler=None, f=None, f1=1.5e-3, f2=12e-3, nf=500, sims=60, sigma=10, _re=None, b=None, o=None, ng=None, norm_f=True, _5sigma=False):
        if b is None:
            b = minbar.Bursts()
        if _re:
            obsid = _re['obsid']
        start_time = T.time()
        if t is not None and y is not None:
            t1 = t
            pass
        elif filename:
            if kepler == True:
                import lcdata
                lc = lcdata.load(filename)
                t = lc.time
                y = lc.xlum
            else:
                lc = fits.open(filename)
                t1 = lc[1].data['TIME']
                t = t1 - t1[0]
                y = lc[1].data['RATE']
        elif obsid:
            b.clear()
            b.obsid(obsid)
            ifb = b.get('obsid')
            if o is None:
                o = minbar.Observations()
            o.clear()
            o.obsid(obsid)
            name = o.get('name')[0]
            obs = minbar.Observation(o[o['entry']]) 
            # _path = obs.instr.lightcurve(obsid)
            # 16s resolution lightcurve
            lc = fits.open(glob.glob(obs.get_path()+'/stdprod/*_n2a.lc.gz')[0])
            # lc = fits.open(glob.glob(obs.get_path()+'/stdprod/*_s2a.lc.gz')[0])
            # 1s resolution lightcurve
            # lc = fits.open(glob.glob(obs.get_path()+'/stdprod/*_n1.lc.gz')[0])
            t1 = lc[1].data['TIME']
            y = lc[1].data['RATE']
            t = t1 - t1[0]
            dt = t1[1] - t1[0]
        else:
            raise AttributeError(f'give me a light curve')

        if np.any(np.isnan(y)) == True:
            print('data cleaning: arrays contain nan data')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]
            print('data cleaning: nan data are clean')
        else:
            print('data cleaning: No nan data')

        self.bg = 'No' 
        if any(y < 0):
        # skip or continue lightcurves with negatives
            if ng == True: # continue
                bglc = fits.open(obs.get_path()+'/bkg_0.125s.lc')
                bg = bglc[1].data['RATE']
                y = y + bg[_int]
                self.bg = bg
            else:
                self.bg = None
                print('Lightcurve has negative values')
                return 

        if np.any(np.isnan(y)) == True:
            print('data cleaning: arrays contain nan data after background correction')
            _int = np.where(np.isnan(y) == False)
            t = t[_int]
            y = y[_int]

# dealing with bursts
        if not obsid:
            tnb = t
            ynb = y
            name = None
        elif (len(np.where(ifb == obsid)[0]) == 0):
            print('data cleaning: No bursts in this observation')
            tnb = t
            ynb = y
        else:
            print(str(len(b.get('bnum'))) +' bursts on this observation')
            obs.get_lc()
            bursttime = (obs.bursts['time'] - obs.mjd.value[0])*86400
            bst = bursttime - 5
            bet = bst + obs.bursts['dur'] * 4 # scaling the time of the duration
            barray = []
            nbarray = []
            a1 = None
            for i in range(len(b.get('bnum'))): # extract burst data and non-burst data
                a = list(abs(t-bst[i])).index(min(abs(t - bst[i])))
                _a = list(abs(t-bet[i])).index(min(abs(t - bet[i])))
                if dt-16 < 1e-5:
                    a = a-1
                    _a = _a+2
                barray.extend(np.r_[a:_a])
                if i == 0: 
                    if a != 0: 
                        nbarray.extend(np.r_[a1:a])
                    else: # for the case of starting in the middle of a burst
                        pass
                else:
                    nbarray.extend(np.r_[a1:a])
                a1 = _a + 1
            if _a == len(t) - 1: # for the case of ending in the middle of a burst
                pass
            else:
                nbarray.extend(np.r_[a1:len(t)])
            self.tb = t[barray]
            self.yb = y[barray]
            tnb = t[nbarray]
            ynb = y[nbarray]
            self.bursttime = bursttime

# divide the light curve  

        tnb_s = []
        ynb_s = []
        for j in Slice(tnb):
            tnb_s.append(tnb[j])
            ynb_s.append(ynb[j])
        self.tnb_s = tnb_s
        self.ynb_s = ynb_s
        ltnb = len(tnb_s)
        self.ltnb = ltnb

        self.obsid = obsid
        self.t = t
        self.y = y
        self.tstart = t1[0]
        self.tnb = tnb
        self.ynb = ynb
        self.name = name
        if not dt:
            dt = t[1] - t[0]
        self.dt = dt
        if f is None:
            f = np.linspace(f1, f2, nf)
        else:
            f = np.array([f])
        self.f = f

        tc = []
        _powall = []
        _npowall = []
        nops = []
        specl = []
        o_modell = []
        resultl = []
        if sims:
            lsigma3 = [] # a list to store all thresholds
            accsynpl = [] # a list of the nth largest power in n simulations
            simsl = [] # a list contain a tuple of (number of simulations, and number of fails)

        for i2 in range(ltnb): # tnb is a list 
            t_c, _, n_of_data, factor, dt, ares  = mc_sim.Fillpoint(t=tnb_s[i2], y=ynb_s[i2], dt=dt)
            tc.append(t_c)
            spec, logspec = mc_sim.Genspec(t=tnb_s[i2], y=ynb_s[i2], dt=dt)
            result, o_model, n_model, norm_fr, guess = mc_sim.Powfit(freq=spec.freq, f=spec.freq, y=spec.power, wf=f, rebin_log=False, factor=factor)
            dat_notrend, _ = detrend(tnb_s[i2], ynb_s[i2], dt=dt)
            _, y_c, _, _, _, _  = mc_sim.Fillpoint(t=tnb_s[i2], y=dat_notrend, dt=dt)
            rws = wavelet_spec(y=y_c, f=f, sigma=sigma, dt=dt)
            specl.append(spec)
            o_modell.append(o_model)
            resultl.append(result)
            # drop the normalisation to spectrum

            if norm_f is True:
                rpower = rws.power / norm_fr[:, np.newaxis]  # dealing with extra [] for 1D f array  
                # breakpoint()
            else:
                rpower = rws.power
            for i5 in range(len(rpower[0])):
                _int = np.where(f < 1/rws.coi[i5])
                rpower[:,i5][_int] = np.nan
            _ind2, _ = np.where(np.isnan(rpower) == False)
            nop = len(_ind2)
            nops.append(nop)
            """Simulation, the number of sims means the number of simulations."""
            if sims:
                i3 = 1
                fi = 0
                if _5sigma == True:
                    sims = int(3.5e6 // nop + 1)
                    # sims = int(1e6 // nop + 1)
                    # sims = self.total_sims // (len(f) * len(t_c)) + 1
                accsynp = []
                while i3 <= sims:
                    # if fi > sims:
                        # raise Exception("too many fails in simulations") 
#                    testtime = time.time() - start_time
                    time, counts = mc_sim.simlc(ares=ares, t=tnb_s[i2], y=ynb_s[i2], dt=dt, N=n_of_data, red_noise=1, o_model=o_model, n_model=n_model, model='n')
                    specs, logspecs = mc_sim.Genspec(t=time, y=counts, dt=dt)
                    results, o_models, _, norm_fs, guesss = mc_sim.Powfit(freq=specs.freq, f=specs.freq, y=specs.power, wf=f, rebin_log=False, factor=factor, exclude=False)
                    if (guesss >= 2*guess) or (guesss <= 0.5*guess):
                        fi += 1
                        continue
                    sdat_notrend, _ = detrend(time, counts, dt=dt) # fill class
                    _, sy, _, _, _, _  = mc_sim.Fillpoint(t=tnb_s[i2], y=sdat_notrend, dt=dt)
                    ws = wavelet_spec(y=sy, f=f, sigma=sigma, dt=dt)
                    if len(f) == 1: 
                        # Case when using single frequency
                        if norm_f is True:
                            norm_pow = ws.power[0] / norm_fs # dealing with extra [] for 1D f array  
                        else:
                            norm_pow = ws.power[0]
                        _int = np.where(f > 1/ws.coi)
                        synp = norm_pow[_int]
                    else:
                        if norm_f is True:
                            norm_pow = ws.power / norm_fs[:, np.newaxis]  
                            # breakpoint()
                        else:
                            norm_pow = ws.power
                        for i4 in range(len(norm_pow[0])):
                            _int = np.where(f < 1/ws.coi[i4])
                            norm_pow[:,i4][_int] = np.nan
                        norm_pow_non = norm_pow.reshape(1, norm_pow.size)[0][~np.isnan(norm_pow.reshape(1, norm_pow.size)[0])]
                    if i3 == 1:
                        synp = norm_pow
                        all_norm_pow_non = norm_pow_non
                    else:
                        synp = np.concatenate((synp, norm_pow), axis=1)
                        all_norm_pow_non = np.concatenate((all_norm_pow_non, norm_pow_non), axis=0)
                    accsynp.append(((i3),np.sort(all_norm_pow_non)[-int(i3)]))
                    i3 += 1
                synpall = synp.reshape(1, synp.size)[0]
                _int2 = np.isnan(synpall)
                synpall = np.sort(synpall[~_int2])
                if _5sigma == True:
                    sigma3 = synpall[-1]
                else:
                    sigma3 = synpall[-sims] # take the nth element having n time simulations
                _pow = rpower
                _npow = rpower / sigma3
                _powall.append(_pow)
                _npowall.append(_npow)
                lsigma3.append(sigma3)
                accsynpl.append(accsynp)
                simsl.append((sims, fi))
            else:
                _powall.append(rpower)
                lsigma3 = None
                synpall = None
                _npowall = None
            self.sigma = lsigma3
            self.synpall = synpall
            self.p = _powall
            self.np = _npowall
            self.tc = tc
            if ltnb == 1:
                self.specl = specl[0]
                self.o_mdell = o_modell[0]
                self.fit_model = resultl[0]
                self.nops = nops[0]
                if sims:
                    self.accsynpl = accsynpl[0]
                    self.simsl = simsl[0]
            else:
                self.specl = specl
                self.o_mdell = o_modell
                self.fit_model = resultl
                self.nops = nops
                if sims:
                    self.accsynpl = accsynpl
                    self.simsl = simsl
            self.logspec = logspec
        self.finish_time = T.time() - start_time
        print(f'Finish time = {self.finish_time}')

# Plot without burst
    def plot_nob(self): # put self arguments
        plt.plot(self.tnb, self.ynb)
        if self.ltnb > 1:
            for s in range(self.ltnb):
                plt.plot(self.tnb_s[s],self.ynb_s[s], alpha=0.5)
        plt.ylabel('Counts/s')
        plt.xlabel('Time (s)')
        plt.show()

 
# plot only burst
    def plot_b(self):
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.plot(self.tb, self.yb, 'r.')
        plt.ylabel('Counts/s', fontsize=15)
        plt.xlabel('Time (s)', fontsize=15)
        plt.tight_layout()
        plt.show()
 
    def plot_lc(self,
                astart = None,
                aend = None,
                tstart = None,
                tend = None
               ):
        ii = slice(astart, aend)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.plot(self.t[ii], self.y[ii],'.')
        plt.ylabel('Counts/s', fontsize=15)
        plt.xlabel('Time (s)', fontsize=15)
        plt.tight_layout()
        plt.show()

    def plot_wspec(
            self,
            astart = None,
            aend = None,
            tstart = None,
            tend = None,
            savef_path = None,
            ):
    
        if savef_path:
            plt.ioff()
        fig, ax = plt.subplots(2, sharex=True)
        self.fig = fig
        self.ax = ax
        f = self.f 
        t = self.tnb_s
        y = self.ynb_s
        tnb = self.tnb
        ynb = self.ynb
        dt = self.dt
        ta = self.t
        ya = self.y
        p = self.p
        tc = self.tc
        sigma = self.sigma
        np = self.np

        ax[0].plot(tnb, ynb)
        for i in range(self.ltnb): 
            if np:
                ax[1].contourf(tc[i], f, np[i], cmap=plt.cm.viridis)
                ax[1].contour(tc[i], f, np[i], [1], colors='k')
            else:
                ax[1].contourf(tc[i], f, p[i], cmap=plt.cm.viridis)


        # vmin = 0
        # vmax = 0
        # if vmax >= np.max(norm_pow):
        #     pass
        # else:
        #     vmax = np.max(norm_pow)
        # cm = ax[1].contourf(_f.tc, f, norm_pow, cmap=plt.cm.viridis, vmin=vmin, vmax=(int(vmax)+1))
        # ax[1].contourf(tc, f, p, cmap=plt.cm.viridis)

        ax[0].set_ylabel('Counts/s')
        fig.subplots_adjust(hspace=0.05)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Frequency (Hz)')
        fig.suptitle(f'{self.name} obsid: {self.obsid}')
        if savef_path:
            plt.savefig(savef_path+'/wavelet_spec.png')
#        fig.colorbar(cm, ax=ax)
### norm_pow may need to modity, as this only has the elements for the last Ekoop
