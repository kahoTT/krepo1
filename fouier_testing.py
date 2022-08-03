# Testing power spectrum for light curves with gaps

import numpy as np
import stingray
import matplotlib.pyplot as plt
from mc_sim import simLC
import kwavelet

class TestLc(object):
	def __init__(self, t=None, y=None, input_data=True):
		if input_data is False:
			t = np.arange(0, 1000, 0.1)
			y1 = np.random.uniform(-1, 1, size=len(t))
			y = y1 + 20
		lc = stingray.Lightcurve(t, y, input_counts=False, skip_checks=False)
		spec = stingray.Powerspectrum(lc, norm='None')   
		power = abs(spec.power)
		sum_pow = sum(power)
		self.t = t
		self.y = y
		self.freq1 = spec.freq
		self.pow1 = power
		l = simLC(t=t, y=y)
		self.lmodel = l.lmodel

		fig, ax = plt.subplots(4, figsize=(8,10))
		self.fig = fig
		self.ax = ax
		ax[0].plot(t,y)
		ax[0].set_xticklabels([])
		ax[1].plot(spec.freq, power)
		ax[1].plot(spec.freq, l.lmodel)
		ax[1].set_xscale('log')
		ax[1].set_yscale('log')
		ax[1].set_ylabel('Abs power')
		ax[1].set_title(f'Power Sum = {sum_pow:.2f}')
		
		y2 = y
		y2[0:500] = y.mean()
#		y2[3000:4500] = 20
#		y2[8000:9000] = 20

		lc2 = stingray.Lightcurve(t, y2, input_counts=False, skip_checks=True)
		spec2 = stingray.Powerspectrum(lc2, norm='None')   
		power2 = abs(spec2.power)
		sum_pow2 = sum(power2)
		self.freq2 = spec2.freq
		self.pow2 = power2
		l2 = simLC(t=t, y=y2)
		self.lmodel2 = l2.lmodel


		ax[2].plot(t,y2)
		ax[2].set_xticklabels([])
		ax[3].plot(spec2.freq, power2)
		ax[3].plot(spec2.freq, l2.lmodel)
		ax[3].set_xscale('log')
		ax[3].set_yscale('log')
		ax[3].set_ylabel('Abs power')
		ax[3].set_title(f'Power Sum2 = {sum_pow2:.2f}, sum1/sum2 = {sum_pow/sum_pow2:.2f}')

# red_noise = 0, no exclude, use the model without any boosting seems to be the best, with problem occurs for both red_noise = 0 or 1 
class Test1613(object):
	def __init__(self, red_noise=1, testno = 5):
		w = kwavelet.analysis(obsid='60032-05-02-00', test=0)
		t2 = w.tnb
		y2 = w.ynb
		_int = np.where((t2> 18500) & (t2<20650))
#		_int = np.where(t2> 18500) 
		t = t2[_int]
		y = y2[_int]
		fig, ax = plt.subplots()
		ax.set_yscale('log')
		ax.set_xscale('log')
		for i in range(0,testno,1):
			s = simLC(t = t, y=y, exclude=False, red_noise=red_noise, model = 'o')
#			ax.plot(s.time, s.counts, label=f'{i}', alpha = 0.6,)
			s2 = simLC(s.time, s.counts, exclude=False, red_noise=red_noise, model = 'o', gen=False)
			if i == 0:
				a = s2.omodel
			else:
				b = s2.omodel
				a = np.vstack((a,b))
#			ax.plot(s2.freq, s2.pow, label=f'{i}', ds='steps-mid')
		print(np.mean(a, axis=0))
		ax.plot(s.freq, s.omodel, label='real data')
		ax.plot(s.freq, s.omodel, 'r.', label='real data')
		ax.plot(s2.freq, np.mean(a, axis=0))
#		ax.legend()