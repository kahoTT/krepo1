import numpy as np
import stingray
import matplotlib.pyplot as plt

class TestLc(object):
	def __init__(self):
		t = np.arange(0, 1000, 0.1)
		y1 = np.random.uniform(-1, 1, size=len(t))
		y = y1 + 20
		lc = stingray.Lightcurve(t, y, input_counts=False, skip_checks=True)
		spec = stingray.Powerspectrum(lc, norm='None')   
		power = abs(spec.power)
		sum_pow = sum(power)
		self.t = t
		self.y = y
		self.freq1 = spec.freq
		self.pow1 = power

		fig, ax = plt.subplots(4, figsize=(8,10))
		self.fig = fig
		self.ax = ax
		ax[0].plot(t,y)
		ax[0].set_xticklabels([])
		ax[1].plot(spec.freq, power)
		ax[1].set_xscale('log')
		ax[1].set_yscale('log')
		ax[1].set_ylabel('Abs power')
		ax[1].set_title(f'Power Sum = {sum_pow:.2f}')
		

		y2 = y
		y2[1500:3000] = 20
		y2[3000:4500] = 20
		y2[8000:9000] = 20

		lc2 = stingray.Lightcurve(t, y2, input_counts=False, skip_checks=True)
		spec2 = stingray.Powerspectrum(lc2, norm='None')   
		power2 = abs(spec2.power)
		sum_pow2 = sum(power2)
		self.freq2 = spec2.freq
		self.pow2 = power2


		ax[2].plot(t,y2)
		ax[2].set_xticklabels([])
		ax[3].plot(spec2.freq, power2)
		ax[3].set_xscale('log')
		ax[3].set_yscale('log')
		ax[3].set_ylabel('Abs power')
		ax[3].set_title(f'Power Sum2 = {sum_pow2:.2f}, sum1/sum2 = {sum_pow/sum_pow2:.2f}')