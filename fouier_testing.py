import numpy as np
import stingray
import matplotlib.pyplot as plt

t = np.arange(0, 1000, 0.1)
y1 = np.random.uniform(-1, 1, size=len(t))
y = y1 - 20
lc = stingray.Lightcurve(t, y, input_counts=False, skip_checks=True)
spec = stingray.Powerspectrum(lc, norm='None')   
spec.power = abs(spec.power)
fig, ax = plt.subplots(2)
fig = fig
ax = ax
ax[0].plot(t,y)
ax[1].plot(spec.freq, spec.power)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Abs power')
plt.ylabel('Abs power')
plt.show()


