import numpy as np
import stingray
import matplotlib.pyplot as plt

class simLC(object):
    def __init__(self, t=None, y=None, dt=None, input_counts=False, norm='None'):
        self.norm = norm
        if dt is None:
            dt = t[1] - t[0]       
        lc = stingray.Lightcurve(t, y, input_counts=input_counts)
        spec = stingray.Powerspectrum(lc, norm=norm)   
        spec.power = abs(spec.power)
        self.spec_power = spec.power 
        self.fre = spec.freq

    def plot_spec(self):
        fig, ax = plt.subplots()
        ax.plot(self.fre, self.spec_power, ds='steps-mid')
        plt.xscale('log')
        plt.yscale('log')
        if self.norm == 'None':
            plt.ylabel('Abs power')
        else:
            plt.ylabel(self.norm + ' power')
        plt.xlabel('Frequency (Hz)')
        plt.show()