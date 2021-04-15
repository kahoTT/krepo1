import numpy as np

class simLC(object):
    def __init__(self, lc, dt):
        c = lc * dt
        simlc = np.random.poisson(c) / dt
        self.lc = simlc
