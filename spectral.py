"""
Make periodogram of lightcurve
"""

import lcdata

from power import Power

class Spectral(Power):
    def __init__(self, filename):
        lc = lcdata.load(filename)
        self.time = lc.time
        self.data = lc.xlum
        super().__init__()
