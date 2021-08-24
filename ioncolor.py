import matplotlib.pyplot as plt

class IonColor(object):
	def __init__(self, n=23):
		self.n = n

	def __call__(self, ion):
		cm = plt.get_cmap('gist_rainbow', self.n)
		idx = 47 * ion.Z + ion.A
		return cm(idx % self.n)