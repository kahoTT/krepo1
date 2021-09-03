import matplotlib.pyplot as plt

class IonColor(object):
	def __init__(self, n=10):
		self.n = n

# 'Set1' 'gist_rainbow' 'tab20'
	def __call__(self, ion):
		cm = plt.get_cmap('tab10', self.n)
		idx = 47 * ion.Z + ion.A
		return cm(idx % self.n)