import matplotlib.pyplot as plt

class IonColor(object):
	def __init__(self, n=20):
		self.n = n

# 'Set1' 'gist_rainbow' 'tab20', 'tab10'
	def __call__(self, ion):
		cm = plt.get_cmap('tab20', self.n)
		idx = 47 * ion.Z + ion.A
		return cm(idx % self.n)