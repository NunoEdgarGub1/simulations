
import numpy as np
from matplotlib import pyplot as plt

class BootStrap ():

	def __init__ (self, n_boots):
		self.y = None
		self.bootstrap_y = None
		self.n_boots = n_boots
		self._N = -1

	def set_y (self, y):
		if (len(y)>0):
			self.y = y
			self._N = len(self.y)
		else:
			print ('y must be a valid array!')

	def run_bootstrap (self):

		if (self._N > 0):
			self.bootstrap_y = np.zeros((self.n_boots, self._N))
			self.yB = np.zeros(self.n_boots)
			self.std_B = np.zeros(self.n_boots)

			for alpha in np.arange(self.n_boots):
				ind = np.random.randint (0, self._N, self._N)
				self.bootstrap_y [alpha, :] = self.y[ind]
				self.yB [alpha] = np.mean (self.y[ind])
				self.std_B [alpha] = np.std(self.y[ind])

			plt.plot (self.std_B)
			plt.ylabel ('holevo var bootstrap')
			plt.show()

			self.mean_bootstrap = np.mean (self.yB)
			self.std_bootstrap = np.mean(self.std_B)
			self.err_std_bootstrap = np.std (self.std_B)

		else:
			print ('Unspecified input data!')

	def run_bootstrap_holevo (self):

		if (self._N > 0):
			self.bootstrap_y = np.zeros((self.n_boots, self._N)) + 1j*np.zeros((self.n_boots, self._N))
			self.yB = np.zeros(self.n_boots)
			self.hB = np.zeros(self.n_boots)

			for alpha in np.arange(self.n_boots):
				ind = np.random.randint (0, self._N, self._N)
				self.bootstrap_y [alpha, :] = self.y[ind]
				self.hB [alpha] = np.abs(np.mean(self.y[ind]))**(-2)-1

			self.meanH_bootstrap = np.mean(self.hB)
			self.errH_bootstrap = np.std (self.hB)

		else:
			print ('Unspecified input data!')



	def print_results (self):
		print ('Bootstrap over '+str(self.n_boots)+' resamples')
		print ('Mean: '+str(self.mean_bootstrap)+ '  StDev: '+str(self.std_bootstrap))
		print ('Error on StDev: '+str(self.err_std_bootstrap))

	def histogram (self):
		plt.figure()
		plt.hist(self.std_B)
		plt.xlabel ('bootstrap std')
		plt.show()





