
import numpy as np
import pylab as plt
from matplotlib import cm

x = np.linspace (0, 7*6.28, 1000)
f = 0.5*np.linspace(0.97, 1.03, 9)

y_sum = np.zeros(1000)

col_idx = [0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2, 0.1]
colors = cm.PuBu(col_idx)
f1 = plt.figure(figsize=(13,5))
for i in np.arange (9):
	y = np.cos (f[i]*x)
	y_sum = y_sum + y
	plt.plot (x, 0.5*(1+y), color = colors[i], linewidth=2)
f1.savefig ('D:/Research/fluct_narrow.svg')
plt.show()

f2 = plt.figure(figsize=(13,5))
plt.plot (x, 0.5*(1+y_sum/9.), linewidth=5, color = 'RoyalBlue')
f2.savefig ('D:/Research/fluct_narrow_sum.svg')
plt.show()
