
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

mpl.rc('xtick', labelsize=22) 
mpl.rc('ytick', labelsize=22)

R_list = np.array([0.8, 0.9, 0.98])
#Flist = [10, 25, 100]
Flist = np.pi*R_list**0.5/(1-R_list)
print Flist

L = 1e-6
n = 2.5

lam = 1e-9*np.linspace (600, 900,10000)
phi = 4*np.pi*n*L/lam

colormap = plt.get_cmap('YlGnBu')
color_list = [colormap(k) for k in np.linspace(0.4, 1, len(Flist))]


plt.figure (figsize = (15,6))
i = 0
for F in Flist:
	T = 1./(1.+(4*F**2/(np.pi**2))*(np.sin(phi/2.)**2))

	plt.plot (lam*1e9, T, color = color_list [i], linewidth=3)
	i += 1

plt.xlabel ('wavelength [nm]', fontsize = 22)
plt.ylabel ('transmission', fontsize = 22)
plt.savefig ('C:/Users/cristian/Research/__curr_work/__teaching/Nanophotonics/cristian_2017/farby_perot.svg')
plt.show()