
import numpy as np
import pylab as plt
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

wC = 0
wQD = 0
w = np.linspace(-1, 1, 100000)*20e9

#Edo Waks
k = 35.9e9
g = 10.2e9 
y = 2.9e9
r = 1 - (k*(1j*(wQD-w)+y/2.))/((1j*(wQD-w)+y/2.)*(1j*(wC-w)+k/2.)+g**2)
r0 = 1 - (k*(1j*(wQD-w)+y/2.))/((1j*(wQD-w)+y/2.)*(1j*(wC-w)+k/2.))
fom_waks = np.cos(np.angle (r/r0))

#John Rarity
k = 1000e9
g = 5.6e9 
y = 70e6
r = 1 - (k*(1j*(wQD-w)+y/2.))/((1j*(wQD-w)+y/2.)*(1j*(wC-w)+k/2.)+g**2)
r0 = 1 - (k*(1j*(wQD-w)+y/2.))/((1j*(wQD-w)+y/2.)*(1j*(wC-w)+k/2.))
fom_rarity = np.cos(np.angle (r/r0))

plt.figure(figsize=(15,5))
plt.plot (w*1e-9, np.angle (r), 'crimson', linewidth=2)
plt.plot (w*1e-9, np.angle (r0), 'k', linewidth = 2)
plt.show()

plt.figure(figsize=(15,5))
plt.plot (w*1e-9, fom_waks, 'RoyalBlue', linewidth = 3)
plt.plot (w*1e-9, fom_rarity, 'crimson', linewidth = 3)
plt.xlabel ('detuning [GHz]', fontsize=20)
plt.ylabel ('cos[phi_R-phi_L]', fontsize=20)
plt.show()