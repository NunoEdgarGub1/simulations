
import numpy as np
import pylab as plt
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

c1 = 0.5
c2 = 0.5

gamma = 0.5
kappa = 1

plt.figure (figsize = (8,5))

for g in [0.01, 0.5, 5]:
    
    a1 = -0.5*(0.5*gamma+0.5*kappa)+0.5*((0.5*gamma+0.5*kappa)**2-4*g**2)**0.5
    a2 = -0.5*(0.5*gamma+0.5*kappa)-0.5*((0.5*gamma+0.5*kappa)**2-4*g**2)**0.5

    t = np.linspace (0, 5, 1000)
    wf = c1*np.exp(a1*t)+c2*np.exp(a2*t)

    prob = np.abs (wf)**2

    plt.plot (t, prob, linewidth =2)
    
plt.ylabel ('Prob [|e>]', fontsize = 18)
plt.xlabel ('time (a.u.)', fontsize = 18)
plt.show()