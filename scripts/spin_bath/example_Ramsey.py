#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os, sys, logging
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

folder = '/Users/eleanorscerri/Documents/GitHub'
sys.path.append (folder)

from simulations.libs.spin import nuclear_spin_bath as NBath
reload (NBath)

clus_size=5
B = 1e-6

print ("------- CLUSTER SIZE: ", clus_size)
print("------ MAGNETIC FIELD: ", B, "Tesla")

sig = []
tau = []

exp = NBath.CentralSpinExp_cluster (nr_spins = 25, auto_save = True)
#exp.set_workfolder (r'/Users/eleanorscerri/Desktop')
exp.set_workfolder (r'C:/Users/cristian/Research/')
exp.set_cluster_size(g=clus_size)

exp.set_thresholds(A=500e3, sparse=10)
exp.set_magnetic_field(Bz=B, Bx=0, By=0)
exp.generate_bath(concentration=0.011)

tau = np.linspace(0,2e-5,15)
for t in tau:
    sig.append(exp.Ramsey_clus(t,0))

y_ind = exp.Ramsey_indep_Nspins (tau = tau)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.figure (figsize = (13,5))
plt.plot(tau*1e6,.5*(1-np.array(sig)), 'royalblue')
plt.plot (tau*1e6, 0.5*(1 - y_ind), 'o', color = 'crimson')
plt.show()


# In[ ]:




