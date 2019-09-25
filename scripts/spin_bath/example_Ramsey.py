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

clus_size=3
B = 1e-6

print ("------- CLUSTER SIZE: ", clus_size)
print("------ MAGNETIC FIELD: ", B, "Tesla")

sig = []
tau = []

exp = NBath.CentralSpinExp_cluster (nr_spins = 7, auto_save = True)
exp.set_workfolder (r'/Users/eleanorscerri/Desktop')
exp.set_cluster_size(g=clus_size)

exp.set_thresholds(A=500e3, sparse=10)
exp.set_magnetic_field(Bz=B, Bx=0, By=0)
exp.set_cluster_size(g=3)
exp.generate_bath(concentration=0.011)

for t in np.linspace(0,2e-5,100):
    sig.append(exp.Ramsey_clus(t,0))
    tau.append(t*1e3)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.plot(tau,.5*(1-np.array(sig)))
plt.show()


# In[ ]:




