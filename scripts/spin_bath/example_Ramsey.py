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
B = 40e-3

print ("------- CLUSTER SIZE: ", clus_size)
print("------ MAGNETIC FIELD: ", B, "Tesla")

sig = []
tau = []

exp = NBath.CentralSpinExp_cluster (nr_spins = 25, auto_save = True)
exp.set_workfolder (r'C:/Users/cristian/Research/')
exp.set_cluster_size(g=clus_size)

exp.set_thresholds(A=400e3, sparse=10)
exp.set_magnetic_field(Bz=B, Bx=0, By=0)
exp.generate_bath(concentration=0.011)

t0 = 190e-6
t1 = 200e-6
tau = np.linspace (t0, t1, 1000)
y_ind = exp.dynamical_decoupling_indep_Nspins (S1=1, S0=0, tau = tau, nr_pulses = 32)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.figure (figsize = (13,5))
#plt.xlim (203, 204)
plt.plot (tau*1e6, y_ind, 'o', color = 'crimson')
plt.show()


# In[ ]:




