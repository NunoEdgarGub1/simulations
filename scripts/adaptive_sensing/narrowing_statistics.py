
import numpy as np
import random
from matplotlib import rc, cm
import os, sys
import h5py
import logging, time
import sys

from matplotlib import pyplot as plt

sys.path.append ('/Users/dalescerri/Documents/GitHub')

from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import tracking_sequence as trackSeq
from importlib import reload

reload (qtrack)
reload (trackSeq)

# total simulation time
time_interval = 5000e-6
# time required for each Ramsey for spin init and read-out (everything but sensing)
overhead = 0e-6

N = 5 #this will be over-written automatically later, if fixN = False
# k-th sensing time is repeated for Mk = G +f*(K-k) times
G = 5
F = 3
# nr of protocol repetitions (for statistics)
nr_reps = 1
trialno = 0
Nstep = 5

#Read-out fidelities for spin 0 and spin 1
fid0 = 1.
fid1 = 1.


track = True

folder = 'C:/'
max_steps = 10
nr_reps = 3
M=3

results = np.zeros((nr_reps, 2*M*max_steps+1))
i = 0

exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, folder=folder, trial=trialno)
exp.set_spin_bath (cluster=np.zeros(7), nr_spins=7, concentration=0.01, verbose=True, do_plot = False, eng_bath=False)
exp.set_msmnt_params (tau0 = 1e-6, T2 = exp.T2star, G=5, F=3, N=10)
exp.initialize()

while (i <nr_reps):

    print ("Repetition nr: ", i+1)
  
    if not exp.skip: 
        exp.reset_unpolarized_bath()  
        exp.initialize()   
        exp.adaptive_2steps (M=M, target_T2star = 5000e-6, max_nr_steps=max_steps, do_plot = True, do_debug = True)
        l = len (exp.T2starlist)
        results [i, :l] = exp.T2starlist/exp.T2starlist[0]
        i += 1

    else:
        exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, folder=folder, trial=trialno)
        exp.set_spin_bath (cluster=np.zeros(7), nr_spins=7, concentration=0.01, verbose=True, do_plot = False, eng_bath=False)
        exp.set_msmnt_params (tau0 = 1e-6, T2 = exp.T2star, G=5, F=3, N=10)
        exp.initialize()

print ('Processing results statistics...')
nr_bins = 100
res_hist = np.zeros((nr_bins+1, 2*M*max_steps+1))
bin_edges = np.zeros((nr_bins+1, 2*M*max_steps+1))

for j in range(2*M*max_steps+1):
    a = np.histogram(results[:nr_bins,j], bins=np.linspace (0, 15, nr_bins+1))
    res_hist [:len(a[0]), j] = a[0]
    bin_edges [:len(a[1]), j] = a[1]

[X, Y] = np.meshgrid (np.arange(2*M*max_steps+1), np.linspace (0, 15, nr_bins+1))

plt.pcolor (X, Y, res_hist)
plt.xlabel ('nr of narrowing steps', fontsize = 18)
plt.ylabel ('T2*/T2*_init', fontsize = 18)
plt.show()

# things to do:
# - save data for debugging
# - how do I reset the bath to unpolarized? It would be good to compare 
# different runs of the protocol on the same bath, in addition to 
# runs of the protocols on different baths
# - need to save some sort of movie that can help us identify problems



