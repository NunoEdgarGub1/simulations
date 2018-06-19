
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

while trialno < 10:
    '''
    If eng_bath==True (engineered bath), set nr_spins to a larger number that how many spins are required. 
    
    len(cluster) determines the nuber of 'seed' spins.
    The values in cluster denotes the number of neighbours selected from nr_spins spins based on proximity to the seed spins.
    
    '''
    exp = qtrack.TimeSequenceQ(time_interval=100e-6, overhead=0, folder=folder, trial=trialno)
    exp.set_spin_bath (cluster=np.zeros(7), nr_spins=7, concentration=0.01, verbose=True, do_plot = False, eng_bath=False)
    exp.set_msmnt_params (tau0 = 1e-6, t_read = 10e-6, T2 = exp.T2star, G=5, F=3, N=10)
    exp.initialize()
    
    if not exp.skip:      
        exp.qTracking (do_debug = True, M=3, nr_steps = Nstep)
        trialno+=1
