
import numpy as np
import random
from matplotlib import rc, cm
import os, sys
import h5py
import logging, time
import sys

from matplotlib import pyplot as plt
from scipy import signal


from analysis_simulations.libs.adaptive_sensing import adaptive_tracking as track_lib
from analysis_simulations.libs.adaptive_sensing import tracking_sequence as trackSeq

reload (track_lib)
#reload (track_libOH)
reload (trackSeq)

# total simulation time
time_interval = 5000e-6
# time required for each Ramsey for spin init and read-out (everything but sensing)
overhead = 10e-6

#dB = time variation of classical signal in MHz*(Hz)^0.5 [corresponding to kappa in Eq 12 in my PRA]
dfB = 10 #[=1MHz*(Hz^0.5)]

fixN = False
N = 5 #this will be over-written automatically later, if fixN = False
# k-th sensing time is repeated for Mk = G +f*(K-k) times
G = 5
F = 3
# nr of protocol repetitions (for statistics)
nr_reps = 1

# T2*
T2 = 100e-6
#Read-out fidelities for spin 0 and spin 1
fid0 = 1.
fid1 = 1.

track = True


# The folder we are passing here is defined when you run setup.py in the beginning.
# Please do not add any user-specific folder in the code

s = trackSeq.SequenceStatistics(folder = root_folder)

s.set_sim_params (reps=nr_reps, time_interval = time_interval, overhead = overhead)
s.set_msmnt_params (G = 5, F = F, dfB = dfB*1e6, N=4, tau0=20e-9, T2 = 100e-6, fid0=fid0, fid1=fid1)
s.G_adptv = G
s.nr_sensing_reps = 0
if fixN:
	s.set_msmnt_params (G = 5, F = F, dfB = dfB*1e6, N=N, tau0=20e-9, T2 = 100e-6, fid0=fid0, fid1=fid1)
else:
	s.automatic_set_N (track=track, dfB = dfB*1e6)
	print 'N = ', s.N

if track:
	s.calculate_fom_thresholds()
	s.fom_threshold = s.fom_array[2]
s.generate_simulation(track = track, do_save = False, description = '', 
					title = '',
					do_plot = True, do_debug = False)
