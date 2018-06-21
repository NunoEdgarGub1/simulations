
import os, sys
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = 'C:/Users/cristian/Research/Work-Data/')
exp.set_sim_params (nr_reps=1)
exp.set_msmnt_params (M=3, N=10, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 7, concentration = 0.01)
exp.simulate_different_bath (max_steps = 3, do_plot=True, do_debug = True, do_save=False)
#exp.analysis (nr_bins=50)
