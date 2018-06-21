
import os, sys
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics ()
exp.set_sim_params (nr_reps=100)
exp.set_msmnt_params (M=5, N=10, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 6, concentration = 0.01)
exp.simulate_different_bath (max_steps = 6)
exp.analysis (nr_bins=50)

# probably there's more info in the initial ramsey, than just T2*
# for example, can we see a double-peaked distribution?