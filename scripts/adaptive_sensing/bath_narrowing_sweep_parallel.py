
import os, sys, logging, time
import numpy as np
import multiprocessing
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

#folder = '/home/dalescerri/Documents/Git/GitTest'
folder = '/Users/dalescerri/Documents/GitHub'
sys.path.append (folder)

from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = 'Results/cluster/10spins/')
exp.set_log_level (logging.INFO)
logging.basicConfig (level=logging.INFO)
exp.set_sim_params (nr_reps=100)
exp.set_msmnt_params (N=7, G=3, F=1, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 7, concentration = .01)
exp.set_plot_settings (do_save=True, do_show=False, save_analysis = False)
exp.set_bath_validity_conditions (A = 1e6, sparse = 10)
exp.save_bath_evolution (False)

max_steps = 100
n_bath = 1
num_cores = 1


for j in range(n_bath):
    print('bath', j+1)
    nbath = exp.generate_bath()
		
    def bath_par (scheme):
        print ('RUN:',scheme)
        exp.set_msmnt_params (N=10, G=3, F=1, tau0=1e-6, fid0=1., fid1=0.)
        exp.alpha = 1
        exp.strategy = 'int'
        nbath.reset_bath_unpolarized()
        exp.simulate (funct_name = scheme, max_steps = max_steps,
        string_id = 'plusPi2_bath'+str(j), do_save = True)
        #exp.analysis (nr_bins=25)

        return None


    Parallel(n_jobs=num_cores)(delayed(bath_par)(scheme) for scheme in ['adaptive_1step_bon'])

