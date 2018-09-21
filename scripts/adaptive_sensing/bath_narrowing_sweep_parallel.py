
import os, sys, logging, time
import numpy as np
import multiprocessing
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

#folder = '/Users/dalescerri/Documents/GitHub'
#sys.path.append (folder)

from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = '/home/qpl/Research/Work-Data/10spins')
exp.set_log_level (logging.INFO)
logging.basicConfig (level=logging.INFO)
exp.set_sim_params (nr_reps=100)
#exp.set_msmnt_params (N=10, G=3, F=5, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 10, concentration = 0.01)
exp.set_plot_settings (do_save=False, do_show=False, save_analysis = True)
exp.set_bath_validity_conditions (A = 1e6, sparse = 10)
exp.save_bath_evolution (False)

max_steps = 100
num_cores = 1

nbath = exp.generate_bath()


def bath_par (G, F, scheme):

    print ('RUN:',G,F)
    exp.set_msmnt_params (N=7, G=G, F=F, tau0=1e-6, fid0=1., fid1=0.)
    exp.alpha = 1
    exp.strategy = 'int'
    nbath.reset_bath_unpolarized()
    exp.simulate (funct_name = scheme, max_steps = max_steps, nBath = nbath,
                string_id = 'capp_plusPi2_bath'+str(0), do_save = True)
    #exp.analysis (nr_bins=25)

    time.sleep (5)



    '''
    print ("NON ADAPTIVE K, adaptive phase")
    nbath.reset_bath_unpolarized()
    exp.set_msmnt_params (N=7, G=3, F=F, tau0=1e-6, fid0=1., fid1=0.)
    exp.simulate (funct_name = 'non_adaptive_k', max_steps = max_steps, nBath = nbath,
                string_id = 'onlyAdd90_bath'+str(i), do_save = True)
    exp.analysis (nr_bins=25)

    time.sleep (60)

    print ("FULLY NON ADAPTIVE")
    exp.set_msmnt_params (N=7, G=3, F=F, tau0=1e-6, fid0=1., fid1=0.)
    nbath.reset_bath_unpolarized()
    exp.simulate (funct_name = 'fully_non_adaptive', max_steps = max_steps, nBath = nbath,
                string_id = 'onlyAdd90_bath'+str(i), do_save = True)
    exp.analysis (nr_bins=25)

    time.sleep (60)
    '''
    
    return None #exp.analysis (nr_bins=25)
    
Parallel(n_jobs=num_cores)(delayed(bath_par)(g, f, 'adaptive_1step') for g in [3] for f in [1])

