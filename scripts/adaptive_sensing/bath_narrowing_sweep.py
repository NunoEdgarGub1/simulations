
import os, sys, logging, time
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt

folder = '/Users/dalescerri/Documents/GitHub'

sys.path.append (folder)

from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = 'C:/Users/cristian/Research/Work-Data/')
exp.set_log_level (logging.INFO)
logging.basicConfig (level=logging.INFO)
exp.set_sim_params (nr_reps=100)
exp.set_msmnt_params (N=7, G=1, F=0, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 7, concentration = 0.01)
exp.set_plot_settings (do_save=True, do_show=False, save_analysis = True)
exp.set_bath_validity_conditions (A = 1e6, sparse = 10)
exp.save_bath_evolution (False)

max_steps = 100


for i in range(10):
    print ("BATH nr ", i+1)
    nbath = exp.generate_bath()

    for F in [1]:

        print ("FULLY ADAPTIVE")
        exp.set_msmnt_params (N=7, G=3, F=F, tau0=1e-6, fid0=1., fid1=0.)
        nbath.reset_bath_unpolarized()
        exp.simulate (funct_name = 'adaptive_1step_capp', max_steps = max_steps, nBath = nbath,
                    string_id = 'capp_plusPi2_bath'+str(i), do_save = True)
        exp.analysis (nr_bins=25)

        time.sleep (5)

        '''
        print ("FULLY ADAPTIVE")
        exp.set_msmnt_params (N=7, G=2, F=F, tau0=1e-6, fid0=1., fid1=0.)
        nbath.reset_bath_unpolarized()
        exp.simulate (funct_name = 'adaptive_1step', max_steps = max_steps, nBath = nbath,
                    string_id = 'capp_plusPi2_bath'+str(i), do_save = True)
        exp.analysis (nr_bins=25) 

        time.sleep (60)
        '''


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

