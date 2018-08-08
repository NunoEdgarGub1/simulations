
import os, sys, logging, time
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = '/home/qpl/Research/Work-Data/')
exp.set_log_level (logging.INFO)
logging.basicConfig (level=logging.INFO)
exp.set_sim_params (nr_reps=100)
exp.set_msmnt_params (N=7, G=1, F=0, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 7, concentration = 0.01)
exp.set_plot_settings (do_save=False, do_show=False, save_analysis = True)
exp.set_bath_validity_conditions (A = 1e6, sparse = 10)
exp.save_bath_evolution (False)

max_steps = 100


for i in range(10):
    print ("BATH nr ", i+1)
    nbath = exp.generate_bath()

    if (i<2):
        exp.set_plot_settings (do_save=True, do_show=False, save_analysis = True)
    else:
        exp.set_plot_settings (do_save=False, do_show=False, save_analysis = True)


    for alpha in [0.5, 0.75, 1., 1.25, 1.5]:
        exp.set_msmnt_params (N=7, G=3, F=2, tau0=1e-6, fid0=1., fid1=0.)
        exp.alpha = alpha
        exp.strategy = 'int'
        nbath.reset_bath_unpolarized()
        exp.simulate (funct_name = 'adaptive_1step', max_steps = max_steps, nBath = nbath,
                    string_id = '_alpha='+str(exp.alpha)+'_str='+exp.strategy+'_bath'+str(i), do_save = True)
        exp.analysis (nr_bins=25)

        time.sleep (20)


        exp.set_msmnt_params (N=7, G=3, F=2, tau0=1e-6, fid0=1., fid1=0.)
        exp.alpha = alpha
        exp.startegy='round'
        nbath.reset_bath_unpolarized()
        exp.simulate (funct_name = 'adaptive_1step', max_steps = max_steps, nBath = nbath,
                    string_id = '_alpha='+str(exp.alpha)+'_str='+exp.strategy+'_bath'+str(i), do_save = True)
        exp.analysis (nr_bins=25)

        time.sleep (20)


