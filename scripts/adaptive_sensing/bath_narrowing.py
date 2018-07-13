
import os, sys, logging
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
from importlib import reload

reload (qtrack)
reload (expStat)

exp = expStat.ExpStatistics (folder = 'C:/Users/cristian/Research/Work-Data/')
exp.set_log_level (logging.DEBUG)
logging.basicConfig (level=logging.DEBUG)
exp.set_sim_params (nr_reps=3)
exp.set_msmnt_params (N=7, G=1, F=0, tau0=1e-6, fid0=1., fid1=0.)
exp.set_bath_params (nr_spins = 7, concentration = 0.01)
exp.set_plot_settings (do_save=True, do_show=True)
exp.set_semiclassical (False)
exp.set_bath_validity_conditions (A = 1e6, sparse = 10)

exp.simulate (funct_name = 'adaptive_1step', max_steps = 50, 
        string_id = 'G1F0_onlyAdd90', do_save = True)
exp.analysis (nr_bins=25)
