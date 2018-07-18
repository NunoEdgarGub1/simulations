
import os, sys, logging, time
import numpy as np
from matplotlib import rc, cm
from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from simulations.libs.adaptive_sensing import exp_statistics as expStat
import multiprocessing as mp
from importlib import reload

reload (qtrack)
reload (expStat)

def run_simulations (args):
    exp = expStat.ExpStatistics (folder = 'C:/Users/cristian/Research/Work-Data/')
    exp.set_log_level (logging.INFO)
    logging.basicConfig (level=logging.INFO)
    exp.set_sim_params (nr_reps=3)
    exp.set_msmnt_params (N=7, G=1, F=0, tau0=1e-6, fid0=1., fid1=0.)
    exp.set_bath_params (nr_spins = 7, concentration = 0.01)
    exp.set_plot_settings (do_save=False, do_show=False, save_analysis = True)
    exp.set_bath_validity_conditions (A = 1e6, sparse = 10)
    exp.save_bath_evolution (False)
    nbath = exp.generate_bath()

    max_steps = 30

    G = args[0]
    F = args[1]
    idx = args[2]
    
    exp.set_msmnt_params (N=7, G=G, F=F, tau0=1e-6, fid0=1., fid1=0.)
    nbath.reset_bath_unpolarized()
    print ("Starting simulation!")
    exp.simulate (funct_name = 'adaptive_1step', max_steps = 100, nBath = nbath,
                string_id = 'capp_plusPi2_bath'+str(idx), do_save = True)
    exp.analysis (nr_bins=25)

params = [(2, 1, 0),(2, 2, 1), (2, 3, 2)]


if __name__ == "__main__":

    nr_cpu =  mp.cpu_count()

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = mp.Pool (processes = 3)
    pool.map (run_simulations, params)





