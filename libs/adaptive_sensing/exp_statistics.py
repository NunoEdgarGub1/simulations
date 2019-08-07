
import numpy as np
import random
from matplotlib import rc, cm
import os, sys
import h5py
import logging, time
import sys
import matplotlib

from matplotlib import pyplot as plt
from simulations.libs.adaptive_sensing import qTracking as qtrack
from tools import data_object as DO
from importlib import reload

reload (qtrack)
reload (DO)

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


class ExpStatistics (DO.DataObjectHDF5):

    def __init__ (self, folder = 'D:/Research/WorkData/'):
        self.folder = folder
        self.auto_set = False
        self._called_modules = []

        self._A_thr = None
        self._sparse_thr = 10

        self._save_plots = False
        self._show_plots = False
        self._save_analysis = False
        self._save_bath_evol = False

        self.log = logging.getLogger ('qTrack_stats')
        self._log_level = logging.INFO 
        logging.basicConfig (level = self._log_level)

    def set_log_level (self, value):
        self._log_level = value
        self.log.setLevel (self._log_level)

    def set_sim_params (self, nr_reps, overhead=0, do_hahn = True):
        self.nr_reps = nr_reps
        self.overhead = overhead
        self._called_modules = []
        self.do_hahn = do_hahn

    def set_msmnt_params (self, F=5, G=1, N=8, tau0=20e-9, fid0=1., fid1=0.):
        self.N = N
        self.tau0 = tau0
        self.fid0 = fid0
        self.fid1 = fid1
        self.F = F
        self.G = G
        self.K = N-1

    def set_bath_params (self, nr_spins=7, concentration=0.01, cluster_size=3):
        self.nr_spins = nr_spins
        self.conc = concentration
        self.cluster_size = cluster_size
	
    def set_hahn_tauarr (self, hahn_tauarr):
        self.hahn_tauarr = hahn_tauarr
	
    def set_magnetic_field (self, Bz, Bx, By):
        self.Bz = Bz
        self.Bx = Bx
        self.By = By
	
    def set_inter (self, inter=True):
        self.inter = inter

    def save_bath_evolution (self, value):
        self._save_bath_evol = value

    def print_parameters(self):
        print ("##### Parameters:")
        for k in self.__dict__.keys():
            if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
                print (' - - ', k, ': ', self.__dict__[k])

    def set_plot_settings (self, do_show = False, do_save = False, save_analysis = False):
        self._save_plots = do_save
        self._show_plots = do_show
        self._save_analysis = save_analysis

    def __save_values(self, obj, file_handle):
        for k in obj.__dict__.keys():
            if ((type (obj.__dict__[k]) is float) or (type (obj.__dict__[k]) is int) or (type (obj.__dict__[k]) is str)):
                file_handle.attrs[k] = obj.__dict__[k] 

    def __generate_file (self, title = ''):

        fName = time.strftime ('%Y%m%d_%H%M%S')+ '_qTrack_G'+str(self.G)+'F'+str(self.F)+'_'+title
        newpath = os.path.join (self.folder, fName) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if not os.path.exists(os.path.join(newpath, fName+'.hdf5')):
            mode = 'w'
        else:
            mode = 'r+'
            print ('Output file already exists!')

        f = h5py.File(os.path.join(newpath, fName+'.hdf5'), mode)
        for k in self.__dict__.keys():
            if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
                f.attrs[k] = self.__dict__[k]
				
        name = os.path.join(newpath, fName+'.hdf5')

        return f, newpath, name

    def set_bath_validity_conditions (self, A=None, sparse=None):
        self._A_thr = A
        self._sparse_thr = sparse

    def generate_bath (self, folder):
        exp = qtrack.BathNarrowing (time_interval=0, overhead = 0, folder=self.folder)
        exp.set_bath_validity_conditions (A=self._A_thr, sparse=self._sparse_thr)
        exp.generate_spin_bath (folder=folder, do_hahn = self.do_hahn, hahn_tauarr = self.hahn_tauarr, nr_spins=self.nr_spins,
                    concentration=self.conc, cluster_size=self.cluster_size, Bx = self.Bx, By = self.By, Bz = self.Bz,
					inter = self.inter, store_evol_dict = self._save_bath_evol)
        print(self.folder)
        return exp.nbath

    def _generate_new_experiment (self, hahn_tauarr, folder, nBath = None):

        exp = qtrack.BathNarrowing (time_interval=0, overhead = 0, folder=folder)
        exp.set_log_level (self._log_level)
        exp.set_bath_validity_conditions (A=self._A_thr, sparse=self._sparse_thr)
        exp._save_plots = self._save_plots
        if (nBath == None):
            exp.generate_spin_bath (folder = folder, do_hahn = self.do_hahn, hahn_tauarr = self.hahn_tauarr, nr_spins=self.nr_spins,
                    concentration=self.conc, cluster_size=self.cluster_size, Bx = self.Bx, By = self.By, Bz = self.Bz,
					store_evol_dict = self._save_bath_evol)
        else:
            a = exp.load_bath (nBath)
            if not(a):
                self.log.warning ("Generate a new bath")
                exp.generate_spin_bath (folder = folder, hahn_tauarr = self.hahn_tauarr, nr_spins=self.nr_spins,
                    concentration=self.conc, cluster_size=self.cluster_size, Bx = self.Bx, By = self.By, Bz = self.Bz,
					store_evol_dict = self._save_bath_evol)
        exp.reset()
        exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, 
                G=self.G, F=self.F, N=self.N)
        exp.target_T2star = 2**(exp.K)*exp.tau0
        exp.set_flip_prob (0)
        exp.initialize()
        exp.set_plot_settings (do_show = self._show_plots, 
                do_save = self._save_plots, do_save_analysis = self._save_analysis)
        return exp

    def simulate (self, funct_name, max_steps, nr_seqs, batch_length = 5, string_id = '',
                do_save = False):

        self._called_modules.append('simulate')
        self.results = np.zeros((self.nr_reps, nr_seqs*max_steps))
        newpath = self.folder
		
        if do_save:
            batch_no=0
            f, newpath, name = self.__generate_file (title = '_'+funct_name+'_'+string_id+'_'+'batch_%d'%batch_no)

        for i in range(self.nr_reps):

            print ("Repetition nr: ", i+1)

            if i%batch_length == 0 and i>0:
                if do_save:
                    batch_no+=1
                    f, newpath, name = self.__generate_file (title = '_'+funct_name+'_'+string_id+'_'+'batch_%d'%batch_no)

            exp = self._generate_new_experiment (hahn_tauarr = np.linspace(0,2e-2,10), folder = newpath, nBath = self.generate_bath(newpath))
            exp.reset()
            exp.initialize()

            if do_save:
                print('run',i)
                grp_name = 'nbath_%d'%(i+1)
                self.save_object_all_vars_to_file (obj = exp.nbath, file_name = name, group_name = grp_name)
                self.save_object_params_list_to_file (obj = exp.nbath, file_name = name, group_name = grp_name,
                params_list= ['Ao', 'Ap', 'Azx', 'Azy', 'values_Az_kHz', 'r_ij', 'theta_ij'])

            try:
                exp.alpha = self.alpha
                exp.strategy = self.strategy
            except:
                pass
                
            exp.curr_rep = i
            a = getattr(exp, funct_name) (max_nr_steps=max_steps, nr_narrow_seqs=nr_seqs)
            l = len (exp.T2starlist)
            if (l<=nr_seqs*max_steps):              
                self.results [i, :l] = (exp.T2starlist[:l]/exp.T2starlist[0])
                self.results [i, l:nr_seqs*max_steps] = (exp.T2starlist[-1]/exp.T2starlist[0])*np.ones(nr_seqs*max_steps-l)
            else:
                self.results [i, :nr_seqs*max_steps] = (exp.T2starlist[:nr_seqs*max_steps]/exp.T2starlist[0])

            if do_save:
                rep_nr = str(i).zfill(len(str(self.nr_reps)))
                grp_name ='rep_'+rep_nr
                self.save_object_all_vars_to_file (obj = exp, file_name = name, group_name = grp_name)
                self.save_object_params_list_to_file (obj = exp, file_name = name, group_name = grp_name,
                        params_list= ['BayesianMean','BayesianSTD','QuantumMean','QuantumSTD',
						'BayesianMax','QuantumMax','T2starlist','conv_step','outcomes_list', 'tau_list', 'phase_list'])

        f.close()
        print ("Simulation completed.")

        self.total_steps = nr_seqs*max_steps
        self.newpath = newpath

    def analysis (self, nr_bins=100):
        self.log.info ('Processing results statistics...')
        max_value = np.max(self.results)
        res_hist = np.zeros((nr_bins+1, self.total_steps))
        bin_edges = np.zeros((nr_bins+1, self.total_steps))

        for j in range(self.total_steps):
            a = np.histogram(self.results[:nr_bins,j], bins=np.linspace (0, max_value, nr_bins+1))
            res_hist [:len(a[0]), j] = a[0]
            bin_edges [:len(a[1]), j] = a[1]

        [X, Y] = np.meshgrid (np.arange(self.total_steps), 
                np.linspace (0, max_value, nr_bins+1))

        plt.figure ()
        plt.pcolor (X, Y, res_hist)
        plt.xlabel ('nr of narrowing steps', fontsize = 18)
        plt.ylabel ('T2* (us)', fontsize = 18)
        if self._save_analysis:
            plt.savefig(os.path.join(self.newpath+'/analysis.png'))
            
        if self._show_plots:
            plt.show()


