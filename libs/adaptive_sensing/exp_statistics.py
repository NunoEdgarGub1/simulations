
import numpy as np
import random
from matplotlib import rc, cm
import os, sys
import h5py
import logging, time
import sys
import matplotlib
#import msvcrt

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
		self._semiclassical = False

	def set_sim_params (self, nr_reps, overhead=0):
		self.nr_reps = nr_reps
		self.overhead = overhead
		self._called_modules = []

	def set_msmnt_params (self, F=5, G=1, N=8, tau0=20e-9, fid0=1., fid1=0.):
		self.N = N
		self.tau0 = tau0
		self.fid0 = fid0
		self.fid1 = fid1
		self.F = F
		self.G = G
		self.K = N-1

	def set_bath_params (self, nr_spins=7, concentration=0.01):
		self.nr_spins = nr_spins
		self.conc = concentration

	def print_parameters(self):
		print ("##### Parameters:")
		for k in self.__dict__.keys():
			if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
				print (' - - ', k, ': ', self.__dict__[k])

	def set_plot_saving (self, value):
		self._save_plots = value

	def __save_values(self, obj, file_handle):
		for k in obj.__dict__.keys():
			if ((type (obj.__dict__[k]) is float) or (type (obj.__dict__[k]) is int) or (type (obj.__dict__[k]) is str)):
				file_handle.attrs[k] = obj.__dict__[k] 

	def __generate_file (self, title = ''):

		fName = time.strftime ('%Y%m%d_%H%M%S')+ '_qTrack'+title
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

		return f, newpath

	def set_semiclassical (self, value=True):
		self._semiclassical = True

	def simulate_same_bath (self, funct_name, max_steps, string_id = '', 
				do_save = False, do_plot = False, do_debug = False):

		self._called_modules.append('simulate')
		R = int(self.G*self.N*self.F*self.N*(self.N-1)/2)
		self.results = np.zeros((self.nr_reps, R))
		newpath = self.folder

		if do_save:
			f, newpath = self.__generate_file (title = '_'+funct_name+'_'+string_id)

		exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, folder=newpath)
		exp.semiclassical = self._semiclassical
		exp._save_plots = self._save_plots
		exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
				 concentration=self.conc, verbose=do_debug, do_plot = do_plot, eng_bath=False)
		exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=self.G, F=self.F, N=10)
		exp.target_T2star = 2**(exp.K)*exp.tau0
		exp.set_flip_prob (0)
		exp.initialize()


		i = 0
		while (i < self.nr_reps):

			print ("Repetition nr: ", i+1)

			if not exp.skip:
				
				exp.reset_unpolarized_bath()
				exp.initialize()
				exp.semiclassical = self._semiclassical
				exp.nbath.print_nuclear_spins()
				exp.curr_rep = i
				a = getattr(exp, funct_name) (max_nr_steps=max_steps, 
						do_plot = do_plot, do_debug = do_debug)
				l = len (exp.T2starlist)
				self.results [i, :l] = exp.T2starlist/exp.T2starlist[0]
				self.results [i, l:R] = (exp.T2starlist[-1]/exp.T2starlist[0])*np.ones(R-l)
				i += 1

				if do_save:
					rep_nr = str(i).zfill(len(str(self.nr_reps)))
					grp = f.create_group('rep_'+rep_nr)
					self.save_object_all_vars_to_file (obj = exp, f = grp)
					self.save_object_params_list_to_file (obj = exp, f = grp, 
							params_list= ['T2starlist', 'outcomes_list', 'tau_list', 'phase_list'])
					grp_nbath = grp.create_group ('nbath')
					self.save_object_all_vars_to_file (obj = exp.nbath, f = grp_nbath)
					self.save_object_params_list_to_file (obj = exp.nbath, f = grp_nbath, 
							params_list= ['Ao', 'Ap', 'Azx', 'Azy', 'values_Az_kHz', 'r_ij', 'theta_ij'])

			else:
				
				exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, folder=newpath)
				exp.semiclassical = self._semiclassical
				exp._save_plots = self._save_plots
				exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
						 concentration=self.conc, verbose=do_debug, do_plot = do_plot, eng_bath=False)
				exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=self.G, F=self.F, N=10)
				exp.target_T2star = 2**(exp.K)*exp.tau0
				exp.set_flip_prob (0)
				exp.initialize()

		if do_save:
			f.close()

		self.total_steps = l
		self.newpath = newpath


	def simulate_different_bath (self, funct_name, max_steps, string_id = '', 
				do_save = False, do_plot = False, do_debug = False, semiclassical = False):

		self._called_modules.append('simulate')
		R = int(self.G*self.N*self.F*self.N*(self.N-1)/2)
		self.results = np.zeros((self.nr_reps, R))
		newpath = self.folder

		if do_save:
			f, newpath = self.__generate_file (title = '_'+funct_name+'_'+string_id)

		i = 0
		while (i < self.nr_reps):
			
			#try:
			exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, folder=newpath)
			exp.semiclassical = semiclassical
			exp._save_plots = self._save_plots
			exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
					 concentration=self.conc, verbose=do_debug, do_plot = do_plot, eng_bath=False)
			exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=self.G, F=self.F, N=10)
			exp.target_T2star = 2**(exp.K)*exp.tau0
			exp.set_flip_prob (0)
			exp.initialize()

			print ("Repetition nr: ", i+1)
			exp.curr_rep = i

			if not exp.skip:
				exp.nbath.print_nuclear_spins()
				a = getattr(exp, funct_name) (max_nr_steps=max_steps, 
						do_plot = do_plot, do_debug = do_debug)
				l = len (exp.T2starlist)
				self.results [i, :l] = exp.T2starlist/exp.T2starlist[0]
				self.results [i, l:R] = (exp.T2starlist[-1]/exp.T2starlist[0])*np.ones(R-l)
				i += 1

				if do_save:
					rep_nr = str(i).zfill(len(str(self.nr_reps)))
					grp = f.create_group('rep_'+rep_nr)
					self.save_object_all_vars_to_file (obj = exp, f = grp)
					self.save_object_params_list_to_file (obj = exp, f = grp, 
							params_list= ['T2starlist', 'outcomes_list', 'tau_list', 'phase_list'])
					grp_nbath = grp.create_group ('nbath')
					self.save_object_all_vars_to_file (obj = exp.nbath, f = grp_nbath)
					self.save_object_params_list_to_file (obj = exp.nbath, f = grp_nbath, 
							params_list= ['Ao', 'Ap', 'Azx', 'Azy', 'values_Az_kHz', 'r_ij', 'theta_ij'])

			#except Exception as e: 
			#	print(e)

		if do_save:
			f.close()

		self.total_steps = l
		self.newpath = newpath

	def analysis (self, nr_bins=100):
		print ('Processing results statistics...')
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
		plt.ylabel ('T2*/T2*_init', fontsize = 18)
		if self._save_plots:
			plt.savefig(os.path.join(self.newpath+'/analysis.png'))
		plt.show()


