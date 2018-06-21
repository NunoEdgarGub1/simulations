
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

	def set_sim_params (self, nr_reps, overhead=0):
		self.nr_reps = nr_reps
		self.overhead = overhead
		self._called_modules = []

	def set_msmnt_params (self, M, N=8, tau0=20e-9, fid0=1., fid1=0.):
		self.N = N
		self.tau0 = tau0
		self.fid0 = fid0
		self.fid1 = fid1
		self.M = M

	def set_bath_params (self, nr_spins=7, concentration=0.01):
		self.nr_spins = nr_spins
		self.conc = concentration

	def print_parameters(self):
		print ("##### Parameters:")
		for k in self.__dict__.keys():
			if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
				print (' - - ', k, ': ', self.__dict__[k])

	def __save_values(self, obj, file_handle):
		for k in obj.__dict__.keys():
			if ((type (obj.__dict__[k]) is float) or (type (obj.__dict__[k]) is int) or (type (obj.__dict__[k]) is str)):
				file_handle.attrs[k] = obj.__dict__[k] 

	def __generate_file (self, title = ''):
		fName = time.strftime ('%Y%m%d_%H%M%S')+ '_qTrack'

		if not os.path.exists(os.path.join(self.folder, fName+'.hdf5')):
			mode = 'w'
		else:
			mode = 'r+'
			print ('Output file already exists!')

		f = h5py.File(os.path.join(self.folder, fName+'.hdf5'), mode)
		for k in self.__dict__.keys():
			if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
				f.attrs[k] = self.__dict__[k]

		return f

	def simulate_same_bath (self, max_steps, string_id = '', 
				do_save = False, do_plot = False, do_debug = False):

		self._called_modules.append('simulate')		

		if do_save:
			f = self.__generate_file (title = string_id)

		trialno = 0

		self.results = np.zeros((self.nr_reps, 2*self.M*max_steps+1))
		i = 0

		exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, 
				folder=self.folder, trial=0)
		exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
				 concentration=self.conc, verbose=do_debug, do_plot = do_plot, eng_bath=False)
		exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=5, F=3, N=10)
		exp.set_flip_prob (0)
		exp.initialize(do_plot = do_plot)

		while (i < self.nr_reps):

			print ("Repetition nr: ", i+1)

			if not exp.skip:
				exp.reset_unpolarized_bath()
				exp.initialize()
				exp.nbath.print_nuclear_spins()
				#here we could do it general, with a general function
				# passed as a string
				exp.adaptive_2steps (M=self.M, target_T2star = 5000e-6, 
						max_nr_steps=max_steps, do_plot = do_plot, do_debug = do_debug, do_save = do_save)
				l = len (exp.T2starlist)
				self.results [i, :l] = exp.T2starlist/exp.T2starlist[0]
				i += 1
			else:
				exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, 
						folder=self.folder, trial=trialno)
				exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
					 	concentration=self.conc, verbose=True, do_plot = False, eng_bath=False)
				exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=5, F=3, N=10)
				exp.set_flip_prob (0)
				exp.initialize(do_plot=do_plot)
				if do_save:
					# what parameters do we have to save??
					# - parameters of the bath
					# - measurement outcomes
					# - bath evolution
					# - evolution of T2*
					rep_nr = str(r).zfill(dig)
					grp = f.create_group('rep_'+rep_nr)
					DO.save_object_to_file (obj = exp, f = grp)

		if do_save:
			f.close()

	def simulate_different_bath (self, max_steps, string_id = '', 
				do_save = False, do_plot = False, do_debug = False):

		self._called_modules.append('simulate')		

		if do_save:
			f = self.__generate_file (title = string_id)

		trialno = 0

		self.results = np.zeros((self.nr_reps, 2*self.M*max_steps+1))
		i = 0

		while (i < self.nr_reps):

			#try:
			exp = qtrack.BathNarrowing (time_interval=100e-6, overhead=0, 
					folder=self.folder, trial=0)
			exp.set_spin_bath (cluster=np.zeros(self.nr_spins), nr_spins=self.nr_spins,
					 concentration=self.conc, verbose=do_debug, do_plot = do_plot, eng_bath=False)
			exp.set_msmnt_params (tau0 = self.tau0, T2 = exp.T2star, G=5, F=3, N=10)
			exp.set_flip_prob (0)
			exp.initialize()

			print ("Repetition nr: ", i+1)

			if not exp.skip:
				exp.nbath.print_nuclear_spins()
				exp.adaptive_2steps (M=self.M, target_T2star = 5000e-6, 
						max_nr_steps=max_steps, do_plot = do_plot, do_debug = do_debug)
				l = len (exp.T2starlist)
				self.results [i, :l] = exp.T2starlist/exp.T2starlist[0]
				i += 1

				if do_save:
					# what parameters do we have to save??
					# - parameters of the bath
					# - measurement outcomes
					# - bath evolution
					# - evolution of T2*
					rep_nr = str(i).zfill(len(str(self.nr_reps)))
					grp = f.create_group('rep_'+rep_nr)
					self.save_object_to_file (obj = exp, f = grp)
					grp_nbath = grp.create_group ('nbath')
					self.save_object_to_file (obj = exp.nbath, f = grp_nbath)
					# NO, saving everything becomes big too quick
					# we need a more specialized approach
					# where we state which params we want to save

			#except Exception as e: 
			#	print(e)

		if do_save:
			f.close()



	def analysis (self, nr_bins=100):
		print ('Processing results statistics...')
		res_hist = np.zeros((nr_bins+1, 2*self.M*self.max_steps+1))
		bin_edges = np.zeros((nr_bins+1, 2*self.M*self.max_steps+1))

		for j in range(2*self.M*self.max_steps+1):
			a = np.histogram(self.results[:nr_bins,j], bins=np.linspace (0, 15, nr_bins+1))
			res_hist [:len(a[0]), j] = a[0]
			bin_edges [:len(a[1]), j] = a[1]

		[X, Y] = np.meshgrid (np.arange(2*self.M*self.max_steps+1), 
				np.linspace (0, 15, nr_bins+1))

		plt.pcolor (X, Y, res_hist)
		plt.xlabel ('nr of narrowing steps', fontsize = 18)
		plt.ylabel ('T2*/T2*_init', fontsize = 18)
		plt.show()


