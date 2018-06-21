
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
						max_nr_steps=max_steps, do_plot = do_plot, do_debug = do_debug)
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
					grp = f.create_group(rep_nr)
					self.__save_values (obj = exp, file_handle = grp)

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

			try:
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
						rep_nr = str(r).zfill(dig)
						grp = f.create_group(rep_nr)
						self.__save_values (obj = exp, file_handle = grp)
			except:
				pass

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




'''
	def load (self, stamp, folder):

		self.folder = folder
		name = toolbox.file_in_folder(folder=self.folder, timestamp = stamp)
		self.name = name
		f = h5py.File(os.path.join(self.folder, name),'r')
		for k in f.attrs.keys():
			setattr (self, k, f.attrs [k])

		try:
			dig = len(str(self.reps))
		except:
			dig = len('99')
			self.reps = 100

		if ('adptvTracking' in name):
			self.track_dict = {}
			self.track = True
		else:
			self.no_track_dict = {}
			self.track = False

		for r in np.arange (self.reps):
			rep_nr = str(r).zfill(dig)

			if ('adptvTracking' in name):
				grp = f['/'+rep_nr+'_track']
				self.track_dict[str(r)] = {}
				for k in grp.keys():
					self.track_dict[str(r)][k] = grp[k].value
				if ('figure_of_merit' in grp.keys()):
					self.fom_exists = True
				else:
					self.fom_exists = False
			else:
				try:
					grp = f['/'+rep_nr+'_no_track']
				except:
					grp = f['/0'+rep_nr+'_no_track']
				self.no_track_dict[str(r)] = {}
				for k in grp.keys():
					self.no_track_dict[str(r)][k] = grp[k].value

		try:
			self.fom_array = f['fom_array'].value
		except:
			pass
		f.close()
			
	def analysis (self, reps = 'all'):
		self._called_modules.append('analysis')

		if self.track:
			dictionary = self.track_dict
		else:
			dictionary = self.no_track_dict

		if (reps =='all'):
			reps = np.arange(self.reps)

		#discard instances that go out of range (|fB|>24 MHz)
		n_samples = 0
		msqe = 0
		new_reps = []
		for i in reps:
			d = dictionary[str(i)]
			ind_over = np.where(np.abs(d['set_B_field'])>24e6)
			if (len(ind_over [0]) == 0):
				new_reps.append(i)
		print ("Discarding out-of-bound instances. Number of usable instances:", len(new_reps),'/',len(reps))
		reps = new_reps
		self.usable_reps = reps
		self.reps = len(reps)

		self.processed_dict = {}
		n_used = 0
		n_total = 0
		self.wf_estim_std = np.zeros(len(self.usable_reps))
		self.avg_sensing_time = np.zeros(len(self.usable_reps))
		self.rmse = np.zeros(len(self.usable_reps))

		idx = 0
		for i in self.usable_reps:
			d = dictionary[str(i)]
			self.processed_dict[str(idx)] = {}
			self.processed_dict[str(idx)]['set_B_field'] = d['set_B_field']
			self.processed_dict[str(idx)]['est_field'] = d['est_B_field']
			self.processed_dict[str(idx)]['time_track'] = d['time_track']

			data = self.processed_dict[str(idx)]
			t = data['time_track']
			field = data['set_B_field']
			est_field = data['est_field']
			estim_var = (1./(t[-1]))*np.sum((np.abs(field[1:]-est_field[1:])**2)*(t[1:]-t[:-1]))
			estim_std = (estim_var)**0.5
			self.wf_estim_std [idx] = estim_std
			self.rmse [idx] = np.mean(np.abs(field-est_field))

			t = d['time_track']
			dt = t[1:]-t[:-1]
			self.avg_sensing_time [idx] = np.mean(dt)
			if (idx==0):
				max_range = np.max(dt)*2
			nr_occurs, bin_edges = np.histogram (dt, range = (0, max_range), bins=1000)
			if idx==0:
				nr_occur_sensing_time = nr_occurs 
			else:
				nr_occur_sensing_time = nr_occur_sensing_time + nr_occurs
			
			idx += 1

		self.nr_occur_sensing_time = nr_occur_sensing_time
		self.hist_bin_edges = 0.5*(bin_edges[1:]+bin_edges[:-1])

		return self.wf_estim_std, 0


	def plot_wf_estim_std_vs_rep (self, do_plot = True):

		if do_plot:
			plt.figure(figsize=(15, 6))
			plt.plot (self.wf_estim_std*1e-3, 'RoyalBlue', linewidth =3)
			plt.plot (self.wf_estim_std*1e-3, 'o', color='crimson')
			plt.ylabel ('waveform estimation error [kHz]', fontsize=18)
			plt.xlabel ('Repetition number', fontsize=18)
			plt.show()

			nr_occurs, bin_edges = np.histogram (self.wf_estim_std*1e-3, bins=1000)
			bins = 0.5*(bin_edges[1:]+bin_edges[:-1])
			print (len(bins),len(nr_occurs))
			plt.figure (figsize = (20, 5))
			plt.plot (bins, nr_occurs, 'RoyalBlue', linewidth = 2)
			plt.plot (bins, nr_occurs, 'o', color = 'RoyalBlue', markersize = 2)
			plt.xlabel ('waveform error [kHz]', fontsize=18)
			plt.ylabel ('fraction of occurences', fontsize=18)
			plt.show()

		print ("Average waveform estimation error: ", np.mean (self.wf_estim_std*1e-3), " -- std: ", np.std(self.wf_estim_std*1e-3))
		ind = np.where (np.abs(self.wf_estim_std-np.mean(self.wf_estim_std))<3*np.std(self.wf_estim_std))
		print ("Excluding outliers: ", np.mean (self.wf_estim_std[ind]*1e-3), " -- std: ", np.std(self.wf_estim_std[ind]*1e-3))
		if self.track:		
			return np.mean (self.wf_estim_std*1e-3), 0
		else:
			return np.mean (self.wf_estim_std*1e-3)

	def hist_sensing_times (self, reps = 'all', do_plot = True):

		plt.figure(figsize=(20, 5))
		self.nr_occur_sensing_time = self.nr_occur_sensing_time/np.sum(self.nr_occur_sensing_time+0.)
		plt.plot (self.hist_bin_edges*1e6, self.nr_occur_sensing_time, 'RoyalBlue', linewidth = 2)
		plt.plot (self.hist_bin_edges*1e6, self.nr_occur_sensing_time, 'o', color = 'crimson')
		plt.xlabel ('sensing times [us]', fontsize=18)
		plt.ylabel ('fraction of occurences', fontsize=18)
		plt.axis ([0, 100, 0, 1.1*max(self.nr_occur_sensing_time)])
		plt.show()

'''
