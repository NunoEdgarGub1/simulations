
import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit
import re

import matplotlib.pyplot as plt
import inspect
from scipy import signal
from simulations.libs.math import statistics as stat
from tools import toolbox_delft as toolbox
from simulations.libs.adaptive_sensing import adaptive_tracking as track_libOH
from tools import data_object as DO
from importlib import reload


reload (stat)
reload (track_libOH)
reload (DO)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


class ExpStatistics (DO.DataObjectHDF5):

	def __init__ (self, folder = 'D:/Research/WorkData/adptv_tracking_sim/'):
		self.folder = folder
		self.auto_set = False
		self._called_modules = []

	def set_sim_params (self, nr_reps, overhead=0):
		self.reps = nr_reps
		self.overhead = overhead
		self._called_modules = []

	def set_msmnt_params (self, M, N=8, tau0=20e-9, T2 = 100e-6, fid0=1., fid1=0.):
		self.N = N
		self.tau0 = tau0
		self.T2 = T2
		self.fid0 = fid0
		self.fid1 = fid1
		self.F = F
		self.G = G
		self.K = N-1

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
		fName = time.strftime ('%Y%m%d_%H%M%S')+ '_adptvTracking'

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

	def simulate_same_bath (self, string_id = '', do_save = False):

		self._called_modules.append('simulate')		

		if do_save:
			f = self.__generate_file (title = string_id)

		for r in np.arange(self.reps):
			sys.stdout.write(str(r)+', ')	

			T = track_libOH.TimeSequence_overhead (time_interval = self.time_interval, overhead = self.overhead, folder = self.folder)
			T.track = track
			T.set_msmnt_params (N = self.N, G = self.G, F = self.F, T2 = self.T2, fid0 = self.fid0, fid1 = self.fid1)

			attr_list = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
			for a in attr_list:
				setattr (T, a, getattr(self,a))

			T.reset_called_modules()
			T.setup_protocol()

			T.curr_rep = r
			T.prev_estim = 0 
			T.init_apriori()
			T.curr_B_idx = 0

			if track:
				T.simulate (track = True, do_plot = do_plot, do_debug=do_debug)
				T.curr_B_idx = 0

				if do_save:
					rep_nr = str(r).zfill(dig)
					grp = f.create_group(rep_nr+'_track')
					self.__save_values (obj = T, file_handle = grp)
			else:
				T.simulate(track = False, do_plot = do_plot)
				T.dur_sens_time = (T.tau0*(T.G*(2**T.N-1)+T.F*(2**T.N-T.N-1)))
				T.dur_OH_time = (T.G*T.N+0.5*(T.F*T.N*(T.N-1)))*T.OH
				if do_save:
					rep_nr = str(r).zfill(dig)
					grp = f.create_group(rep_nr+'_no_track')
					self.__save_values (obj = T, file_handle = grp)

		if do_save:

			try:
				f.create_dataset ('fom_array', data = self.fom_array)
			except:
				print ("fom_array not found")

			grp = f.create_group('code')

			for i in T._called_modules:
				try:
					grp.attrs[i] = inspect.getsource(getattr(T, i))
				except:
					print ("Non-existing function: ", i)
			f.close()

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

