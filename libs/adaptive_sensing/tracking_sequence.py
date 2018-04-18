
import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit
import re

#sys.path.append ('/Users/dalescerri/Documents/GitHub')

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


class Analyze (DO.DataObjectHDF5):

	def __init__(self, name_tag = '', folder = 'D:/Research/WorkData/adptv_tracking_sim/'):
		self.folder = folder
		self.analysis_dict = {}

	def do_analysis (self, name, stamps_list, sweep_par):
		analysis_dict = {}

		s = SequenceStatistics ()

		ind = 0

		if (len(stamps_list)==1):
			print ("Checking for files with stamp: ", stamps_list[0])
			all_files = [f[:-5] for f in os.listdir(self.folder) if stamps_list[0] in f]
			all_files.sort()
			stamps_list = all_files

			print ("Nr of files found: ", len(stamps_list))

		self.wf_estim_rmse = np.zeros(len(stamps_list))
		self.wf_estim_rmse_std = np.zeros(len(stamps_list))
		self.sweep_par = np.zeros(len(stamps_list))
		self.avg_sensing_time = np.zeros(len(stamps_list))
		self.std_sensing_time = np.zeros(len(stamps_list))

		for i in stamps_list:

			print ("Checking stamp: ", i)
			analysis_dict[i] = {}
			s.load (folder = self.folder, stamp = i)

			for k in s.__dict__.keys():
				if (type(s.__dict__[k]) in [int, float, str]):
					analysis_dict[i][k] = s.__dict__[k]
				elif isinstance(s.__dict__[k], np.float64):
					analysis_dict[i][k] = s.__dict__[k]
				elif isinstance(s.__dict__[k], np.int32):
					analysis_dict[i][k] = s.__dict__[k]
				elif isinstance(s.__dict__[k], np.ndarray):
					analysis_dict[i][k] = s.__dict__[k]

			s.analysis (reps = 'all')
			analysis_dict [i]['waveform_estimation_rmse'] = np.mean(s.wf_estim_std)
			analysis_dict [i]['waveform_estimation_rmse_std'] = np.std(s.wf_estim_std)
			self.wf_estim_rmse[ind] = analysis_dict [i]['waveform_estimation_rmse']
			self.wf_estim_rmse_std[ind] = analysis_dict [i]['waveform_estimation_rmse_std']
			self.sweep_par[ind] = analysis_dict[i][sweep_par]
			self.avg_sensing_time[ind] = np.mean(s.avg_sensing_time)
			self.std_sensing_time[ind] = np.std(s.avg_sensing_time)

			ind += 1

		analysis_dict['result'] = {'sweep_par': self.sweep_par, 'sweep_par_name':sweep_par, 
									'waveform_estimation_rmse':self.wf_estim_rmse,
								 	'waveform_estimation_rmse_std':self.wf_estim_rmse_std,
								 	'avg_sensing_time':self.avg_sensing_time,
								 	'std_sensing_time':self.std_sensing_time}
		for i in s._called_modules:
			analysis_dict['info']['code_'+i] = inspect.getsource(getattr(s, i))
		self.analysis_dict[name] = analysis_dict


	def plot_analysis (self, x_scale=1, sweep_par_label = ''):
		
		#plot waveform estimation rmse vs sweep_parameter
		plt.figure(figsize = (12,5))
		plt.plot (self.sweep_par*x_scale, self.wf_estim_rmse*1e-3, color='RoyalBlue')
		plt.errorbar (self.sweep_par*x_scale, self.wf_estim_rmse*1e-3, 
						yerr=self.wf_estim_rmse_std*1e-3, fmt='o', markersize=10, color = 'RoyalBlue')
		plt.plot (self.sweep_par*x_scale, self.wf_estim_rmse*1e-3, 'o', markersize=10, color = 'crimson')
		plt.ylabel ('waveform estimation error [kHz]', fontsize=18)
		plt.xlabel (sweep_par_label, fontsize=18)
		plt.axis('tight')
		plt.show()

		#plot avegrage sensing time vs sweep_parameter
		plt.figure(figsize = (12,5))
		plt.plot (self.sweep_par*x_scale, self.avg_sensing_time*1e6, color='RoyalBlue')
		plt.errorbar (self.sweep_par*x_scale, self.avg_sensing_time*1e6, 
						yerr=self.std_sensing_time*1e6, fmt='o', markersize=10, color = 'RoyalBlue')
		plt.plot (self.sweep_par*x_scale, self.avg_sensing_time*1e6, 'o', markersize=10, color = 'crimson')
		plt.ylabel ('average sensing time [us]', fontsize=18)
		plt.xlabel (sweep_par_label, fontsize=18)
		plt.show()

		return self.sweep_par, self.wf_estim_rmse, self.wf_estim_rmse_std

	def save_analysis (self, file_name):
		fName = time.strftime ('%Y%m%d_%H%M%S')+ '_analysis'+file_name+'.hdf5'
		f = h5py.File(os.path.join(self.folder, fName), 'w')
		self.save_dict_to_file (d = self.analysis_dict, file_handle = f)
		f.close()

	def load_analysis_results (self, stamp, folder):
		self.folder = folder
		self.name = stamp
		name = toolbox.file_in_folder(folder=self.folder, timestamp = stamp)
		f = h5py.File(os.path.join(self.folder, name),'r')
		k = f.keys()
		grp = f[k[0]]
		grp2 = grp['result']
		for k in grp2.keys():
			setattr (self, k, grp2[k].value)


class SequenceStatistics (DO.DataObjectHDF5):

	def __init__ (self, folder = 'D:/Research/WorkData/adptv_tracking_sim/'):
		self.folder = folder
		self.auto_set = False
		self._called_modules = []

	def set_sim_params (self, reps, time_interval, overhead):
		self.reps = reps
		self.time_interval = time_interval
		self.overhead = overhead
		self._called_modules = []

	def set_msmnt_params (self, G, F, dfB, N=8, tau0=20e-9, T2 = 100e-6, fid0=0.75, fid1=0.01):
		self.N = N
		self.tau0 = tau0
		self.T2 = T2
		self.fid0 = fid0
		self.fid1 = fid1
		self.F = F
		self.G = G
		self.K = N-1
		self.dfB = dfB

		self.points = 2**(self.N+1)+3
		self.discr_steps = 2*self.points+1
		self.fB_max = 1./(2*tau0)
		self.n_points = 2**(self.N+5)
		self.beta = np.linspace (-self.fB_max, self.fB_max, self.n_points+1)
		self.beta = self.beta[:-1]
		self._called_modules = []

	def automatic_set_N (self, track, dfB):

		'''
		Sets automatically the value of N (K = N-1), based on Eq 14 in PRA 95, 052348 (2017)
		'''

		N1 = int(np.log2(self.T2/self.tau0))
		diff = -1
		K = N1
		csi = 1 - self.fid0

		if not(track):
			while diff<0:
				K -= 1
				N = K+1
				R = self.G*N + self.F*N*(N-1)/2.
				a = (3/(4*np.pi**2))*(2**(-2*K)*(1+csi*N))/(self.G*self.tau0**2)
				b = (self.dfB**2)*((self.G+self.F)*(2**K)*self.tau0+R*self.overhead)
				diff = a-2*b
		else:
			while diff<0:
				K -= 1
				N = K+1
				R = self.G_adptv
				a = (3/(4*np.pi**2))*(2**(-2*K)*(1+csi*N))/(self.G_adptv*self.tau0**2*self.dfB**2)
				b = ((self.G_adptv)*(2**K)*self.tau0+R*self.overhead)
				diff = a-2*b
		N = N+1 
		self.N = N
		self.set_msmnt_params (G=self.G, F=self.F, dfB=dfB, N=N, tau0=self.tau0, T2 = self.T2, fid0=self.fid0, fid1=self.fid1)
		self._called_modules.append('automatic_set_N')
		print ("# Automatic settings: N=", self.N)

	def calculate_fom_thresholds (self, alpha = 1.):

		'''
		Calculate the thresholds for the figure of merit, according to Eq. 22 in PRA 95, 052348 (2017)

		Input
		alpha 	[float]
		'''

		self.fom_array = np.zeros(self.N)
		csi = 0*(1 - self.fid0)
		for n in np.arange(self.N):
			self.fom_array[n] = alpha*(2.**(-(self.N-(n+1)))/self.tau0)
		self.auto_set = True
		self._called_modules.append('calculate_fom_thresholds')
		print ("-----------------------------------------------")
		print ("FOM thresholds (MHz): ", self.fom_array*1e-6)

	def print_parameters(self):
		print ("##### Parameters:")
		for k in self.__dict__.keys():
			if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
				print (' - - ', k, ': ', self.__dict__[k])

	def __save_values(self, obj, file_handle):
		for k in obj.__dict__.keys():
			if ((type (obj.__dict__[k]) is float) or (type (obj.__dict__[k]) is int) or (type (obj.__dict__[k]) is str)):
				file_handle.attrs[k] = obj.__dict__[k] 
		file_handle.create_dataset ('time_track', data = obj.time_scale)
		file_handle.create_dataset ('set_B_field', data = obj.field)
		file_handle.create_dataset ('est_B_field', data = obj.est_field)
		try:
			file_handle.create_dataset ('msmnt_times', data = obj.msmnt_times)
		except:
			pass

		rms = (np.mean(np.abs(obj.field-obj.est_field)**2))**0.5
		file_handle.attrs['rms'] = rms
		rms_std = (np.std(np.abs(obj.field-obj.est_field)**2))**0.5
		file_handle.attrs['rms_std'] = rms_std

		if hasattr (obj, 'fom'):
			file_handle.create_dataset ('figure_of_merit', data = obj.fom)

		if hasattr (obj, 'field_std'):
			file_handle.create_dataset ('set_B_field_std', data = obj.field_std)


	def plot_simulation (self, rep, color=None, do_save=False):

		matplotlib.rc('xtick', labelsize=22) 
		matplotlib.rc('ytick', labelsize=22)

		data = self.processed_dict[str(rep)]
		t = data['time_track']
		field = data['set_B_field']
		est_field = data['est_field']

		x = len(t)

		if color==None:
			if self.track:
				color = 'crimson'
			else:
				color = 'RoyalBlue'
		fig = plt.figure(figsize=(20,5))
		plt.plot (t[:x]*1e3, field[:x]*1e-6, color='darkgray', linewidth = 3)
		plt.plot (t[:x]*1e3, est_field[:x]*1e-6, 'o', markersize=3, color=color, markeredgecolor=color)
		plt.plot (t[:x]*1e3, est_field[:x]*1e-6, color=color, linewidth = 1)
		plt.xlabel ('time [ms]', fontsize = 20)
		plt.ylabel (' frequency [MHz]', fontsize = 20)
		plt.axis('tight')
		if do_save:
			fig.savefig (self.folder+'figure.svg')
		plt.show()

		estim_var = (1./(t[-1]))*np.sum((np.abs(field[1:]-est_field[1:])**2)*(t[1:]-t[0:-1]))
		print ("Waveform estimation std: ", 1e-3*(estim_var)**0.5, ' kHz')
		print ("RMSE: ", np.mean(np.abs(field-est_field))*1e-3, ' kHz')

		plt.figure(figsize=(20,5))
		plt.plot (t*1e3, np.abs(field-est_field)*1e-3, color='crimson', linewidth = 1)
		plt.plot (t*1e3, np.abs(field-est_field)*1e-3, 'o', markersize=2, color='RoyalBlue')
		plt.axis ('tight')
		plt.xlabel ('time [ms]', fontsize = 18)
		plt.ylabel ('|B_{est} - B| [kHz]', fontsize = 18)
		plt.show()


	def generate_simulation (self, track, do_save = False, description = '', title = '', do_plot = False, period = None, do_debug = False):

		self._called_modules.append('generate_simulation')		
		if do_save:
			if track:
				fName = time.strftime ('%Y%m%d_%H%M%S')+ '_adptvTracking'
			else:
				fName = time.strftime ('%Y%m%d_%H%M%S')+ '_noTracking'

			fName = fName+ '_fid'+str(int(self.fid0*100))+'_'+title+'_OH'+str(self.overhead*1e6)+'us'+'_dfB='+str(self.dfB*1e-6)+'MHz_N'+str(self.N)

			if not os.path.exists(os.path.join(self.folder, fName+'.hdf5')):
				mode = 'w'
			else:
				mode = 'r+'
				print ('Output file already exists!')
			
			f = h5py.File(os.path.join(self.folder, fName+'.hdf5'), mode)
			
			for k in self.__dict__.keys():
				if ((type (self.__dict__[k]) is float) or (type (self.__dict__[k]) is int) or (type (self.__dict__[k]) is str)):
					f.attrs[k] = self.__dict__[k]
			f.attrs['overhead'] = self.overhead

			
			dig = len(str(self.reps))

		print ('Repetitions: ')

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

