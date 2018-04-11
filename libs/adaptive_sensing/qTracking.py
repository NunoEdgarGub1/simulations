import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit
from importlib import reload

import matplotlib.pyplot as plt
from scipy import signal
from simulations.libs.math import statistics as stat
from analysis.libs.tools import toolbox
from simulations.libs.adaptive_sensing import adaptive_tracking as adptvTrack
from simulations.libs.spin import diluted_NSpin_bath_py3 as NSpin

reload (NSpin)
reload (adptvTrack)
reload (stat)

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


class TimeSequenceQ (adptvTrack.TimeSequence_overhead):

	def __init__ (self, time_interval, overhead, folder):
		self.time_interval = time_interval
		self._B_dict = {}
		self.kappa = None
		self.curr_fB_idx = 0
		self.OH = overhead
		self.set_fB = []
		self.est_fB = []
		self.curr_fB = 0
		self.plot_idx = 0

		# The "called modules" is  a list that tracks which functions have been used
		# so that they are saved in the output hdf5 file.
		# When you look back into old data, it's nice to have some code recorded
		# to make sure you know how the data was generated (especially if you tried out diffeent things)
		self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase']
		self.folder = folder

	def set_spin_bath (self, nr_spins, concentration, verbose = False):
		
		# Spin bath initialization
		self.nbath = NSpin.CentralSpinExperiment()
		self.nbath.set_experiment(nr_lattice_sites=False, 
				nr_nuclear_spins=nr_spins, concentration = concentration)

		if verbose:
			self.nbath.print_nuclear_spins()

	def reset_called_modules(self):
		self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase']

	def ramsey (self, t=0., theta=0., do_plot = False):

		'''
		Ramsey experiment simulation
		Calculates the probabilities p0 and p1 to get 0/1 and draws an outcome with the probabilities

		Input:
		t 		[ns]		sensing time
		theta	[rad]		Ramsey phase

		Returns:
		Ramsey outcome
		'''

		az0, pd0 = np.real(self.nbath.get_probability_density())

		m = self.nbath.Ramsey (tau=t, phi = theta)

		az, pd = np.real(self.nbath.get_probability_density())

		if do_plot:
			title = 'Ramsey: tau = '+str(int(t*1e9))+' ns -- phase: '+str(int(theta*180/3.14))+' deg'
			plt.figure (figsize = (8,4))
			plt.plot (az0, pd0, linewidth=2, color = 'RoyalBlue')
			plt.plot (az, pd, linewidth=2, color = 'crimson')
			plt.xlabel ('frequency hyperfine (kHz)', fontsize=18)
			plt.ylabel ('probability', fontsize=18)
			plt.title (title, fontsize=18)
			plt.show()

		return m

	def set_magnetic_field (self, dfB):
		self.dfB = dfB

	def calc_acc_phase (self, t_units, do_debug=False):
		pass

	def adptv_tracking_single_step (self, k, M, do_debug=False):

		t_i = int(2**k)
		ttt = -2**(k+1)
		t0 = self.running_time					

		phase_cappellaro = 0.5*np.angle (self.p_k[ttt+self.points])
		m_list = []
		for m in range(M):
			m_res = self.ramsey (theta=phase_cappellaro, t = t_i*self.tau0, do_plot=do_debug)
			m_list.append(m_res)	
			self.bayesian_update (m_n = m_res, phase_n = phase_cappellaro, t_n = t_i, do_plot=do_debug)
			if do_debug:
				print ("Estimation step: t_units=", t_i, "    -- res:", m_res)

		return m_list


	def adaptive_tracking_estimation (self, do_plot = False, do_debug=False):

		'''
		Simulates adaptive tracking protocol

		Input: do_plot [bool], do_debug [bool]
		'''

		self._called_modules.append('adaptive_tracking_estimation')
		fom = 1e12
		idx = 0
		total_time = 0
		self.field = []
		self.field_std = []
		self.time_tags = []
		self.fom = []
		self.est_field = []
		self.prev_estim = self.curr_fB
		rep_array = np.arange(self.nr_sensing_reps)
		self.k_array = self.K-np.arange(self.K+1)
		sigma_reps_idx = 0

		self.init_apriori ()
		estim_time = 0
		self.initialize_in_zero(do_debug=do_debug)
		fom = 0
		rep_idx = 0
		est_field = 0
		estim_time = 0

		while self.running_time<self.time_interval:

			fom = 100e6
			rep_idx = 0
			sensing_time_idx = np.mod(idx, self.N)
			
			t_i = int(2**(self.k_array[sensing_time_idx]))
			BBB = []

			nr_sensing_reps= int(self.G_adptv + self.F*sensing_time_idx)
			if nr_sensing_reps<1:
				nr_sensing_reps=1
			while ((rep_idx < nr_sensing_reps) and (fom>0.15*self.fom_array[sensing_time_idx])):#(fom<self.fom_threshold)):
				self.convolve_prob_distribution (t = 1*(t_i+self.OH/self.tau0), do_plot = do_debug)
				dt = self.adptv_tracking_single_step (sensing_time_idx = sensing_time_idx, do_debug=do_debug)

				fom = self.figure_of_merit_std()
				rep_idx += 1

				total_time = total_time + dt
				estim_time = estim_time + dt
				#BBB = np.hstack((BBB, self.curr_fB_array))

			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)
			if (fom<0.15*self.fom_array[sensing_time_idx]):
				if (sensing_time_idx >0):
					idx=idx-1
				else:
					idx = 0
			else:
				idx +=1

			self.fom.append(fom)
			#self.est_field.append(est_field)
			self.time_tags.append(total_time)
			#self.field.append (np.mean(BBB))
			#self.field_std.append(np.std(BBB))

		self.prev_estim = est_field
		self.field = np.asarray(self.field)
		self.time_tags = np.asarray(self.time_tags)
		#self.est_field = np.asarray(self.est_field)
		self.fom = np.asarray(self.fom)
		self.field_std = np.asarray(self.field_std)
		self.time_scale = self.time_tags*self.tau0

		if do_plot:
			plt.figure(figsize=(20,6))
			plt.subplot (2,1,1)
			sf = (3**0.5*2**(-self.N))/(2*np.pi*self.tau0)
			plt.plot (self.time_scale*1e3, self.est_field*1e-6, 'crimson')		
			plt.fill_between (self.time_scale*1e3, 1e-6*(self.est_field-sf), 1e-6*(self.est_field+sf), color = 'crimson', alpha=0.3)
			plt.plot (self.time_scale*1e3, self.est_field*1e-6, 'o', color='crimson', markersize=4)		
			plt.fill_between (self.time_scale*1e3, 1e-6*(self.field-self.field_std), 1e-6*(self.field+self.field_std), color = 'RoyalBlue')
			plt.xlabel ('time [msec]', fontsize=16)
			plt.ylabel ('[MHz]', fontsize=16)
			plt.axis('tight')
			plt.subplot (2,1,2)
			plt.semilogy (self.time_scale*1e3,self.fom, color='lightsteelblue', linewidth=1)
			plt.semilogy (self.time_scale*1e3,self.fom, 'o', markersize=3, color='RoyalBlue')
			plt.ylabel ('FOM', fontsize=16)
			plt.axis('tight')
			plt.show()

	def qTracking (self, do_plot = False, do_debug=False):

		'''
		Simulates adaptive tracking protocol

		Input: do_plot [bool], do_debug [bool]
		'''

		self._called_modules.append('adaptive_tracking_estimation')
		self.init_apriori ()
		estim_time = 0
		self.initialize_in_zero(do_debug=do_debug)
		fom = 0
		rep_idx = 0
		est_field = 0
		estim_time = 0

		while self.running_time<self.time_interval:

			fom = 100e6
			rep_idx = 0
			sensing_time_idx = np.mod(idx, self.N)
			
			t_i = int(2**(self.k_array[sensing_time_idx]))
			BBB = []

			nr_sensing_reps= int(self.G_adptv + self.F*sensing_time_idx)
			if nr_sensing_reps<1:
				nr_sensing_reps=1
			while ((rep_idx < nr_sensing_reps) and (fom>0.15*self.fom_array[sensing_time_idx])):#(fom<self.fom_threshold)):
				self.convolve_prob_distribution (t = 1*(t_i+self.OH/self.tau0), do_plot = do_debug)
				dt = self.adptv_tracking_single_step (sensing_time_idx = sensing_time_idx, do_debug=do_debug)

				fom = self.figure_of_merit_std()
				rep_idx += 1

				total_time = total_time + dt
				estim_time = estim_time + dt
				#BBB = np.hstack((BBB, self.curr_fB_array))

			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)
			if (fom<0.15*self.fom_array[sensing_time_idx]):
				if (sensing_time_idx >0):
					idx=idx-1
				else:
					idx = 0
			else:
				idx +=1

			self.fom.append(fom)
			#self.est_field.append(est_field)
			self.time_tags.append(total_time)
			#self.field.append (np.mean(BBB))
			#self.field_std.append(np.std(BBB))

		self.prev_estim = est_field
		self.field = np.asarray(self.field)
		self.time_tags = np.asarray(self.time_tags)
		#self.est_field = np.asarray(self.est_field)
		self.fom = np.asarray(self.fom)
		self.field_std = np.asarray(self.field_std)
		self.time_scale = self.time_tags*self.tau0

		if do_plot:
			plt.figure(figsize=(20,6))
			plt.subplot (2,1,1)
			sf = (3**0.5*2**(-self.N))/(2*np.pi*self.tau0)
			plt.plot (self.time_scale*1e3, self.est_field*1e-6, 'crimson')		
			plt.fill_between (self.time_scale*1e3, 1e-6*(self.est_field-sf), 1e-6*(self.est_field+sf), color = 'crimson', alpha=0.3)
			plt.plot (self.time_scale*1e3, self.est_field*1e-6, 'o', color='crimson', markersize=4)		
			plt.fill_between (self.time_scale*1e3, 1e-6*(self.field-self.field_std), 1e-6*(self.field+self.field_std), color = 'RoyalBlue')
			plt.xlabel ('time [msec]', fontsize=16)
			plt.ylabel ('[MHz]', fontsize=16)
			plt.axis('tight')
			plt.subplot (2,1,2)
			plt.semilogy (self.time_scale*1e3,self.fom, color='lightsteelblue', linewidth=1)
			plt.semilogy (self.time_scale*1e3,self.fom, 'o', markersize=3, color='RoyalBlue')
			plt.ylabel ('FOM', fontsize=16)
			plt.axis('tight')
			plt.show()


	def nontracking_estimation_routine (self):
		self._called_modules.append('nontracking_estimation_routine')
		tau = 2**(self.k_array)
		t = np.zeros (self.K+1)
		res_idx = 0
		m_res = 0

		total_time = 0
		self.msmnt_idx = 0
		for i,k in enumerate(self.k_array):

			t[i] = int(2**k)
			ttt = -2**(k+1)					
			m_total = 0

			MK = self.G+self.F*(self.K-k)

			for m in np.arange(MK):
				if (m_res == 0):
					phase_inc_swarm = self.u0 [res_idx]
				else:
					phase_inc_swarm = self.u1 [res_idx]

				phase_cappellaro = 0.5*np.angle (self.p_k[ttt+self.points])
				phase = phase_cappellaro + phase_inc_swarm
				
				f0, time_tag, dt = self.calc_acc_phase (t_units = int(t[i]))

				self.field.append(f0)
				self.time_tags.append(time_tag)
				self.msmnt_idx +=1

				m_res = self.ramsey (theta=phase, t = t[i]*self.tau0)					
				self.bayesian_update (m_n = m_res, phase_n = phase, t_n = 2**k)
				res_idx = res_idx + 1
				total_time = total_time + dt
		return total_time

	def non_tracking_estimation (self, do_plot=False):
		self._called_modules.append('non_tracking_estimation')
		idx = 0
		total_time = 0

		# these arrays record the 'real' values of the magnetic field over time
		self.field = []
		self.field_std = []
		self.time_tags = []

		self.est_field = np.array([])
		self.msmnt_times = []
		self.k_array = self.K-np.arange(self.K+1)
		self.curr_fB = 1e6*(np.random.rand()-0.5)*5
		self.init_apriori()
		
		while self.running_time<self.time_interval:
			self.init_apriori ()
			#self.convolve_prob_distribution (t = sensing_t+OH_t, do_plot = False)
			dt = self.nontracking_estimation_routine ()
			total_time = total_time + dt
			self.msmnt_times.append(dt)

			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)
			self.est_field = np.hstack ((self.est_field, est_field*np.ones(self.msmnt_idx)))

		self.field = np.asarray(self.field)
		self.time_tags = np.asarray(self.time_tags)
		self.time_scale = self.time_tags*self.tau0
		self.msmnt_times = np.asarray (self.msmnt_times)

		if do_plot:
			plt.figure(figsize=(20,8))
			plt.subplot (2,1,1)
			plt.plot (self.time_scale*1e3, self.est_field*1e-6, 'o', markersize=4,color='crimson')		
			plt.plot (self.time_scale*1e3, self.field*1e-6, 'royalblue')
			plt.subplot (2,1,2)
			plt.plot (self.time_scale*1e3, np.abs(self.est_field-self.field)*1e-6, 'royalblue')		
			plt.show()
			print "RMSE: ", np.mean(np.abs(self.est_field-self.field))*1e-3, ' kHz'


	def simulate(self, track, do_save = False, do_plot = False, kappa = None, do_debug=False):
		self.k_array = self.K-np.arange(self.K+1)
		self.init_apriori ()

		total_units = 0
		self.curr_step = -1

		self.prev_estim = 0
		self.total_time = np.array([])

		self.running_time = 0
		self.nr_estimations = 0

		while (self.running_time < self.time_interval):
			if not(track):
				self.init_apriori ()

			self.curr_step = self.curr_step + 1

			if track:
				self.adaptive_tracking_estimation(do_plot=do_plot, do_debug=do_debug)
			else:
				self.non_tracking_estimation (do_plot=do_plot)
		self.nr_time_steps = self.curr_step

