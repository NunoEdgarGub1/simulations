
import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit

import matplotlib.pyplot as plt
from scipy import signal
from analysis_simulations.libs.math import statistics as stat
from analysis_simulations.libs.tools import toolbox

from analysis_simulations.libs.adaptive_sensing import adaptive_tracking as tracking_lib

reload (stat)
reload (tracking_lib)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

class TimeSequence_overhead (tracking_lib.TimeSequence):

	def __init__ (self, time_interval, overhead):
		self.time_interval = time_interval
		self._B_dict = {}
		self.kappa = None
		self.curr_B_idx = 0
		self.filter_b, self.filter_a = signal.butter(2, 0.01, analog=False)
		self.OH = overhead
		self.set_B = []
		self.est_B = []
		self.curr_B = 0
		self.plot_idx = 0
		self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase', 'est_routine_singleStep']

	def reset_called_modules(self):
		self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase', 'est_routine_singleStep']

	def initialize_in_zero (self, init_value = 0., do_debug=False):
		self.init_apriori()
		t = np.zeros(self.K+1)
		self.k_array = self.K-np.arange(self.K+1)
		self.curr_B = init_value

		for i,k in enumerate(self.k_array):
			ttt = -2**(k+1)					
			MK = self.G+self.F*(self.K-k)

			for m in np.arange(MK):
				phase = 0.5*np.angle (self.p_k[ttt+self.points])
				self.curr_acc_phase = 2*np.pi*(2**k)*self.tau0*init_value
				m_res = self.ramsey (theta=phase, t = (2**k)*self.tau0)									
				self.bayesian_update (m_n = m_res, phase_n = phase, t_n = 2**k)

		if do_debug:
			self.plot_probability_distr(zoom=True)

	def ramsey (self, t=0., theta=0.):

		A = 0.5*(self.fid0 + self.fid1)
		B = 0.5*(self.fid1 - self.fid0)		

		p0 = (1-A)-B*np.exp(-(t/self.T2)**2)*np.cos(self.curr_acc_phase+theta)
		p1 = A+B*np.exp(-(t/self.T2)**2)*np.cos(self.curr_acc_phase+theta)
		np.random.seed()
		result = np.random.choice (2, 1, p=[p0, p1])
		return result[0]	

	def set_magnetic_field (self, dB):
		self.dB = dB

	def calc_acc_phase (self, t_units, do_debug=False):
		B = np.zeros(t_units)
		B[0] = self.curr_B
		for i in np.arange(t_units-1):
			B[i+1] = B[i] + self.dB*((self.tau0)**0.5)*np.random.randn()

		self.curr_acc_phase = 2*np.pi*self.tau0*np.sum(B)
		avg_B = np.mean(B)
		self.curr_B = B[-1] + self.dB*((self.OH)**0.5)*np.random.randn()
		if do_debug:
			print "Check dB (total): ", 1e-6*((self.curr_B - B[0])/(self.tau0*t_units+self.OH)**0.5), " MHz"
			print "Check dB (overhead): ", 1e-6*((self.curr_B - B[-1])/(self.OH)**0.5), " MHz"
		self.running_time = self.running_time + t_units*self.tau0+self.OH
		self.curr_B_array = B
		return np.mean(B), self.running_time/self.tau0, t_units*self.tau0+self.OH

	def est_routine_singleStep (self, sensing_time_idx, do_debug=False, room_temp = False):
		k = self.k_array[sensing_time_idx]
		t_i = int(2**k)
		ttt = -2**(k+1)
		t0 = self.running_time					

		phase_cappellaro = 0.5*np.angle (self.p_k[ttt+self.points])

		m = 0
		f0, time_tag, dt = self.calc_acc_phase (t_units = t_i, do_debug=do_debug)
		m_res = self.ramsey (theta=phase_cappellaro, t = t_i*self.tau0)	
		self.bayesian_update (m_n = m_res, phase_n = phase_cappellaro, t_n = t_i, do_plot=do_debug)
		if do_debug:
			print "Estimation step: t_units=", t_i, "    -- res:", m_res, '--- elapsed_time: ', (self.running_time -t0)*1e3, "ms"

		return dt/self.tau0


	def fully_adaptive_estimation (self, do_plot = False, do_debug=False):
		self._called_modules.append('fully_adaptive_estimation')
		fom = 1e12
		idx = 0
		total_time = 0
		self.field = []
		self.field_std = []
		self.time_tags = []
		self.fom = []
		self.est_field = []
		self.prev_estim = self.curr_B
		rep_array = np.arange(self.nr_sensing_reps)
		self.k_array = self.K-np.arange(self.K+1)
		sigma_reps_idx = 0

		self.init_apriori ()
		estim_time = 0
		self.initialize_in_zero(init_value = 1e6*(np.random.rand()-0.5)*10, do_debug=do_debug)
		#self.initialize_in_zero(init_value = 1e6*(1.3), do_debug=do_debug)
		fom = 0
		rep_idx = 0
		est_field = 0
		estim_time = 0
		sigma = -100


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
				dt = self.est_routine_singleStep (sensing_time_idx = sensing_time_idx, do_debug=do_debug)

				fom = self.figure_of_merit_std()
				rep_idx += 1

				total_time = total_time + dt
				estim_time = estim_time + dt
				BBB = np.hstack((BBB, self.curr_B_array))

			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)
			if (fom<0.15*self.fom_array[sensing_time_idx]):
				if (sensing_time_idx >0):
					idx=idx-1
				else:
					idx = 0
			else:
				idx +=1

			self.fom.append(fom)
			self.est_field.append(est_field)
			self.time_tags.append(total_time)
			self.field.append (np.mean(BBB))
			self.field_std.append(np.std(BBB))

		self.prev_estim = est_field
		self.field = np.asarray(self.field)
		self.time_tags = np.asarray(self.time_tags)
		self.est_field = np.asarray(self.est_field)
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

			ind = np.where(self.fom>0)
			plt.figure(figsize=(20,10))
			plt.subplot (3,1,1)
			plt.plot (self.time_scale[ind]*1e3, self.est_field[ind]*1e-6, 'o', markersize=4,color='crimson')		
			plt.plot (self.time_scale*1e3, self.field*1e-6, 'royalblue')
			ind = np.where(self.fom>self.fom_threshold)
			plt.subplot (3,1,2)
			plt.plot (self.time_scale[ind]*1e3, self.est_field[ind]*1e-6, 'o', markersize=4, color='crimson')		
			plt.plot (self.time_scale*1e3, self.field*1e-6, 'royalblue')
			ind = np.where(self.fom>np.abs(np.mean(self.fom)-np.std(self.fom)))
			plt.subplot (3,1,3)
			plt.plot (self.time_scale[ind]*1e3, self.est_field[ind]*1e-6, 'o', markersize=5, color='crimson')		
			plt.plot (self.time_scale*1e3, self.field*1e-6, 'royalblue')
			plt.show()

			plt.figure(figsize = (20,5))
			plt.subplot(3,1,1)
			plt.plot (np.abs(self.est_field - self.field)*1e-3)
			print "Mean error, no fom:", np.mean(np.abs(self.est_field - self.field))
			x = np.arange(len(self.est_field))
			ind = np.where(self.fom>self.fom_threshold)
			plt.subplot(3,1,2)
			plt.plot (np.abs(self.est_field[ind] - self.field[ind])*1e-3)
			print "Mean error, fom=fom_thr="+str(self.fom_threshold), ": ", np.mean(np.abs(self.est_field[ind] - self.field[ind]))
			x = np.arange(len(self.est_field))
			ind = np.where(self.fom>np.abs(np.mean(self.fom)-np.std(self.fom)))
			plt.subplot(3,1,3)
			plt.plot (np.abs(self.est_field[ind] - self.field[ind])*1e-3)
			print "Mean error, fom=mean(fom)-std(fom)="+str(int(np.mean(self.fom)-np.std(self.fom)))+": ", np.mean(np.abs(self.est_field[ind] - self.field[ind]))
			plt.show()


	def est_routine_fullSeq (self):
		self._called_modules.append('est_routine_fullSeq')
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
				
				f0, time_tag, dt = self.calc_acc_phase (t_units = t[i])

				self.field.append(f0)
				self.time_tags.append(time_tag)
				self.msmnt_idx +=1

				m_res = self.ramsey (theta=phase, t = t[i]*self.tau0)					
				self.bayesian_update (m_n = m_res, phase_n = phase, t_n = 2**k)
				res_idx = res_idx + 1
				total_time = total_time + dt
		return total_time

	def fixed_time_estimation (self, track=False, do_plot=False):
		self._called_modules.append('fixed_time_estimation')
		idx = 0
		total_time = 0
		self.field = []
		self.field_std = []
		self.time_tags = []
		self.est_field = np.array([])
		self.msmnt_times = []
		self.k_array = self.K-np.arange(self.K+1)
		self.curr_B = 1e6*(np.random.rand()-0.5)*5
		self.init_apriori()
		
		while self.running_time<self.time_interval:
			self.init_apriori ()
			#self.convolve_prob_distribution (t = sensing_t+OH_t, do_plot = False)
			dt = self.est_routine_fullSeq ()
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


	def simulate(self, track, protocol, do_save = True, do_plot = False, kappa = None, do_debug=False):
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
			if (protocol [:10] == 'fixed_time'):
				sequence = None
				self.fixed_time_estimation (do_plot=do_plot)
			elif (protocol == 'fully_adaptive'):
				self.fully_adaptive_estimation(do_plot=do_plot, do_debug=do_debug)

		self.nr_time_steps = self.curr_step

