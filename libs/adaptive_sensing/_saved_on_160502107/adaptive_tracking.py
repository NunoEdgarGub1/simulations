
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

reload (stat)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

class TimeSequence ():

	def __init__ (self, nr_time_steps):
		self.nr_time_steps = nr_time_steps
		self.plot_idx = 0

	def set_msmnt_params (self, G, F, N=6, tau0=20e-9, T2 = 1e-6, fid0=0.88, fid1=0.01):
		self.N = N
		self.tau0 = tau0
		self.T2 = T2
		self.fid0 = fid0
		self.fid1 = fid1
		self.F = F
		self.G = G
		self.K = N-1

		self.points = 2**(self.N+1)+3
		self.discr_steps = 2*self.points+1
		self.B_max = 1./(2*tau0)
		self.n_points = 2**(self.N+1)
		self.beta = np.linspace (-self.B_max, self.B_max, self.n_points+1)
		self.beta = self.beta[:-1]
		self.init_apriori()

	def set_magnetic_field (self, B0, dB):
		self.B = np.zeros (self.nr_time_steps)
		self.B0 = B0
		self.B[0] = B0
		self.dB = dB

	def init_apriori (self):
		self.p_k = np.zeros (self.discr_steps)+1j*np.zeros (self.discr_steps)
		self.p_k[self.points] = 1/(2.*np.pi)

	def bayesian_update(self, m_n, phase_n, t_n,repetition = None, do_plot=False):

		if do_plot:
			y_old, b_mean = self.return_p_f()
			y_old = y_old/np.sum(y_old)
			m_old = max(y_old)

		p_old = np.copy(self.p_k)
		p0 = p_old*((1-m_n)-((-1)**m_n)*(self.fid0+self.fid1)/2.) 
		p1 = ((-1)**m_n)*(self.fid0-self.fid1)*0.25*(np.exp(1j*(phase_n))*np.roll(p_old, shift = -t_n)) 
		p2 = ((-1)**m_n)*(self.fid0-self.fid1)*0.25*(np.exp(-1j*(phase_n))*np.roll(p_old, shift = +t_n)) 
		p = p0+p1+p2
		p = (p/np.sum(np.abs(p)**2)**0.5)
		p = p/(2*np.pi*np.real(p[self.points]))
		self.p_k = np.copy (p)

		if do_plot:

			p0 = 0.5-0.5*np.cos(2*np.pi*self.beta*t_n*self.tau0+phase_n)
			p1 = 0.5+0.5*np.cos(2*np.pi*self.beta*t_n*self.tau0+phase_n)
			cb_min = np.min(self.curr_B_array)
			cb_max = np.max(self.curr_B_array)

			fom = self.figure_of_merit()
			y, b_mean = self.return_p_f()
			y = y/np.sum(y)
			m = max(y)

			massimo = max([m_old, m])

			fig = plt.figure(figsize = (10,4))
			plt.plot (self.beta*1e-6, y_old, '--', color='dimgray', linewidth = 2)
			plt.axvspan (cb_min*1e-6, cb_max*1e-6, alpha=0.5, color='green')
			plt.plot (self.beta*1e-6, y, 'k', linewidth=4)
			#plt.plot (self.beta*1e-6, massimo*p0/max(p0), color='crimson', linewidth=1)
			plt.fill_between (self.beta*1e-6, 0, massimo*p0/max(p0), color='crimson', alpha = 0.15)
			#plt.plot (self.beta*1e-6, massimo*p1/max(p1),  color = 'RoyalBlue', linewidth=1)
			plt.fill_between (self.beta*1e-6, 0, massimo*p1/max(p1), color='RoyalBlue', alpha = 0.15)
			plt.axis([b_mean*1e-6-0.5,b_mean*1e-6+0.5, 0, massimo*1.1])
			#plt.axis([-6., -4., 0, massimo*1.1])
			plt.title ('Bayesian update, res = '+str(m_n)+' -- fom (red-curve): '+str(int(fom)), fontsize=15)
			fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_bayesian_update.svg', dpi=20)
			fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_bayesian_update.png', dpi=50)
			self.plot_idx += 1
			plt.show()

	def setup_protocol (self):
		self.total_nr_msmnts = self.G*(2**(self.K+1)-1) + self.F*(2**(self.K+1)-2-self.K)
		self.nr_results = int((self.K+1)*(self.G + self.F*self.K/2.))
		
		folder = 'D:/Research/bonato-lab/analysis_simulations/scripts/simulations/adaptive_sensing/swarm_optimization/'
		file_old = folder+'phases_G'+str(self.G)+'_F'+str(self.F)+'/swarm_opt_G='+str(self.G)+'_F='+str(self.F)+'_K='+str(self.K)+'.npz'
		round_fid = int(round(self.fid0*100))
		file_new = folder+'incr_fid'+str(round_fid)+'_G'+str(self.G)+'/incr_fid'+str(round_fid)+'_G'+str(self.G)+'F'+str(self.F)+'_K='+str(self.K)+'.npz'
			
		do_it = True
		if os.path.exists (file_new):
			swarm_incr_file = file_new
		elif os.path.exists (file_old):
			swarm_incr_file = file_old
		else:
			print 'ATTENTION!!! No file found for swarm optimization...'
			#print file_new
			#print file_old
			do_it = False
			
		if do_it:
			swarm_opt_pars = np.load (swarm_incr_file)
			self.u0 = swarm_opt_pars['u0']
			self.u1 = swarm_opt_pars['u1']

	def return_p_f (self):
		self.renorm_p_k()
		y = np.fft.fftshift(np.abs(np.fft.fft(self.p_k, self.n_points))**2)
		prob = y/np.sum(np.abs(y))
		m = np.sum(self.beta*prob)
		return prob, m

	def renorm_p_k (self):
		self.p_k=self.p_k/(np.sum(np.abs(self.p_k)**2)**0.5)
		self.p_k = self.p_k/(2*np.pi*np.real(self.p_k[self.points]))

	def return_std (self, do_print=False):
		self.renorm_p_k()
		Hvar = (2*np.pi*np.abs(self.p_k[self.points-1]))**(-2)-1
		std_H = ((Hvar**0.5)/(2*np.pi*self.tau0))
		fom = self.figure_of_merit()
		if do_print:
			print "Std (Holevo): ", std_H*1e-3 , ' kHz --- fom = ', fom
		return  std_H, fom

	def figure_of_merit (self):
		self.renorm_p_k()
		fom = int(np.abs(1./(1-2*np.pi*np.abs(self.p_k[self.points-1]))))
		return fom

	def figure_of_merit_std (self):
		self.renorm_p_k()
		Hvar = (2*np.pi*np.abs(self.p_k[self.points-1]))**(-2) - 1
		fom = Hvar**0.5/(2*np.pi*self.tau0)
		return fom

	def plot_probability_distr (self, zoom = False):
		prob, m = self.return_p_f()

		plt.figure(figsize=(20,5))
		plt.plot (self.beta*1e-6, prob, 'royalblue', linewidth = 3)
		plt.xlabel ('f_B [MHz]', fontsize = 15)
		if zoom:
			plt.axis ([m*1e-6-2, m*1e-6+2, 0, max(prob)])
		plt.show()

	def convolve_prob_distribution(self, t, dB_conv = None, do_plot = False):
		self.renorm_p_k()

		if do_plot:
			fig = plt.figure(figsize = (10,4))
			y, b_mean = self.return_p_f()
			ym = max(y)
			plt.plot (self.beta*1e-6, y, '--', color='dimgray', linewidth = 2)
			self.return_std(do_print=True)

		if (dB_conv == None):
			dB_conv = self.dB
		k = np.arange(self.discr_steps)-self.points
		#dphi = 2*np.pi*dB_conv*self.tau0
		#gauss_conv = np.exp(-(k**2)*(dphi**2)*(t*self.tau0)/2.)
		gauss_conv = np.exp(-2*((k*np.pi*dB_conv*self.tau0)**2)*(t*self.tau0))
		gauss_conv = gauss_conv/np.sum(gauss_conv)
		self.old_p_k = np.copy (self.p_k)
		self.p_k = self.p_k*gauss_conv
		self.renorm_p_k()

		if do_plot:
			y, b_mean = self.return_p_f()
			plt.plot (self.beta*1e-6, y, 'k', linewidth=4)
			#plt.axis([-6., -4., 0, ym*1.1])
			plt.axis([b_mean*1e-6-0.5,b_mean*1e-6+0.5, 0, ym])
			plt.title ('Convolution, t~'+str(int(t*self.tau0*10e6)/10.)+'us  -- fom = '+str(self.figure_of_merit()), 
														fontsize=15)

			fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_convolution.svg')
			fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_convolution.png')
			self.plot_idx += 1
			plt.show()
			self.return_std(do_print=True)

class TimeSequence_set_B (TimeSequence):

	def __init__ (self, time_interval):
		self.time_interval = time_interval
		self._B_dict = {}
		self.kappa = None
		self.curr_B_idx = 0
		self.filter_b, self.filter_a = signal.butter(2, 0.01, analog=False)

	def ramsey (self, t=0., theta=0.):

		A = 0.5*(self.fid0 + self.fid1)
		B = 0.5*(self.fid1 - self.fid0)		

		p0 = (1-A)-B*np.exp(-(t/self.T2)**2)*np.cos(self.curr_acc_phase+theta)
		p1 = A+B*np.exp(-(t/self.T2)**2)*np.cos(self.curr_acc_phase+theta)

		np.random.seed()
		result = np.random.choice (2, 1, p=[p0, p1])
		return result[0]	

	def set_initial_Bfield(self):
		self.nr_units = int(self.time_interval/self.tau0)
		self.total_time = np.array([])

		B = np.zeros(int(1.2*self.nr_units))
		B[0] = 0
		for i in np.arange(len(B)-1):
			B[i+1] = B[i] + self.dB*((self.tau0)**0.5)*np.random.randn()
		self.initial_B_field = signal.filtfilt(self.filter_b, self.filter_a, B)
		#self.initial_B_field = B
		self.curr_B_idx = 0

	def set_magnetic_field (self, dB):
		self.dB = dB
		self.set_initial_Bfield()

	def calc_acc_phase (self, t_units):
		fff = self.initial_B_field[self.curr_B_idx:self.curr_B_idx+t_units]
		self.curr_acc_phase = 2*np.pi*self.tau0*np.sum(fff)
		self.curr_B = self.initial_B_field[self.curr_B_idx+t_units]
		self.curr_B_idx = self.curr_B_idx + t_units
		return fff

	def est_routine_fullSeq (self, track, sequence, step_idx = 1):
		k_array = self.K-np.arange(self.K+1)
		tau = 2**(k_array)
		t = np.zeros (self.K+1)
		res_idx = 0
		m_res = 0

		total_time = 0
		for i,k in enumerate(k_array):

			t[i] = int(2**(k))
			ttt = -2**(k+1)					
			m_total = 0
			if track:
				if (step_idx==0):
					MK = self.G+self.F*(self.K-k)
				else:
					MK = sequence [i-1]
			else:
				MK = self.G+self.F*(self.K-k)

			for m in np.arange(MK):
				if ((step_idx==0) or not(track)):
					if (m_res == 0):
						phase_inc_swarm = self.u0 [res_idx]
					else:
						phase_inc_swarm = self.u1 [res_idx]

				phase_cappellaro = 0.5*np.angle (self.p_k[ttt+self.points])
				if ((step_idx==0) or not(track)):
					phase = phase_cappellaro + phase_inc_swarm
				else:
					phase = phase_cappellaro 

				f0 = self.calc_acc_phase (t_units = t[i])
				try:
					self.field = np.hstack((self.field, f0))
				except:
					self.field = []
				m_res = self.ramsey (theta=phase, t = t[i]*self.tau0)					
				self.bayesian_update (m_n = m_res, phase_n = phase, t_n = 2**(k))
				res_idx = res_idx + 1
				total_time = total_time + t[i]
		return total_time

	def est_routine_singleStep (self, sensing_time_idx):
		k_array = self.K-np.arange(self.K+1)
		k = k_array[sensing_time_idx]
		t_i = int(2**(k))
		ttt = -2**(k+1)					
		phase_cappellaro = 0.5*np.angle (self.p_k[ttt+self.points])
		f0 = self.calc_acc_phase (t_units = t_i)
		try:
			self.field = np.hstack((self.field, f0))
		except:
			self.field = []
		m_res = self.ramsey (theta=phase_cappellaro, t = t_i*self.tau0)					
		self.bayesian_update (m_n = m_res, phase_n = phase_cappellaro, t_n = t_i)
		return t_i

	def self_tuning_estimation (self, kappa = None, sigma_check = True):
		fom = 0
		k_array = self.K-np.arange(self.K+1)
		idx = 0
		total_time = 0
		self.field = []
		#self.init_apriori ()

		if (self.curr_step == 0):
			total_time = total_time + self.est_routine_fullSeq (track=True, step_idx=0,
					 sequence=[2,2,2,2,2,3,3,3,3,3])
			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)

		if (kappa and np.mod (self.curr_step, kappa)==0):
			#print self.curr_step, " --- kappa check!!!"
			if (self.fid0 > 0.96):
				sq = [1,1,1,1,2,2,2,2,3,3]
			else:
				sq = [2,2,2,2,2,3,3,3,3,3]
			total_time = total_time + self.est_routine_fullSeq (track=True, step_idx=1,
					 sequence=sq)
			est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)
		else:		
			while (((fom<self.fom_threshold) or (idx < 1*self.N)) and (idx < 5*self.N)):
				total_time = total_time + self.est_routine_singleStep (sensing_time_idx = np.mod(idx, self.N))
				#print self.curr_rep, self.curr_step
				#self.plot_probability_distr()
				fom = (1-np.abs(self.p_k[self.points-1])/np.abs(self.p_k[self.points]))**(-1)
				est_field = -np.angle(self.p_k[self.points-1])/(2*np.pi*self.tau0)

				if (sigma_check and (fom>self.fom_threshold)):
					if (np.abs(est_field - self.prev_estim)> self.sigma_check_thr*self.dB*(total_time*self.tau0)**0.5):
						fom = 0
						if [self.curr_rep, self.curr_step] not in self.sigma_check_list:
							self.sigma_check_list.append ([self.curr_rep, self.curr_step])
				idx = idx + 1
		self.prev_estim = est_field
		#print "-----------  New estimation!!!!", self.prev_estim

		beta = np.linspace (-self.B_max, self.B_max, self.n_points)
		y = np.fft.fftshift(np.abs(np.fft.fft(self.p_k, self.n_points))**2)
		prob = y/np.sum(y) 
		#idx = np.argmax(prob)
		#B = beta[idx]
		B = np.sum(prob*beta)
		self.est_field = B*np.ones(len(self.field))
		self.est_field = B*np.ones(len(self.field))
		return total_time, B

	def fixed_time_estimation (self, sequence, track, auto_correct = False, step_idx = 1):

		if not(track):
			self.init_apriori ()

		total_time = 0
		self.field = []
		total_time = total_time + self.est_routine_fullSeq (track=track, step_idx=step_idx, sequence=sequence)
		fom = (1-np.abs(self.p_k[self.points-1])/np.abs(self.p_k[self.points]))**(-1)
		if auto_correct:
			rep_seq_idx = 0
			while ((fom<self.fom_threshold) and (rep_seq_idx<5) and track):
				total_time = total_time + self.est_routine_fullSeq (track=track, step_idx=step_idx, sequence=[1,1,1,1,1,1,1,1,1,1])
				fom = (1-np.abs(self.p_k[self.points-1])/np.abs(self.p_k[self.points]))**(-1)
				rep_seq_idx = rep_seq_idx + 1

		beta = np.linspace (-self.B_max, self.B_max, self.n_points)
		y = np.fft.fftshift(np.abs(np.fft.fft(self.p_k, self.n_points))**2)
		prob = y/np.sum(y)
		idx = np.argmax(prob)
		B = beta[idx]
		#B = np.sum(prob*beta)
		self.est_field = B*np.ones(len(self.field))
		return total_time, B

	def simulate(self, track, protocol, do_save = True, do_plot = False, kappa = None):
		self.init_apriori ()

		self.campo = np.array([])
		self.campo_stimato = np.array([])

		total_units = 0
		self.curr_step = -1

		self.prev_estim = 0
		self.sigma_check_list = []

		while (total_units < self.nr_units):
			if not(track):
				self.init_apriori ()

			self.curr_step = self.curr_step + 1
			if (protocol == 'self_tuning'):
				t, B = self.self_tuning_estimation (kappa = kappa)
			elif (protocol [:10] == 'fixed_time'):
				if protocol [11:] == 'seq1':
					sequence = [1,1,1,1,1,1,1,1,1,1]
				t, B = self.fixed_time_estimation (sequence = sequence, track = track, auto_correct = False)
			total_units = total_units + t
			self.campo = np.hstack((self.campo, self.field))
			self.campo_stimato = np.hstack((self.campo_stimato, self.est_field))
			self.total_time = np.hstack ((self.total_time, np.array([t])))

			if track:
				self.convolve_prob_distribution(t, do_plot = False)	

		self.nr_time_steps = self.curr_step
		if (protocol == 'self_tuning'):
			self.sigma_check_array = np.zeros ([len(self.sigma_check_list), 2])
			for i in np.arange(len(self.sigma_check_list)):
				self.sigma_check_array [i, 0] = self.sigma_check_list [i][0]
				self.sigma_check_array [i, 1] = self.sigma_check_list [i][1]

