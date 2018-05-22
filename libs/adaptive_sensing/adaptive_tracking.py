import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit

import matplotlib.pyplot as plt
from scipy import signal
from simulations.libs.math import statistics as stat
from importlib import reload

reload (stat)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

class TimeSequence ():

	def __init__ (self, nr_time_steps):
		self.nr_time_steps = nr_time_steps
		self.plot_idx = 0

	def set_msmnt_params (self, G, F, N=6, tau0=20e-9, T2 = 1e-6, fid0=0.88, fid1=1.):

		'''
		Set measurement parameters

		Input:
		G, F 	[int]: 	define number of repetitions for each sensing time
		N 		[int]:	number of sensing times, 2^N*t0, 2^(N-1) t0, ... , t0 --- N = K+1
		tau0	[ns]:	minimum sensing time
		T2		[ns]:	spin coherence time (T2*)
		fid0 	[0-1]:	readout fidelity for spin 0, fid0 = 1 is perfect fidelity
		fid1	[0-1]: 	readout fidelity for spin 1, fid1 = 1 is perfect fidelity
		'''

		# Experimental parameters
		self.N = N
		self.tau0 = tau0
		self.T2 = T2
		self.fid0 = fid0
		self.fid1 = fid1
		self.F = F
		self.G = G
		self.K = N-1

		# Quantities required for computation (ex: discretization space)
		self.points = 2**(self.N+1)+3
		self.discr_steps = 2*self.points+1
		self.fB_max = 1./(tau0)
		self.n_points = 2**(self.N+1)
		self.beta = np.linspace (-self.fB_max, self.fB_max, self.discr_steps)
		self.init_apriori()


	def set_magnetic_field (self, fB0, dfB):
		self.fB = np.zeros (self.nr_time_steps)
		self.fB0 = fB0
		self.fB[0] = fB0
		self.dfB = dfB

	def init_apriori (self):

		'''
		Creates an initial uniform probability distribution
		(in Fourier space, only p[0] != 0)
		'''

		#p_k is the probability distribution in Fourier space
		self.p_k = np.zeros (self.discr_steps)+1j*np.zeros (self.discr_steps)
		self.p_k[self.points] = 1/(2.*np.pi)

	def bayesian_update(self, m_n, phase_n, t_n,repetition = None, do_plot=False):

		'''
		Performs the Bayesian update in Fourier space

		m_n 	[0 or 1]	measurement outcome (0 or 1)
		phase_n [rad]		phase of Ramsey experiment
		t_n 	[ns]		sensing time of Ramsey experiment

		do_plot [bool]		DEBUG: plot probability distribution before and after Bayesian update
		'''

		if do_plot:
			y_old, b_mean = self.return_p_fB()
			y_old = y_old/np.sum(y_old)
			m_old = max(y_old)

		p_old = np.copy(self.p_k)
		print('p_old length',len(p_old))
		print('p_old max check',list(p_old).index(max(p_old)), max(p_old))
		p0 = p_old*((1-m_n)-((-1)**m_n)*(self.fid0+1.-self.fid1)/2.) 
		p1 = ((-1)**m_n)*(self.fid0-1.+self.fid1)*0.25*(np.exp(1j*(phase_n))*np.roll(p_old, shift = -t_n)) 
		p2 = ((-1)**m_n)*(self.fid0-1.+self.fid1)*0.25*(np.exp(-1j*(phase_n))*np.roll(p_old, shift = +t_n)) 
		p = p0+p1+p2
		p = (p/np.sum(np.abs(p)**2)**0.5)
		p = p/(2*np.pi*np.real(p[self.points]))
		self.p_k = np.copy (p)#np.abs(np.copy (p)) ###########

		if do_plot:

			p0 = 0.5-0.5*np.cos(2*np.pi*self.beta*t_n*self.tau0+phase_n)
			p1 = 0.5+0.5*np.cos(2*np.pi*self.beta*t_n*self.tau0+phase_n)

			try:
				cb_min = np.min(self.curr_fB_array)
				cb_max = np.max(self.curr_fB_array)
				fom = self.figure_of_merit()
			except:
				pass

			y, b_mean = self.return_p_fB()
			y = y/np.sum(y)
			m = max(y)

			massimo = max([m_old, m])

			fig = plt.figure(figsize = (10,4))
			plt.plot (self.beta*1e-6, y_old, '--', color='dimgray', linewidth = 2)
			try:
				plt.axvspan (cb_min*1e-6, cb_max*1e-6, alpha=0.5, color='green')
			except:
				pass
			plt.plot (self.beta*1e-6, y, 'k', linewidth=4)
			plt.fill_between (self.beta*1e-6, 0, massimo*p0/max(p0), color='crimson', alpha = 0.15)
			plt.fill_between (self.beta*1e-6, 0, massimo*p1/max(p1), color='RoyalBlue', alpha = 0.15)
			plt.axis([b_mean*1e-6-0.5,b_mean*1e-6+0.5, 0, massimo*1.1])
			plt.title ('Bayesian update, res = '+str(m_n)+' -- fom (red-curve): '+str(int(fom)), fontsize=15)
			#fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_bayesian_update.svg', dpi=20)
			#fig.savefig ('D:/Research/WorkData/adptv_tracking_sim/figure_protocollo/'+str(self.plot_idx)+'_bayesian_update.png', dpi=50)
			self.plot_idx += 1
			plt.show()


	def return_p_fB (self):

		'''
		Returns probability distribution in real space

		Outputs
		p_fB 	[array]		probab distrib in real-space
		m 		[float]		average value of probability distribution
		'''

		self.renorm_p_k()
		y = np.fft.fftshift(np.abs(np.fft.fft(self.p_k, self.discr_steps))**2)
		p_fB = y/np.sum(np.abs(y))
		m = np.sum(self.beta*p_fB)
		return p_fB, m

	def renorm_p_k (self):
		# self.p_k = self.p_k/(np.sum(np.abs(self.p_k)**2)**0.5)
		# self.p_k = self.p_k/(2*np.pi*np.real(self.p_k[self.points]))
		self.p_k = self.p_k/(2*np.pi*np.sum(np.abs(self.p_k)**2)**0.5)

	def return_std (self, verbose=False):

		'''
		Returns:
		std_H 		standard deviation for the frequency f_B (calculated from p_{-1})
		fom 		figure of merit	
		'''

		self.renorm_p_k()
		print ("|p_(-1)| = ", np.abs(self.p_k[self.points-1]))
		Hvar = (np.abs(self.p_k[self.points-1]))**(-2)-1
		std_H = ((Hvar**0.5)/(2*np.pi*self.tau0))
		fom = self.figure_of_merit()
		if verbose:
			print ("Std (Holevo): ", std_H*1e-3 , ' kHz --- fom = ', fom)
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

		'''
		Plots current probability distribution P (fB)

		Input
		zoom		[bool]		if zoom=True, plots only region around mean (+/- 2MHz)
		'''

		prob, m = self.return_p_fB()

		plt.figure(figsize=(20,5))
		plt.plot (self.beta*1e-6, prob, 'royalblue', linewidth = 3)
		plt.xlabel ('f_B [MHz]', fontsize = 15)
		if zoom:
			plt.axis ([m*1e-6-2, m*1e-6+2, 0, max(prob)])
		plt.show()

	def convolve_prob_distribution(self, t, dfB_conv = None, do_plot = False):

		'''
		Convolves probability distribution to take into account signal fluctuations

		Input
		t 		[ns]		evolution time
		dfB_conv [##]		\kappa of Wiener process (if None, it uses the pre-defined value)

		do_plot [bool]
		'''

		self.renorm_p_k()

		if do_plot:
			fig = plt.figure(figsize = (10,4))
			y, b_mean = self.return_p_fB()
			ym = max(y)
			plt.plot (self.beta*1e-6, y, '--', color='dimgray', linewidth = 2)
			self.return_std(do_print=True)

		if (dfB_conv == None):
			dfB_conv = self.dfB
		k = np.arange(self.discr_steps)-self.points
		gauss_conv = np.exp(-2*((k*np.pi*dfB_conv*self.tau0)**2)*(t*self.tau0))
		gauss_conv = gauss_conv/np.sum(gauss_conv)
		self.old_p_k = np.copy (self.p_k)
		self.p_k = self.p_k*gauss_conv
		self.renorm_p_k()

		if do_plot:
			y, b_mean = self.return_p_fB()
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

class TimeSequence_overhead (TimeSequence):

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

	def reset_called_modules(self):
		self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase']

	def setup_protocol (self):
		self.total_nr_msmnts = self.G*(2**(self.K+1)-1) + self.F*(2**(self.K+1)-2-self.K)
		self.nr_results = int((self.K+1)*(self.G + self.F*self.K/2.))
		
		folder = self.folder+'/analysis_simulations/scripts/simulations/adaptive_sensing/swarm_optimization/'

		file_old = folder+'phases_G'+str(self.G)+'_F'+str(self.F)+'/swarm_opt_G='+str(self.G)+'_F='+str(self.F)+'_K='+str(self.K)+'.npz'
		round_fid = int(round(self.fid0*100))
		file_new = folder+'incr_fid'+str(round_fid)+'_G'+str(self.G)+'/incr_fid'+str(round_fid)+'_G'+str(self.G)+'F'+str(self.F)+'_K='+str(self.K)+'.npz'
			
		do_it = True
		if os.path.exists (file_new):
			swarm_incr_file = file_new
		elif os.path.exists (file_old):
			swarm_incr_file = file_old
		else:
			print ('ATTENTION!!! No file found for swarm optimization...')
			do_it = False
			
		if do_it:
			swarm_opt_pars = np.load (swarm_incr_file)
			self.u0 = swarm_opt_pars['u0']
			self.u1 = swarm_opt_pars['u1']

	def initialize_in_zero (self, init_value = 0., do_debug=False):

		'''
		Initializes probability distribution corresponding to an initial value [default = 0]

		Input: init_value (float)
		do_debug: plots probability distribution
		'''

		self.init_apriori()
		t = np.zeros(self.K+1)
		self.k_array = self.K-np.arange(self.K+1)
		self.curr_fB = init_value

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

		'''
		Ramsey experiment simulation
		Calculates the probabilities p0 and p1 to get 0/1 and draws an outcome with the probabilities

		Input:
		t 		[ns]		sensing time
		theta	[rad]		Ramsey phase

		Returns:
		Ramsey outcome
		'''

		A = 0.5*(self.fid0 + 1.-self.fid1)
		B = 0.5*(1.-self.fid1 - self.fid0)		

		Bp = np.exp(-(t/self.T2)**2)*np.cos(self.curr_acc_phase+theta)
		p0 = (1-A)-B*Bp
		p1 = A+B*Bp
		np.random.seed()
		result = np.random.choice (2, 1, p=[p0, p1])
		return result[0]	

	def set_magnetic_field (self, dfB):
		self.dfB = dfB

	def calc_acc_phase (self, t_units, do_debug=False):

		'''
		Calculates the phase accumulated by the sensor over time

		Input
		t_units 	[int]		sensing time in units of tau0

		Returns
		1) mean of f_B over the sensing time
		2) units of running time
		3) sensing time in ns (including overhead)
		'''

		fB = np.zeros(t_units)

		# creates a random sequence of 'real' values for fB,
		# with statistics specified by self.dfB
		# and time values multiple of the smallest sensing time tau_0
		fB[0] = self.curr_fB
		for i in np.arange(t_units-1):
			fB[i+1] = fB[i] + self.dfB*((self.tau0)**0.5)*np.random.randn()

		self.curr_acc_phase = 2*np.pi*self.tau0*np.sum(fB)
		avg_fB = np.mean(fB)

		self.curr_fB = fB[-1] + self.dfB*((self.OH)**0.5)*np.random.randn()
		self.running_time = self.running_time + t_units*self.tau0+self.OH
		self.curr_fB_array = fB

		if do_debug:
			print ("Check dfB (total): ", 1e-6*((self.curr_fB - fB[0])/(self.tau0*t_units+self.OH)**0.5), " MHz")
			print ("Check dfB (overhead): ", 1e-6*((self.curr_fB - fB[-1])/(self.OH)**0.5), " MHz")

		return np.mean(fB), self.running_time/self.tau0, t_units*self.tau0+self.OH

	def adptv_tracking_single_step (self, sensing_time_idx, do_debug=False, room_temp = False):
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
			print ("Estimation step: t_units=", t_i, "    -- res:", m_res, '--- elapsed_time: ', (self.running_time -t0)*1e3, "ms")

		return dt/self.tau0


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
		self.initialize_in_zero(init_value = 1e6*(np.random.rand()-0.5)*10, do_debug=do_debug)
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
				dt = self.adptv_tracking_single_step (sensing_time_idx = sensing_time_idx, do_debug=do_debug)

				fom = self.figure_of_merit_std()
				rep_idx += 1

				total_time = total_time + dt
				estim_time = estim_time + dt
				BBB = np.hstack((BBB, self.curr_fB_array))

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
			print ("RMSE: ", np.mean(np.abs(self.est_field-self.field))*1e-3, ' kHz')


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

