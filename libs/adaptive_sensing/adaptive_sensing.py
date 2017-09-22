import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import logging, time, timeit

import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

class EstimationSequenceRT ():

	def __init__ (self, p0, V):
		self.p0 = p0 	#p hoton collection efficiency
		self.V = V   	# odmr contrast

	def set_msmnt_params (self, G, F, N=6, tau0=20e-9, T2 = 1e-6):

		'''
		Set measurement parameters

		Input:
		G, F 	[int]: 	define number of repetitions for each sensing time
		N 		[int]:	number of sensing times, 2^N*t0, 2^(N-1) t0, ... , t0 --- N = K+1
		tau0	[ns]:	minimum sensing time
		T2		[ns]:	spin coherence time (T2*)
		'''

		# Experimental parameters
		self.N = N
		self.tau0 = tau0
		self.T2 = T2
		self.F = F
		self.G = G
		self.K = N-1
		self.k_array = self.K-np.arange(self.K+1)

		self.total_nr_msmnts = self.G*(2**(self.K+1)-1) + self.F*(2**(self.K+1)-2-self.K)
		self.nr_results = int((self.K+1)*(self.G + self.F*self.K/2.))

		# Quantities required for computation (ex: discretization space)
		self.points = 2**(self.N+3)+3
		self.discr_steps = 2*self.points+1
		self.fB_max = 1./(2*tau0)
		self.n_points = 2**(self.N+1)
		self.beta = np.linspace (-self.fB_max, self.fB_max, self.discr_steps)
		self.init_apriori()

	def _load_swarm_phases (self):
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
			print 'ATTENTION!!! No file found for swarm optimization...'
			do_it = False
			
		if do_it:
			swarm_opt_pars = np.load (swarm_incr_file)
			self.u0 = swarm_opt_pars['u0']
			self.u1 = swarm_opt_pars['u1']

	def set_nr_reps (self, R):
		self.R = float(R)

		p0 = self.p0*(1+self.V)
		p1 = self.p0*(1-self.V)
		self.threshold = 0.5*self.R*0.5*(p0+p1)

	def set_magnetic_field (self, fB):
		self.fB = fB

	def init_apriori (self):

		'''
		Creates an initial uniform probability distribution
		(in Fourier space, only p[0] != 0)
		'''

		self.pf = np.ones (self.discr_steps)
		self.pf = self.pf/np.sum(self.pf)

		self.p_k = np.fft.ifftshift(np.fft.ifft(self.pf))
		self.renorm_p_k()

	def renorm_p_k (self):
		self.p_k=self.p_k/(np.sum(np.abs(self.p_k)**2)**0.5)
		self.p_k = self.p_k/(2*np.pi*np.real(self.p_k[self.points]))

	def return_p_fB (self):

		'''
		Returns probability distribution in real space

		Outputs
		p_fB 	[array]		probab distrib in real-space
		m 		[float]		average value of probability distribution
		'''

		y = np.fft.fftshift(np.abs(np.fft.fft(self.p_k))**2)
		p_fB = y/np.sum(np.abs(y))
		m = np.sum(self.beta*p_fB)
		return p_fB, m

	def bayesian_update_photon (self, r, t, phase, do_plot=False):

		y_old = np.copy (self.pf)
		mu = r/(self.R+0.)
		s2 = 2*mu*(1-mu)/self.R
		phi = 2*np.pi*self.beta*t + phase
		x = self.V*np.cos(phi)
		p = 0.5*self.p0*(1 + x)
		gauss = np.exp(-(p-mu)**2/s2)

		self.pf = self.pf*gauss
		self.pf = self.pf/np.sum(np.abs(self.pf))
		self.p_k = np.fft.ifftshift(np.fft.ifft(self.pf))
		self.renorm_p_k()

		if do_plot:
			fig = plt.figure(figsize = (20,4))
			plt.plot (self.beta*1e-6, y_old, '--', color='dimgray', linewidth = 2)
			plt.plot (self.beta*1e-6, self.pf, 'k', linewidth=4)
			plt.title ('Bayesian update, nr photons = '+str(r)+'/'+str(int(self.R)), fontsize=15)
			plt.show()

	def bayesian_update (self, m_n, phase_n, t_n, do_plot=False):

		'''
		Performs the Bayesian update in Fourier space

		m_n 	[0 or 1]	measurement outcome (0 or 1)
		phase_n [rad]		phase of Ramsey experiment
		t_n 	[ns]		sensing time of Ramsey experiment

		do_plot [bool]		DEBUG: plot probability distribution before and after Bayesian update
		'''

		#print 'Bayesian update, outcome: ', m_n, phase_n, t_n
		fid0 = 1
		fid1 = 1

		p_old = np.copy(self.p_k)
		y_old, hhhjh = self.return_p_fB()

		p0 = p_old*((1-m_n)-((-1)**m_n)*(fid0+1.-fid1)/2.) 
		p1 = ((-1)**m_n)*(fid0-1.+fid1)*0.25*(np.exp(1j*(phase_n))*np.roll(p_old, shift = -t_n)) 
		p2 = ((-1)**m_n)*(fid0-1.+fid1)*0.25*(np.exp(-1j*(phase_n))*np.roll(p_old, shift = +t_n)) 
		p = p0+p1+p2
		p = (p/np.sum(np.abs(p)**2)**0.5)
		p = p/(2*np.pi*np.real(p[self.points]))
		self.p_k = np.copy (p)

		if do_plot:

			y, b_mean = self.return_p_fB()
			y = y/np.sum(y)
#
			fig = plt.figure(figsize = (10,4))
			plt.plot (self.beta*1e-6, y_old, '--', color='dimgray', linewidth = 2)
			plt.plot (self.beta*1e-6, y, 'k', linewidth=4)
			plt.show()

	def return_std (self, do_print=False):

		'''
		Returns:
		std_H 		standard deviation for the frequency f_B (calculated from p_{-1})
		fom 		figure of merit	
		'''

		self.renorm_p_k()
		Hvar = (2*np.pi*np.abs(self.p_k[self.points-1]))**(-2)-1
		std_H = ((Hvar**0.5)/(2*np.pi*self.tau0))
		fom = self.figure_of_merit()
		if do_print:
			print "Std (Holevo): ", std_H*1e-3 , ' kHz --- fom = ', fom
		return  std_H, fom

	def ramsey (self, t=0., phase=0.):

		'''
		Ramsey experiment simulation

		Input:
		t 		[ns]		sensing time
		phase	[rad]		Ramsey phase
		R 		[int]		number of Ramsey repetitions

		Returns:
		Number of detectd photons (r)
		'''

		# probability to detect one photon
		prob_success = 0.5*self.p0*(1 + self.V*np.cos(2*np.pi*self.fB*t + phase))
		prb0 = 0.5*(1 + np.cos(2*np.pi*self.fB*t + phase))
		np.random.seed()
		r = np.random.binomial (n = self.R, p = prob_success)
		#print 'Nr photon: ', r,  '[prob_0 = ', prb0,']'
		return r	

	def estimate (self, protocol, do_plot = False):
		self.init_apriori()
		tau = 2**(self.k_array)
		t = np.zeros (self.K+1)

		for i,k in enumerate(self.k_array):

			t[i] = int(2**k)
			ttt = -2**(k+1)					
			m_total = 0

			MK = self.G+self.F*(self.K-k)

			for m in np.arange(MK):

				phase = m*np.pi/MK
				r = self.ramsey (phase=phase, t = t[i]*self.tau0)					

				if (protocol == 'bayesian'):
					self.bayesian_update_photon (r=r, phase = phase, t = t[i]*self.tau0, do_plot = do_plot)
				elif (protocol == 'threshold'):
					if (r >= self.threshold):
						self.bayesian_update (m_n = 0, phase_n=phase, t_n=int(t[i]), do_plot = do_plot)
					else:
						self.bayesian_update (m_n = 1, phase_n=phase, t_n=int(t[i]), do_plot = do_plot)
				else:
					print 'Unknown estimation protocol'

		if (protocol == 'threshold'):
			self.pf, kkkk = self.return_p_fB()

		m = np.sum(self.pf*self.beta)
		return m



	def repeated_experiment (self, nr_reps):
		
		sqe = np.zeros(nr_reps)
		for i in np.arange(nr_reps):
			if (np.mod (i,100) == 0):
				sys.stdout.write(str(i)+'/'+str(nr_reps)+', ')	
			fb = (np.random.rand()-0.5)*40
			self.set_magnetic_field (5*1e6)
			mmm = self.estimate ()
			sqe[i] = np.abs(mmm-fb*1e6)**2
			print fb, mmm*1e-6, 1e-6*sqe[i]**0.5

		print ''

		error = 1e-3*(np.mean (sqe))**0.5
		print ' ---------------------- R = ', self.R
		print 'Error: ', error, ' kHz'
		
		plt.hist(1e-6*(sqe**0.5), bins='auto')
		plt.show()
		print " "
		return error

	def check_threshold (self):

		p0 = self.p0*(1+self.V)
		p1 = self.p0*(1-self.V)
		st0 = []
		st1 = []
		for i in np.arange (5000):
			np.random.seed()
			st0.append(np.random.binomial (n = self.R, p = p0))
			st1.append(np.random.binomial (n = self.R, p = p1))

		plt.hist(st0, bins='auto')
		plt.hist(st1, bins='auto')
		plt.show()

def simulate_bayesian_reps ():
	Nsims = 1000
	R_values = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000, 12500, 15000, 17500, 20000]
	sigma = []

	for R in R_values:
		print '######### R = ', R
		seq = np.zeros(Nsims)
		for i in np.arange(Nsims):
			if (np.mod (i,100) == 0):
				sys.stdout.write(str(i)+'/'+str(Nsims)+', ')	
			s = EstimationSequenceRT (p0=.1, V = .15)
			s.set_msmnt_params (N = 6, G=5, F=3)
			s.set_nr_reps(R=R)
			if (i==0):
				print 'Threshold = ', s.threshold
			fb = (np.random.rand()-0.5)*40
			s.set_magnetic_field (fb*1e6)
			m = s.estimate (protocol = 'threshold', do_plot=False)
			seq[i] = 1e-3*np.abs(m-s.fB)

		plt.hist(seq, bins='auto')
		plt.show()
		err = np.mean(seq)
		print 'Average error: ', err, 'kHz'
		sigma.append (err)


	plt.plot (R_values, sigma, 'ob')
	plt.plot (R_values, sigma, 'RoyalBlue')
	plt.xlabel ('nr of Ramseys per step')
	plt.ylabel ('std [MHz]')
	plt.show()

	plt.semilogy (R_values, sigma, 'ob')
	plt.semilogy (R_values, sigma, 'RoyalBlue')
	plt.xlabel ('nr of Ramseys per step')
	plt.ylabel ('std [MHz]')
	plt.show()

	return R_values, sigma

R, s = simulate_bayesian_reps()