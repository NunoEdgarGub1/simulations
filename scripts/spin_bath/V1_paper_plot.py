import numpy as np
import pylab as plt
import matplotlib
import os
import cPickle

from os.path import isfile, join
from analysis_simulations.libs.spin import rabi
from matplotlib.colors import ListedColormap


matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

def return_value(k, string):
    v_idx = k.find(string)
    returnvalue = 0
    for i in range(1,10):        
        try:
            returnvalue = float(k[v_idx-i:v_idx])
        except:
            pass
    return returnvalue

class V1_SiC ():

	def __init__(self, B_Gauss, verbose = True):
		self.ZFS = 4e6
		self.g = 2*14e9
		self.B = B_Gauss*1e-4
		E32m = -1.5*self.g*self.B
		E32p = +1.5*self.g*self.B
		E12m = self.ZFS-0.5*self.g*self.B
		E12p = self.ZFS+0.5*self.g*self.B
		self.E1 = E32m
		self.E2 = E12m
		self.E3 = E12p
		self.E4 = E32p
		if verbose:
			print "### Transitions: "
			print " 	-3/2 <--> -1/2, frq: ", abs(E12m-E32m)*1e-6, ' MHz'
			print " 	-1/2 <--> +1/2, frq: ", abs(E12p-E12m)*1e-6, ' MHz'
			print " 	+1/2 <--> +3/2, frq: ", abs(E12p-E32p)*1e-6, ' MHz'
		self.PL_sweep = None
		self.v_factor = 1e-4
		self.n_exp = 2
		self.set_intensity_transitions(1.,1.,1.)

	def set_folders (self, data_folder, work_folder):
		self.data_folder = data_folder
		self.folder = work_folder

	def set_colormap (self, colormap='viridis'):
		self.cmap = colormap

	def set_decoherence_exponent (self, n_exp):
		self.n_exp = n_exp

	def set_intensity_transitions (self, I_m32_m12=1, I_m12_p12=1, I_p12_p32=1):
		alpha0 = I_m32_m12
		beta0 = I_m12_p12
		gamma0 = I_p12_p32
		self.alpha = alpha0/alpha0
		self.beta = beta0/alpha0
		self.gamma = gamma0/alpha0

	def set_ODMR_pars (self, polariz_array, ODMR_contrast_array, verbose = False):
		#init_polarization is a four-element array which stores the occupancy of the 
		#four ms spin sublevels at start.
		#Spin sublevels are ordered as [-3/2, -1/2, +1/2, +3/2]
		#For example, if we completely polarize into +/-3/2, then the array
		#would be [1,0,0,1]
		self.init_polarization = polariz_array/(np.sum(polariz_array)+0.)
		self.odmrC = ODMR_contrast_array

		if verbose:
			print
			print " ---- Simulation parameters ----"
			print "Initial polarization: ", self.init_polarization
			print "Spin-dependent PL:    ", self.odmrC
		#print self.init_polarization

	def set_decay (self, t1, t2):
		#self.decay_function = 0.5*np.exp (-self.t/t1) + 0.5*np.exp(-self.t/t2)
		#self.decay_function = 0.5*np.exp (-(self.t/t1)**2) + 0.5*np.exp(-self.t/t2)
		self.decay_function = 0.2*(1+(self.t/t1)**2)**(-0.25) + 0.8*np.exp(-self.t/t2)

	def rabi (self, f, Omega, do_plot = False, do_fft = False):
		t = self.t
		H = np.array ([[self.E1+3*f/2., self.alpha*Omega*(3**2/4.), 0, 0],
				[self.alpha*Omega*(3**0.5/4.), self.E2 + f/2., self.beta*Omega*0.5, 0],
				[0,self.beta*Omega*0.5, self.E3-f/2., self.gamma*Omega*(3**0.5/4.)],
				[0, 0, self.gamma*Omega*(3**0.5/4.), self.E4-3*f/2.]])

		w, V = np.linalg.eigh(H)
		#The eigenvectors are in the columns of V: V[:,i] is the eigenvector corresponding
		#to the eigenvalue w[i]
		V = np.transpose (V)

		init_state_array = [0,1,2,3]
		p1 = np.zeros(len(t))
		p2 = np.zeros(len(t))
		p3 = np.zeros(len(t))
		p4 = np.zeros(len(t))
		for init_state in init_state_array:
			factor = self.init_polarization [init_state]

			C = np.linalg.pinv(V)
			e0 = (np.exp(1j*2*np.pi*w[0]*t))
			e1 = (np.exp(1j*2*np.pi*w[1]*t))
			e2 = (np.exp(1j*2*np.pi*w[2]*t))
			e3 = (np.exp(1j*2*np.pi*w[3]*t))

			psi_f1 = np.abs(e0*C[init_state,0]*V[0, 0]+e1*C[init_state,1]*V[1, 0]+e2*C[init_state,2]*V[2, 0]+e3*C[init_state,3]*V[3, 0])**2
			psi_f2 = np.abs(e0*C[init_state,0]*V[0, 1]+e1*C[init_state,1]*V[1, 1]+e2*C[init_state,2]*V[2, 1]+e3*C[init_state,3]*V[3, 1])**2
			psi_f3 = np.abs(e0*C[init_state,0]*V[0, 2]+e1*C[init_state,1]*V[1, 2]+e2*C[init_state,2]*V[2, 2]+e3*C[init_state,3]*V[3, 2])**2
			psi_f4 = np.abs(e0*C[init_state,0]*V[0, 3]+e1*C[init_state,1]*V[1, 3]+e2*C[init_state,2]*V[2, 3]+e3*C[init_state,3]*V[3, 3])**2

			#Euristically introduce decoherence
			psi_f1 = (psi_f1-np.mean(psi_f1)*np.ones(len(psi_f1)))*self.decay_function + np.mean(psi_f1)*np.ones(len(psi_f1))
			psi_f2 = (psi_f2-np.mean(psi_f2)*np.ones(len(psi_f2)))*self.decay_function + np.mean(psi_f2)*np.ones(len(psi_f2))
			psi_f3 = (psi_f3-np.mean(psi_f3)*np.ones(len(psi_f3)))*self.decay_function + np.mean(psi_f3)*np.ones(len(psi_f3))
			psi_f4 = (psi_f4-np.mean(psi_f4)*np.ones(len(psi_f4)))*self.decay_function + np.mean(psi_f4)*np.ones(len(psi_f4))

			norm = psi_f1 + psi_f2 + psi_f3 + psi_f4
			psi_f1 = psi_f1/norm
			psi_f2 = psi_f2/norm
			psi_f3 = psi_f3/norm
			psi_f4 = psi_f4/norm
			p1 = p1 + factor*psi_f1
			p2 = p2 + factor*psi_f2
			p3 = p3 + factor*psi_f3
			p4 = p4 + factor*psi_f4


		PL = (self.odmrC[0]*p1+self.odmrC[1]*p2+self.odmrC[2]*p3+self.odmrC[3]*p4)

		self.p1 = p1/(p1+p2+p3+p4)
		self.p2 = p2/(p1+p2+p3+p4)
		self.p3 = p3/(p1+p2+p3+p4)
		self.p4 = p4/(p1+p2+p3+p4)

		self.PL = PL 
		
		if do_plot:
			plt.figure(figsize = (15,5))
			plt.plot (t*1e6, PL, 'crimson', linewidth = 2)
			plt.xlabel ('time [us]', fontsize=15)
			plt.ylabel ('PL [kcounts]', fontsize = 15)
			plt.axis([min(t)*1e6, max(t)*1e6, 0, max(PL)*1.1])
			plt.show()

		return PL

	def rabi_sweep_drive (self, init_frq, end_frq, nr_steps, Omega, do_plot = True, add_colorbar = False):
		self.drive_frq = np.linspace (init_frq, end_frq, nr_steps)
		self.PL_sweep = np.zeros((nr_steps, len(self.t)))

		i = 0
		for f in self.drive_frq:
			PL = self.rabi (f = f, Omega = Omega, do_plot = False)
			self.PL_sweep [i,:] = PL
			i += 1

		self.Omega = Omega


	def calc_fft (self, in_fft = 0, config = 'modulo_squared'):
		if (self.PL_sweep == None):
			print "No Rabi data found"
		else:
			i = 0
			for f in self.drive_frq:
				PL = self.PL_sweep [i,:]

				if config == 'real_part':
					fff = np.fft.ifftshift(np.real(np.fft.fft(PL)))
				elif config == 'phase':
					fff = np.fft.ifftshift(np.angle(np.fft.fft(PL)))
				elif config == 'modulo_squared':
					fff = np.fft.ifftshift(np.abs(np.fft.fft(PL))**2)
				else:
					print "Unknown FFT setting. Calculating modulo_squared"
					fff = np.fft.ifftshift(np.abs(np.fft.fft(PL))**2)
	
				fff = fff/np.sum(fff)
				fff = fff[len(self.t)/2+in_fft:]

				if (i==0):
					self.PL_fft = fff
				else:
					self.PL_fft = np.vstack((self.PL_fft, fff))
				i += 1
			self.df = 1./max(self.t)
			freq = np.fft.ifftshift(np.fft.fftfreq(len(self.t)))
			r_frq = freq*self.df*len(self.t)
			self.r_frq = r_frq[len(self.t)/2+in_fft:]

	def plot_time_domain_rabi (self, add_colorbar=False, xlim =[0, 1], save_fig = False):
		X, Y = plt.meshgrid (self.t*1e6, self.drive_frq*1e-6)
		fig = plt.figure(figsize=(3,5))
		plt.imshow (self.PL_sweep, vmin = np.min(np.min(self.PL_sweep)), vmax= np.max(np.max(self.PL_sweep)),
				extent = [min(self.t*1e6), max(self.t*1e6), min(self.drive_frq*1e-6), max(self.drive_frq*1e-6)],
				aspect = 'auto', cmap = 'viridis', interpolation = 'nearest')
		plt.xlabel ('time [us]', fontsize = 18)
		plt.ylabel ('driving frequency [MHz]', fontsize = 18)
		plt.axis([xlim[0], xlim[1], self.drive_frq[0]*1e-6, self.drive_frq[-1]*1e-6])
		if add_colorbar:
			plt.colorbar()
		if save_fig:
			fig.savefig (self.folder+'/'+save_fig, dpi = 400)
		plt.show()

	def plot_fft (self, do_renorm = False, add_colorbar=False, save_fig = False, xlim = [0,30], figsize = [4,9]):

		X, Y = plt.meshgrid (self.r_frq*1e-6, self.drive_frq*1e-6)
		self.PL_fft[np.isnan(self.PL_fft)] = 0

		fig = plt.figure(figsize=(figsize[0], figsize[1]))
		plt.imshow (self.PL_fft, vmin = np.min(np.min(self.PL_fft)), vmax= 0.8*np.max(np.max(self.PL_fft)),
				extent = [min(self.r_frq*1e-6), max(self.r_frq*1e-6), min(self.drive_frq*1e-6), max(self.drive_frq*1e-6)],
				aspect = 'auto', cmap = 'viridis', interpolation = 'nearest')		
		plt.xlabel ('Rabi frequency [MHz]', fontsize = 18)
		plt.ylabel ('driving frequency [MHz]', fontsize = 18)
		if add_colorbar:
			plt.colorbar()
		plt.axis([xlim[0], xlim[1], min(self.drive_frq*1e-6), max(self.drive_frq*1e-6)])
		if save_fig:
			fig.savefig (self.folder+'/'+save_fig, dpi = 400)
		plt.show()

	def getRabiData(self, filename):
	    with open(filename,'rb') as f:
	        d=cPickle.load(f)
	        return d['measurement']['tau'], d['measurement']['spin_state']  

	def load_data (self, dBm):

		data_path=self.data_folder + '/Rabi_160_190MHz_-'+str(45-dBm)+'dBm/'
		file_list = [f for f in os.listdir(data_path) if (isfile(join(data_path, f)) and f.endswith('.pys'))]
		nr_steps = len(file_list)

		os.chdir(data_path)

		self.data_dict = {}
		self.RF_frq_array = np.zeros (len(file_list))
		self.drive_frq = np.zeros(nr_steps)

		i = 0
		for g in file_list:
		    tau, spin_state = self.getRabiData(g) 
		    RF_frq = return_value(g,'MHz')
		    extr_dbm = return_value(g,'dBm')
		    self.RF_frq_array [i] = RF_frq
		    self.data_dict[str(RF_frq)+'MHz'] = {}
		    self.data_dict[str(RF_frq)+'MHz']['tau'] = tau
		    self.data_dict[str(RF_frq)+'MHz']['PL'] = spin_state#/np.mean(spin_state)
		    self.drive_frq[i] = RF_frq*1e6
		    i +=1

		self.drive_frq = np.sort(self.drive_frq)
		self.t = tau*1e-9

		#convert to the same format as simulations
		self.PL_sweep = np.zeros((nr_steps, len(tau)))
		for i in np.arange(nr_steps):
			self.PL_sweep [i, :] = self.data_dict[str(self.drive_frq[i]*1e-6)+'MHz']['PL']

	def plot_state_occupancy(self, do_save = False, text=''):

		fig = plt.figure (figsize = (6,6))
		plt.subplot (2,1,1)
		plt.plot (self.t*1e6, self.p1, 'darkcyan', linewidth = 4, label = 'Sz = -3/2')
		plt.plot (self.t*1e6, self.p4,'--', color='crimson', linewidth = 4,label = 'Sz = +3/2')
		plt.ylim ([0, 0.5])
		plt.legend()
		plt.xlabel ('time [us]', fontsize=15)
		plt.ylabel ('Occupation probability', fontsize = 15)
		plt.subplot (2,1,2)
		plt.plot (self.t*1e6, self.p2, 'navy', linewidth = 4, label = 'Sz = -1/2')
		plt.plot (self.t*1e6, self.p3, '--', color='darkred', linewidth = 4, label = 'Sz = +1/2')
		plt.ylim ([0, 0.5])
		plt.legend()
		plt.xlabel ('time [us]', fontsize=15)
		plt.ylabel ('Occupation probability', fontsize = 15)
		if do_save:
			fig.savefig (self.folder+'/occupancy_'+text+'.svg', dpi = 400)
		plt.show()



V1 = V1_SiC(B_Gauss=62.5)
#Experimental data are stored in data_folder, 
#figures are saved in work_folder
V1.set_folders (data_folder = 'D:/Research/WorkData/VSi_V1_rabi/', 
			work_folder = 'D:/Research/__current_work/__ongoing_research/SiC_spectroscopy/')
V1.set_colormap ('viridis')

plot_exp_fft = False
plot_sim_fft = True
plot_time_domain_rabi = False
plot_state_occ = False

#FFT plots of experimental data
#------------------------------
#Loads experimental data, calculate FFT (config = 'modulo_squared', 'real_part') and plots it
#If save_fig is False, then the figure is not saved. Otherwise, it is saved with the name specified
# as save_fig parameter
if plot_exp_fft:
	V1.load_data(dBm = 23)
	V1.calc_fft(in_fft = 2, config = 'modulo_squared')
	V1.plot_fft(figsize = [3, 5], xlim = [0, 20], save_fig = 'exp_23dBm.svg')
	V1.load_data(dBm = 26)
	V1.calc_fft(in_fft = 2, config = 'modulo_squared')
	V1.plot_fft(figsize = [3, 5], xlim = [0, 20], save_fig = 'exp_26dBm.svg')
	V1.load_data(dBm = 29)
	V1.calc_fft(in_fft = 2, config = 'modulo_squared')
	V1.plot_fft(figsize = [3, 5], xlim = [0, 20], save_fig = 'exp_29dBm.svg')


#FFT plots simulations
#---------------------
if plot_sim_fft:
	o1 = 6.e6
	for o in [o1, o1*(2**0.5), o1*2]:
		V1.t = np.arange (0, 1.e-6, 2.e-9) 
		V1.set_intensity_transitions (I_m32_m12=1., I_m12_p12=1., I_p12_p32=1.)
		V1.set_ODMR_pars (polariz_array = [1.,0.,0.,1.], ODMR_contrast_array=[1.5,1.,1.,1.5], verbose = True)
		V1.set_decay (t1 = 100e-9, t2= 100e-9)
		V1.rabi_sweep_drive (init_frq=160e6, end_frq=190e6, nr_steps=60, Omega=o, do_plot = False, add_colorbar = False)
		V1.calc_fft(in_fft=1, config = 'modulo_squared')
		V1.plot_fft(figsize = [3, 5], xlim = [0, 20], save_fig = 'sim_O='+str(o*1e-6)+'MHz.svg')


if plot_time_domain_rabi:
	V1.load_data(dBm = 26)
	V1.plot_time_domain_rabi (add_colorbar=False, xlim =[0, 1], save_fig = 'time_domain.svg')

if plot_state_occ:
	
	Omega= 15e6
	for i in [172.4e6, 176.4e6, 180.4e6]:
		V1.t = np.arange (0, .5e-6, 2.e-9) 
		V1.set_intensity_transitions (I_m32_m12=1., I_m12_p12=1., I_p12_p32=1.)
		V1.set_ODMR_pars (polariz_array = [1.,0.,0.,1.], ODMR_contrast_array=[1.5,1.,1.,1.5], verbose = True)
		V1.set_decay (t1 = 200e-9, t2= 200e-9)
		V1.rabi (f=i, Omega = Omega, do_plot = False)
		V1.plot_state_occupancy(do_save = True, text = 'Omega='+str(Omega*1e-6)+'__f='+str(i*1e-6))