
import numpy as np
import pylab as plt

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 

#idea is to create an object-oriented approach, where I add electron spin object, nuclear spin objects
# and I can then calculate the effect of the nuclear spin on the electronic spin or viceversa
# all ab-initio

class NSpin ():

	def __init__ (self, specie, loc):

		self._species = specie
		self._x = loc[0]
		self._y = loc[1]
		self._z = loc[2]

		species_list = ['13C', '14N', '15N', '19F', '29Si']
		gm_list = [0,1,2,3,4]
		self._gm_ratio = gm_list [[i for i,x in enumerate(species_list) if x == specie] [0]]
		print (self._gm_ratio)

class CentralSpinExperiment ():

	def __init__ (self):

	    #Constants
	    #lattice parameter
	    self.a0 = 3.57 * 10**(-10)
	    # Hyperfine related constants
	    self.gam_el = 1.760859 *10**11 #Gyromagnetic ratio rad s-1 T-1
	    self.gam_n = 67.262 *10**6 #rad s-1 T-1
	    self.hbar = 1.05457173*10**(-34)
	    self.mu0 = 4*np.pi*10**(-7)

	    self.prefactor = self.mu0*self.gam_el*self.gam_n/(4*np.pi)*self.hbar**2 /self.hbar/(2*np.pi) #Last /hbar/2pi is to convert from Joule to Hz

	def generate_NSpin_distr (self, conc=0.02, N=25, do_sphere = True):

	    pi = np.pi

	    ##Carbon Lattice Definition
	    #Rotation matrix to get b along z-axis
	    Rz=np.array([[np.cos(pi/4),-np.sin(pi/4),0],[np.sin(pi/4),np.cos(pi/4),0],[0,0,1]])
	    Rx=np.array([[1,0,0],[0,np.cos(np.arctan(np.sqrt(2))),-np.sin(np.arctan(np.sqrt(2)))],[0,np.sin(np.arctan(np.sqrt(2))),np.cos(np.arctan(np.sqrt(2)))]])
	    # Basis vectors
	    a = np.array([0,0,0])
	    b = self.a0/4*np.array([1,1,1])
	    b = Rx.dot(Rz).dot(b)
	    # Basisvectors of Bravais lattice
	    i = self.a0/2*np.array([0,1,1])
	    i = Rx.dot(Rz).dot(i)
	    j = self.a0/2*np.array([1,0,1])
	    j = Rx.dot(Rz).dot(j)
	    k = self.a0/2*np.array([1,1,0])
	    k = Rx.dot(Rz).dot(k)

	    # define position of NV in middle of the grid
	    NVPos = round(N/2) *i +round(N/2)*j+round(N/2)*k

	    #Initialise
	    L_size = 2*(N)**3-2 # minus 2 for N and V positions
	    Ap = np.zeros(L_size) #parallel
	    Ao = np.zeros(L_size) # perpendicular component
	    r = np.zeros(L_size)
	    x = np.zeros(L_size)
	    y = np.zeros(L_size)
	    z = np.zeros(L_size)
	    o=0
	    #Calculate Hyperfine strength for all gridpoints
	    for n in range(N):
	        for m in range(N):
	            for l in range(N):
	                if (n== round(N/2) and m==round(N/2) and l == round(N/2)) :#Omit the Nitrogen and the Vacancy centre in the calculations
	                    o+=0
	                else:
	                    pos1 = n*i + m*j+l*k - NVPos
	                    pos2 = pos1 + b
	                    r[o] = np.sqrt(pos1.dot(pos1))
	                    Ap[o] =self.prefactor*np.power(r[o],-3)*(3*np.power(pos1[2],2)*np.power(r[o],-2)-1)
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos1[0],2)+np.power(pos1[1],2))*pos1[2]*np.power(r[o],-2))
	                    x[o] = pos1[0]
	                    y[o] = pos1[1]
	                    z[o] = pos1[2]
	                    o +=1
	                    r[o] = np.sqrt(pos2.dot(pos2))
	                    Ap[o] = self.prefactor*np.power(r[o],-3)*(3*np.power(pos2[2],2)*np.power(r[o],-2)-1)
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos2[0],2)+np.power(pos2[1],2))*pos2[2]*np.power(r[o],-2))
	                    x[o] = pos2[0]
	                    y[o] = pos2[1]
	                    z[o] = pos2[2]
	                    o+=1
	    # Generate different NV-Objects by randomly selecting which gridpoints contain a carbon.
	    
	    if do_sphere == True:
	        #zipped = zip(r,Ap,Ao,x,y,z)
	        #zipped.sort() # sort list as function of r
	        zipped = sorted(zip(r,Ap,Ao,x,y,z))
	        zipped = zipped[0:int(len(r)/2)] # only take half of the occurences
	        r = np.asarray([r_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	        Ap = np.asarray([Ap_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	        Ao = np.asarray([Ao_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	        x = np.asarray([x_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	        y = np.asarray([y_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	        z = np.asarray([z_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
	    
	    
	    for p in range(N):
	        # here we choose the grid points that contain a carbon 13 spin, dependent on concentration
	        Sel = np.where(np.random.rand(int(L_size/2)) < conc)
	        Ap_NV =[ Ap[u] for u in Sel]
	        Ao_NV =[ Ao[u] for u in Sel]
	        r_NV = [ r[u] for u in Sel]
	        # NV_list.append(A_NV[0]) #index 0 is to get rid of outher brackets in A_NV0
	    self._nr_nucl_spins = len(Ap_NV[0])
	    print ("Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice.")
	    return Ap_NV[0], Ao_NV[0] , r_NV[0]

	def set_spin_bath (self, Ap, Ao):

		self.Ap = Ap
		self.Ao = Ao
		self._nr_nucl_spins = len(self.Ap)

	def set_B (self, Bp, Bo):

		self.Bp = Bp
		self.Bo = Bo

	def plot_spin_bath_info (self):

		A = (self.Ap**2+self.Ao**2)**0.5
		phi = np.arccos((self.Ap)/A)*180/np.pi


		plt.plot (A/1000., 'o', color='Crimson')
		plt.xlabel ('nuclear spins', fontsize=15)
		plt.ylabel ('hyperfine [kHz]', fontsize=15)
		plt.show()


		plt.plot (phi, 'o', color = 'RoyalBlue')
		plt.xlabel ('nuclear spins', fontsize=15)
		plt.ylabel ('angle [deg]', fontsize=15)
		plt.show()


	def _set_pars (self, tau):

		self.hp_1 = self.Bp - self.Ap/self.gam_n
		self.ho_1 = self.Bo - self.Ao/self.gam_n
		self.h_1 = (self.hp_1**2+self.ho_1**2)**0.5
		self.hp_0 = self.Bp*np.ones (self._nr_nucl_spins)
		self.ho_0 = self.Bo*np.ones (self._nr_nucl_spins)
		self.h_0 = (self.hp_0**2+self.ho_0**2)**0.5
		self.phi_01 = np.arccos((self.hp_0*self.hp_1 + self.ho_0*self.ho_1)/(self.h_0*self.h_1))


	def FID (self, tau):

		self.L = np.zeros ((self._nr_nucl_spins, len(tau)))
		self.L_fid = np.ones (len(tau)) 
		self._set_pars (tau=tau)

		for i in np.arange(self._nr_nucl_spins):
			th_0 = self.gam_n*self.h_0[i]*tau
			th_1 = self.gam_n*self.h_1[i]*tau
			self.L[i, :] = np.cos (th_0/2.)*np.cos (th_1/2.) + \
						np.sin (th_0/2.)*np.sin (th_1/2.)*np.cos (self.phi_01[i])
			#plt.plot (tau, self.L[i, :])
		#plt.show()

		for i in np.arange(self._nr_nucl_spins):
			self.L_fid = self.L_fid * self.L[i, :]

		plt.figure (figsize = (20,5))
		plt.plot (tau*1e6, self.L_fid, linewidth =2, color = 'RoyalBlue')
		plt.xlabel ('free evolution time [us]', fontsize = 15)
		plt.title ('Free induction decay', fontsize = 15)
		plt.show()

	def __set_h_vector(self, tau, S1 = 1., S0 = 0.):
		self.L = np.zeros ((self._nr_nucl_spins, len(tau)))
		self.L_hahn = np.ones (len(tau)) 
		self.L = np.zeros ((self._nr_nucl_spins, len(tau)))
		self.L_dd = np.ones (len(tau)) 
		self._set_pars (tau=tau)

		self.hp_1 = self.Bp*np.ones (self._nr_nucl_spins) + S1*self.Ap/self.gam_n
		self.ho_1 = self.Bo*np.ones (self._nr_nucl_spins) + S1*self.Ao/self.gam_n
		self.h_1 = (self.hp_1**2+self.ho_1**2)**0.5
		self.hp_0= self.Bp*np.ones (self._nr_nucl_spins) + S0*self.Ap/self.gam_n
		self.ho_0 = self.Bo*np.ones (self._nr_nucl_spins) + S0*self.Ao/self.gam_n
		self.h_0 = (self.hp_0**2+self.ho_0**2)**0.5

		self.phi_01 = np.arccos((self.hp_0*self.hp_1 + self.ho_0*self.ho_1)/(self.h_0*self.h_1))



	def Hahn_eco (self, tau, S1=1, S0=0):

		self.__set_h_vector (tau=tau, S1=S1, S0=S0)

		for i in np.arange(self._nr_nucl_spins):
			th_0 = self.gam_n*self.h_0[i]*tau
			th_1 = self.gam_n*self.h_1[i]*tau
			a1 = np.sin(self.phi_01[i])**2
			a2 = np.sin(th_0)**2
			a3 = np.sin(th_1)**2

			self.L[i, :] = np.ones(len(tau)) -2*a1*a2*a3

			#plt.plot (tau, self.L[i, :])
		#plt.show()

		for i in np.arange(self._nr_nucl_spins):
			self.L_hahn = self.L_hahn * self.L[i, :]

		plt.figure (figsize=(30,10))
		plt.plot (tau*1e6, self.L_hahn, 'RoyalBlue')
		plt.plot (tau*1e6, self.L_hahn, 'o')
		plt.xlabel ('time (us)', fontsize=18)
		plt.axis ([min(tau*1e6), max(tau*1e6), -0.05, 1.05])
		plt.title ('Hahn echo')
		plt.show()

	def dynamical_decoupling (self, nr_pulses, tau, S1=1, S0=0):

		self.N = nr_pulses
		self.__set_h_vector (tau=tau, S1=S1, S0=S0)
		k = int(self.N/2)

		#plt.figure (figsize=(50,10))

		for i in np.arange(self._nr_nucl_spins):
			th_0 = self.gam_n*self.h_0[i]*tau
			th_1 = self.gam_n*self.h_1[i]*tau
			alpha = np.arctan ((np.sin(th_0/2.)*np.sin(th_1/2.)*np.sin(self.phi_01[i]))/(np.cos(th_0/2.)*np.cos(th_1/2.) - np.sin(th_0/2.)*np.sin(th_1/2.)*np.cos(self.phi_01[i])))
			theta = 2*np.arccos (np.cos(th_0)*np.cos(th_1) - np.sin(th_0)*np.sin(th_1)*np.cos(self.phi_01[i]))

			if np.mod (self.N, 2) == 0:
				a1 = (np.sin(alpha))**2
				a2 = np.sin(k*theta/2.)**2
				self.L[i, :] = np.ones(len(tau)) -2*a1*a2
			else:
				print ("Not yet")
			#plt.plot (tau*1e6, self.L[i, :])
		#plt.show()


		for i in np.arange(self._nr_nucl_spins):
			self.L_dd = self.L_dd * self.L[i, :]

		plt.figure (figsize=(50,10))
		plt.plot (tau*1e6, self.L_dd, 'RoyalBlue')
		plt.plot (tau*1e6, self.L_dd, 'o')
		plt.title ('Dynamical Decoupling  -  S0 = '+str(S0)+', S1 = '+str(S1), fontsize=25)
		plt.show()



#n1 = NSpin (specie= '15N', loc = [5,6,7])

exp = CentralSpinExperiment ()

Ap, Ao, r = exp.generate_NSpin_distr (N = 20, conc = 0.02)
exp.set_spin_bath (Ap, Ao)
exp.plot_spin_bath_info ()
Bz_gauss = 300
exp.set_B (Bp=Bz_gauss*1e-4, Bo = 0)
exp.FID (tau = np.linspace (1, 100000, 10000)*1e-9)

tau_max= 1e-3
exp.Hahn_eco (S1=1, S0=0, tau = np.linspace (0, tau_max, 100000))
exp.Hahn_eco (S1=-0.5, S0=0.5, tau = np.linspace (0, tau_max, 100000))
exp.Hahn_eco (S1=-1.5, S0=-0.5, tau = np.linspace (0, tau_max, 100000))


t0 = 0e-6
t1 = 10e-6
exp.dynamical_decoupling (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=0.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)

t0 = 7e-6
t1 = 10e-6
exp.dynamical_decoupling (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=0.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)

#things to do:
#1) expand to defects different than NV in diamond (for example, S=3/2)
#2) expand to nuclei different than 13C (19F, 29Si)
#   - need to re-write the bath generation function to include multiple specied with different concentrations,
#		plot them in 3D (myavi). Need to modify it such that first I create the specific lattice, given the 
#		concentrations, and then calculate the hyperfines (for non-interacting systems)
#3) include two-body clusters contribution. Just keep non-interacting hamiltonian, but include interactons only 
#		for spins that are sufficiently close by
#4) q-theory to include effect on nuclear spins (diagonalize sparse matrix up to selected cluster size)
#		- possibly use QuTip?



#tests
