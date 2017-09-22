
import numpy as np
import pylab as plt
import math as mt
import copy as cp
import scipy.linalg as lin
import scipy.spatial.distance as dst
import numpy.random as ran

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18)


class NSpin ():

	def __init__ (self, specie, loc):

		self._species = specie
		self._x = loc[0]
		self._y = loc[1]
		self._z = loc[2]

		species_list = ['13C', '14N', '15N', '19F', '29Si']
		gm_list = [0,1,2,3,4]
		self._gm_ratio = gm_list [[i for i,x in enumerate(species_list) if x == specie] [0]]
		print self._gm_ratio

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
	    Aox = np.zeros(L_size) # perpendicular component
	    Aoy = np.zeros(L_size) # perpendicular component
	    r = np.zeros(L_size)
	    x = np.zeros(L_size)
	    y = np.zeros(L_size)
	    z = np.zeros(L_size)
	    o=0
	    #Calculate Hyperfine strength for all gridpoints
	    #A[o] changed to 1-3cos^2 and not 3cos^2 - 1        
	    for n in range(N):
	        for m in range(N):
	            for l in range(N):
	                if (n== round(N/2) and m==round(N/2) and l == round(N/2)) :#Omit the Nitrogen and the Vacancy centre in the calculations
	                    o+=0
	                else:
	                    pos1 = n*i + m*j+l*k - NVPos
	                    pos2 = pos1 + b
	                    r[o] = np.sqrt(pos1.dot(pos1))
	                    Ap[o] =self.prefactor*np.power(r[o],-3)*(1-3*np.power(pos1[2],2)*np.power(r[o],-2))
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos1[0],2)+np.power(pos1[1],2))*pos1[2]*np.power(r[o],-2))
	                    if pos1[0] != 0:
	                        Aox[o] = Ao[o]*np.cos(np.arctan(pos1[1]/pos1[0]))
	                        Aoy[o] = Ao[o]*np.sin(np.arctan(pos1[1]/pos1[0]))
	                    else:
	                        Aox[o] = 0
	                        Aoy[o] = Ao[o]                            
	                    x[o] = pos1[0]
	                    y[o] = pos1[1]
	                    z[o] = pos1[2]
	                    o +=1
	                    r[o] = np.sqrt(pos2.dot(pos2))
	                    Ap[o] = self.prefactor*np.power(r[o],-3)*(3*np.power(pos2[2],2)*np.power(r[o],-2)-1)
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos2[0],2)+np.power(pos2[1],2))*pos2[2]*np.power(r[o],-2))
	                    if pos1[0] != 0:
	                        Aox[o] = Ao[o]*np.cos(np.arctan(pos1[1]/pos1[0]))
	                        Aoy[o] = Ao[o]*np.sin(np.arctan(pos1[1]/pos1[0]))
	                    else:
	                        Aox[o] = 0
	                        Aoy[o] = Ao[o]
	                    x[o] = pos2[0]
	                    y[o] = pos2[1]
	                    z[o] = pos2[2]
	                    o+=1
	    # Generate different NV-Objects by randomly selecting which gridpoints contain a carbon.
	    
	    if do_sphere == True:
	        zipped = zip(r,Ap,Ao,Aox,Aoy,x,y,z)
	        zipped.sort() # sort list as function of r
	        zipped = zipped[0:len(r)/2] # only take half of the occurences
	        r = np.asarray([r_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        Ap = np.asarray([Ap_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        Ao = np.asarray([Ao_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        Aox = np.asarray([Aox_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        Aoy = np.asarray([Aoy_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        x = np.asarray([x_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        y = np.asarray([y_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	        z = np.asarray([z_s for r_s,Ap_s,Ao_s,Aox_s,Aoy_s,x_s,y_s,z_s in zipped])
	    
	    
	    for p in range(N):
	        # here we choose the grid points that contain a carbon 13 spin, dependent on concentration
	        Sel = np.where(np.random.rand(L_size/2) < conc)
	        Ap_NV =[ Ap[u] for u in Sel]
	        Ao_NV =[ Ao[u] for u in Sel]
	        Aox_NV =[ Aox[u] for u in Sel]
	        Aoy_NV =[ Aoy[u] for u in Sel]
			
			#T2 time approximations for high and low external fields according to Maze 2012 paper
	        T2_h = sum(Ap_NV[0][u]**2 for u in range(len(Ap_NV[0])))**-0.5
	        T2_l = sum(Ap_NV[0][u]**2 + Ao_NV[0][u]**2 for u in range(len(Ap_NV[0])))**-0.5             
	        r_NV = [ r[u] for u in Sel]
	        # NV_list.append(A_NV[0]) #index 0 is to get rid of outher brackets in A_NV0
	    self._nr_nucl_spins = len(Ap_NV[0])
	    print "Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice."
	    print T2_h,T2_l
	    return Ap_NV[0], Ao_NV[0] , Aox_NV[0] , Aoy_NV[0] , r_NV[0] , T2_h*1e6, T2_l*1e6

	def set_spin_bath (self, Ap, Ao, Aox, Aoy, T2h, T2l):

		self.Ap = Ap
		self.Ao = Ao
		self.Aox = Aox
		self.Aoy = Aoy
		self.T2h = T2h
		self.T2l = T2l        
		self._nr_nucl_spins = len(self.Ap)

	def set_B (self, Bp, Bo):

		self.Bp = Bp
		self.Bo = Bo
        
	def set_B_Cart (self, Bx, By, Bz):

		self.Bz = Bz
		self.By = By
		self.Bx = Bx

	def gaussian(self, x, mu, sig):
		return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) #(1 / (np.sqrt(2*np.pi)*sig)) * 
        
	def plot_spin_bath_info (self):

		#A = (self.Ap**2+self.Ao**2)**0.5
		#comp = (self.Aox**2+self.Aoy**2)**0.5 - np.abs(self.Ao)
		phi = np.arccos((self.Ap)/A)*180/np.pi
		Brang = np.linspace(-2.5*2*np.sqrt(2*np.log(2))/self.T2l,2.5*2*np.sqrt(2*np.log(2))/self.T2l,max(1/self.T2l,1000))


		plt.plot (A/1000., 'o', color='Crimson')
		plt.xlabel ('nuclear spins', fontsize=15)
		plt.ylabel ('hyperfine [kHz]', fontsize=15)
		plt.show()

		plt.plot (comp/1000., 'o', color='g')
		plt.xlabel ('comp', fontsize=15)
		plt.ylabel ('hyperfine [kHz]', fontsize=15)
		plt.show()

		plt.plot (phi, 'o', color = 'RoyalBlue')
		plt.xlabel ('nuclear spins', fontsize=15)
		plt.ylabel ('angle [deg]', fontsize=15)
		plt.show()

	#gives Larmor vector
	def larm_vec (self):
		lar_1 = np.zeros((len(self.Ap),3))
		lar_0 = np.zeros((len(self.Ap),3))
		for f in range(len(self.Ap)):
			lar_1[f] = self.gam_n*np.array([self.Bx,self.By,self.Bz])+np.array([self.Aox[f],self.Aoy[f],self.Ap[f]])
			lar_0[f] = self.gam_n*np.array([self.Bx,self.By,self.Bz])
		return lar_0, lar_1

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

        
	def Hahn_eco (self, tau):

		self.L = np.zeros ((self._nr_nucl_spins, len(tau)))
		self.L_hahn = np.ones (len(tau)) 
		self._set_pars (tau=tau)

		self.hp_1 = self.Bp - self.Ap/self.gam_n
		self.ho_1 = self.Bo - self.Ao/self.gam_n
		self.h_1 = (self.hp_1**2+self.ho_1**2)**0.5
		self.hp_0 = self.Bp*np.ones (self._nr_nucl_spins)
		self.ho_0 = self.Bo*np.ones (self._nr_nucl_spins)
		self.h_0 = (self.hp_0**2+self.ho_0**2)**0.5
		self.phi_01 = np.arccos((self.hp_0*self.hp_1 + self.ho_0*self.ho_1)/(self.h_0*self.h_1))

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
		plt.plot (tau, self.L_hahn, 'RoyalBlue')
		plt.plot (tau, self.L_hahn, 'o')
		plt.title ('Hahn echo')
		plt.show()

	def dynamical_decoupling (self, nr_pulses, tau):

		self.N = nr_pulses
		self.L = np.zeros ((self._nr_nucl_spins, len(tau)))
		self.L_dd = np.ones (len(tau)) 
		self._set_pars (tau=tau)

		self.hp_1 = self.Bp + 1.5*self.Ap/self.gam_n
		self.ho_1 = self.Bo + 1.5*self.Ao/self.gam_n
		self.h_1 = (self.hp_1**2+self.ho_1**2)**0.5
		#self.hp_0 = self.Bp*np.ones (self._nr_nucl_spins)
		#self.ho_0 = self.Bo*np.ones (self._nr_nucl_spins)
		self.hp_0= self.Bp - 0.5*self.Ap/self.gam_n
		self.ho_0 = self.Bo - 0.5*self.Ao/self.gam_n
		self.h_0 = (self.hp_0**2+self.ho_0**2)**0.5
		self.phi_01 = np.arccos((self.hp_0*self.hp_1 + self.ho_0*self.ho_1)/(self.h_0*self.h_1))
		k = int(self.N/2)

		plt.figure (figsize=(50,10))

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
				print "Not yet"
			plt.plot (tau, self.L[i, :])
		plt.show()


		for i in np.arange(self._nr_nucl_spins):
			self.L_dd = self.L_dd * self.L[i, :]

		plt.figure (figsize=(50,10))
		plt.plot (tau, self.L_dd, 'RoyalBlue')
		plt.plot (tau, self.L_dd, 'o')
		plt.title ('Dynamical Decoupling')
		plt.show()

class TimeEvol ():
    
	def __init__ (self):

		# Pauli matrices
		self.sx = np.array([[0,1],[1,0]])
		self.sy = np.array([[0,-complex(0,1)],[complex(0,1),0]])
		self.sz = np.array([[1,0],[0,-1]])
		self.In = .5*np.array([self.sx,self.sy,self.sz])

		# current density matrix for nuclear spin bath
		self._curr_rho = []


	def set_experiment(self, nr_spins, concentration = .1):   
		'''
		Sets up spin bath and external field

		Input:
		nr_spins 		[integer]	- number of nuclear spins in simulation
		concentration 	[float]		- concnetration of nuclei with I>0
		'''

		self.exp = CentralSpinExperiment ()
		self.Ap, self.Ao, self.Aox, self.Aoy, self.r, self.T2h, self.T2l = \
				self.exp.generate_NSpin_distr (conc = concentration, N = nr_spins, do_sphere=True)
		
		#modified previous code to give necessary Cartesian components of hf vector (not just Ap and Ao)
		self.exp.set_spin_bath (self.Ap, self.Ao, self.Aox, self.Aoy, self.T2h, self.T2l)
		self.exp.set_B_Cart (Bx=0, By=0 , Bz=1)
		self.Larm = self.exp.larm_vec ()
		self._nr_nucl_spins = self.exp._nr_nucl_spins
		#hyperfine vector
		self.HFvec = np.array([[self.Aox[j], self.Aoy[j], self.Ap[j]] for j in range(self._nr_nucl_spins)]) 

		#Creating 2**N * 2**N spin Pauli matrices
		self.In_tens = np.zeros((self._nr_nucl_spins,3,2**self._nr_nucl_spins,2**self._nr_nucl_spins),dtype=complex)
		#initial bath density matrix
		self._curr_rho = np.eye(2**self._nr_nucl_spins)/np.trace(np.eye(2**self._nr_nucl_spins))  

	def _op_var(self, Op):
		'''
		Calculates the variance of operator Op with density matric rho

		Input:
		Op: matrix representing operator (whatever dimension)
		'''
		return np.sqrt(np.trace(Op.dot(Op.dot(self._curr_rho))) - np.trace(Op.dot(self._curr_rho))**2)

	def _op_mean(self, Op):
		'''
		Calculates the mean of operator Op with density matric rho

		Input:
		Op: matrix representing operator (whatever dimension)
		'''
		return np.trace(Op.dot(self._curr_rho))
	
	def _overhauser_op(self):
		
		'''
		Creates Overhauser operator
		'''
		
		self._over_op = np.zeros((3,2**self._nr_nucl_spins,2**self._nr_nucl_spins),dtype=complex)
		
		for j in range(self._nr_nucl_spins):
			Q1 = np.eye(2**j)
			Q2 = np.eye(2**(self._nr_nucl_spins-(j+1)))

			for k in range(3):
				self.In_tens[j][k] = np.kron(np.kron(Q1,self.In[k]),Q2)
	
		for k in range(3): 
			self._over_op[k] = sum(self.HFvec[j][k]*self.In_tens[j][k] for j in range(self._nr_nucl_spins))
		return self._over_op

	
	def _U_op(self, ms, tau): 
		'''
		Returns matrix element <ms|U*rho*U|ms> 
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''

		Umsi = np.zeros((len(self.Ap),2,2),dtype=complex)
		for g in range(len(self.Ap)):
			Umsi[g] = lin.expm(-complex(0,1)*sum(self.Larm[ms][g][h]*self.In[h] for h in range(3)) *tau)

		Ums = Umsi[0]
		for g in range(1,len(self.Ap)):
			Ums = np.kron(Ums, Umsi[g])

		return Ums
	
	def Ramsey (self, tau):
		'''
		Performs a single Ramsey experiment:
		(1) calculates probabilities [p0, p1] to get ms=0,1
		(2) performs a measurement: returns either 0 (or 1) with probablity p0(p1)
		(3) updates the density matrix depending on the measurement outcome in (2)

		Input: 
		tau [seconds]: Ramsey sensing time

		Output: outcome {0/1} of Ramsey experiment
		'''

		U0 = self._U_op(0, tau)-self._U_op(1, tau)
		U1 = self._U_op(0, tau)+self._U_op(1, tau)
		
		#Ramsey result probabilities
		p0 = .25*np.trace(U0.dot(self._curr_rho.dot(U0.conj().T))).real
		p1 = .25*np.trace(U1.dot(self._curr_rho.dot(U1.conj().T))).real
		print 'Probablity to get 0 (%): ', int(p0*100)

		ms = ran.choice([1,0],p=[p1, p0])
		#evolution operator depending on Ramsey result:
		U = self._U_op(0, tau)+((-1)**(ms+1))*self._U_op(1, tau)
	
		rhoN_new = (U).dot(self._curr_rho.dot((U).conj().T))/ np.trace((U).dot(self._curr_rho.dot((U).conj().T)))
		self._curr_rho = rhoN_new

		return ms

	def plot_field_distribution (self):

		h = [self._curr_rho[i,i] for i in range(2**self._nr_nucl_spins)]

		plt.figure(figsize = (12, 8))
		plt.plot (h, 'o', color='royalblue')
		plt.show()

	def Nruns(self, nr_steps, tau_0, do_plot):

		'''
		Runs full experimental sequence

		Input:
		nr_steps 	[integer]:	number of steps
		tau_0 		[seconds]:	Ramsey free-evolution time
		do_plot 	[boolean]
		'''

		self.eigvals = []
		self.Over_mean = [[] for j in range(3)]
		self.Over_var = [[] for j in range(3)]

		for g in range(nr_steps+1):

			self.tau = tau_0 #should be changed tau_k depending on adaptive protocol

			if (g>0):
				outcome = self.Ramsey(self.tau) 
				print 'Ramsey experiment  - outcome: ', outcome
			
			#self.plot_field_distribution()

			for j in range(3):
				self.Over_mean[j].append(self._op_mean(self._overhauser_op()[j]))
				self.Over_var[j].append(self._op_var(self._overhauser_op()[j]))
			
			self.eigvals.append(sorted(np.linalg.eig(self._curr_rho)[0], reverse=True))#sorted eigenvalues
	
		if do_plot:
		
			ax_labels = ['x','y','z']

			colormap = plt.get_cmap('hot')
			color_list = [colormap(k) for k in np.linspace(0, 0.8, nr_steps+1)]
			
			#analytical values of intitial Overhauser Op components to double check
			init_over_ticks = [.5* sum(self.HFvec[j][k]**2 for j in range(self._nr_nucl_spins))**.5 \
								for k in range(3)]

			fig = plt.figure(figsize = (15,6))
			ax = fig.add_subplot(1, 1, 1)
			for j in range(nr_steps+1): 
				ax.plot(range(2**self._nr_nucl_spins),self.eigvals[j], linewidth = 2, color=color_list[j])
			plt.xlabel ('nuclear spin number', fontsize = 20)
			plt.show()
			
			fig2 = plt.figure(figsize = (15,6))
			ax2 = fig2.add_subplot(1, 1, 1)
			for j in range(3):
				ax2.plot(range(nr_steps+1),np.divide(self.Over_var[j],1000),
						label=ax_labels[j]+'- component variance')
			ax2.scatter(np.zeros(3), np.divide(init_over_ticks,1000), color='red',
						label='analytical initial variance')
			plt.ylabel (r'$\sigma^{x,y,z}_{Over}$ (kHz)', fontsize=18)
			plt.legend(fontsize=15)
			plt.xlabel ('step nr', fontsize=18)

			
			fig3 = plt.figure(figsize = (15,6))
			ax3 = fig3.add_subplot(1, 1, 1)
			for j in range(3):
				ax3.plot(range(nr_steps+1),self.Over_mean[j],
						label=ax_labels[j]+'- component mean')
			plt.ylabel (r'$\mu^{x,y,z}_{Over}$ (Hz)', fontsize=18)
			plt.xlabel ('step nr', fontsize=18)
			plt.legend(fontsize=15)
			plt.show()

        
#n1 = NSpin (specie= '15N', loc = [5,6,7])


exp2 = TimeEvol()
exp2.set_experiment(nr_spins=4)

exp2.Nruns(nr_steps=5, tau_0=1e-6, do_plot=True)
