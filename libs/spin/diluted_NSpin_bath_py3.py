
#########################################################
# Library to simulate diluted nuclear spin baths
# 
# Created: 2017 
# Cristian Bonato, c.bonato@hw.ac.uk
# Dale Scerri, 
#
# Relevant Literature
# J. Maze NJP
# 
#########################################################

import numpy as np
from numpy import *
import pylab as plt
import math as mt
import copy as cp
import operator as op
import scipy.linalg as lin
import scipy.spatial.distance as dst
import numpy.random as ran
import tabulate as tb

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18)


class NSpinBath ():

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
 

	def generate_NSpin_distr (self, conc =0.02, Nspins = 7, do_sphere = True):
		# this is just a quick hack, we need to put in a specific number of spins
		N = int (((Nspins/conc+2)/2)**(1/3))
		N = int (((Nspins/conc))**(1/3))
		print ("Calculating with N = ", N)
		return self.generate_NSpin_distr_latticeSites (conc=conc, N=N, do_sphere = do_sphere)

	def generate_NSpin_distr_latticeSites (self, conc=0.02, N=25, do_sphere = True):

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
	        zipped = list(zip(r,Ap,Ao,Aox,Aoy,x,y,z))
	        zipped.sort(key = lambda t: t[0]) #sort list as a function of r
	        zipped = zipped[0:int(len(r)/2)] # only take half of the occurences
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
	        Sel = np.where(np.random.rand(int(L_size/2)) < conc)
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
	    print ("Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice.")
	    print ("T2* -- high field: , ", int(T2_h*1e9), " ns")
	    print ("T2* -- low field: ", int(T2_l*1e9), " ns")
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

		A = (self.Ap**2+self.Ao**2)**0.5
		comp = (self.Aox**2+self.Aoy**2)**0.5 - np.abs(self.Ao)
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
		#gam are in rad s-1 T-1, convert to Hz T-1
		for f in range(len(self.Ap)):
			lar_1[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])+np.array([self.Aox[f],self.Aoy[f],self.Ap[f]])
			lar_0[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])
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
				print ("Not yet")
			plt.plot (tau, self.L[i, :])
		plt.show()

		for i in np.arange(self._nr_nucl_spins):
			self.L_dd = self.L_dd * self.L[i, :]

		plt.figure (figsize=(50,10))
		plt.plot (tau, self.L_dd, 'RoyalBlue')
		plt.plot (tau, self.L_dd, 'o')
		plt.title ('Dynamical Decoupling')
		plt.show()

class CentralSpinExperiment ():
    
	def __init__ (self):

		# Pauli matrices
		self.sx = np.array([[0,1],[1,0]])
		self.sy = np.array([[0,-complex(0,1)],[complex(0,1),0]])
		self.sz = np.array([[1,0],[0,-1]])
		self.In = .5*np.array([self.sx,self.sy,self.sz])

		# current density matrix for nuclear spin bath
		self._curr_rho = []

		# "evolution dictionary": stores data for each step
		self._evol_dict = {}

	def gaussian(self, x, mu, sig):
		return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

	def print_nuclear_spins (self):

		T = [['', 'Ap (kHz)', 'Ao (kHz)', 'r (A)'], ['------', '------', '------', '------']]

		for i in np.arange(self._nr_nucl_spins):
			T.append ([i, int(self.Ap[i]*1e-2)/10, int(self.Ao[i]*1e-2)/10, int(self.r[i]*1e11)/10])

		print(tb.tabulate(T, stralign='center'))


	def set_experiment(self, nr_lattice_sites, nr_nuclear_spins = False, concentration = .1):   
		'''
		Sets up spin bath and external field

		Input:
		nr_spins 		[integer]	- number of nuclear spins in simulation
		concentration 	[float]		- concnetration of nuclei with I>0
		'''

		self._curr_step = 0
		self.exp = NSpinBath ()
		if (nr_nuclear_spins == False):
			self.Ap, self.Ao, self.Aox, self.Aoy, self.r, self.T2h, self.T2l = \
					self.exp.generate_NSpin_distr_latticeSites (conc = concentration, N = nr_lattice_sites, do_sphere=True)
		elif (nr_lattice_sites == False):
			self.Ap, self.Ao, self.Aox, self.Aoy, self.r, self.T2h, self.T2l = \
					self.exp.generate_NSpin_distr (conc = concentration, Nspins = nr_nuclear_spins, do_sphere=True)
		else:
			print ("Error! You need to set eithr nr_lattice_sites or nr_nuclear_spins to False")

		#modified previous code to give necessary Cartesian components of hf vector (not just Ap and Ao)
		self.exp.set_spin_bath (self.Ap, self.Ao, self.Aox, self.Aoy, self.T2h, self.T2l)
		self.exp.set_B_Cart (Bx=0, By=0 , Bz=1)
		self.Larm = self.exp.larm_vec ()
		self._nr_nucl_spins = self.exp._nr_nucl_spins
		print ("Number of nuclear spins: ", self._nr_nucl_spins)
		
		#hyperfine vector
		self.HFvec = np.array([[self.Aox[j], self.Aoy[j], self.Ap[j]] for j in range(self._nr_nucl_spins)])

		#Creating 2**N * 2**N spin Pauli matrices
		self.In_tens = np.zeros((self._nr_nucl_spins,3,2**self._nr_nucl_spins,2**self._nr_nucl_spins),dtype=complex)
		#initial bath density matrix
		self._curr_rho = np.eye(2**self._nr_nucl_spins)/np.trace(np.eye(2**self._nr_nucl_spins))

		self.exp.plot_spin_bath_info()
		pd = np.real(self.get_probability_density())
		self.values_Az_kHz = pd[0]
		stat = self.get_overhauser_stat()
		self._evol_dict ['0'] = {
			#'rho': self._curr_rho,
			'mean_OH': np.real(stat[0]),
			'std_OH': np.real(stat[1]),
			'prob_Az': pd[1],
			'outcome': None,
		}		


	def _op_sd(self, Op):
		'''
		Calculates the standard deviation of operator Op with density matric rho

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
		
		
	def _H_op(self, ms):
		'''
		Returns matrix element H_ms
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''

		Hmsi = np.zeros((self._nr_nucl_spins,2,2),dtype=complex)
		for g in range(self._nr_nucl_spins):
			Hmsi[g] = sum(self.Larm[ms][g][h]*self.In[h] for h in range(3))

		Hms = Hmsi[0]
		for g in range(1,self._nr_nucl_spins):
			Hms = np.kron(Hms, Hmsi[g])

		return Hms
	
	def _U_op(self, ms, tau): 
		'''
		Returns matrix element U_ms
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''

		Umsi = np.zeros((self._nr_nucl_spins,2,2),dtype=complex)
		for g in range(self._nr_nucl_spins):
			Umsi[g] = lin.expm(-complex(0,1)*sum(self.Larm[ms][g][h]*self.In[h] for h in range(3)) *tau)

		Ums = Umsi[0]
		for g in range(1,self._nr_nucl_spins):
			Ums = np.kron(Ums, Umsi[g])

		return Ums
	
	def Ramsey (self, phi, tau):
		'''
		Performs a single Ramsey experiment:
		(1) calculates probabilities [p0, p1] to get ms=0,1
		(2) performs a measurement: returns either 0 (or 1) with probablity p0(p1)
		(3) updates the density matrix depending on the measurement outcome in (2)

		Input: 
		phi [radians]: Rotation angle of the spin readout basis
		tau [seconds]: Ramsey sensing time

		Output: outcome {0/1} of Ramsey experiment
		'''
		U0 = multiply(np.exp(-complex(0,1)*phi/2),self._U_op(0, tau)) - multiply(np.exp(complex(0,1)*phi/2),self._U_op(1, tau))
		U1 = multiply(np.exp(-complex(0,1)*phi/2),self._U_op(0, tau)) + multiply(np.exp(complex(0,1)*phi/2),self._U_op(1, tau))
		
		#Ramsey result probabilities
		p0 = .25*np.trace(U0.dot(self._curr_rho.dot(U0.conj().T))).real
		p1 = .25*np.trace(U1.dot(self._curr_rho.dot(U1.conj().T))).real
		#print 'Probablity to get 0 (%): ', int(p0*100)

		ms = ran.choice([1,0],p=[p1, p0])
		print ('Ramsey outcome: ', ms)
		#evolution operator depending on Ramsey result:
		U = multiply(np.exp(-complex(0,1)*phi/2),self._U_op(0, tau))+((-1)**(ms+1))*multiply(np.exp(complex(0,1)*phi/2),self._U_op(1, tau))
	
		rhoN_new = (U).dot(self._curr_rho.dot((U).conj().T))/ np.trace((U).dot(self._curr_rho.dot((U).conj().T)))
		self._curr_rho = rhoN_new
		
		# update evolution dictionary
		self._curr_step += 1
		pd = np.real(self.get_probability_density())
		stat = self.get_overhauser_stat()
		self._evol_dict [str(self._curr_step)] = {
			#'rho': self._curr_rho,
			'mean_OH': stat[0],
			'std_OH': stat[1],
			'prob_Az': pd[1],
			'outcome': ms,
		}

		return ms
	
	def get_probability_density(self):
		'''
		(1) Calculates eigenvalues (Az) and (normalized) eigenvectors (|Az>) of the Overhauser Operator z component
		(2) Sorts list of P(Az) = eigvec_prob[j] = Tr(|Az><Az| rho) according to sorted Az list (to plot later on)
		(3) returns list of eigvals and P(Az) list
		
		Output:
		eigvals       [kHz]: sorted Az list
		eigvec_prob        : Tr(|Az><Az| rho) list sorted according to Az list
		
		Note:
		I believe this is what's being shown in Fig.1 of DOI: 10.1103/PhysRevA.74.032316, although they start from a Gaussian distribution. Discuss.
		'''
		
		eigvals, eigvecs = np.linalg.eig(self._overhauser_op()[2])
		eigvecs = [x for (y,x) in sorted(zip(eigvals,eigvecs), key=lambda pair: pair[0])]
		eigvals = multiply(1e-3, sorted(eigvals))
		
		#Calculate Tr(|Az><Az| rho)
		eigvec_prob = np.zeros(2**self._nr_nucl_spins,dtype=complex)
		for j in range(2**self._nr_nucl_spins):
			eigvec_prob[j] = np.trace(self._curr_rho.dot(np.outer(np.conjugate(eigvecs[j]),eigvecs[j])))
			
		return eigvals, eigvec_prob

	def get_values_Az (self):
		return self.values_Az_kHz

	def plot_curr_probability_density (self, title = ''):
		az, pd = np.real(self.get_probability_density())

		plt.figure (figsize = (10,6))
		plt.plot (az, pd, linewidth=2, color = 'RoyalBlue')
		plt.xlabel ('frequency hyperfine (kHz)', fontsize=18)
		plt.ylabel ('probability', fontsize=18)
		plt.title (title, fontsize=18)
		plt.show()

	
	def get_overhauser_stat (self, component=None):
		'''
		Calculates mean and standard deviation of Overhauser field

		Input:
		component: 1,2,3

		Output:
		mean, standard_deviation
		'''

		if component in [0,1,2]:
			return self._op_mean(self._overhauser_op()[component]), self._op_sd(self._overhauser_op()[component])
		else:
			m = np.zeros(3)
			s = np.zeros(3)
			for j in range(3):
				m[j] = np.real(self._op_mean(self._overhauser_op()[j]))
				s[j] = np.real(self._op_sd(self._overhauser_op()[j]))
			return m, s


	def plot_bath_evolution (self):

		y = self.values_Az_kHz
		x = np.arange(self._curr_step+1)

		[X, Y] = np.meshgrid (x,y)

		# make 2D matrix with prob(Az) as function of time
		M = np.zeros ([len(y), self._curr_step+1])
		for j in range(self._curr_step+1):
			M [:, j] = np.ndarray.transpose(self._evol_dict[str(j)]['prob_Az'])

		plt.figure (figsize = (15, 8));
		plt.pcolor (X, Y, M)
		plt.xlabel ('step number', fontsize=22)
		plt.ylabel ('Az (kHz)', fontsize=22)
		plt.show()
