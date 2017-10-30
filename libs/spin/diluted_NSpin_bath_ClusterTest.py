
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
import itertools as it
import scipy.linalg as lin
import scipy.spatial.distance as dst
import numpy.random as ran
import time as time

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
	    self.ZFS = 2.87*10**9

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
	    Azx = np.zeros(L_size) # perpendicular component
	    Azy = np.zeros(L_size) # perpendicular component
	    
	    #Elements for dC correction of Cnm:
	    Axx = np.zeros(L_size)
	    Ayy = np.zeros(L_size)
	    Axy = np.zeros(L_size)
	    Ayx = np.zeros(L_size)
	    Axz = np.zeros(L_size)
	    Ayz = np.zeros(L_size)
	    r = np.zeros(L_size)
	    theta = np.zeros(L_size)
	    phi = np.zeros(L_size)
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
	                    #Ap[o] = self.prefactor*np.power(r[o],-3)*(1-3*np.power(pos1[2],2)*np.power(r[o],-2))
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos1[0],2)+np.power(pos1[1],2))*pos1[2]*np.power(r[o],-2))                            
	                    x[o] = pos1[0]
	                    y[o] = pos1[1]
	                    z[o] = pos1[2]
	                    if x[o] != 0:
	                        phi[o] = np.arctan(y[o]/x[o])
	                    else:
	                        phi[o] = np.pi/2
	                           
	                    if r[o] != 0:
	                        theta[o] = np.arccos(z[o]/r[o])
	                    else:
	                        print 'Error: nuclear spin overlapping with NV centre'
	                            
	                    #if x[o] != 0:
	                    #    Azx[o] = Ao[o]*np.cos(phi[o])
	                    #    Azy[o] = Ao[o]*np.sin(phi[o])
	                    #else:
	                    #    Azx[o] = 0
	                    #    Azy[o] = Ao[o]
	                    #Elements for dC correction of Cnm:
	                    Axx[o] = self.prefactor*np.power(r[o],-3)*(1-3*(np.sin(theta[o])**2)*(np.cos(phi[o])**2))
	                    Ayy[o] = self.prefactor*np.power(r[o],-3)*(1-3*(np.sin(theta[o])**2)*(np.sin(phi[o])**2))
	                    Axy[o] = self.prefactor*np.power(r[o],-3)*(-1.5*(np.sin(theta[o])**2)*(np.sin(2*phi[o])))
	                    Ayx[o] = Axy[o]
	                    Axz[o] = self.prefactor*np.power(r[o],-3)*(-3*np.cos(theta[o])*np.sin(theta[o])*np.cos(phi[o]))
	                    Ayz[o] = self.prefactor*np.power(r[o],-3)*(-3*np.cos(theta[o])*np.sin(theta[o])*np.sin(phi[o]))
	                    Azx[o] = Axz[o]
	                    Azy[o] = Ayz[o]
	                    Ap[o] = self.prefactor*np.power(r[o],-3)*(1-3*np.cos(theta[o])**2)
	                    o +=1
	                    r[o] = np.sqrt(pos2.dot(pos2))
	                    #Ap[o] = self.prefactor*np.power(r[o],-3)*(1-3*np.power(pos2[2],2)*np.power(r[o],-2))
	                    Ao[o] = self.prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos2[0],2)+np.power(pos2[1],2))*pos2[2]*np.power(r[o],-2))
	                    x[o] = pos2[0]
	                    y[o] = pos2[1]
	                    z[o] = pos2[2]
	                    if x[o] != 0:
	                        phi[o] = np.arctan(y[o]/x[o])
	                    else:
	                        phi[o] = np.pi/2
	                           
	                    if r[o] != 0:
	                        theta[o] = np.arccos(z[o]/r[o])
	                    else:
	                        print 'Error: nuclear spin overlapping with NV centre' 
	                            
	                    #if x[o] != 0:
	                    #    Azx[o] = Ao[o]*np.cos(phi[o])
	                    #    Azy[o] = Ao[o]*np.sin(phi[o])
	                    #else:
	                    #    Azx[o] = 0
	                    #    Azy[o] = Ao[o]
	                    #Elements for dC correction of Cnm:
	                    Axx[o] = self.prefactor*np.power(r[o],-3)*(1-3*(np.sin(theta[o])**2)*(np.cos(phi[o])**2))
	                    Ayy[o] = self.prefactor*np.power(r[o],-3)*(1-3*(np.sin(theta[o])**2)*(np.sin(phi[o])**2))
	                    Axy[o] = self.prefactor*np.power(r[o],-3)*(-1.5*(np.sin(theta[o])**2)*(np.sin(2*phi[o])))
	                    Ayx[o] = Axy[o]
	                    Axz[o] = self.prefactor*np.power(r[o],-3)*(-3*np.cos(theta[o])*np.sin(theta[o])*np.cos(phi[o]))
	                    Ayz[o] = self.prefactor*np.power(r[o],-3)*(-3*np.cos(theta[o])*np.sin(theta[o])*np.sin(phi[o]))
	                    Azx[o] = Axz[o]
	                    Azy[o] = Ayz[o]
	                    Ap[o] = self.prefactor*np.power(r[o],-3)*(1-3*np.cos(theta[o])**2)
	                    o+=1
	    # Generate different NV-Objects by randomly selecting which gridpoints contain a carbon.
	
	    if do_sphere == True:
	        zipped = zip(r,Ap,Ao,Axx,Ayy,Axy,Ayx,Axz,Ayz,Azx,Azy,x,y,z,theta,phi)
	        zipped.sort() # sort list as function of r
	        zipped = zipped[0:len(r)/2] # only take half of the occurences
	        r = np.asarray([r_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Ap = np.asarray([Ap_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Ao = np.asarray([Ao_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Axx = np.asarray([Axx_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Ayy = np.asarray([Ayy_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Axy = np.asarray([Axy_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Ayx = np.asarray([Ayx_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Axz = np.asarray([Axz_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Ayz = np.asarray([Ayz_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Azx = np.asarray([Azx_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        Azy = np.asarray([Azy_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        x = np.asarray([x_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        y = np.asarray([y_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        z = np.asarray([z_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        theta = np.asarray([theta_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])
	        phi = np.asarray([phi_s for r_s,Ap_s,Ao_s,Axx_s,Ayy_s,Axy_s,Ayx_s,Axz_s,Ayz_s,Azx_s,Azy_s,x_s,y_s,z_s,theta_s,phi_s in zipped])    
	    
	    for p in range(N):
	        # here we choose the grid points that contain a carbon 13 spin, dependent on concentration
	        Sel = np.where(np.random.rand(L_size/2) < conc)
	        Ap_NV =[ Ap[u] for u in Sel]
	        Ao_NV =[ Ao[u] for u in Sel]
	        Axx_NV =[ Axx[u] for u in Sel]
	        Ayy_NV =[ Ayy[u] for u in Sel]
	        Axy_NV =[ Axy[u] for u in Sel]
	        Ayx_NV =[ Ayx[u] for u in Sel]
	        Axz_NV =[ Axz[u] for u in Sel]
	        Ayz_NV =[ Ayz[u] for u in Sel]
	        Azx_NV =[ Azx[u] for u in Sel]
	        Azy_NV =[ Azy[u] for u in Sel]        
	        x_NV = [ x[u] for u in Sel]
	        y_NV = [ y[u] for u in Sel]
	        z_NV = [ z[u] for u in Sel]          
	        r_NV = [ r[u] for u in Sel]
	        theta_NV = [ theta[u] for u in Sel]
	        phi_NV = [ phi[u] for u in Sel]
	        # NV_list.append(A_NV[0]) #index 0 is to get rid of outher brackets in A_NV0
	    self._nr_nucl_spins = len(Ap_NV[0])
	    print theta_NV[0]
 	       
	    pair_lst = list(it.combinations(range(self._nr_nucl_spins), 2))
	    r_ij_C = np.zeros(len(pair_lst)) #length is nC2, n = #nuc spins
	    theta_ij_C = np.zeros(len(pair_lst)) #length is nC2, n = #nuc spins
	    phi_ij_C = np.zeros(len(pair_lst)) #length is nC2, n = #nuc spins
	
		#Calculate nuclear-nuclear angles and
	    for j in range(len(pair_lst)): #sum over nC2
	        r_ij = np.array([x_NV[0][pair_lst[j][1]] - x_NV[0][pair_lst[j][0]],
	                         y_NV[0][pair_lst[j][1]] - y_NV[0][pair_lst[j][0]],
	                         z_NV[0][pair_lst[j][1]] - z_NV[0][pair_lst[j][0]]])
	
	        if r_ij[0] != 0:
	            phi_ij_C[j] = np.arctan(r_ij[1]/r_ij[0])
	        else:
	            phi_ij_C[j] = np.pi/2
	
	        r_ij_C[j] = np.sqrt(r_ij.dot(r_ij))
	    
	        if r_ij_C[j] != 0:
	            theta_ij_C[j] = np.arccos(r_ij[2]/r_ij_C[j])
	        else:
	            print 'Error: %d nuclear spin pair overlapping'%j
	        
		geom_lst = [r_ij_C , theta_ij_C , phi_ij_C] #all parameters to calculate nuclear bath couplings
		dC_lst = [[Axx_NV[0],Axy_NV[0],Axz_NV[0]],[Ayx_NV[0],Ayy_NV[0],Ayz_NV[0]]] #additional hf values to calculate dC 
	
	    print "Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice."
	    return Ap_NV[0], Ao_NV[0] , Azx_NV[0] , Azy_NV[0] , r_NV[0] , pair_lst , geom_lst , dC_lst

	def set_spin_bath (self, Ap, Ao, Azx, Azy):

		self.Ap = Ap
		self.Ao = Ao
		self.Azx = Azx
		self.Azy = Azy      
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

	#gives Larmor vector. hf_approx condition set to normal condition before we agree if setting azx and azy is a valid approximation...
	def larm_vec (self, hf_approx, clus):
		'''
		Calculates Larmor vectors.
		
		Input:
		hf_approx	[boolean]	-high Bz field approximation: neglects Azx and Azy components
		clus		[boolean]	-apply corrections to HF vector due to secular approximation
								 in disjoint cluster method c.f. DOI:10.1103/PhysRevB.78.094303
		
		'''
	
		lar_1 = np.zeros((len(self.Ap),3))
		lar_0 = np.zeros((len(self.Ap),3))
		
		#gam are in rad s-1 T-1, convert to Hz T-1
		for f in range(len(self.Ap)):
			if hf_approx:
				lar_1[f] = (self.gam_n/(2*np.pi))*(1+self.gam_el/(self.ZFS*self.gam_n))*np.array([self.Bx,self.By,self.Bz])+np.array([0,0,self.Ap[f]])
				lar_0[f] = (self.gam_n/(2*np.pi))*(1-2*self.gam_el/(self.ZFS*self.gam_n))*np.array([self.Bx,self.By,self.Bz])
		
			else:
				lar_1[f] = (self.gam_n/(2*np.pi))*(1+self.gam_el/(self.ZFS*self.gam_n))*np.array([self.Bx,self.By,self.Bz])+np.array([self.Azx[f],self.Azy[f],self.Ap[f]])
				lar_0[f] = (self.gam_n/(2*np.pi))*(1-2*self.gam_el/(self.ZFS*self.gam_n))*np.array([self.Bx,self.By,self.Bz])
	
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
	
	def set_experiment(self, nr_spins, concentration = .1, hf_approx = False, clus = True):
		'''
		Sets up spin bath and external field

		Input:
		nr_spins 		[integer]	- number of nuclear spins in simulation
		concentration 	[float]		- concnetration of nuclei with I>0
		hf_approx       [boolean]   - high Bz field approximation: neglects Azx and Azy components
		clus			[boolean]	- apply corrections to HF vector due to secular approximation
									  in disjoint cluster method c.f. DOI:10.1103/PhysRevB.78.094303
		
		'''

		self._curr_step = 0
		self.exp = NSpinBath ()
		self.Ap, self.Ao, self.Azx, self.Azy, self.r , self.pair_lst , self.geom_lst , self.dC_list= \
				self.exp.generate_NSpin_distr (conc = concentration, N = nr_spins, do_sphere=True)
	
		self._hf_approx = hf_approx
		self._clus = clus

		#modified previous code to give necessary Cartesian components of hf vector (not just Ap and Ao)
		self.exp.set_spin_bath (self.Ap, self.Ao, self.Azx, self.Azy)
		self.exp.set_B_Cart (Bx=0, By=0 , Bz=.005)

		
		self.Larm = self.exp.larm_vec (self._hf_approx, self._clus)
		self._nr_nucl_spins = self.exp._nr_nucl_spins

		#hyperfine vector
		self.HFvec = np.array([[self.Azx[j], self.Azy[j], self.Ap[j]] for j in range(self._nr_nucl_spins)])

		#Creating 2**N * 2**N spin Pauli matrices. For full cluster only, not disjoint
		self.In_tens = np.zeros((self._nr_nucl_spins,3,2**self._nr_nucl_spins,2**self._nr_nucl_spins),dtype=complex)
		for j in range(self._nr_nucl_spins):
			Q1 = np.eye(2**j)
			Q2 = np.eye(2**(self._nr_nucl_spins-(j+1)))

			for k in range(3):
				self.In_tens[j][k] = np.kron(np.kron(Q1,self.In[k]),Q2)

		#Run group algo for next step
		self._group_algo()

		#Creating 2**g * 2**g spin Pauli matrices. For disjoint cluster only, not disjoint
		self.In_tens_disjoint = [[[] for l in range(len(self._grp_lst[j]))] for j in range(len(self._grp_lst))]
		for l in range(len(self._grp_lst)):
			for j in range(len(self._grp_lst[l])):
				Q1 = np.eye(2**j)
				Q2 = np.eye(2**(len(self._grp_lst[l])-(j+1)))
				
				for k in range(3):
					self.In_tens_disjoint[l][j].append(np.kron(np.kron(Q1,self.In[k]),Q2))
				
		#initial bath density matrix
		self._curr_rho = np.eye(2**self._nr_nucl_spins)/np.trace(np.eye(2**self._nr_nucl_spins))
		
		#Create sub matrices based on result of group algo
		if self._clus:
			self._block_rho = []
			for j in range(len(self._grp_lst)):
				self._block_rho.append(np.multiply(np.eye(2**len(self._grp_lst[j])),(2**-len(self._grp_lst[j]))))

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

	def _Cmn (self):
		'''
		Calculates Cmn tensor for every pair in self.pair_lst
		'''
	
		self.r_ij = self.geom_lst[0]
		self.theta_ij = self.geom_lst[1]
		self.phi_ij = self.geom_lst[2]

		Carr = [[[] for j in range(3)] for k in range(len(self.pair_lst))]
		
		for j in range(len(self.pair_lst)):
			Carr[j][0].append(self.prefactor*np.power(self.r_ij[j],-3)*(1-3*(np.sin(self.theta_ij[j])**2)*(np.cos(self.phi_ij[j])**2))) #xx
			Carr[j][0].append(self.prefactor*np.power(self.r_ij[j],-3)*(-1.5*(np.sin(self.theta_ij[j])**2)*(np.sin(2*self.phi_ij[j])))) #xy
			Carr[j][0].append(self.prefactor*np.power(self.r_ij[j],-3)*(-3*np.cos(self.theta_ij[j])*np.sin(self.theta_ij[j])*np.cos(self.phi_ij[j]))) #xz
			
			Carr[j][1].append(Carr[j][0][1]) #yx
			Carr[j][1].append(self.prefactor*np.power(self.r_ij[j],-3)*(1-3*(np.sin(self.theta_ij[j])**2)*(np.sin(self.phi_ij[j])**2))) #yy
			Carr[j][1].append(self.prefactor*np.power(self.r_ij[j],-3)*(-3*np.cos(self.theta_ij[j])*np.sin(self.theta_ij[j])*np.sin(self.phi_ij[j]))) #yz
			
			Carr[j][2].append(Carr[j][0][2]) #zx
			Carr[j][2].append(Carr[j][1][2]) #zy
			Carr[j][2].append(self.prefactor*np.power(self.r_ij[j],-3)*(1-3*(np.cos(self.theta_ij[j])**2))) #zz
				
		return Carr
		
	def _dCmn (self, ms, m, n):
		'''
		Constructs the dCmn matrix (DOI:10.1103/PhysRevB.78.094303 Eq.A4) for the correct nuc-nuc inter. in the secular approx.
		
		The values of dg should vary 0-15 for nuclei close to spin
		
		Input:
		
		ms  [0/1]	:electron spin state
		m,n [int]	:nuclear spins m and n
		
		'''
		
		dgm = -(2-3*ms)*self.gam_el/(self.gam_n*self.ZFS) * \
				np.array([[self.dC_list[0][0][m],self.dC_list[0][1][m],self.dC_list[0][2][m]],
						  [self.dC_list[1][0][m],self.dC_list[1][1][m],self.dC_list[1][2][m]],
						  [0,0,0]])
		
		dgn = -(2-3*ms)*self.gam_el/(self.gam_n*self.ZFS) * \
				np.array([[self.dC_list[0][0][n],self.dC_list[0][1][n],self.dC_list[0][2][n]],
						  [self.dC_list[1][0][n],self.dC_list[1][1][n],self.dC_list[1][2][n]],
						  [0,0,0]])
		
		dCmn = -((self.ZFS*self.gam_n/self.gam_el)**2 / (self.ZFS*(2-3*ms))) * (dgm.T).dot(dgn)

		return dCmn

	def _C_merit(self):
		'''
		sqrt(C^xx_mn **2 + C^yy_mn **2 + C^zz_mn **2) calculated for each pair for sorting. c.f. DOI:10.1103/PhysRevB.78.094303
		
		'''
	
		Cij = [np.sqrt(sum(self._Cmn()[j][k][k]**2 for k in range(3))) for j in range(len(self.pair_lst))]
		
		pair_lst_srt = [x for (y,x) in sorted(zip(Cij,self.pair_lst), key=lambda pair: pair[0], reverse=True)]
		Cij_srt = sorted(Cij, reverse=True)
		
		return Cij_srt, pair_lst_srt


	def _group_algo(self, g=3):
		'''
		Returns a list of groups for which we will calculate In.Cnm.Im based on DOI:10.1103/PhysRevB.78.094303 grouping algorithm:
		
		Input:
		g 		[int]		max. no. of spins in each group
		
		self._grp_lst is intialized to contain every sorted pair, sorted by _C_merit
		
		for spin pair (i,j):
			(1) if i not in any group:
				   Set group(i) to the next group no. available
				
			(2)	if j not in any group:
					Set group(j) to the next group no. available
				
			(3)	if group(i) != group(j):
					if length(group(i)) + length(group(i)) < g:
						Set group(k) of each element k with group(k) = group(i) or group(j) in the existing group list, to group(k) = min(group(i),group(j))
			
		
		(4) Update self.group_lst with new groups
		self._sorted_pairs = possible pair combinations for a given group (ex if group = [0,1,2], sorted pairs = [(0,1),(0,2),(1,2)])
		self._ind_arr = index of each pair in self._sorted_pairs in original pairs list self.pair_lst to find corresponding Cmn later on
		
		'''
		
		self._sorted_pairs = self._C_merit()[1]
		C = self._C_merit()[0]
		self._grp_lst = [[self._sorted_pairs[j][0],self._sorted_pairs[j][1]] for j in range(len(self._sorted_pairs))]
		ind = [[] for j in range(self._nr_nucl_spins)]
		check_lst = []


		for j in range(len(self._grp_lst)):
			#print self._grp_lst[j]
			
			#(1)
			if ind[self._grp_lst[j][0]] == []:
				ind[self._grp_lst[j][0]] = [next(index for index, value in enumerate(self._grp_lst) if self._grp_lst[j][0] in value)]

			#(2)
			if ind[self._grp_lst[j][1]] == []:
				ind[self._grp_lst[j][1]] = [next(index for index, value in enumerate(self._grp_lst) if self._grp_lst[j][1] in value)]
		
			#(3)
			if (ind[self._grp_lst[j][0]] != ind[self._grp_lst[j][1]] and ind.count(ind[self._grp_lst[j][0]])+ind.count(ind[self._grp_lst[j][1]]) <= g):
				for itemno in range(len(ind)):
					if ind[itemno] == ind[self._grp_lst[j][0]] or ind[itemno] == ind[self._grp_lst[j][1]]:
						#print ind[itemno], '-->', min(ind[self._grp_lst[j][0]],ind[self._grp_lst[j][1]])
						ind[itemno] = min(ind[self._grp_lst[j][0]],ind[self._grp_lst[j][1]])
	
		#(4)
		self._grp_lst = [[] for j in range(max([ind[k][0] for k in range(len(ind))])+1)]
		for j in range(self._nr_nucl_spins):
			self._grp_lst[ind[j][0]].append(j)
		self._grp_lst = [x for x in self._grp_lst if x != []]
		
		#create new pair list
		self._sorted_pairs = []

		for k in range(len(self._grp_lst)):
			if len(self._grp_lst[k]) > 1:
				self._sorted_pairs.append(list(it.combinations(self._grp_lst[k], 2)))

		self._ind_arr = [[] for j in range(len(self._sorted_pairs))]
		
		for j in range(len(self._sorted_pairs)):
			for k in range(len(self._sorted_pairs[j])):
				self._ind_arr[j].append(self.pair_lst.index(self._sorted_pairs[j][k]))

		#new list of sorting parameter values (not used)
		Cmer_arr = [[C[j] for j in self._ind_arr[k]] for k in range(len(self._ind_arr))]

	def _op_sd(self, Op):
		'''
		Calculates the standard deviation of operator Op with density matric rho

		Input:
		Op: matrix representing operator (whatever dimension)
		'''
		return np.sqrt(np.trace(Op.dot(Op.dot(self._curr_rho))) - np.trace(Op.dot(self._curr_rho))**2)

	def _diag_kron(self,a,b): #Use ONLY for diagonal matrices
		c = np.zeros((len(a)*len(b),len(a)*len(b)),dtype=complex)
		for j in range(1,len(c)+1):
			c[j-1][j-1] = a[int(mt.floor((j-1)/2))][int(mt.floor((j-1)/2))]*b[int((j-1)%2)][int((j-1)%2)]
		return c

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

		Hmsi = np.zeros((self._nr_nucl_spins,2**self._nr_nucl_spins,2**self._nr_nucl_spins),dtype=complex)
		for g in range(self._nr_nucl_spins):
			Hmsi[g] = sum(self.Larm[ms][g][h]*self.In_tens[g][h] for h in range(3))

		Hms = sum(Hmsi[g] for g in range(self._nr_nucl_spins))

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
		if self._hf_approx:
			for g in range(self._nr_nucl_spins):
				Umsi[g] = np.diag(np.diag(lin.expm(-complex(0,1)*sum(self.Larm[ms][g][h]*self.In[h] for h in range(2,3)) *tau)))
	
			Ums = Umsi[0]
		
			for g in range(1,self._nr_nucl_spins):
				Ums = self._diag_kron(Ums, Umsi[g])
			
		else:
			for g in range(self._nr_nucl_spins):
				Umsi[g] = lin.expm(-complex(0,1)*sum(self.Larm[ms][g][h]*self.In[h] for h in range(3)) *tau)

			Ums = Umsi[0]
			
			for g in range(1,self._nr_nucl_spins):
				Ums = np.kron(Ums, Umsi[g])

		return Ums
	


	def Ramsey (self, tau, phi):
		'''
		Performs a single Ramsey experiment:
		(1) calculates probabilities [p0, p1] to get ms=0,1
		(2) performs a measurement: returns either 0 (or 1) with probablity p0(p1)
		(3) updates the density matrix depending on the measurement outcome in (2)

		Input: 
		phi  [radians]				: Rotation angle of the spin readout basis
		U_in = [U(ms=0),U(ms=1)]	: input list of evol. ops. depending on whether

		Output: outcome {0/1} of Ramsey experiment
		'''
		
		startTime = time.time()
		
		if self._clus:
			U_in = [self._U_op_clus(0, tau), self._U_op_clus(1, tau)]
		
		else:
			U_in = [self._U_op(0, tau), self._U_op(1, tau)]
		
		U0 = multiply(np.exp(-complex(0,1)*phi/2),U_in[0]) - multiply(np.exp(complex(0,1)*phi/2),U_in[1])
		U1 = multiply(np.exp(-complex(0,1)*phi/2),U_in[0]) + multiply(np.exp(complex(0,1)*phi/2),U_in[1])
		
		#Ramsey result probabilities
		
		if self._hf_approx and not self._clus:
			p0 = .25*sum(U0[j][j]*self._curr_rho[j][j]*np.conjugate(U0[j][j]) for j in range(2**self._nr_nucl_spins)).real
			p1 = .25*sum(U1[j][j]*self._curr_rho[j][j]*np.conjugate(U1[j][j]) for j in range(2**self._nr_nucl_spins)).real

		else:
			p0 = .25*np.trace(U0.dot(self._curr_rho.dot(U0.conj().T))).real
			p1 = .25*np.trace(U1.dot(self._curr_rho.dot(U1.conj().T))).real

		print [p0,p1]
		ms = ran.choice([1,0],p=[p1, p0])
		print 'Ramsey outcome: ', ms
		#evolution operator depending on Ramsey result:
		U = multiply(np.exp(-complex(0,1)*phi/2),U_in[0])+((-1)**(ms+1))*multiply(np.exp(complex(0,1)*phi/2),U_in[1])
		
		if self._hf_approx and not self._clus:
			rhoN_new = np.zeros((2**self._nr_nucl_spins,2**self._nr_nucl_spins), dtype=complex)
			for j in range(2**self._nr_nucl_spins):
				rhoN_new[j][j] = U[j][j]*self._curr_rho[j][j]*np.conjugate(U[j][j])
			self._curr_rho = rhoN_new/np.trace(rhoN_new)
		
		else:
			rhoN_new = U.dot(self._curr_rho.dot(U.conj().T))/ np.trace(U.dot(self._curr_rho.dot(U.conj().T)))
			self._curr_rho = rhoN_new
		
		print 'Elapsed time', time.time() - startTime

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



	def Hahn_Echo_test (self, tauarr, phi):
		'''
		Caclulates signal for spin echo

		Input: 
		tauarr  [array]		: time array for spin echo
		phi  [radians]		: Rotation angle of the spin readout basis

		'''
		
		self.arr_test = []
		self.arr_test_clus = []
		
		for t in tauarr:
			U_in = [self._U_op(0, t), self._U_op(1, t)]
			
			U0 = multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).dot(multiply(np.exp(complex(0,1)*phi/2),U_in[1]))
			U1 = (multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).conj().T).dot(multiply(np.exp(complex(0,1)*phi/2),U_in[1]).conj().T)

			sig = .5*(1+np.trace(U0.dot(self._curr_rho.dot(U1))).real)
			self.arr_test.append(sig)
			
			sig_clus = 1
			
			for j in range(len(self._grp_lst)):
				U_in_clus = [self._U_op_clus_test(j, 0, t), self._U_op_clus_test(j, 1, t)]
				
				U0_clus = multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).dot(multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]))
				U1_clus = (multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).conj().T).dot(multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]).conj().T)
				
				sig_clus *= np.trace(U0_clus.dot(self._block_rho[j].dot(U1_clus)))

			self.arr_test_clus.append(.5*(1 + sig_clus.real))
	
		plt.figure (figsize=(30,10))
		plt.plot (tauarr, self.arr_test, 'RoyalBlue',label='Independent')
		plt.plot (tauarr, self.arr_test, 'o')
		plt.plot (tauarr, self.arr_test_clus, 'Red')
		plt.plot (tauarr, self.arr_test_clus, 'o',label='Clusters')
		plt.legend()
		plt.title ('Hahn echo')
		plt.show()


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
		
			#takes the non zero element from each eigvector in the sorted list
			dum_var = [i for i, e in enumerate(eigvecs[j]) if e != 0][0]
			eigvec_prob[j] = self._curr_rho[dum_var,dum_var]

		return eigvals, eigvec_prob
	
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

		fig = plt.figure(figsize=(10, 5.2))

		ax = fig.add_subplot(111)
		ax.set_title('Density matrix - Re')
		plt.imshow(self._curr_rho.real)
		ax.set_aspect('equal')
		cax = fig.add_axes([1.2*0.12, 0.1, 1.2*0.78, 0.8])
		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.patch.set_alpha(0)
		cax.set_frame_on(False)
		plt.colorbar(orientation='vertical')
		plt.show()
		
		fig = plt.figure(figsize=(10, 5.2))

		ax = fig.add_subplot(111)
		ax.set_title('Density matrix - Im')
		plt.imshow(self._curr_rho.imag)
		ax.set_aspect('equal')
		cax = fig.add_axes([1.2*0.12, 0.1, 1.2*0.78, 0.8])
		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.patch.set_alpha(0)
		cax.set_frame_on(False)
		plt.colorbar(orientation='vertical')
		plt.show()


class SpinExp_cluster1 (CentralSpinExperiment):
	'''
	For now we'll ignore spins which are too close to the centre.
	Check what nuclear-electron distance this approximation corresponds to...
	'''

	def __init__ (self):
	
		self.gam_el = 1.760859 *10**11 #Gyromagnetic ratio rad s-1 T-1
		self.gam_n = 67.262 *10**6 #rad s-1 T-1
		self.hbar = 1.05457173*10**(-34)
		self.mu0 = 4*np.pi*10**(-7)
		self.ZFS = 2.87*10**9

		self.prefactor = self.mu0*(self.gam_n**2)/(4*np.pi)*self.hbar**2 /self.hbar/(2*np.pi) #Last /hbar/2pi is to convert from Joule to Hz

		# Pauli matrices
		self.sx = np.array([[0,1],[1,0]])
		self.sy = np.array([[0,-complex(0,1)],[complex(0,1),0]])
		self.sz = np.array([[1,0],[0,-1]])
		self.In = .5*np.array([self.sx,self.sy,self.sz])
		#list of grouped spins, possible pair comb's in each group, and respective index in self.pair_lst (to find corresponding Cmn values)
		self._grp_lst, self._sorted_pairs, self._ind_arr = [], [], []
		# current density matrix for nuclear spin bath.
		# Now you won't keep all the elements but only the diagonal ones
		self._curr_rho = []
		# "evolution dictionary": stores data for each step
		self._evol_dict = {}

	def _H_op_clus(self, ms):
		'''
		Calculates the Hamiltonian in the presence of non-zero nuclear-nuclear interaction.
		
		Input:
		ms 		[0/1]		:  electron spin state
		
		'''

		groups = [x for x in self._grp_lst if len(x) > 1]
		
		Hc = []
		
		for m in range(len(groups)):
			group = groups[m]
			pair_ind = self._ind_arr[m]
			pair = self._sorted_pairs[m]
			
			#add correction terms based on self._dCmn (DOI:10.1103/PhysRevB.78.094303 Eq.A4, DOI: 10.1126/science.1131871 Suppl. Info)
			Hc.append(sum(sum(sum(self._Cmn()[pair_ind[index]][cartm][cartn]*self.In_tens[pair[index][0]][cartm].dot(self.In_tens[pair[index][1]][cartn])
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind)))
			+ sum(sum(sum(self._dCmn(ms, pair[index][0], pair[index][1])[cartm][cartn]*self.In_tens[pair[index][0]][cartm].dot(self.In_tens[pair[index][1]][cartn])
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind))))

	
		Htot = self._H_op(ms) + sum(Hc[m] for m in range(len(groups)))
	
		return Htot
		

	def _H_op_clus_disjoint(self, group, ms):
		'''
		Calculates the Hamiltonian in the presence of non-zero nuclear-nuclear interaction  for disjoint approach.
		pair_enum is the pair in group corresponding to pair_ind from the sorting algorithm
		
		Input:
		group 	[int]	:  group number based on sorting algorithm
		ms 		[0/1]	:  electron spin state
		
		'''
		
		Hmsi = []
		
		for j in range(len(self._grp_lst[group])):
			Hmsi.append(sum(self.Larm[ms][self._grp_lst[group][j]][h]*self.In_tens_disjoint[group][j][h] for h in range(3)))

		Hms = sum(Hmsi[j] for j in range(len(self._grp_lst[group])))
		
		if len(self._grp_lst[group])>1:
		#for m in range(len(groups)):
			#group = groups[m]
			pair_ind = self._ind_arr[group]
			pair = self._sorted_pairs[group]
			pair_enum = list(it.combinations(range(len(self._grp_lst[group])),2))
			
			#add correction terms based on self._dCmn (DOI:10.1103/PhysRevB.78.094303 Eq.A4, DOI: 10.1126/science.1131871 Suppl. Info)
			Hc = (sum(sum(sum(self._Cmn()[pair_ind[index]][cartm][cartn]*self.In_tens_disjoint[group][pair_enum[index][0]][cartm].dot(self.In_tens_disjoint[group][pair_enum[index][1]][cartn])
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind)))
			+ sum(sum(sum(self._dCmn(ms, pair[index][0], pair[index][1])[cartm][cartn]*self.In_tens_disjoint[group][pair_enum[index][0]][cartm].dot(self.In_tens_disjoint[group][pair_enum[index][1]][cartn])
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind))))
		
			Hms = Hms+Hc

		return Hms
	

	def _U_op_clus(self, ms, tau):
		'''
		Returns matrix element U_ms
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		:  electron spin state
		tau 	[seconds]	:  free-evolution time Ramsey
		'''
		
		return lin.expm(-complex(0,1)* self._H_op_clus(ms) *tau)
	

	def _U_op_clus_disjoint(self, group, ms, tau):
		'''
		Returns matrix element U_ms for disjoint approach
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''
		
		return lin.expm(-complex(0,1)* self._H_op_clus_disjoint(group, ms) *tau)

	def Ramsey_clus_disjoint (self, tau, phi):
		'''
		Performs a single Ramsey experiment:
		(1) Calculates tr(U1* U0 rho_block) for each dum density matrix
		(2) Multiplies results to get probability of getting ms=0 or 1
		(3) updates the sub density matrices depending on the measurement outcome in (2), and constructs new density matrix

		Input: 
		tau  [s]					: free evolution time
		phi  [radians]				: Rotation angle of the spin readout basis

		Output: outcome {0/1} of Ramsey experiment
		'''
		
		startTime = time.time()
		
		sig = 1 #seed value for total sig
		
		#calculate Prod(tr(U1* U0 rho_block))
		for j in range(len(self._grp_lst)):
			U_in = [self._U_op_clus_disjoint(j, 0, tau), self._U_op_clus_disjoint(j, 1, tau)]
			
			U0 = multiply(np.exp(-complex(0,1)*phi/2),U_in[0])
			U1 = multiply(np.exp(complex(0,1)*phi/2),U_in[1])
			
			sig *= np.trace(U0.dot(self._block_rho[j].dot(U1.conj().T)))
		
		#calculate probability given by 1 +/- Prod(tr(U1* U0 rho_block))
		p1 = .5*(1+sig.real)
		p0 = .5*(1-sig.real)
		print [p0,p1]
		ms = ran.choice([1,0],p=[p1, p0])
		print 'Ramsey outcome: ', ms
		
		#Ppropagate sub density matrices based on Ramsey result. 
		#Then calculate full density matrix
		for j in range(len(self._grp_lst)):
			#evolution operator depending on Ramsey result:
			U_in = [self._U_op_clus_disjoint(j, 0, tau), self._U_op_clus_disjoint(j, 1, tau)]
			U = multiply(np.exp(-complex(0,1)*phi/2),U_in[0])+((-1)**(ms+1))*multiply(np.exp(complex(0,1)*phi/2),U_in[1])
			
			rhoN_new = U.dot(self._block_rho[j].dot(U.conj().T))
			self._block_rho[j] = rhoN_new
			self._block_rho[j] = self._block_rho[j]/np.trace(self._block_rho[j])
			if j==0:
				self._curr_rho = self._block_rho[j]
			else:
				self._curr_rho = np.kron(self._curr_rho,self._block_rho[j])
		
		print 'Elapsed time', time.time() - startTime
		
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



#		self._evol_dict ['0'] = {
#			#'rho': self._curr_rho,
#			'mean_OH': np.real(stat[0]),
#			'std_OH': np.real(stat[1]),
#			'prob_Az': pd[1],
#			'outcome': None,
#		}
