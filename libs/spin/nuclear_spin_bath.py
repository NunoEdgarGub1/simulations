
#########################################################
# Library to simulate diluted nuclear spin baths
# 
# Created: 2017 
# Cristian Bonato, c.bonato@hw.ac.uk
# Dale Scerri, ds32@hw.ac.uk
#
# Relevant Literature
# J. Maze NJP DOI: 10.1088/1367-2630/14/10/103041 [1]
# J. Maze disjoint cluster method DOI: 10.1103/PhysRevB.78.094303 [2]
# 
#########################################################

import numpy as np
from numpy import *
from operator import mul
from scipy.optimize import curve_fit
import pylab as plt
import math as mt
import copy as cp
import operator as op
import itertools as it
import scipy.linalg as lin
import scipy.spatial.distance as dst
import scipy.signal as sg
import numpy.random as ran
import time as time
import random as rand
import tabulate as tb
import functools as ft
import logging


import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
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
	    self.log = logging.getLogger ('nBath')
	    logging.basicConfig (level = logging.INFO)

	def generate_NSpin_distr (self, conc=0.02, N=25, do_sphere = True):
	
	    '''
	    Delft code to generate nuclear spins.
		
	    1) Generate lattice of size L_size
	    2) Populate every lattice point with spin.
	    3) Select N random spins (old code generated random no. of spins based on concentration)
		
	    Input:
	    conc        [float]     : concentration of nuclear spins (if random number is generated)
	    N           [int]       : number of spins to be generated (if fixed number of spins is generated)
	    do_sphere   [Boolean]   : ???
		
	    Output:
	    Ap_NV[0]    [array]  : Parallel hyperfine tensor component (Azz)
	    Ao_NV[0]    [array]  : Planar hyperfine tensor component
	    Azx_NV[0]   [array]  : Azx hyperfine tensor component
	    Azy_NV[0]   [array]  : Azy hyperfine tensor component
	    r_NV[0]     [array]  : NV-nuclear separations
	    pair_lst    [list]   : All nuclear spin pairings. Used to select relevant pairs in cluster method
	    geom_lst    [list]   : All parameters to calculate nuc-nuc bath couplings
	    dC_lst      [list]   : Additional elements for dCmn matrix ([2]). See _dCmn()
	    T2_h        [float]  : T2* in high field limit
	    T2_l        [float]  : T2* in low field limit
		
		
	    '''

	    self.log.info ("Generating spin bath...")
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
	                        self.log.error ('Error: nuclear spin overlapping with NV centre')
	                            
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
	                        self.log.error ('Error: nuclear spin overlapping with NV centre') 
	                            
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
	        zipped = sorted(zipped, key=lambda r: r[0]) # sort list as function of r
	        zipped = zipped[0:int(len(r)/2)] # only take half of the occurences
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
	    
	    # here we choose the grid points that contain a carbon 13 spin
	    Sel = []
	    while(len(Sel)<N):
	        lst_sel=[sel for sel in np.where(np.random.rand(int(L_size/2)) < conc)[0].tolist() if sel not in Sel]
	        Sel+=lst_sel
	        #print("Sel", Sel)

	    #Sel = (np.array(rand.sample(list(range(int(L_size/2))), N)),)#np.where(np.random.rand(int(L_size/2)) < conc)
	    Sel = (np.array(Sel)[:N],)
	    #np.random.shuffle(Sel)
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

	    #print(Sel)
	    theta_NV = [ theta[u] for u in Sel]
	    phi_NV = [ phi[u] for u in Sel]
	    # NV_list.append(A_NV[0]) #index 0 is to get rid of outher brackets in A_NV0
	    T2_h = sum(Ap_NV[0][u]**2 for u in range(len(Ap_NV[0])))**-0.5
	    T2_l = sum(Ap_NV[0][u]**2 + Ao_NV[0][u]**2 for u in range(len(Ap_NV[0])))**-0.5
	
	    self._nr_nucl_spins = len(Ap_NV[0])
	    self.log.info ("Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice.")
	    self.log.info ("T2* -- high field: {0} us".format(T2_h*1e6))
	    self.log.info ("T2* -- low field: {0} us".format (T2_l*1e6))
	    self.T2star_lowField = T2_l
	    self.T2star_highField = T2_h
        
	    if self._nr_nucl_spins <= 1:
	        pair_lst = []
	        r_ij_C = []
	        theta_ij_C = []
	        phi_ij_C = []
	        geom_lst = []
	        dC_lst = []

            
	    else:
	        pair_lst = list(it.combinations(list(range(self._nr_nucl_spins)), 2))
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
	                self.log.error ('Error: %d nuclear spin pair overlapping'%j)
                                                            
	        geom_lst = [r_ij_C , theta_ij_C , phi_ij_C] #all parameters to calculate nuclear bath couplings
	        dC_lst = [[Axx_NV[0],Axy_NV[0],Axz_NV[0]],[Ayx_NV[0],Ayy_NV[0],Ayz_NV[0]]] #additional hf values to calculate dC

	        #self.log.info ("Created "+str(self._nr_nucl_spins)+" nuclear spins in the lattice.")
	        return Ap_NV[0], Ao_NV[0] , Azx_NV[0] , Azy_NV[0] , r_NV[0] , pair_lst , geom_lst , dC_lst, T2_h, T2_l



	def set_spin_bath (self, Ap, Ao, Azx, Azy):

		self.Ap = Ap
		self.Ao = Ao
		self.Azx = Azx
		self.Azy = Azy      
		self._nr_nucl_spins = len(self.Ap)

	def set_B (self, Bp, Bo):

		self.Bp = Bp
		self.Bo = Bo
		self.Bx, self.By, self.Bz = self.set_B_Cart (Bx=Bo, Bz=Bp, By=0)
	
	def set_B_Cart (self, Bx, By, Bz):

		self.Bz = Bz
		self.By = By
		self.Bx = Bx
	
		return self.Bx, self.By, self.Bz
        
	def plot_spin_bath_info (self):

		A = (self.Ap**2+self.Ao**2)**0.5
		phi = np.arccos((self.Ap)/A)*180/np.pi

		plt.hist (A/1000., bins=100,color='Crimson')
		plt.xlabel ('hyperfine [kHz]', fontsize=15)
		plt.show()


	#gives Larmor vector. hf_approx condition set to normal condition before we agree if setting azx and azy is a valid approximation...
	def larm_vec (self, hf_approx):
	
		'''
		Calculates Larmor vectors.
		
		Input:
		hf_approx	[boolean]	-high Bz field approximation: neglects Azx and Azy components
		
		'''
	
		lar_1 = np.zeros((len(self.Ap),3))
		lar_0 = np.zeros((len(self.Ap),3))
		
		#gam are in rad s-1 T-1, convert to Hz T-1
		for f in range(len(self.Ap)):
			if hf_approx:
				lar_1[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])+np.array([0,0,self.Ap[f]])
				lar_0[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])
		
			else:
				lar_1[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])+np.array([self.Azx[f],self.Azy[f],self.Ap[f]])
				lar_0[f] = (self.gam_n/(2*np.pi))*np.array([self.Bx,self.By,self.Bz])
	

		self.log.debug ('Larmor frequency:', (self.gam_n/(2*np.pi) *self.Bz + self.Ap.mean())**-1)
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

		for i in np.arange(self._nr_nucl_spins):
			self.L_fid = self.L_fid * self.L[i, :]

		plt.figure (figsize = (20,5))
		plt.plot (tau*1e6, self.L_fid, linewidth =2, color = 'RoyalBlue')
		plt.xlabel ('free evolution time [us]', fontsize = 15)
		plt.title ('Free induction decay', fontsize = 15)
		plt.show()

	def __set_h_vector(self, tau, S1 = 1., S0 = 0.):
		print ("Bp = ", self.Bp)
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



	def Hahn_echo (self, tau, S1=1, S0=0):

		print ("Hahn echo - magnetic field: ", self.Bp)

		self.__set_h_vector (tau=tau, S1=S1, S0=S0)

		for i in np.arange(self._nr_nucl_spins):
			th_0 = self.gam_n*self.h_0[i]*tau
			th_1 = self.gam_n*self.h_1[i]*tau
			a1 = np.sin(self.phi_01[i])**2
			a2 = np.sin(th_0)**2
			a3 = np.sin(th_1)**2

			self.L[i, :] = np.ones(len(tau)) -2*a1*a2*a3

		for i in np.arange(self._nr_nucl_spins):
			self.L_hahn = self.L_hahn * self.L[i, :]

		plt.figure (figsize=(30,10))
		plt.plot (tau*1e6, self.L_hahn, 'RoyalBlue')
		plt.plot (tau*1e6, self.L_hahn, 'o')
		plt.xlabel ('time (us)', fontsize=18)
		plt.axis ([min(tau*1e6), max(tau*1e6), -0.05, 1.05])
		plt.title ('Hahn echo')
		plt.show()

		return self.L_hahn

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


class CentralSpinExperiment ():
    
	def __init__ (self):

		# Pauli matrices
		self.sx = np.array([[0,1],[1,0]])
		self.sy = np.array([[0,-complex(0,1)],[complex(0,1),0]])
		self.sz = np.array([[1,0],[0,-1]])
		self.In = .5*np.array([self.sx,self.sy,self.sz])
		self.msArr = []
		self._A_thr = None
		self._sparse_thr = 10
		self.close_cntr = 0
		self.sparse_distribution = False
		self._store_evol_dict = False

		self.gam_el = 1.760859 *10**11 #Gyromagnetic ratio rad s-1 T-1
		self.gam_n = 67.262 *10**6 #rad s-1 T-1
		self.hbar = 1.05457173*10**(-34)
		self.mu0 = 4*np.pi*10**(-7)
		self.ZFS = 2.87*10**9
		self.msArr=[]
		self.flipArr = []

		self.prefactor = self.mu0*(self.gam_n**2)/(4*np.pi)*self.hbar**2 /self.hbar/(2*np.pi) #Last /hbar/2pi is to convert from Joule to Hz

		self.nbath = NSpinBath ()

		# current density matrix for nuclear spin bath
		self._curr_rho = []

		# "evolution dictionary": stores data for each step
		self._evol_dict = {}
		self._store_evol_dict = True

		self.log = logging.getLogger ('nBath')
		logging.basicConfig (level = logging.INFO)

	def set_thresholds (self, A, sparse):
		self._A_thr = A
		self._sparse_thr = sparse

	def set_magnetic_field (self, Bz, Bx):
		self._Bz = Bz
		self._Bx = Bx
		self.nbath.set_B (Bp=self._Bz, Bo = self._Bx)

	def set_log_level (self, value):
		self.log.setLevel (value)

	def deactivate_evol_dict (self):
		self._store_evol_dict = False

	def gaussian(self, x, mu, sig):
		return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

	def print_nuclear_spins (self):

		T = [['', 'Ap (kHz)', 'Ao (kHz)', 'r (A)'], ['------', '------', '------', '------']]

		for i in np.arange(self._nr_nucl_spins):
			T.append ([i, int(self.Ap[i]*1e-2)/10, int(self.Ao[i]*1e-2)/10, int(self.r[i]*1e11)/10])

		print(tb.tabulate(T, stralign='center'))

	def kron_test(self, mat, N, reverse=False):
		'''
		Works out np.kron(np.eye(N), mat) or np.kron(mat,np.eye(N)) if reverse==True
		
		Input:
		
		mat    [array] : matrix to be tensored with identity
		N      [int]   : dimension of identity matrix
		
		'''
		
		m,n = mat.shape
		rang = np.arange(N)
		if reverse:
			prod = np.zeros((m,N,n,N),dtype=mat.dtype)
			prod[:,rang,:,rang] = mat
		else:
			prod = np.zeros((N,m,N,n),dtype=mat.dtype)
			prod[rang,:,rang,:] = mat
		prod.shape = (m*N,n*N)

		return prod

	def generate (self, nr_spins, concentration = .1,
				hf_approx = False, clus = True, single_exp = False, do_plot = False, Bp=0.001):
		'''
		Sets up spin bath and external field

		Input:
		nr_spins 		[integer]	: number of nuclear spins in simulation
		concentration 	[float]		: concnetration of nuclei with I>0
		hf_approx       [boolean]   : high Bz field approximation: neglects Azx and Azy components
		
		clus			[boolean]	: apply corrections to HF vector due to secular approximation
									  in disjoint cluster method c.f. DOI:10.1103/PhysRevB.78.094303
									  
		single_exp      [boolean]   : if False, doesn't update the probability distribution and 
									  can apply the disjoint cluster methods. Otherwise, full dynamics
									  have to be considered due to the required Overhauser operator.
		
		'''

		self._curr_step = 0
		self.Ap, self.Ao, self.Azx, self.Azy, self.r , self.pair_lst , self.geom_lst , self.dC_list, self.T2h, self.T2l= \
				self.nbath.generate_NSpin_distr (conc = concentration, N = nr_spins, do_sphere=True)
	
		self._hf_approx = hf_approx
		self._clus = clus

		#modified previous code to give necessary Cartesian components of hf vector (not just Ap and Ao)
		self.nbath.set_spin_bath (self.Ap, self.Ao, self.Azx, self.Azy)
		# Why do we need to set B hard-coded? Cristian
		self.Bp=Bp
		self.Bx, self.By, self.Bz = self.nbath.set_B_Cart (Bx=0, By=0 , Bz=self.Bp)
		self.nbath.set_B (Bp=self.Bp, Bo=0)

		self.Larm = self.nbath.larm_vec (self._hf_approx)
		self._nr_nucl_spins = int(self.nbath._nr_nucl_spins)
		self.nbath.plot_spin_bath_info()
		
		close_cntr = 0

		if not(self._A_thr == None):
			self.log.debug("Checking hyperfine threshold...")
			for j in range(self._nr_nucl_spins):
				if np.abs(self.Ap[j]) > self._A_thr:
					close_cntr +=1
			
			self.log.warning ('{0} spins with |A| > {1}MHz.'.format(close_cntr, self._A_thr*1e-6))
		self.close_cntr = close_cntr

		#hyperfine vector
		self.HFvec = np.array([[self.Azx[j], self.Azy[j], self.Ap[j]] for j in range(self._nr_nucl_spins)])

		self.In_tens = [[] for j in range(self._nr_nucl_spins)]

		if not single_exp:
			for j in range(self._nr_nucl_spins):
				Q1 = np.diag([1 for f in range(2**j)])
				Q2 = np.diag([1 for f in range(2**(self._nr_nucl_spins-(j+1)))])
				
				for k in range(3):
					self.In_tens[j].append(self.kron_test(self.kron_test(self.In[k],2**j),2**(self._nr_nucl_spins-(j+1)), reverse=True))#append(np.kron(np.kron(Q1,self.In[k]),Q2))
	
		#Run group algo for next step
		self._group_algo()

		#Creating 2**g * 2**g spin Pauli matrices. For disjoint cluster only
		self.In_tens_disjoint = [[[] for l in range(len(self._grp_lst[j]))] for j in range(len(self._grp_lst))]
		for l in range(len(self._grp_lst)):
			for j in range(len(self._grp_lst[l])):
				Q1 = np.eye(2**j)
				Q2 = np.eye(2**(len(self._grp_lst[l])-(j+1)))

				for k in range(3):
					self.In_tens_disjoint[l][j].append(np.kron(np.kron(Q1,self.In[k]),Q2))

		if not single_exp:
			self._curr_rho = np.diag([2**-self._nr_nucl_spins for j in range(2**self._nr_nucl_spins)])

		#Create sub matrices based on result of group algo
		self._block_rho = []
		for j in range(len(self._grp_lst)):
			self._block_rho.append(np.multiply(np.eye(2**len(self._grp_lst[j])),(2**-len(self._grp_lst[j]))))

		if do_plot:
			self.nbath.plot_spin_bath_info()

		if not single_exp:
			pd = np.real(self.get_probability_density())

			if not(self._sparse_thr == None):
				az, p_az = self.get_probability_density()
				az2 = np.roll(az,-1)
				if max(az2[:-1]-az[:-1]) > self._sparse_thr:
					self.log.debug ('Sparse distribution:{0} kHz'.format(max(az2[:-1]-az[:-1])))
					self.sparse_distribution = True
				else:
					self.sparse_distribution = False

			self.values_Az_kHz = pd[0]
			stat = self.get_overhauser_stat()
			if self._store_evol_dict:
				self._evol_dict ['0'] = {
					'mean_OH': np.real(stat[0]),
					'std_OH': np.real(stat[1]),
					'prob_Az': pd[1],
					'outcome': None,
				}


	def reset_bath_unpolarized (self, do_plot = True):

		self.log.debug ("Reset bath...")
		self._evol_dict = {}
		self._curr_rho = np.eye(2**self._nr_nucl_spins)/np.trace(np.eye(2**self._nr_nucl_spins))

		pd = np.real(self.get_probability_density())
		self.values_Az_kHz = pd[0]
		stat = self.get_overhauser_stat()
		self._evol_dict = {}
		if self._store_evol_dict:
			self._evol_dict ['0'] = {
				'mean_OH': np.real(stat[0]),
				'std_OH': np.real(stat[1]),
				'prob_Az': pd[1],
				'outcome': None,
			}


	def FID_indep_Nspins (self, tau):
		self.nbath.FID (tau=tau)

	def Hahn_echo_indep_Nspins (self, S1, S0, tau):
		self.nbath.Hahn_echo (tau=tau, S1=S1, S0=S0)

	def dynamical_decoupling_indep_Nspins (self, S1, S0, tau, nr_pulses):
		self.nbath.dynamical_decoupling (tau=tau, S1=S1, S0=S0, nr_pulses=nr_pulses)


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
		
		self.Cz_mean = np.mean([Carr[j][2] for j in range(len(self.pair_lst))])
		
		return Carr
		
	def _dCmn (self, ms, m, n):
		'''
		Constructs the dCmn matrix ([2]) for the correct nuc-nuc inter. in the secular approx.
		
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
		sqrt(C^xx_mn **2 + C^yy_mn **2 + C^zz_mn **2) calculated for each pair for sorting [2]
		
		'''
		
		Cmn = self._Cmn()
	
		Cij = [np.sqrt(sum(Cmn[j][k][k]**2 for k in range(3))) for j in range(len(self.pair_lst))]
		
		pair_lst_srt = [x for (y,x) in sorted(zip(Cij,self.pair_lst), key=lambda pair: pair[0], reverse=True)]
		Cij_srt = sorted(Cij, reverse=True)
		self.T2est = np.mean([(Cij[j]**-1) for j in range(self._nCr(self._nr_nucl_spins,2))])
		
		return Cij_srt, pair_lst_srt

	def _group_algo(self, g=3):
		'''
		Returns a list of groups for which we will calculate In.Cnm.Im based on [2] grouping algorithm:
		
		Input:
		g 		[int]		max. no. of spins in each group. Set to g=1 for full dynamics without grouping
		
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
		
		self.log.info ("Running grouping algorithm...")
		if self._nr_nucl_spins == 1:
			self._grp_lst = [[0]]
		
		self._sorted_pairs = self._C_merit()[1]
		C = self._C_merit()[0]
		self._grp_lst = [[self._sorted_pairs[j][0],self._sorted_pairs[j][1]] for j in range(len(self._sorted_pairs))]
		ind = [[] for j in range(self._nr_nucl_spins)]
		check_lst = []
		
		if g==1:
			self._grp_lst = [[j for j in range(self._nr_nucl_spins)]]#[[j] for j in range(self._nr_nucl_spins)]
		
		else:
			for j in range(len(self._grp_lst)):
				
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
							ind[itemno] = min(ind[self._grp_lst[j][0]],ind[self._grp_lst[j][1]])
		
			#(4)
			self._grp_lst = [[] for j in range(max([ind[k][0] for k in range(len(ind))])+1)]
			for j in range(self._nr_nucl_spins):
				self._grp_lst[ind[j][0]].append(j)
			self._grp_lst = [x for x in self._grp_lst if x != []]
		
		#create new pair list
		self._sorted_pairs_test = []

		for k in range(len(self._grp_lst)):
			if len(self._grp_lst[k]) > 1:
				self._sorted_pairs_test.append(list(it.combinations(self._grp_lst[k], 2)))
				
			else:
				self._sorted_pairs_test.append([-1])
	
		self._ind_arr = [[] for j in range(len(self._sorted_pairs_test))]
		
		for j in range(len(self._sorted_pairs_test)):
			if self._sorted_pairs_test[j][0]!=-1:
				for k in range(len(self._sorted_pairs_test[j])):
					self._ind_arr[j].append(self._sorted_pairs.index(self._sorted_pairs_test[j][k]))

			else:
				self._ind_arr[j].append(-1)

		#print(self._ind_arr)
		self._ind_arr_unsrt = [[] for j in range(len(self._sorted_pairs_test))]
		
		for j in range(len(self._sorted_pairs_test)):
			if self._sorted_pairs_test[j][0]!=-1:
				for k in range(len(self._sorted_pairs_test[j])):
					self._ind_arr_unsrt[j].append(self.pair_lst.index(self._sorted_pairs_test[j][k]))

			else:
				self._ind_arr_unsrt[j].append(-1)

		#new list of sorting parameter values (not used)
		Cmer_arr = [[C[j] for j in self._ind_arr[k]] for k in range(len(self._ind_arr)) if self._ind_arr[k][0]!=-1]
		ind_test = [[self._ind_arr[k][j] for j in range(len(self._ind_arr[k]))] for k in range(len(self._ind_arr)) if self._ind_arr[k][0]!=-1]
		
		self.log.info ("Completed.")
		self.log.debug ('unsorted index array', self._ind_arr_unsrt)
		self.log.debug ('grouped', self._grp_lst)
		self.log.debug ('nuc-nuc coupling strength', Cmer_arr)


	def _op_sd(self, Op):
		'''
		Calculates the standard deviation of operator Op with density matric rho

		Input:
		Op: matrix representing operator (whatever dimension)
		'''
		
		SD = np.sqrt(np.trace(Op.dot(Op.dot(self._curr_rho))) - np.trace(Op.dot(self._curr_rho))**2)
		
		return SD
	
	def _nCr(self, n, k):
		k = min(k, n-k)
		n = ft.reduce(op.mul, range(n, n-k, -1), 1)
		d = ft.reduce(op.mul, range(1, k+1), 1)
		return n//d


	def _op_mean(self, Op):
		'''
		Calculates the mean of operator Op with density matric rho

		Input:
		Op: matrix representing operator (whatever dimension)
		'''
		
		Mean = np.trace(Op.dot(self._curr_rho))
		
		return Mean
	
	
	def _overhauser_op(self):
		
		'''
		Creates Overhauser operator for updating probability distribution.
		Does not admit clustering due to sum over different clusters having different dimensions
		
		Output:
		self._over_op  [list] :  list of Overhauser operator Cartesian components
		
		'''

		self._over_op = []
		
		for j in range(3):
			self._over_op.append(sum(self.HFvec[k][j]*self.In_tens[k][j] for k in range(self._nr_nucl_spins)))

		return self._over_op


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
		eigvecs = [x for (y,x) in sorted(zip(eigvals,eigvecs), key=lambda pair: pair[0], reverse=True)]
		eigval_prob = multiply((2*np.sqrt(np.pi))**-1 * 1e-3, sorted(eigvals))#multiply((2*np.pi)**-1 * 1e-3, sorted(eigvals))
		
		#Calculate Tr(|Az><Az| rho)
		eigvec_prob = np.zeros(2**self._nr_nucl_spins,dtype=complex)
		for j in range(2**self._nr_nucl_spins):
		
			#takes the non zero element from each eigvector in the sorted list
			dum_var = [i for i, e in enumerate(eigvecs[j]) if e != 0][0]
			eigvec_prob[j] = self._curr_rho[dum_var,dum_var]


		return eigval_prob, eigvec_prob
		
	def get_values_Az (self):
		return self.values_Az_kHz

	def get_histogram_Az (self, nbins = 50):
		hist, bin_edges = np.histogram (self.get_values_Az(), nbins)
		bin_ctrs = 0.5*(bin_edges[1:]+bin_edges[:-1])
		return hist, bin_ctrs
		
	def plot_curr_probability_density (self, title = ''):
		az, pd = np.real(self.get_probability_density())

		plt.figure (figsize = (10,6))
		plt.plot (az, pd, 'o', color = 'RoyalBlue')
		plt.xlabel ('(kHz)', fontsize = 18)
		plt.xlabel ('frequency hyperfine (kHz)', fontsize=18)
		plt.ylabel ('probability', fontsize=18)
		plt.title (title, fontsize=18)
		plt.show()

	def get_overhauser_stat (self, component=None):
		'''
		Calculates mean and standard deviation of Overhauser field

		Input:
		component [int]: 1,2,3

		Output:
		mean, standard_deviation [int]
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
			
			
	def Hahn_Echo_clus (self, tauarr, phi, tol=1e-6, do_compare=True):
		'''
		Caclulates signal for spin echo with disjoint clusters [2]

		Input: 
		tauarr  [array]		: time array for spin echo
		phi     [radians]	: rotation angle of the spin readout basis
		tol     [float]     : tolerance factor for peak finding in non-interacting case 
		                      (see below)
		
		1) First calculate Hahn echo without nuc-nuc interactions to find where roughly the peaks
		should be for the interacting case, as only a rough envelope is required to calculate
		T2. 
		2) Select every 5th peak for lower comp. time with peak within [1-tol, 1].
		3) Create arrays of points in the neighbourhood of the maxima positions (newtauarr)
		4) For each array in newtauarr, ca;lculate Hahn echo signal for the interacting case,
		and select maximum from each subset.
		5) Fit Gaussian to the new data to get approximate value of T2
		

		'''
		
		self.arr_test = []
		self.arr_test_clus = []
		maxsigntauarr = []
		count=0
		
		print('Group list', self._grp_lst, self._ind_arr_unsrt)
		
		Hahn_an = self.nbath.Hahn_echo(tauarr)
		peaktopeak = max([tauarr[s] for s in range(len(Hahn_an)) if (Hahn_an[s] >= 1-tol and s%5==0)])

		signallist = [n*peaktopeak for n in range(20)]

		newtauarr = []
		for signal in signallist[:1]:
			newtauarr.append(np.linspace(signal,signal+.5*.1*peaktopeak,5).tolist())
		
		for signal in signallist[1:]:
			newtauarr.append(np.linspace(signal-.5*.1*peaktopeak,signal+.5*.1*peaktopeak,5).tolist())
		
		for sublist in newtauarr:
		
			sig_clus_lst = []
		
			for t in sublist:
		
				print(count/(len(newtauarr)*len(sublist)) * 100,'%')
				count+=1
			
				if do_compare:
				
					U_in = [self._U_op_int(0, t), self._U_op_int(1, t)]
					
					U0 = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in[1]))
					U1 = (np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).conj().T).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in[1]).conj().T)

					sig = np.trace(U0.dot(self._curr_rho.dot(U1))).real
					self.arr_test.append(sig)
				
				sig_clus = 1
				
				for j in range(len(self._grp_lst)):
					U_in_clus = [self._U_op_clus(j, 0, t), self._U_op_clus(j, 1, t)]
					
					U0_clus = np.multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]))
					U1_clus = (np.multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).conj().T).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]).conj().T)
				
					sig_clus *= np.trace(U0_clus.dot(self._block_rho[j].dot(U1_clus)))
		
				sig_clus_lst.append(sig_clus)

			self.arr_test_clus.append(max(sig_clus_lst).real)
			maxsigntauarr.append(sublist[sig_clus_lst.index(max(sig_clus_lst))])
			print(self.arr_test_clus)

		#popt,pcov = curve_fit(self.gaus,np.array(maxsigntauarr),np.array(self.arr_test_clus))
		
		#self.T2echo = abs(popt[0])
		#print ("T2 - Hahn echo: ", self.T2echo)
		
	def Hahn_echo (self, tau, phi=0):
		'''
		Caclulates signal for spin echo with disjoint clusters [2]

		Input: 
		tauarr  [array]		: time array for spin echo
		phi     [radians]	: rotation angle of the spin readout basis
		
		1) First calculate Hahn echo without nuc-nuc interactions to find where roughly the peaks
		should be for the interacting case, as only a rough envelope is required to calculate
		T2. 
		2) Select every 5th peak for lower comp. time with peak within [1-tol, 1].
		3) Create arrays of points in the neighbourhood of the maxima positions (newtauarr)
		4) For each array in newtauarr, ca;lculate Hahn echo signal for the interacting case,
		and select maximum from each subset.
		5) Fit Gaussian to the new data to get approximate value of T2
		

		'''
		
		self.arr_test = []
		self.arr_test_clus = []
		sig_clus_lst = []
		
		for t in tau:
				
			sig_clus = 1
		
			for j in range(len(self._grp_lst)):
				U_in_clus = [self._U_op_clus(j, 0, t), self._U_op_clus(j, 1, t)]
			
				U0_clus = np.multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]))
				U1_clus = (np.multiply(np.exp(-complex(0,1)*phi/2),U_in_clus[0]).conj().T).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in_clus[1]).conj().T)
		
				sig_clus *= np.trace(U0_clus.dot(self._block_rho[j].dot(U1_clus)))

			#print ("H-E: ", t, sig_clus)
			#sig_clus_lst.append(sig_clus)
			self.arr_test_clus.append(sig_clus.real)

		plt.figure()
		plt.plot (tau*1e6, self.arr_test_clus)
		plt.show()


	def _H_op_clus(self, group, ms):
		'''
		
		Updated method to generate Hamiltonian, works for both clustered/full dynamics approach
		
		Calculates the Hamiltonian in the presence of non-zero nuclear-nuclear interaction.
		pair_enum is the pair in group corresponding to pair_ind from the sorting algorithm
		
		Input:
		group 	[int]	:  group number based on sorting algorithm
		ms 		[0/1]	:  electron spin state
		
		'''

		Hms = sum(sum(self.Larm[ms][self._grp_lst[group][j]][h]*self.In_tens_disjoint[group][j][h] for j in range(len(self._grp_lst[group]))) for h in range(3))

		if len(self._grp_lst[group])>1:
			
			pair_ind = self._ind_arr_unsrt[group]
			pair = self._sorted_pairs_test[group]
			pair_enum = list(it.combinations(list(range(len(self._grp_lst[group]))),2))
		
			Hmsi = []
		
			Cmn = self._Cmn()
		
			dCmn = [self._dCmn(ms, pair[index][0], pair[index][0]) for index in range(len(pair_ind))]
			
			Hms = sum(sum(self.Larm[ms][self._grp_lst[group][j]][h]*self.In_tens_disjoint[group][j][h] for j in range(len(self._grp_lst[group]))) for h in range(3))
			
			Hc = (sum(sum(sum(np.asarray(csr_matrix.todense(Cmn[pair_ind[index]][cartm][cartn]*csr_matrix(self.In_tens_disjoint[group][pair_enum[index][0]][cartm]).dot(csr_matrix(self.In_tens_disjoint[group][pair_enum[index][1]][cartn]))))
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind)))
			+ sum(sum(sum(np.asarray(csr_matrix.todense(dCmn[index][cartm][cartn]*csr_matrix(self.In_tens_disjoint[group][pair_enum[index][0]][cartm]).dot(csr_matrix(self.In_tens_disjoint[group][pair_enum[index][1]][cartn]))))
			for cartn in [0,1,2])
			for cartm in [0,1,2])
			for index in range(len(pair_ind)))
			)

			return Hms + Hc #+(self.ZFS-self.gam_el*self.Bz)*ms*np.eye(2**len(self._grp_lst[group]))

		else:
			
			Hms = sum(self.Larm[ms][self._grp_lst[group][0]][h]*self.In_tens_disjoint[group][0][h] for h in range(3))

			return Hms #+(self.ZFS-self.gam_el*self.Bz)*ms*np.eye(2**len(self._grp_lst[group]))


	def _U_op_clus(self, group, ms, tau):
		'''
		
		Updated method to generate Hamiltonian, works for both clustered/full dynamics approach
		
		Returns matrix element U_ms
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''
		
		H = self._H_op_clus(group, ms)
		
		U = lin.expm(np.around(-np.multiply(complex(0,1)*tau,H),10))
		
		return U

	def _H_op_int(self, ms):
		'''
		
		Updated method to generate Hamiltonian, works for both clustered/full dynamics approach
		
		Calculates the Hamiltonian in the presence of non-zero nuclear-nuclear interaction.
		pair_enum is the pair in group corresponding to pair_ind from the sorting algorithm
		
		Input:
		group 	[int]	:  group number based on sorting algorithm
		ms 		[0/1]	:  electron spin state
		
		'''
		
		pair_ind = [j for j in range(self._nCr(self._nr_nucl_spins,2))]
		pair = self.pair_lst
		pair_enum = list(it.combinations(list(range(self._nr_nucl_spins)),2))
		
		Hmsi = []
		
		Cmn = self._Cmn()
		
		dCmn = [self._dCmn(ms, pair[index][0], pair[index][1]) for index in range(len(pair_ind))]
		
		Hms = sum(sum(np.multiply(self.Larm[ms][j][h],self.In_tens[j][h]) for j in range(self._nr_nucl_spins)) for h in range(3))
		
		Hc = (sum(sum(sum(np.asarray(csr_matrix.todense(Cmn[index][cartm][cartn]*csr_matrix(self.In_tens[pair_enum[index][0]][cartm]).dot(csr_matrix(self.In_tens[pair_enum[index][1]][cartn]))))
		for cartn in [0,1,2])
		for cartm in [0,1,2])
		for index in range(self._nCr(self._nr_nucl_spins,2)))
		+ sum(sum(sum(np.asarray(csr_matrix.todense(dCmn[index][cartm][cartn]*csr_matrix(self.In_tens[pair_enum[index][0]][cartm]).dot(csr_matrix(self.In_tens[pair_enum[index][1]][cartn]))))
		for cartn in [0,1,2])
		for cartm in [0,1,2])
		for index in range(self._nCr(self._nr_nucl_spins,2)))
		)


		return Hms + Hc #+ (self.ZFS-self.gam_el*self.Bz)*ms*np.eye(2**len(self._grp_lst[group]))

	def _U_op_int(self, ms, tau):
		'''
		
		Updated method to generate Hamiltonian, works for both clustered/full dynamics approach
		
		Returns matrix element U_ms
		(to be used to calculate the evolution in the Ramsey)

		Input:
		ms 		[0/1]		electron spin state
		tau 	[seconds]	free-evolution time Ramsey
		'''
		
		H = self._H_op_int(ms)
		
		U = lin.expm(-np.multiply(complex(0,1)*tau,H))
		
		return U


	def plot_bath_evolution (self):

		if self._store_evol_dict:

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
			for j in y:
				plt.axhline(j,c='r',alpha=0.1,ls=':')
			plt.colorbar(orientation='vertical')
			plt.show()

		else:

			self.log.warning ("Evolution dictionary is not updated.")


class FullBathDynamics (CentralSpinExperiment):

	def __init__ (self):
	
		super()
		self._store_evol_dict = False

		self.gam_el = 1.760859 *10**11 #Gyromagnetic ratio rad s-1 T-1
		self.gam_n = 67.262 *10**6 #rad s-1 T-1
		self.hbar = 1.05457173*10**(-34)
		self.mu0 = 4*np.pi*10**(-7)
		self.ZFS = 2.87*10**9
		self.msArr=[]
		self.flipArr = []

		self.prefactor = self.mu0*(self.gam_n**2)/(4*np.pi)*self.hbar**2 /self.hbar/(2*np.pi) #Last /hbar/2pi is to convert from Joule to Hz

		# Pauli matrices
		self.sx = np.array([[0,1],[1,0]])
		self.sy = np.array([[0,-complex(0,1)],[complex(0,1),0]])
		self.sz = np.array([[1,0],[0,-1]])
		self.In = .5*np.array([self.sx,self.sy,self.sz])
		# current density matrix for nuclear spin bath.
		# Now you won't keep all the elements but only the diagonal ones
		self._curr_rho = []
		self._curr_rho_test = []
		# "evolution dictionary": stores data for each step
		self._evol_dict = {}

		self.log = logging.getLogger ('nBath')
		logging.basicConfig (level = logging.INFO)

	def Hahn_Echo (self, tauarr, phi, do_compare=True):
		'''
		Caclulates signal for spin echo

		Input: 
		tauarr  [array]		: time array for spin echo
		phi  [radians]		: Rotation angle of the spin readout basis

		'''
		
		self.hahn_sig = []
		self.arr_test_clus = []
		self.arr_test_clus2 = []
		count = 0
		
		for t in tauarr:
		
			print(count/len(tauarr) *100,'%')
			count+=1

			U_in = [self._U_op_int(0, t), self._U_op_int(1, t)]
			
			U0 = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in[1]))
			U1 = (np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]).conj().T).dot(np.multiply(np.exp(complex(0,1)*phi/2),U_in[1]).conj().T)
			
			sig = np.trace(U0.dot((self._curr_rho/np.trace(self._curr_rho)).dot(U1)))

			self.hahn_sig.append(sig.real)
		
	
		plt.figure (figsize=(10,5))
		plt.plot (tauarr, self.hahn_sig, 'Red', label='Interacting')
		plt.plot (tauarr, self.hahn_sig, 'o',ms=3)
		plt.legend(fontsize=15)
		plt.title ('Hahn echo')
		plt.show()
	
	
	def gaus(self,x,sigma):
		return np.exp(-x**2/(2*sigma**2))
		

	def Ramsey (self, tau,  phi, flip_prob = 0, t_read = 0):
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
		
		phi = phi
		tflip = 0 #get finite value if flip happens, else set to 0
		
		#calculate Prod(tr(U1* U0 rho))
		U_in = [self._U_op_int(0, tau), self._U_op_int(1, tau)]
		
		U0 = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0])
		U1 = np.multiply(np.exp(complex(0,1)*phi/2),U_in[1])
		
		sig = np.trace(U0.dot((self._curr_rho/np.trace(self._curr_rho)).dot(U1.conj().T)))
		
		U0 = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]) - np.multiply(np.exp(complex(0,1)*phi/2),U_in[1])
		U1 = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0]) + np.multiply(np.exp(complex(0,1)*phi/2),U_in[1])

		#calculate probability given by 1 +/- Prod(tr(U1* U0 rho_block))
		p1 = round(.5*(1+sig.real),5)
		p0 = round(.5*(1-sig.real),5)
		
		ms = ran.choice([1,0],p=[p1,p0])
		self.log.debug ('Ramsey outcome: {0}'.format(ms))
		
		if ms==1:
			ms = ran.choice([1,0],p=[1-flip_prob, flip_prob])
			if ms==0:
				self.log.warning ('************************** FLIPPED SPIN ****************************',)
				self.flipArr.append(len(self.msArr))
				
				#random flip time during readout
				tflip = np.random.uniform(0,t_read)
		
		#Propagate sub density matrices based on Ramsey result. Then calculate full density matrix
		#tflip = 0 means no flip happened during readout, only a phase is picked up by the bath
		
		if tflip==0:
			U = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0])+((-1)**(ms+1))*np.multiply(np.exp(complex(0,1)*phi/2),U_in[1])
			
			U_read = self._U_op_int(ms, t_read)
			U_read_phi = np.multiply(np.exp(((-1)**(ms+1))*complex(0,1)*phi/2),U_read)
			
			U_glob = U_read_phi.dot(U)
			
			self._curr_rho = U_glob.dot(self._curr_rho.dot(U_glob.conj().T))
		
			self._curr_rho = self._curr_rho/(np.trace(self._curr_rho).real)
			self.msArr.append(ms)
			
		else:
			#ms=1 for 10e-6 - tflip
			U_noflip = self._U_op_int(1, tflip)
			#flip at tflip, ms=0
			U_flip = self._U_op_int(0, t_read - tflip)
		
			U = np.multiply(np.exp(-complex(0,1)*phi/2),U_in[0])+((-1)**(ms+1))*np.multiply(np.exp(complex(0,1)*phi/2),U_in[1])
			U_noflip_phi = np.multiply(np.exp(complex(0,1)*phi/2),U_noflip)
			U_flip_phi = np.multiply(np.exp(-complex(0,1)*phi/2),U_flip)
			U_glob = U_flip.dot(U_noflip.dot(U))
			
			self._curr_rho = U_glob.dot(self._curr_rho.dot(U_glob.conj().T))
			self._curr_rho = self._curr_rho/(np.trace(self._curr_rho).real)
			
			#To feed incorrect result to Bayesian update
			ms=1
			self.msArr.append(ms)
		
		#now = time.time()
		#print 'Ram', now-program_starts
		
		# update evolution dictionary
		self._curr_step += 1

		if self._store_evol_dict:
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

	def FreeEvo (self, ms, tau):
		
		#Propagate sub density matrices based on Ramsey result. Then calculate full density matrix
		U = self._U_op_int(ms, tau)
		self._curr_rho = U.dot(self._curr_rho.dot(U.conj().T))
		self._curr_rho = self._curr_rho/(np.trace(self._curr_rho).real)

		ms = float('nan') #mt.nan
		self.msArr.append(ms)
		
		#now = time.time()
		#print 'Ram', now-program_starts
		
		# update evolution dictionary
		self._curr_step += 1

		if self._store_evol_dict:
			pd = np.real(self.get_probability_density())
			stat = self.get_overhauser_stat()
			self._evol_dict [str(self._curr_step)] = {
				#'rho': self._curr_rho,
				'mean_OH': stat[0],
				'std_OH': stat[1],
				'prob_Az': pd[1],
				'outcome': ms,
			}

