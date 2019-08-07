# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:40:48 2019

@author: Karen
"""
import math 
import numpy as np
import cmath
import os, sys
import matplotlib.pyplot as plt
folder = '/Users/dalescerri/Documents/GitHub/QPL'
sys.path.append (folder)

from simulations.libs.adaptive_sensing import adaptive_tracking
from importlib import reload
reload (adaptive_tracking)

reps = 2 # number of repititions (between these diffusion occurs)
j = cmath.sqrt(-1) # imaginary unit
fb = 12.5*10**6 # set the larmor precession due to external magnetic field in -20 to 20 MHz range
N=6# number of different sensing times
tau_min = 20*10**(-9) #minimum sensing time
G = 5
F = 1
F0 = 1 # readout fidelity for ms = 0
F1 = 0.993 # readout fidelity for ms = 1
T2star = 5*10**(-6) #or 96 microseconds is the other value used in the paper

adtr = adaptive_tracking.TimeSequence(nr_time_steps = 0)
adtr.set_msmnt_params(G=G, F=G, N=N, tau0=tau_min, T2 = 1e-3, fid0=F0, fid1=F1)

"""Start with a uniform probability distribution"""
mid_point = 2**(N+1)+ 3#2**(N+4)
No_of_points = 2*mid_point+1
print("Karen",No_of_points)
#nr_discr = 2*mid_point+1
range_fHz = 1/tau_min#50*10**6 # frequency range
fHz = np.linspace (-range_fHz/2, range_fHz/2, No_of_points) # array of frequencies

"""Fourier transform of uniform distribution- means we start with a delta function in Fourier space"""
p_k = np.zeros (No_of_points)+1j*np.zeros (No_of_points)
p_k[mid_point] = 1/(2.*np.pi) # starting probability distribution in fourier space


"""Define a Ramsey function which produces a measurment outcome, mu,
   based on the fidelities and set external magnetic field"""
   
def Ramsey (theta, tau):
    P0 = (1+F0-F1)/2 + (F0+F1-1)/2*(math.exp(-(tau/T2star)**2)*math.cos(2*np.pi*fb*tau+theta))
    mu = np.random.choice(a=[0, 1],size = 1,replace = True, p=[P0, 1-P0])
    return int(mu)


"""Defining a diffusion function, where the probability distribution broadens over time"""

def Diffusion_frequency_space(mu, T2_est = 1e-3, tf = 10e-3):
    if mu == 0:
        tau_arr = np.linspace(0,tf,No_of_points)
        Hahn = np.exp(-(2*tau_arr/T2_est)**3)
        Hahn_FT = np.abs(np.fft.fft(Hahn)) #moving to frequency space
        Hahn_FT = np.roll(Hahn_FT, shift = mid_point)/np.sum(np.abs(Hahn_FT))# make the peak in the midpoint and not at the edges of the graph
        #plt.figure()
        #plt.plot(fHz, Hahn_FT)
        return Hahn_FT
    
    
"""Defining the limited adaptive protocol bayesian update"""

def Bayesian_update (res, theta, tau, tn, m, n, p_k):
    p_oldk = p_k
    p0 =(1+((-1)**res)*(F0-F1))*(p_oldk/2)
    p1 =((-1)**m_n)*(math.exp(-((tau/T2star)**2)))*((F0+F1-1)/4)*((cmath.exp(j*(res*np.pi+theta)))*np.roll(p_oldk, shift = -tn))
    p2 =((-1)**m_n)*(math.exp(-((tau/T2star)**2)))*((F0+F1-1)/4)*((cmath.exp(-j*(res*np.pi+theta)))*np.roll(p_oldk, shift = +tn))
    p_k =p0+p1+p2
    
    """Make sure p_k is normalised"""
    p_k = p_k/(2*np.pi*np.sum(np.abs(p_k)**2)**0.5)
    
    """Go from Fourier space to real space"""
    y = np.fft.fftshift(np.abs(np.fft.fft(p_k, No_of_points))**2)
    P = y/np.sum(np.abs(y))

    return p_k , P

"""Defining the equivalent to the Bayesian update, this time with gaussian approximations"""

P_G = (np.zeros (No_of_points)+1j*np.zeros (No_of_points))+2*np.pi # starting uniform distribution

def gauss_Bayesian_update (res, theta, tau, PGold): 
    sigma = 1/(math.sqrt(2)*np.pi*tau) # analytical sigma calculated from taylor exp and small angle approximation
    PG_newbit = np.zeros (No_of_points)+1j*np.zeros (No_of_points)# this is what you multiply your probability distribution by
    
    """Build up a sum of exponentials an an approximation to cosine"""
    for i in range(2**(N)+1): 
        PG_newbit =PG_newbit+ np.exp(-(fHz-(res/2+(i-2**(N-1))-theta/2/np.pi)/(tau))**2/(2.*sigma**2))
    PG_newbit = PG_newbit/np.sum(np.abs(PG_newbit)) #normalise
    
    P_G = PGold*PG_newbit
    P_G= P_G/np.sum(np.abs(P_G)) #normalise
    return P_G


"""Limited adaptive protocol"""
for rep in range(reps):
    for n in range(N+1):
        tn = 2**(n) #sensing time coefficient
        #theta_ctrl = 0.5*cmath.phase((p_k[-tn-1+mid_point]))# capellaro's theta
        Mn = G + F*(n-1)
        for m in range(1, Mn+1):
            #You can put theta_ctrl outside this loop, I found that it made a significant difference having it here
            theta_ctrl = 0.5*cmath.phase((p_k[2*tn+mid_point]))
            mu = Ramsey (theta = theta_ctrl, tau = tn*tau_min)
            adtr.bayesian_update(m_n=mu, phase_n=theta_ctrl, t_n=tn, T2_est = 1e-3, tf = 10e-3, free_evo = False, c_sig = True)
            #[p_k, P] = Bayesian_update (res =mu, theta = theta_ctrl, tau = tn*tau_min,tn=tn, m=m, n=n, p_k=p_k)
            p_k = adtr.p_k
            P = adtr.return_p_fB (T2_est = 1e-3, tf = 10e-3)[0]
            P_G = gauss_Bayesian_update (res=mu, theta = theta_ctrl, tau = tn*tau_min, PGold=P_G)
    
        """Plot figure"""
        plt.figure()
        title = "rep=%d, n=%d, frequency = %2.2fMHz" %(rep, n, fb*10**(-6))
        plt.title(title, fontsize=20)
        plt.plot(fHz*10**(-6), P, label = "adaptive tracking protocol")
        plt.plot(fHz*10**(-6), P_G, label = "from gaussian approximation")
        plt.xlabel("Larmor frequency (MHz)",fontsize=22)
        plt.ylabel("Probability density", fontsize=22)
        plt.legend(loc='upper center', ncol=2, fontsize = 14)
        plt.show()
    
    if mu ==0:
        Hahn_FT = Diffusion_frequency_space(mu=mu)
        P_G = np.convolve(P_G, Hahn_FT, mode="same")
        P_G = P_G/np.sum(abs(P_G))
		
        adtr.bayesian_diffuse(T2_est = 1e-3, tf = 10e-3)
        P = adtr.return_p_fB (T2_est = 1e-3, tf = 10e-3)[0]
        
        """Plot figure"""
        plt.figure()
        title = "Diffused rep =%d, frequency = %2.2fMHz" %(rep, fb*10**(-6))
        plt.title(title, fontsize=20)
        plt.plot(fHz*10**(-6), P, label = "adaptive tracking protocol")
        plt.plot(fHz*10**(-6), P_G, label = "from gaussian approximation")
        plt.xlabel("Larmor frequency (MHz)",fontsize=22)
        plt.ylabel("Probability density", fontsize=22)
        plt.legend(loc='upper center', ncol=2, fontsize = 14)
        plt.show()
