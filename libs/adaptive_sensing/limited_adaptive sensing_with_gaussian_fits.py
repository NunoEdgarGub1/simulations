# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:56:43 2019

@author: Karen Craigie
"""
import math 
import numpy as np
import cmath
import matplotlib.pyplot as plt

j = cmath.sqrt(-1)
fb = 3*10**6 # set the larmor precession due to external magnetic field in -20 to 20 MHz range
N=3 # number of different sensing times
tau_min = 20*10**(-9) #minimum sensing time
G = 5
F = 1
F0 = 1 # readout fidelity for ms = 0
F1 = 1#0.993 # readout fidelity for ms = 1
T2star = 5*10**(-6) #or 96 microseconds is the other value used in the paper

"""Start with a uniform probability distribution"""
mid_point = 2**(N+3)
No_of_points = 2*mid_point+1
fHz = np.linspace (-20*10**6, 20*10**6, No_of_points)

"""Fourier transform - means we start with a delta function in Fourier space"""
p_k = np.zeros (No_of_points)+1j*np.zeros (No_of_points)
p_k[mid_point] = 1/(2.*np.pi)

"""Define a Ramsey function which produces a measurment outcome, mu,
   based on the fidelities and set external magnetic field"""
   
def Ramsey (theta, tau):
    P0 = (1+F0-F1)/2 + (F0+F1-1)/2*(math.exp(-(tau/T2star)**2)*math.cos(2*np.pi*fb*tau+theta))
    mu = np.random.choice(a=[0, 1],size = 1,replace = True, p=[P0, 1-P0])
    return int(mu)

def Bayesian_update (res, theta, tau, tn, m, n, p_k):
    p_oldk = p_k
    p0 =(1+((-1)**res)*(F0-F1))*(p_oldk/2) 
    p1 =(math.exp(-((tau/T2star)**2)))*((F0+F1-1)/4)*((cmath.exp(j*(res*np.pi+theta)))*np.roll(p_oldk, shift = -tn))
    p2 =(math.exp(-((tau/T2star)**2))*((F0+F1-1)/4)*((cmath.exp(-j*(res*np.pi+theta)))*np.roll(p_oldk, shift = +tn))
    p_k =p0+p1+p2
    plt.figure()
    #plt.plot(fHz*10**(-6), p2)
    
    """Make sure p_k is normalised"""
    p_k = p_k/(2*np.pi*np.sum(np.abs(p_k)**2)**0.5)
    
    """Go from Fourier space to real space"""
    y = np.fft.fftshift(np.abs(np.fft.fft(p_k, No_of_points))**2)
    P = y/np.sum(np.abs(y))

    return p_k , P

for n in range(1,2):#N+1) :
    tn = 2**(N-n) #sensing time coefficient
    #theta_ctrl = 0.5*cmath.phase((p_k[-tn-1+mid_point]))# capellaro's theta
    theta_ctrl = 0.5*cmath.phase((p_k[2*tn+mid_point]))
    Mn = G + F*(n-1)
    for m in range(1, Mn+1):
        mu = Ramsey (theta = theta_ctrl, tau = tn*tau_min)
        print(mu)
        [p_k, P] =Bayesian_update (res =mu, theta = theta_ctrl, tau = tn*tau_min,tn=tn, m=m, n=n, p_k=p_k)
        #plt.figure()
        title = "n=%d" %(n)
        plt.title(title, fontsize=20)
        #plt.plot(fHz*10**(-6), p_k)
        plt.plot(fHz*10**(-6), P)
    
    """Plot figure"""
    """plt.figure()
    title = "n=%d" %(n)
    plt.title(title, fontsize=20)
    plt.plot(fHz*10**(-6), P)"""