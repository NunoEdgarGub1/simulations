
import numpy as np
import pylab as plt

mu0 = 4*np.pi*1e-7 #H/m
ge = 2
muB = 9.27400949e-24 #J/T
hbar = 1.0545718e-34 #m^2 kg/s

def average_dipolar_coupling (density_cm3, verbose = True):
	density = density_cm3*1e6
	a = (3./(4*np.pi*density))**(1./3.)
	r_avg = 0.893*a
	dipolar_coupling = mu0*(ge**2)*(muB**2)/(4*np.pi*hbar*r_avg**3) #in Hz

	if verbose:
		print "Mean inter-particle distance: ", r_avg*1e9, " nm"
		print "Avergae dipolar coupling: ", dipolar_coupling*1e-3, " kHz"

	return r_avg, dipolar_coupling
