#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:21:31 2019

@author: Ted S. Santana

Basis

|0> = Ground state
|1> = Symmetric state
|2> = Antisymmetric state
|3> = Excited state
"""

from masterEquation_v1 import masterEquation
import numpy as np
import pylab as pl
import scipy.constants as const
pl.ioff()

hbar = const.hbar / const.elementary_charge * 1e15 # ueV.ns

G = 1.5 # Radiative decay
D = 0.0 # detuning
Om = 0.1*G # Rabi frequency

###### Hamiltonian ######
H = np.zeros((4,4), dtype = np.complex_)
H[0,0] = 0.0;              H[0,1] = Om*np.sqrt(2.0);  H[0,2] = 0.0; H[0,3] = 0.0;
H[1,0] = Om*np.sqrt(2.0);  H[1,1] = D;                H[1,2] = 0.0; H[1,3] = Om*np.sqrt(2.0);
H[2,0] = 0.0;              H[2,1] = 0.0;              H[2,2] = 0.0; H[2,3] = 0.0;
H[3,0] = 0.0;              H[3,1] = Om*np.sqrt(2.0);  H[3,2] = 0.0; H[3,3] = 2.0*D;

###### Sigma operators ######
sigM1 = np.zeros((4,4), dtype = np.complex_)
sigM2 = np.zeros((4,4), dtype = np.complex_)
sigM1[1,3] = np.sqrt(2.0*G)
sigM2[0,1] = np.sqrt(2.0*G)
    
yi = np.zeros((4,4), dtype = np.complex_)
yi[0,0] = 1.0

sys = masterEquation(yi, H, sigM1, sigM2, dt = 1e-2, n=5e2)
    
#sys.get_steady_state()
#sys.trajectory(atol = 1e-8, rtol=1e-7)
sys.g2Func(sigM1, sigM2, atol = 1e-9, rtol=1e-9)

pl.figure(1, figsize = (6,5))
pl.rc('font', **{'family': 'sans', 'serif': ['Computer Modern']})
pl.rc('text', usetex=True)
pl.rcParams['xtick.major.pad']='10'
pl.rcParams['ytick.major.pad']='10'
pl.rcParams['axes.linewidth'] = 2

pl.tick_params('both', length=15, width=2, which='major', labelsize=26, direction='in', top=True, right=True)
pl.tick_params('both', length=7, width=1, which='minor', direction='in', top=True, right=True)
pl.title("g2-superradiance 2", fontsize=16)
pl.plot(sys.time, sys.g2, 'r-', lw = 2)
pl.gca().xaxis.set_major_locator(pl.MultipleLocator(2))
pl.gca().xaxis.set_minor_locator(pl.MultipleLocator(1))
pl.gca().yaxis.set_major_locator(pl.MultipleLocator(0.5))
pl.gca().yaxis.set_minor_locator(pl.MultipleLocator(0.25))
pl.xlabel('$\\tau$ (ns)', fontsize = 26, labelpad=0)
pl.ylabel('$g^{(2)}(\\tau)$', fontsize = 26, labelpad=0)
pl.xlim(np.min(sys.time), np.max(sys.time))
pl.ylim(0, 1.5)

pl.tight_layout()
pl.show()
#pl.savefig('det_dep.pdf')