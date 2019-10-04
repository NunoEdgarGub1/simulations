#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:44:31 2019

@author: Ted S. Santana

Basis

|0> = |00>
|1> = |10>
|2> = |01>
|3> = |11>
"""

from masterEquation_v1 import masterEquation, tensor
import numpy as np
import pylab as pl
import scipy.constants as const
pl.ioff()

hbar = const.hbar / const.elementary_charge * 1e15 # ueV.ns

G1 = 1.5
D1 = 0.0
Om1 = 0.1*G1
G2 = G1
D2 = -D1
Om2 = Om1

H1 = np.zeros((2,2), dtype = np.complex_)
H1[0,0] = -D1/2.0;  H1[0,1] = Om1/2.0;
H1[1,0] = Om1/2.0;  H1[1,1] = D1/2.0;

H2 = np.zeros((2,2), dtype = np.complex_)
H2[0,0] = -D2/2.0;  H2[0,1] = Om2/2.0;
H2[1,0] = Om2/2.0;  H2[1,1] = D2/2.0;

H = tensor(H1, np.eye(2, dtype = np.complex_)) + tensor(np.eye(2, dtype = np.complex_), H2)

sigM = np.zeros((2,2), dtype = np.complex_)
sigM[0,1] = 1.0
sigM1 = tensor(np.sqrt(G1)*sigM, np.eye(2, dtype=np.complex_))
sigM2 = tensor(np.eye(2, dtype=np.complex_), np.sqrt(G2)*sigM)
print(np.real(sigM1), "\n\n", np.real(sigM2))
    
yi = np.zeros((4,4), dtype = np.complex_)
yi[0,0] = 1.0

sys = masterEquation(yi, H, np.sqrt(G1)*sigM1, np.sqrt(G2)*sigM2, dt = 1e-2, n=5e2)
    
#sys.get_steady_state()
#sys.trajectory(atol = 1e-8, rtol=1e-7)
sys.g2Func(np.sqrt(G1)*sigM1, np.sqrt(G2)*sigM2, atol = 1e-9, rtol=1e-9)

pl.figure(1, figsize = (6,5))
pl.rc('font', **{'family': 'sans', 'serif': ['Computer Modern']})
pl.rc('text', usetex=True)
pl.rcParams['xtick.major.pad']='10'
pl.rcParams['ytick.major.pad']='10'
pl.rcParams['axes.linewidth'] = 2

pl.tick_params('both', length=15, width=2, which='major', labelsize=26, direction='in', top=True, right=True)
pl.tick_params('both', length=7, width=1, which='minor', direction='in', top=True, right=True)
pl.title("g2-superradiance", fontsize=16)
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
#pl.savefig('/home/ted/HWU_visit/experiments/superradiance/simulation/det_dep.pdf')