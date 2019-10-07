#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:12:07 2019

@author: Ted S. Santana
"""
print("Running...")

from masterEquation_v1 import masterEquation, tensor, adj
import pylab as pl
import scipy.constants as const

class emitter:
    def __init__(self, wlen, u, r):
        self.wlen = wlen
        self.u = u
        self.r = r
        self.w = 2*pl.pi*const.c/wlen
        self.G = self.Gamma()    

    def Gamma(self):
        self.u = pl.norm(self.u)
        return (2*pl.pi/self.wlen)**3*self.u**2/(3.0*pl.pi*const.epsilon_0*const.hbar)

class emitters(emitter):
    def __init__(self, wlen1, wlen2, u, r1, r2):
        self.QD1 = emitter(wlen1, u, r1)
        self.QD2 = emitter(wlen2, u, r2)
        self.u = u
        self.r = r1-r2
        self.k0 = (self.QD1.w + self.QD2.w)/2.0/const.c

    def F(self, r = None):
        """
        Spatial dependent function.
        See equation 2.91.
        """
        if not r:
            r = self.r
        kr = self.k0*pl.norm(r)
        ur = pl.dot(self.u/pl.norm(u), r/pl.norm(r))
        return 1.5*((1-ur**2)*pl.sin(kr)/kr + (1-3*ur**2)*(pl.cos(kr)/kr**2-pl.sin(kr)/kr**3))
    
    def G(self, n,m, r = None):
        """
        Spontaneous emission and incoherent dipole-dipole coupling
        See equation 2.90.
        """
        if n==m and n==1:
            return self.QD1.G
        elif n==m and n==2:
            return self.QD2.G
        else:
            return pl.sqrt(self.QD1.G * self.QD2.G)*self.F(r)
    
    def Lambda(self, r = None):
        """
        Coherent dipole-dipole interaction
        See equation 2.96.
        """
        if not r:
            r = self.r
        kr = self.k0*pl.norm(r)
        ur = pl.dot(self.u/pl.norm(u), r/pl.norm(r))
        return 0.75*pl.sqrt(self.QD1.G*self.QD2.G)*(-(1-ur**2)*pl.cos(kr)/kr+
                            (1-3*ur**2)*(pl.sin(kr)/kr**2+pl.cos(kr)/kr**3))

pl.ioff()
hbar = const.hbar / const.elementary_charge * 1e15 # ueV.ns

udir = pl.array([1,1j])
u = 1.4*const.elementary_charge*udir/pl.norm(udir)  # dipole moment (x, y)
wlen1 = 964.4                                       # resonance wavelength of the first emitter
wlen2 = 964.4                                       # resonance wavelength of the second emitter
wlenl = 964.4                                       # wavelength of the laser
w1 = 2*pl.pi*const.c/wlen1                          # angular frequency of the first emitter
w2 = 2*pl.pi*const.c/wlen2                          # angular frequency of the second emitter
wl = 2*pl.pi*const.c/wlenl                          # angular frequency of the laser
r1 = -0.4*wlenl * pl.array([0,1])                  # position of QD1 in nm
r2 = +0.4*wlenl * pl.array([0,1])                  # position of QD2 in nm
Om1 = 0.28                                          # Rabi frequency in GHz
Om2 = Om1                                           # Rabi frequency in GHz
print("detuning = {:.3f} ueV".format((w2-wl)*hbar))
print("distance = {:.1f} nm".format(pl.norm(r2-r1)))

QDs = emitters(wlen1, wlen2, u, r1, r2)
#QDs.QD1.G = 1.6
#QDs.QD2.G = 1.8
rates = pl.zeros((2,2), dtype = pl.complex_)
rates[0,0] = QDs.G(1,1); rates[0,1] = QDs.G(1,2);
rates[1,0] = QDs.G(2,1); rates[1,1] = QDs.G(2,2);
print(rates)

################# Operators #################
sigZ = pl.zeros((2,2), dtype = pl.complex_)
sigM = pl.zeros((2,2), dtype = pl.complex_)
sigZ[0,0] = -0.5; sigZ[1,1] = 0.5; 
sigM[0,1] = 1.0
sigZ1 = tensor(sigZ, pl.eye(2))
sigZ2 = tensor(pl.eye(2), sigZ)
sigM1 = tensor(sigM, pl.eye(2))
sigM2 = tensor(pl.eye(2), sigM)
sigP1 = adj(sigM1)
sigP2 = adj(sigM2)

Hs = QDs.Lambda()*pl.dot(sigP1, sigM2)+pl.conj(QDs.Lambda())*pl.dot(sigP2, sigM1)
Hl = -0.5j*(Om1*sigP1 + Om2*sigP2) + 0.5j*(pl.conj(Om1)*sigM1 + pl.conj(Om2)*sigM2)
H = Hs + Hl
H[1,1] = (w2-wl); H[2.2] = (w1-wl); H[3,3] = (w1+w2-2*wl)

################# Master equation #################
yi = pl.zeros((4,4), dtype = pl.complex_)
yi[0,0] = 1
sys = masterEquation(yi, H, sigM1, sigM2, n=2e3, dt=1e-2, rates = rates)
#sys.get_steady_state()
#sys.trajectory(atol = 1e-9, rtol=1e-9)
sys.g2Func(sigM1, sigM2, atol = 1e-9, rtol=1e-9)
#print(rates)

pl.figure(1, figsize = (6,5))
pl.rc('font', **{'family': 'sans', 'serif': ['Computer Modern']})
pl.rc('text', usetex=True)
pl.rcParams['xtick.major.pad']='10'
pl.rcParams['ytick.major.pad']='10'
pl.rcParams['axes.linewidth'] = 2

pl.tick_params('both', length=15, width=2, which='major', labelsize=26, direction='in', top=True, right=True)
pl.tick_params('both', length=7, width=1, which='minor', direction='in', top=True, right=True)
pl.title("g2-superradiance - Ficek", fontsize=16)
pl.plot(sys.time, sys.g2, 'r-', lw = 2)
pl.gca().xaxis.set_major_locator(pl.MultipleLocator(10))
pl.gca().xaxis.set_minor_locator(pl.MultipleLocator(5))
pl.gca().yaxis.set_major_locator(pl.MultipleLocator(0.5))
pl.gca().yaxis.set_minor_locator(pl.MultipleLocator(0.25))
pl.xlabel('$\\tau$ (ns)', fontsize = 26, labelpad=0)
pl.ylabel('$g^{(2)}(\\tau)$', fontsize = 26, labelpad=0)
pl.xlim(min(sys.time), max(sys.time))
#pl.ylim(-0.1, 1.1)
pl.hlines(0, min(sys.time), max(sys.time), colors = 'k', linestyles = '--')
pl.hlines(0.5, min(sys.time), max(sys.time), colors = 'k', linestyles = '--')
pl.hlines(1, min(sys.time), max(sys.time), colors = 'k', linestyles = '--')

pl.tight_layout()
pl.show()
#pl.savefig('det_dep.pdf')