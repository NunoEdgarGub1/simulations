#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 23:43:52 2019

@author: Ted S. Santana

To do list:
    1. The second-ordem correlation function g2Func is normalized "by hand".
    Need to write the proper normalization factor.
"""

import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp
import types

def adj(a):
    """
    Shortcut function to calculate the self-adjoint matrix.
    
    Parameters
    ----------
    a : Complex numpy array
        Matrix whose self-adjoint must be returned
    """
    return np.conj(a).T

def tensor(a, b):
    """
    This function returns the Kronecker product of two matrices after checking
    if they are both square matrices.
    
    Parameters
    ------------
    a, b : Numpy arrays
        Matrices to be multiplied (a times b).
    """
    if (a.shape[0]!=a.shape[1]):
        return ValueError("The first matrix must be square.")
    elif (b.shape[0]!=b.shape[1]):
        return ValueError("The second matrix must be square.")
    else:
        return np.kron(a, b)

def dot(a, b):
    """
    This function performs the multiplication of two matrices.
    
    Parameters
    ----------
    a, b : Numpy arrays
        Matrices to be multiplied (a times b).
    """
    if (a.shape[1]!=b.shape[0]):
        return ValueError("The matrices cannot be multiplied. Please, check their dimensions.")
    M = np.zeros((a.shape[0], b.shape[1]), dtype=np.complex_)
    for n in range(a.shape[0]):
        for m in range(b.shape[1]):
            for i in range(a.shape[1]):
                M[n,m] += a[n,i] * b[i,m]
    return M

class masterEquation:
    """
    Class used to construct and solve the Lindblab master equation given the
    Hamiltonian and the operator(s) involved in the incoherent evolution of the
    open quantum system.
    
    Please, use only numpy arrays.
    
    Parameters
    -----------
    yi : Complex numpy array
        Initial state
    H : Complex numpy array or a function of time returning a complex numpy array
        Hamiltonian in angular frequency units
    *sigmas : List of complex numpy arrays
        Operator(s) for the Lindblad superoperator(s)
    **par n : int
        Number of points to be calculated
    **par dt : float
        Time resolution of the density matrix trajectory
    **par rates : numpy array
        Matrix containing the 
    """
    def __init__(self, yi, H, *sigmas, **par):
        self.H = H
        self.sigmas = sigmas
        self.y = yi
        self.dt = par['dt']
        self.n = np.int(par['n'])
        self.time = np.linspace(0, self.dt*self.n, self.n)
        self.rho = np.zeros((len(yi), len(yi), self.n), dtype = np.complex_)
        self.rho[:,:,0] = yi
        self.steady = np.array([], dtype = np.complex_)
        self.g1 = np.zeros(self.n, dtype = np.complex_)
        self.g2 = np.zeros(self.n, dtype = np.float)
        if not len(par['rates']):
            self.rates = np.ones((len(sigmas), len(sigmas)))
        else:
            self.rates = par['rates']

    def _ME(self, t, yt):
        """
        Function returning the set of differential equations to be solved.
        
        Parameters
        -----------
        t : Float
            Time
        yt : Complex numpy array
            Density matrix
        """
        if type(self.H) == types.FunctionType: # if H is a function
            Haux = self.H(t)
        else:
            Haux = self.H
        yt = yt.reshape(Haux.shape)
        L = np.zeros(Haux.shape, dtype = np.complex_) # Lindblad term
        n, m = (-1,-1)
        for sig1 in self.sigmas:
            n += 1
            m = -1
            for sig2 in self.sigmas:
                m += 1
                L += self.rates[n,m]*(np.dot(np.dot(sig1, yt), adj(sig2))-0.5*(np.dot(np.dot(adj(sig2), sig1), yt)+np.dot(yt, np.dot(adj(sig2), sig1))))
        dydx = -1j*(np.dot(Haux, yt) - np.dot(yt, Haux)) + L # Master equation
        return np.ravel(dydx)
    
    def get_steady_state(self):
        """
        This function returns the steady-state solution of the master equation.
        It manipulates mathematical symbols (sympy) corresponding to the density
        matrix elements to extract the coefficient matrix called A. Then it uses
        the function numpy.linalg.solve() to calculate the steady-state solution.
        """
        if type(self.H) == types.FunctionType:
            Haux = self.H(0)
            print("Warning: calculating the steady-state solution for a quantum system with time-dependent Hamiltonian.")
        else:
            Haux = self.H
        yt = np.array(sy.symbols('y0:{}'.format(Haux.shape[0]**2))) # symbolic density matrix
        yt = yt.reshape(Haux.shape) # in a matrix form
        L = sy.zeros(yt.shape[0],yt.shape[1]) # Lindblad term
        n, m = (-1,-1)
        for sig1 in self.sigmas:
            n += 1
            m = -1
            for sig2 in self.sigmas:
                m += 1
                L += self.rates[n,m]*(np.dot(np.dot(sig1, yt), adj(sig2))-0.5*(np.dot(np.dot(adj(sig2), sig1), yt)+np.dot(yt, np.dot(adj(sig2), sig1))))
        dydx = -1j*(np.dot(Haux, yt) - np.dot(yt, Haux)) + L # Master equation
        dydx[0] = np.trace(yt) # Sum of probabilities must be 1
        dydx = np.ravel(dydx)
        A = np.zeros((len(dydx), len(dydx)), dtype = np.complex_)
        for i in range(len(dydx)):
            expr = dydx[i].expand()
            for j in range(len(dydx)):
                A[i,j] = expr.collect(sy.Symbol('y{}'.format(j))).coeff(sy.Symbol('y{}'.format(j))) # obtaining the coefficient matrix
#        B = np.zeros(16, dtype = np.complex_)
        B = np.zeros(len(dydx), dtype = np.complex_)
        B[0] = 1.0 # accounting for the sum of the diagonal elements in dydx[0]
        try:
            self.steady = np.linalg.solve(A, B)
        except:
            print("\nWarning (singular matrix): using np.linalg.lstsq.\n")
            self.steady, residual, rank, singular = np.linalg.lstsq(A, B, rcond=None)
        self.steady = self.steady.reshape(Haux.shape)
        return self.steady
        
    def trajectory(self, atol = 1e-7, rtol = 1e-4):
        """
        Function to calculate the trajectory of the density matrix.
        
        Parameters
        -----------
        atol : Float
            Absolute tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        rtol : Float
            Relative tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        """
        res = solve_ivp(self._ME, (self.time[0], self.time[-1]), np.ravel(self.y), method='RK45', t_eval=self.time, rtol = rtol, atol = atol)
        self.time = res['t']
        self.rho  = res['y'].reshape((self.H.shape[0], self.H.shape[1], len(self.time)))
        self.steady = self.rho[:,:,-1]
        return self.rho

    def g1Func(self, *sigmas, atol=1e-7, rtol=1e-4):
        """
        This function returns the first-order correlation function from the
        master equation.
        
        Parameters
        -----------
        *sigmas : List of complex numpy arrays
            Operator(s) describing the radiative decay(s).
        atol : Float
            Absolute tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        rtol : Float
            Relative tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        """
        if len(self.steady)==0:
            self.get_steady_state()
        n, m = (-1,-1)
        for sig1 in self.sigmas:
            n += 1
            m = -1
            for sig2 in self.sigmas:
                m += 1
                yi = np.dot(sig1, self.steady)
                self.y = yi
                i=0
                res = solve_ivp(self._ME, (self.time[0], self.time[-1]), np.ravel(self.y), method='RK45', t_eval=self.time, rtol = rtol, atol = atol)
                yaux = res['y'].reshape((yi.shape[0], yi.shape[1], len(self.time)))
                while i < self.n:
                    self.g1[i] += self.rates[n,m]*np.trace(np.dot(yaux[:,:,i], adj(sig2)))
                    i+=1
        return self.g1
    
    def powerSpectrum(self, *sigmas, atol=1e-7, rtol=1e-4):
        """
        This function returns the first-order correlation function from the
        master equation.
        
        Parameters
        -----------
        *sigmas : List of complex numpy arrays
            Operator(s) describing the radiative decay(s).
        atol : Float
            Absolute tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        rtol : Float
            Relative tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        """
        if np.all(self.g1 == 0):
            self.g1Func(*sigmas, atol=atol, rtol=rtol)
        self.spec = 2.0*np.real(np.fft.fft(self.g1))
#        self.spec[0] = self.spec[1]
        self.freq = np.fft.fftfreq(self.n, self.time[1] - self.time[0])
        freq, spec = zip(*sorted(zip(self.freq, self.spec)))
        self.freq = np.array(freq)
        self.spec = np.array(spec)
        return self.freq, self.spec
    
    def g2Func(self, *sigmas, atol=1e-7, rtol=1e-4):
        """
        This function returns the second-order correlation function from the
        master equation.
        
        Parameters
        -----------
        *sigmas : List of complex numpy arrays
            Operator(s) describing the radiative decay(s).
        atol : Float
            Absolute tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        rtol : Float
            Relative tolerance of the Runge-Kutta method in the scipy.integrate.solve_ivp function.
        """
        if len(self.steady)==0:
            self.get_steady_state()
        q, k, n, m = (-1, -1, -1, -1)
        for sig1 in sigmas:
            q += 1
            k = -1
            n = -1
            m = -1
            for sig2 in sigmas:
                k += 1
                n = -1
                m = -1
                for sig3 in sigmas:
                    n += 1
                    m = -1
                    for sig4 in sigmas:
                        m += 1
                        yi = self.rates[q,k]*np.dot(np.dot(sig1, self.steady), adj(sig2))
                        self.y = yi
                        res = solve_ivp(self._ME, (self.time[0], self.time[-1]), np.ravel(self.y), method='RK45', t_eval=self.time, rtol = rtol, atol = atol)
                        yaux = res['y'].reshape((yi.shape[0], yi.shape[1], len(self.time)))
                        i=0
                        while i < self.n:
                            self.g2[i] += np.real(self.rates[n,m]*np.trace(np.dot(yaux[:,:,i], np.dot(adj(sig3), sig4))))
                            i+=1
        self.g2 /= self.g2[-1] # Normalization by hand. Fix it!
        self.time = np.append(-self.time[::-1], self.time[1:])
        self.g2 = np.append(self.g2[::-1], self.g2[1:])
        return self.time, self.g2