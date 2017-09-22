import numpy as np
import matplotlib.pyplot as plt

n = 10000
t = np.linspace (0, 3000, n)*1e-9
F0 = 5e6
phi0 = 0


def IQmod (I, Q):
	R = (I**2+Q**2)**0.5+1e-9
	I=I/R
	Q=Q/R
	y = I*np.cos(2*np.pi*F0*t+phi0) + Q*np.sin(2*np.pi*F0*t+phi0)
	return y

def SSB (fc, A, phase):
	Issb = A*sin (2*np.pi*fc*t+phase)
	Qssb = A*cos (2*np.pi*fc*t+phase)
	
	y = IQmod (I=Issb, Q=Qssb)#Issb*np.cos(2*np.pi*F0*t+phi0) + Qssb*np.sin(2*np.pi*F0*t+phi0)
	return y

I0 = np.zeros(n)
I0 [2000:4000] = 1
I0 [6000:8000] = 1
Q0 = I0
y0 = IQmod (I=I0, Q=Q0)

I = np.zeros(n)
I [2000:4000] = 1
I [6000:8000] = 2
Q = np.zeros(n)
Q [2000:4000] = 1
Q [6000:8000] = 0.5
y = IQmod (I=I, Q=Q)



plt.figure(figsize=(20,5))
plt.plot (t*1e6, y0, color='RoyalBlue', linewidth = 2)
plt.plot (t*1e6, y, color='crimson', linewidth = 2)
plt.title ('IQ modulation', fontsize=18)
plt.show()

A0 = np.zeros(n)
A0 [2000:4000] = 1
A0 [6000:8000] = 1
phase0 = np.zeros(n)
phase0 [2000:4000] = 0
phase0 [6000:8000] = 0
y0 = SSB(fc=1e6, A=A0, phase=phase0)


A = np.zeros(n)
A [2000:4000] = 1
A [6000:8000] = 1
phase = np.zeros(n)
phase [2000:4000] = 0
phase [6000:8000] = np.pi/4.
y = SSB(fc=1e6, A=A, phase=phase)

plt.figure(figsize=(20,5))
plt.plot (y0, color='RoyalBlue', linewidth = 2)
plt.plot (y, color='crimson', linewidth = 2)
plt.title ('SSB modulation', fontsize=18)
plt.show()


