import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

mpl.rc('xtick', labelsize=22) 
mpl.rc('ytick', labelsize=22)

c= 3e8

def reflectance (n1, n2):
	th1 = (np.pi/180.)*np.linspace (0, 90, 10000)
	th2 = np.arcsin((n1*np.sin(th1)/n2))
	rs = (n1*np.cos(th1) - n2*np.cos(th2))/(n1*np.cos(th1) + n2*np.cos(th2))
	ts = 2*n1*np.cos(th1)/(n1*np.cos(th1) + n2*np.cos(th2))

	R = np.abs(rs)**2
	T = ((n2*np.cos(th2))/(n1*np.cos(th1)))*(np.abs(ts)**2)

	plt.figure (figsize = (6,8))
	plt.plot (th1*(180/np.pi), R, 'royalblue', linewidth=5)
	plt.plot (th1*(180/np.pi), T, 'crimson', linewidth=5)
	plt.axis ([0, 90, 0, 1])
	plt.show()

def _infinite_periodic (k1x, k2x, L1, L2):
	C2 = np.cos(k2x*L2)
	S2 = np.sin(k2x*L2)
	A = np.exp(1j*k1x*L1)*(C2+0.5*1j*(k2x/k1x+k1x/k2x)*S2)
	#B = np.exp(-1j*k1x*L1)*(0.5*1j*(k2x/k1x-k1x/k2x)*S2)
	#C = np.exp(1j*k1x*L1)*(-0.5*1j*(k2x/k1x-k1x/k2x)*S2)
	D = np.exp(-1j*k1x*L1)*(C2-0.5*1j*(k2x/k1x+k1x/k2x)*S2)

	Lambda = L1+L2
	K = (1/Lambda)*np.arccos(0.5*np.real(A+D))
	return K

def band_infinite (L1, L2, n1, n2):

	l = 1e-9*np.linspace (330, 200000, 1000000)
	w = 2*np.pi*c/l
	th1 = 0
	th2 = np.arcsin((n1*np.sin(th1)/n2))

	k1x = n1*w/c*((1-(np.sin(th1)**2))**0.5)
	k2x = n2*w/c*((1-(np.sin(th2)**2))**0.5)

	K = _infinite_periodic (k1x=k1x, k2x=k2x, L1=L1, L2=L2)
	K = np.hstack ([-K, K])
	w = np.hstack([w,w])

	Lambda = L1+L2
	fig = plt.figure (figsize = (10,8))
	plt.plot (K*Lambda/(np.pi), w, 'royalblue', linewidth = 5)
	plt.savefig ('C:/Users/cristian/band.svg')
	plt.show()


def reflectivityDBR ()


#reflectance (n1=1.5, n2=1.)
band_infinite (L1= 170e-9, L2 = 100e-9, n1 = 3, n2 = 1.5)
