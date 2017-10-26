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


def toy_bandgap (l0):
	l = np.linspace (400, 1500, 2000)*1e-9
	w = 2*np.pi*3e8/l
	w0 = 2*np.pi*3e8/l0
	phi = 0.5*np.pi*(l0/l)
	n = 1.1

	#A = 9*np.exp(2j*phi)-1
	#B = 3-3*np.exp(-2j*phi)
	#C = -3+3*np.exp(-2j*phi)
	#D = 9*np.exp(-2j*phi)-1

	A_D_2 = (n+1)**2*np.cos(2*phi) - (n-1)**2 # (A+D)/2 
	Lambda = l0/4.

	k = (1./Lambda)*np.arccos(A_D_2)

	plt.figure (figsize=(10,10))
	plt.plot (k*Lambda/np.pi, w/w0, linewidth=3, color = 'royalblue')
	plt.plot (-k*Lambda/np.pi, w/w0, linewidth=3, color = 'royalblue')
	#plt.axis ([-1,1, 1.45, 2.5])
	plt.show()

def DBR (N=5, th0 = 0., l0 = 900, n1=1.5, n2=2, do_plot = False):
	
	l_array = np.linspace (500, 1500, 1000)
	R = np.zeros (len(l_array))

	th1 = np.arcsin(np.sin(th0)/n1)
	th2 = np.arcsin(n1*np.sin(th1)/n2)


	D0 = np.matrix ([[1,1],[np.cos(th0), -np.cos(th0)]])
	D1 = np.matrix ([[1,1],[n1*np.cos(th1), -n1*np.cos(th1)]])
	D2 = np.matrix ([[1,1],[n2*np.cos(th2), -n2*np.cos(th2)]])

	S01 = np.linalg.inv(D0)*D1
	S12 = np.linalg.inv(D1)*D2
	S20 = np.linalg.inv(D2)*D0
	S21 = np.linalg.inv(D2)*D1

	d1 = l0/(4*n1)
	d2 = l0/(4*n2)

	for ind,l in enumerate(l_array):
		phi1 = n1*(2*np.pi/l)*d1
		phi2 = n2*(2*np.pi/l)*d2

		G1 = np.matrix([[np.exp(1j*phi1), 0],[0, np.exp(-1j*phi1)]])
		G2 = np.matrix([[np.exp(1j*phi2), 0],[0, np.exp(-1j*phi2)]])

		S = S20
		for i in np.arange(N):
			S = S*G2*S12*G1*S21

		s21 = S[1,0]
		s11 = S[0,0]

		R[ind] = np.abs(s21/s11)**2

	if do_plot:
		plt.figure (figsize = (12, 6))
		plt.plot (l_array, R, linewidth=3, color='royalblue')
		plt.xlabel ('wavelength (nm)', fontsize = 20)
		plt.ylabel ('reflectivity', fontsize = 20)
		plt.show()

	'''
	plt.figure (figsize = (12, 6))
	plt.plot (2*np.pi*3e8/l_array, R, linewidth=3, color='royalblue')
	plt.xlabel ('frequency (Hz)', fontsize = 20)
	plt.ylabel ('reflectivity', fontsize = 20)
	plt.show()
	'''

	return l_array, R

def compare_DBR_reflectivity (N = [3,5]):

	plt.figure (figsize = (12, 6))

	for n in N:
		l, R = DBR (N=n)
		plt.plot (l, R, linewidth=3)
	plt.xlabel ('wavelength (nm)', fontsize = 20)
	plt.ylabel ('reflectivity', fontsize = 20)
	plt.show()

def compare_DBR_stopband (n2 = [2, 2.5, 3]):

	plt.figure (figsize = (12, 6))

	for iii in n2:
		l, R = DBR (N=10, n2=iii)
		plt.plot (l, R, linewidth=3)
	plt.xlabel ('wavelength (nm)', fontsize = 20)
	plt.ylabel ('reflectivity', fontsize = 20)
	plt.show()


#reflectance (n1=1.5, n2=1.)
#band_infinite (L1= 170e-9, L2 = 100e-9, n1 = 3, n2 = 1.5)
#toy_bandgap (l0=1000e-9)

#DBR(N=10)
compare_DBR_reflectivity (N = [3, 5, 10])
compare_DBR_stopband()
