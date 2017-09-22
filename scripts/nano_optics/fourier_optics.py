
import numpy as np

def fresnel_kernel_1D (fx, d2, lam):
	#This function calculates the free space transfer for diffraction
# 	#by a distance d2. Distances are measured in terms of A.

	beta = 1/lam**2 - fx**2 
	#for large values of d2, the large phase of this function tends to disrupt Fourier transforms.
	# substituting phip = exp(- i * 2 * pi * (d2 .* sqrt(dg)-d2/lam)); seems to correct this problem.
	ind = np.where (beta>0)
	phip = np.zeros (len(fx)) + 1j*np.zeros (len(fx))
	phip [ind] = exp(- 1j * 2 * np.pi * d2 * np.sqrt(beta[ind]))

	return phip

def fresnel_1D ():
	r = 32. #range
	npoints=512.
	step=r/npoints
	l=0.5*10**(-3);
	distance=1000;
	scale = np.arange (-r/2, (r/2)-step, step)
	ftscale=(npoints/r)*scale;
	x = scale
	fx = ftscale

	phase_x = np.ones(len(x))
	phase_x = np.exp(-(x/5)**2)

	dt=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(phase_x)));
	ff=fresnel_kernel_1D (fx, distance, l);

	ft=ff*dt;
	intensity = np.abs(np.fft.ifft(np.fft.fftshift(ft)))**2

	plt.plot (x, np.abs(phase_x)**2, 'b', linewidth =2)
	#plt.plot (x, np.arg(phase_x), ':b', linewidth =2)
	plt.plot (x, intensity, 'r', linewidth =2)
	plt.show()

fresnel_1D ()

