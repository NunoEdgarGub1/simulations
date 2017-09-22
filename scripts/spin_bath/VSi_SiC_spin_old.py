import numpy as np
import pylab as plt
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

def four_level_system_Rabi_FFT (Omega = 1e6, title='', cristian = False):
	ZFS = 4e6
	g = 2*np.pi*14e9
	B_Gauss = 20
	B = B_Gauss*1e-4
	E32m = -1.5*g*B
	E32p = +1.5*g*B
	E12m = ZFS-0.5*g*B
	E12p = ZFS+0.5*g*B
	E1 = E32m
	E2 = E12m
	E3 = E12p
	E4 = E32p
	print "### Transitions: "
	print " -3/2 <--> -1/2, frq: ", abs(E12m-E32m)*1e-6, ' MHz'
	print " -1/2 <--> +1/2, frq: ", abs(E12p-E12m)*1e-6, ' MHz'
	print " +1/2 <--> +3/2, frq: ", abs(E12p-E32p)*1e-6, ' MHz'

	alpha0 = 3**0.5 #strength -3/2 <--> -1/2 transition
	beta0 = 2. #strength -1/2 <--> +1/2 transition
	gamma0 = 3**0.5 #strength +1/2 <--> 3/2 transition
	alpha = alpha0/alpha0
	beta = beta0/alpha0
	gamma = gamma0/alpha0
	print alpha, beta, gamma
	nr_steps = 200
	www = np.linspace (0, 50, 5000)*1e6
	frq = np.linspace (160, 190, nr_steps)*1e6
	f_fft = np.zeros([nr_steps, len (www)])
	t = np.linspace(0, 20e-6, 1000)
	ind = 0
	G = 0

	in_fft = 5
	end_fft_plot = 200
	f_fft = np.zeros((len(frq), len(t)/2-in_fft))
	rabi_t = np.zeros((nr_steps, len(t)))
	ind = 0
	for f in frq:
		H = np.array ([[E1+3*f/2.+G, alpha*Omega, 0, 0],[alpha*Omega, E2 + f/2.+G, beta*Omega, 0],[0,beta*Omega, E3-f/2.+G, gamma*Omega],[0,0, gamma*Omega, E4-3*f/2.+G]])
		w, V = np.linalg.eigh(H)

		init_state_array = [3, 0]
		p1 = np.zeros(len(t))
		p2 = np.zeros(len(t))
		p3 = np.zeros(len(t))
		p4 = np.zeros(len(t))
		for init_state in init_state_array:
			factor = 1.
			if (init_state in [1,2]): 
				factor = 0.

			C = np.linalg.pinv(V)
			psi_f1 = np.abs(np.exp(1j*w[0]*t)*C[0,0]*V[0, init_state]+np.exp(1j*w[1]*t)*C[0,1]*V[1, init_state]+np.exp(1j*w[2]*t)*C[0,2]*V[2, init_state]+np.exp(1j*w[3]*t)*C[0,3]*V[3, init_state])**2
			psi_f2 = np.abs(np.exp(1j*w[0]*t)*C[1,0]*V[0, init_state]+np.exp(1j*w[1]*t)*C[1,1]*V[1, init_state]+np.exp(1j*w[2]*t)*C[1,2]*V[2, init_state]+np.exp(1j*w[3]*t)*C[1,3]*V[3, init_state])**2
			psi_f3 = np.abs(np.exp(1j*w[0]*t)*C[2,0]*V[0, init_state]+np.exp(1j*w[1]*t)*C[2,1]*V[1, init_state]+np.exp(1j*w[2]*t)*C[2,2]*V[2, init_state]+np.exp(1j*w[3]*t)*C[2,3]*V[3, init_state])**2
			psi_f4 = np.abs(np.exp(1j*w[0]*t)*C[3,0]*V[0, init_state]+np.exp(1j*w[1]*t)*C[3,1]*V[1, init_state]+np.exp(1j*w[2]*t)*C[3,2]*V[2, init_state]+np.exp(1j*w[3]*t)*C[3,3]*V[3, init_state])**2
			norm = psi_f1 + psi_f2 + psi_f3 + psi_f4
			psi_f1 = psi_f1/norm
			psi_f2 = psi_f2/norm
			psi_f3 = psi_f3/norm
			psi_f4 = psi_f4/norm
			p1 = p1 + factor*psi_f1
			p2 = p2 + factor*psi_f2
			p3 = p3 + factor*psi_f3
			p4 = p4 + factor*psi_f4

		p1 = p1/len(init_state_array)
		p2 = p2/len(init_state_array)
		p3 = p3/len(init_state_array)
		p4 = p4/len(init_state_array)
		PL = (p1+0.5*p2+0.5*p3+p4)*np.exp(-(t/4e-6)**2)
		rabi_t [ind, :] = PL
		if cristian:
			fff = np.fft.ifftshift(np.abs(np.fft.fft(PL))**2)
			f_fft [ind, :] = fff[len(t)/2+in_fft:]
		else:
			fff = np.fft.rfft(PL)
			fff = fff[15:100]
			if ind==0:
				f_fft = fff
			else:
				f_fft = np.vstack((f_fft, fff))
		ind = ind + 1

	t = t/(2*np.pi)
	df = 1./max(t)
	freq = np.fft.ifftshift(np.fft.fftfreq(PL.size))

	r_frq = freq*df*len(t)
	r_frq = 2*np.pi*r_frq[len(t)/2+in_fft:]
	X, Y = meshgrid (r_frq*1e-6, frq*1e-6)

	do_renorm = 1
	if do_renorm:
		a,b = np.shape (f_fft)
		for i in np.arange(b):
			y = f_fft[:, i]
			f_fft[:, i] = y/np.sum(y)

	plt.figure(figsize=(7, 10))
	#plt.imshow (np.abs(f_fft), aspect='auto', interpolation='none', cmap = 'viridis')
	plt.pcolor (f_fft, cmap= 'viridis')
	plt.set_cmap(viridis)

	plt.title (title, fontsize = 18)
	plt.xlabel ('Rabi frequency [MHzs]', fontsize = 18)
	plt.ylabel ('driving frequency [MHz]', fontsize = 18)
	#plt.axis([2, 30, 160, 190])
	plt.show()

	#X, Y = meshgrid (t*1e6, frq*1e-6)
	#plt.figure(figsize=(10, 10))
	#plt.pcolor (X, Y, rabi_t)
	#plt.xlabel ('time [us]', fontsize = 15)
	#plt.ylabel ('driving frequency [MHz]', fontsize = 15)
	#plt.axis([0, 20, 165, 190])
	#plt.show()

four_level_system_Rabi_FFT(Omega=2e6, title = 'Rabi: 2MHz')
four_level_system_Rabi_FFT(Omega=2**0.5*2e6, title = 'power x2')
four_level_system_Rabi_FFT(Omega=2*2e6, title = 'power x 4')
four_level_system_Rabi_FFT(Omega=8**0.5*2e6, title = 'power x 8')
four_level_system_Rabi_FFT(Omega=16**0.5*2e6, title = 'power x 16')

#Rabi(f=176e6)


