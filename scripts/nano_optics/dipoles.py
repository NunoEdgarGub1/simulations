
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

def sim_2D_phi0 (alpha = 0, beta = 0, n =2.5):

	nt = 1
	theta = np.linspace (-np.pi/2., np.pi/2, 10000)
	theta_t = np.arcsin (n*np.sin(theta))



	#plt.figure()
	#plt.plot (theta*(180/np.pi), theta_t*(180/np.pi))
	#plt.show()

	Es = 0
	Ep = np.sin (alpha - theta)
	I0 = (Es)**2 + (Ep)**2

	print n, nt
	t_s = (2*n*np.cos(theta))/(n*np.cos(theta)+nt*np.cos(theta_t))
	t_p = (2*n*np.cos(theta))/(n*np.cos(theta_t)+nt*np.cos(theta))
	factor = (nt*np.cos(theta_t))/(n*np.cos(theta))

	#Let's look at the transmission coefficients
	#plt.figure()
	#plt.plot (theta*(180/np.pi), (abs(t_s)**2)*factor)
	#plt.plot (theta*(180/np.pi), (abs(t_p)**2)*factor)
	#plt.show()


	I = ((t_s*Es)**2 + ((t_p*Ep)**2))*factor
	ind = np.where(np.isnan (I))
	I [ind] = 0
	#I = I/max(I)

	ax = plt.subplot(121, projection='polar')
	ax.plot(theta, I0, color='r', linewidth=3)
	ax.set_rmax(1.0)
	ax.grid(True)
	ax.set_title("dipole emission", va='bottom')
	ax2 = plt.subplot(122, projection='polar')
	ax2.plot(theta, I, '.r', linewidth=3)
	ax2.set_rmax(1.0)
	ax2.grid(True)
	ax2.set_title("transmitted emission", va='bottom')
	plt.show()

	theta_max = np.arcsin(1./n)

	ratio = np.sum(I)/np.sum(I0)
	print 'Transmitted fraction: ', ratio


def sim_2D_integrated_phi (alpha = 0, beta = 0, n =2.5):

	nt = 1
	theta = np.linspace (-np.pi/2., np.pi/2, 10000)
	theta_t = np.arcsin (n*np.sin(theta))

	Es = (np.pi*(np.sin(alpha)**2))**0.5
	Ep = (np.pi*(np.sin(alpha)**2)*(np.cos(theta)**2)+2*np.pi*(np.cos(alpha)**2)*(np.sin(theta)**2))**0.5
	I0 = (Es)**2 + (Ep)**2

	print n, nt
	t_s = (2*n*np.cos(theta))/(n*np.cos(theta)+nt*np.cos(theta_t))
	t_p = (2*n*np.cos(theta))/(n*np.cos(theta_t)+nt*np.cos(theta))
	factor = (nt*np.cos(theta_t))/(n*np.cos(theta))

	I = ((t_s*Es)**2 + ((t_p*Ep)**2))*factor
	ind = np.where(np.isnan (I))
	I [ind] = 0
	#I = I/max(I)

	ax = plt.subplot(121, projection='polar')
	ax.plot(theta, I0/(2*np.pi), color='r', linewidth=3)
	ax.set_rmax(1.0)
	ax.grid(True)
	ax.set_title("dipole emission", va='bottom')
	ax2 = plt.subplot(122, projection='polar')
	ax2.plot(theta, I, '.r', linewidth=3)
	ax2.set_rmax(1.0)
	ax2.grid(True)
	ax2.set_title("transmitted emission", va='bottom')
	plt.show()

	theta_max = np.arcsin(1./n)

	ratio = np.sum(I)/np.sum(I0)
	print 'Transmitted fraction: ', ratio

def sim_2D_integrated_phi_SIL (alpha = 0, beta = 0, n =2.6, max_NA = 1.):

	nt = 1.
	#t_s = (2*n/(n*np.cos(0)+nt*np.cos(0))
	#t_p = (2*n*np.cos)/(n*np.cos(0)+nt*np.cos(0))
	#factor = (nt*np.cos(0))/(n*np.cos(0))

	NA = np.linspace (0, max_NA, 100)
	ratio = np.zeros (len(NA))
	ratio_sil = np.zeros (len(NA))

	theta = np.arange (-np.pi/2, np.pi/2, 0.001)
	Es = (np.pi*(np.sin(alpha)**2))**0.5
	Ep = (np.pi*(np.sin(alpha)**2)*(np.cos(theta)**2)+2*np.pi*(np.cos(alpha)**2)*(np.sin(theta)**2))**0.5
	Inorm = np.sum((Es)**2 + (Ep)**2)


	ind = 0
	for na in NA:
		#t1 = np.arcsin (na)
		#t2 = np.arcsin (1/n)
		#theta_max = min (t2, t1)
		theta_max = np.arcsin (na/n)
		theta = np.arange (-theta_max, theta_max, 0.001)
		Es = (np.pi*(np.sin(alpha)**2))**0.5
		Ep = (np.pi*(np.sin(alpha)**2)*(np.cos(theta)**2)+2*np.pi*(np.cos(alpha)**2)*(np.sin(theta)**2))**0.5
		I0 = (Es)**2 + (Ep)**2

		theta_max = np.arcsin (na/nt)
		theta = np.arange (-theta_max, theta_max, 0.001)
		Es = (np.pi*(np.sin(alpha)**2))**0.5
		Ep = (np.pi*(np.sin(alpha)**2)*(np.cos(theta)**2)+2*np.pi*(np.cos(alpha)**2)*(np.sin(theta)**2))**0.5
		I = (Es)**2 + (Ep)**2

		ratio [ind] = np.sum(I0)/Inorm
		ratio_sil [ind] = np.sum(I)/Inorm
		ind = ind + 1

	return NA, ratio, ratio_sil




def sim_2D_metasurface (alpha = 0, beta = 0, n =2.5, m = -60):
	# we take on th eintegrated-phi version

	nt = 1
	N = 1000
	theta = np.linspace (-np.pi/2., np.pi/2, N)

	x = L*np.tan(theta)
	dx = (x[1]-x[0])*1e9
	print 'dx = ', dx, ' nm'
	phase = np.pi*m/(np.cos(np.arctan(x/L)))

	#m = -
	n_samples = N
	phase = np.zeros(n_samples)
	phase[n_samples/2:] = np.pi*m*(x[n_samples/2:]-x[n_samples/2+150])/wl
	phase[:n_samples/2] = -np.pi*m*(x[:n_samples/2]-x[n_samples/2-150])/wl
	phase[n_samples/2-150:n_samples/2+150] = 0*phase[n_samples/2-150:n_samples/2+150]
	phase[:50] = 0
	phase[-50:] = 0
	gradient = (phase [1:] - phase [:-1])/(x[1:] - x[:-1])

	theta = (np.pi/180.)*np.linspace (-90, 90, size (gradient))
	theta_t = np.arcsin (n*np.sin(theta_i) + (wl/(2*3.14))*gradient)

	wrapped_phase = phase % (2 * np.pi )
	plt.figure(figsize = (18,6))
	ax = plt.subplot(211)
	ax.plot (x*1e6, phase/(2*np.pi), 'c', linewidth =1)
	ax.plot (x*1e6, phase/(2*np.pi), '.b', linewidth =2)
	ax.set_ylabel ('phase * 2pi', fontsize = 15)
	ax.set_xlim ([-50, 50])
	ax.set_xlabel ('x [micron]', fontsize=15)
	ax2 = plt.subplot(212)
	ax2.plot (x*1e6, wrapped_phase/(2*np.pi), 'c', linewidth =1)
	ax2.plot (x*1e6, wrapped_phase/(2*np.pi), '.b', linewidth =2)
	ax2.set_ylabel ('phase * 2pi', fontsize = 15)
	ax2.set_xlim ([-50, 50])
	ax2.set_xlabel ('x [micron]', fontsize=15)
	plt.show()

	plt.figure(figsize = (7,7))
	plt.plot (theta*180/3.14, theta_t*180/3.14, '.',linewidth = 2)
	plt.xlabel ('emission angle [deg]', fontsize = 15)
	plt.ylabel ('transmission angle [deg]', fontsize = 15)
	plt.xlim ([-90, 90])
	plt.ylim ([-90, 90])
	plt.show()


	Es = (np.pi*(np.sin(alpha)**2))**0.5
	Ep = (np.pi*(np.sin(alpha)**2)*(np.cos(theta)**2)+2*np.pi*(np.cos(alpha)**2)*(np.sin(theta)**2))**0.5
	I0 = (Es)**2 + (Ep)**2

	print n, nt
	t_s = (2*n*np.cos(theta))/(n*np.cos(theta)+nt*np.cos(theta_t))
	t_p = (2*n*np.cos(theta))/(n*np.cos(theta_t)+nt*np.cos(theta))
	factor = (nt*np.cos(theta_t))/(n*np.cos(theta))

	I = ((t_s*Es)**2 + ((t_p*Ep)**2))*factor
	ind = np.where(np.isnan (I))
	I [ind] = 0
	#I = I/max(I)

	#re-binning
	n_digit = 50
	theta_t_bins = np.linspace (-np.pi/2, np.pi/2, n_digit)
	ids = np.digitize (theta_t, theta_t_bins)

	I_new = np.zeros(n_digit)
	for i in np.arange(N-1):
		I_new[ids[i]-1] = I_new[ids[i]-1] + I[i]

	plt.figure(figsize=(12,4))
	ax = plt.subplot(131, projection='polar')
	ax.plot(theta, I0/(2*np.pi), color='r', linewidth=3)
	#ax.set_rmax(1.0)
	ax.grid(True)
	ax.set_title("dipole emission", va='bottom')
	#ax2 = plt.subplot(122, projection='polar')
	#ax2.plot(theta, I, '.r', linewidth=3)
	#ax2.set_rmax(1.0)
	#ax2.grid(True)
	#ax2.set_title("emission, surface", va='bottom')
	ax2 = plt.subplot(132, projection='polar')
	ax2.plot(theta_t, I, '.r', linewidth=3)
	#ax2.set_rmax(1.0)
	ax2.grid(True)
	ax2.set_title("transmitted emission", va='bottom')

	ax3 = plt.subplot(133, projection='polar')
	ax3.plot(theta_t_bins, I_new, 'r', linewidth=3)
	#ax2.set_rmax(1.0)
	ax3.grid(True)
	ax3.set_title("transmitted emission", va='bottom')

	plt.show()

	theta_max = np.arcsin(1./n)

	ratio = np.sum(I)/np.sum(I0)
	print 'Transmitted fraction: ', ratio



#the dipole lies in a material with refractive index n, at a distance d from the surface.
#We are looking at the far-field.

n = 2.5
nt = 1
theta_max = np.arcsin(1./n)

#dipole orientation
alpha = 0*np.pi/2 #Theta (0: vertical, pi/2: horizontal)
beta = 0 #Phi

#sim_2D_phi0 (alpha = alpha, beta = 0, n = n)
#sim_2D_integrated_phi (alpha = alpha, beta = 0, n = n)
#sim_2D_metasurface (alpha = alpha, beta = 0, n = n, m = -30)

dipole_comparison = 1
if dipole_comparison:
	na, r_h, r_sil_h = sim_2D_integrated_phi_SIL (alpha = np.pi/2, beta = beta, n=n)
	na, r_v, r_sil_v = sim_2D_integrated_phi_SIL (alpha = 0, beta = beta, n=n)

	fig = plt.figure()
	plt.plot (na, 0.5*r_h, '--b', linewidth = 5)
	plt.plot (na, 0.5*r_v, '--r',linewidth = 5)
	plt.plot (na, 0.5*r_sil_h, 'b', linewidth = 5)
	plt.plot (na, 0.5*r_sil_v, 'r', linewidth = 5)
	plt.xlabel ('numerical aperture', fontsize = 22)
	plt.ylabel ('collection fraction', fontsize = 22)
	fig.savefig ('d:/dipole_fraction.svg')
	plt.show()

	na_140, r_h_140, r_sil_h = sim_2D_integrated_phi_SIL (alpha = np.pi/2, beta = beta, n=n, max_NA=1.4)
	na_140, r_v_140, r_sil_v = sim_2D_integrated_phi_SIL (alpha = 0, beta = beta, n=n, max_NA=1.4)
	na_85, r_h_85, r_sil_h = sim_2D_integrated_phi_SIL (alpha = np.pi/2, beta = beta, n=n, max_NA=0.85)
	na_85, r_v_85, r_sil_v = sim_2D_integrated_phi_SIL (alpha = 0, beta = beta, n=n, max_NA=0.85)

	fig = plt.figure()
	plt.plot (na_140, 0.5*r_h_140, '--b', linewidth = 5)
	plt.plot (na_140, 0.5*r_v_140, '--r',linewidth = 5)
	plt.plot (na_85, 0.5*r_h_85, 'b', linewidth = 5)
	plt.plot (na_85, 0.5*r_v_85, 'r',linewidth = 5)
	plt.xlabel ('numerical aperture', fontsize = 22)
	plt.ylabel ('collection fraction', fontsize = 22)
	fig.savefig ('d:/dipole_fraction.svg')
	plt.show()



#First, let's look at a 2D picture, taking phi = 0

'''
# create supporting points in polar coordinates
theta = np.linspace(-np.pi, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
T, P = np.meshgrid(theta, phi)

#theta_t = np.arcsin (n*np.sin(T))
#alpha = (np.cos (T)/(n*np.cos(theta_t)))**0.5
#t_s = (2*n*np.cos(T)/(n*np.cos(T)+n2*np.cos(theta_t)))*alpha
#t_p = (2*n*np.cos(T)/(n*np.cos(theta_t)+n2*np.cos(T)))*alpha
t_s = 1
t_p = 1
R = np.sin(T)**2 #vertical dipole
#R = np.sin(P)**2 + (t_p*np.cos(T)*np.cos(P))**2
#R = (np.cos(T))**2

# transform them to cartesian system
X, Y = R*np.sin(T)*np.cos(P), R*np.sin(T)*sin(P)
Z = R*cos(T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, cmap=cm.YlOrRd)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')
ax.view_init (elev = 60., azim = 50)
fig.savefig ('D:/dipolo.svg')
plt.show()
'''




