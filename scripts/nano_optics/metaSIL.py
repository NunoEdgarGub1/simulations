
import numpy as np
import pylab as plt

wl = 900e-9
n = 2.5
m = -0.00005
n_samples = 1000
L = 10e-6

theta_i = (np.pi/180.)*np.linspace (-70, 70, n_samples)
x = L*np.tan(theta_i)
dx = (x[1]-x[0])*1e9
print 'dx = ', dx, ' nm'

#m = -40
#phase = np.pi*m/(np.cos(np.arctan(x[n_samples/2:]/L)))

m = -3
phase = np.zeros(n_samples)
phase[n_samples/2:] = np.pi*m*x[n_samples/2:]/wl
phase[:n_samples/2] = -np.pi*m*x[:n_samples/2]/wl
phase[n_samples/2-150:n_samples/2+150] = 0*phase[n_samples/2-150:n_samples/2+150]
phase = phase - 0.5*np.pi/(np.cos(np.arctan(x/L)))
#m = -2
#phase = 
gradient = (phase [1:] - phase [:-1])/(x[1:] - x[:-1])
print size(gradient), size(theta_i)
theta_i = (np.pi/180.)*np.linspace (-70, 70, size (gradient))
#m = (1./1.1)*np.ones(1000)

theta_t_noMS = np.arcsin (n*np.sin(theta_i))
theta_t_MS = np.arcsin (n*np.sin(theta_i) + (wl/(2*3.14))*gradient)


#phase_paper = (np.pi/(0.55*wl))*x

plt.figure(figsize = (8,8))
plt.plot (theta_i*(180/np.pi), theta_t_MS *(180/np.pi), linewidth =2, label = 'metasurface')
plt.plot (theta_i*(180/np.pi), theta_t_noMS *(180/np.pi), linewidth =2, label = 'no metasurface')
plt.xlabel ('angle incidence [deg]', fontsize=15)
plt.ylabel ('angle refraction [deg]', fontsize = 15)
plt.legend(loc=2)
plt.axis ('equal')
plt.show()



wrapped_phase = phase % (2 * np.pi )


plt.figure(figsize = (18,3))
plt.plot (x*1e6, wrapped_phase/(2*np.pi), 'c', linewidth =1)
plt.plot (x*1e6, wrapped_phase/(2*np.pi), '.b', linewidth =2)
plt.ylabel ('phase * 2pi', fontsize = 15)
#plt.plot (x*1e6, phase_paper, 'r', linewidth =2)
plt.xlabel ('x [micron]', fontsize=15)
plt.show()

plt.plot (x, phase)
plt.show()

plt.plot (gradient*wl/6.28)
plt.ylim ([-2,2])
plt.show()

'''
print 'Digitizing...'
dig_step = 1000
dig_phase = []
for i in np.arange((len(phase)/dig_step)):
	media = np.mean(phase [i*dig_step:(i+1)*dig_step])
	dig_phase = np.hstack ([dig_phase, media*np.ones(dig_step)])


dig_grad = (dig_phase [1:] - dig_phase [:-1])/(x[1:] - x[:-1])

plt.figure (figsize = (15,5))
plt.plot (dig_phase)
plt.show()

theta_i_old = np.copy(theta_i)
theta_i = theta_i[:len(dig_grad)]
theta_t_MS = np.arcsin (n*np.sin(theta_i) + (wl/(2*3.14))*dig_grad)


#phase_paper = (np.pi/(0.55*wl))*x

plt.figure(figsize = (8,8))
plt.plot (theta_i*(180/np.pi), theta_t_MS *(180/np.pi), linewidth =2, label = 'metasurface')
plt.plot (theta_i_old*(180/np.pi), theta_t_noMS *(180/np.pi), linewidth =2, label = 'no metasurface')
plt.xlabel ('angle incidence [deg]', fontsize=15)
plt.ylabel ('angle refraction [deg]', fontsize = 15)
plt.legend(loc=2)
plt.axis ('equal')
plt.show()
'''