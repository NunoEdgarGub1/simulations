
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from simulations.libs.spin import nuclear_spin_bath as NBath
reload (NBath)

exp = NBath.CentralSpinExp_cluster (nr_spins = 25, auto_save = False)
exp.set_workfolder (r'C:\Users\cristian\Research\Work-Data')
#exp.print_nuclear_spins()

B=5000
j = 0
exp.generate_bath (concentration = 0.015, name = 'bath'+str(j))
exp.set_thresholds (A = 500e3, sparse = 10)
exp.set_magnetic_field (Bz=B*1e-4, Bx=0)
print ("------- MAGNETIC FIELD: ", B, " gauss")

tau_max= 10.e-3
tau_ind = np.linspace (0, tau_max, 100000)
tau_cls = np.linspace (0, tau_max, 250)
ind = exp.Hahn_echo_indep_Nspins (S1=1, S0=0, tau = tau_ind, do_plot=False, name = str(B)+'gauss')
clus = exp.Hahn_echo (tau = tau_cls, phi = 0, do_plot = False, name = str(B)+'gauss')

plt.figure (figsize = (15,8))
plt.plot (1e6*tau_ind, ind, color = 'crimson')
plt.plot (1e6*tau_cls, clus, color = 'royalblue')
plt.xlabel ('tau (us)', fontsize=17)
plt.title ('Hahn Echo, B = '+str(B)+ ' Gauss', fontsize=17)
plt.show()


#t0 = 5e-6
#t1 = 15e-6
#exp.dynamical_decoupling_indep_Nspins (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 32)
#exp.dynamical_decoupling_indep_Nspins (S1=0, S0=-1, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
#exp.dynamical_decoupling_indep_Nspins (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
#exp.dynamical_decoupling_indep_Nspins (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)

'''
t0 = 7e-6
t1 = 10e-6
exp.dynamical_decoupling_indep_Nspins (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=0.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
'''
