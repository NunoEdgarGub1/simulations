import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
#sys.path.append (folder)
from simulations.libs.spin import nuclear_spin_bath as NBath
reload (NBath)

# creates the diamond lattice and sets the nr of nuclear spins to be created
exp = NBath.CentralSpinExp_cluster (nr_spins = 35, auto_save = True)
# sets folder where data will be saved
exp.set_workfolder (r'C:\Users\cristian\Research\Work-Data')

# sets maximum size clusters 
# cluster = group of NVs treated with full density matrix
# interaction between clusters is neglected (diluted bath = clusters are far apart)
clus_size=4
exp.set_cluster_size(g=clus_size)

# generates the bath with a given concentration of nuclear spins
# (the total nr of spins was set before)
exp.generate_bath (concentration = 0.011, name = 'bath1')
# prints info about the bath
#exp.print_nuclear_spins()

# sets thresholds
# A: threshold for maximum hyperfine (to avoid strongly-coupled nuclear spins)
exp.set_thresholds (A = 500e3, sparse = 10)
B = 100 # field in gauss
exp.set_magnetic_field (Bz=B*1e-4, Bx=0)
#exp.FID_indep_Nspins (tau = np.linspace (0, 30e-6, 1000))

# HAHN ECHO
# we have two functions for Hahn echo, one that neglects all interactions betweeen 
# nuclear spins (independent nuclear spins), the other uses the cluster approach
tau_max= 5.e-3
ind = exp.Hahn_echo_indep_Nspins (S1=1, S0=0, tau = np.linspace (0, tau_max, 100000), do_plot=False)
nr_points_hahn = 100
clus = exp.Hahn_echo (tau = np.linspace (0, tau_max, nr_points_hahn), phi = 0, do_plot = False)

plt.figure (figsize = (15,8))
plt.plot (1e6*np.linspace (0, tau_max, 100000), ind, color = 'crimson')
plt.plot (1e6*np.linspace (0, tau_max, nr_points_hahn), clus, color = 'royalblue')
plt.xlabel ('tau (us)', fontsize = 15)
plt.ylabel ('probability |0>', fontsize = 15)
plt.show()


