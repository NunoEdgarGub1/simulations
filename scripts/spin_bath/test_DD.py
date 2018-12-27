
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from simulations.libs.spin import nuclear_spin_bath as NBath
reload (NBath)

#exp = NBath.NSpinBath()

#Ap_NV, Ao_NV, Azx_NV, Azy_NV, r_NV, pair_lst, geom_lst, dC_lst, T2_h, T2_l = exp.generate_NSpin_distr (N = 30, conc = 0.02, eng_bath_cluster = [])
#exp.set_spin_bath (Ap=Ap_NV, Ao = Ao_NV, Azx = Azx_NV, Azy = Azy_NV)
#exp.plot_spin_bath_info ()
#exp.set_B (Bp=0.001, Bo =0.000)
#exp.FID (tau = np.linspace (1, 100000, 10000)*1e-9)

exp = NBath.CentralSpinExp_cluster (nr_spins = 35, auto_save = True)
exp.set_workfolder (r'C:\Users\cristian\Research\Work-Data')
exp.generate_bath (concentration = 0.011, name = 'bath1')
#exp.print_nuclear_spins()

for B in [1, 100, 400, 1000, 5000]:
    exp.set_thresholds (A = 500e3, sparse = 10)
    exp.set_magnetic_field (Bz=B*1e-4, Bx=0)
    print ("------- MAGNETIC FIELD: ", B, " gauss")
    #exp.FID_indep_Nspins (tau = np.linspace (0, 30e-6, 1000))

    tau_max= 5.e-3
    ind = exp.Hahn_echo_indep_Nspins (S1=1, S0=0, tau = np.linspace (0, tau_max, 100000), do_plot=False)
    nr_points_hahn = 3
    clus = exp.Hahn_echo (tau = np.linspace (0, tau_max, nr_points_hahn), phi = 0, do_plot = False)

    plt.figure (figsize = (15,8))
    plt.plot (1e6*np.linspace (0, tau_max, 100000), ind, color = 'crimson')
    plt.plot (1e6*np.linspace (0, tau_max, nr_points_hahn), clus, color = 'royalblue')
    plt.show()


    #t0 = 0e-6
    #t1 = 10e-6
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
