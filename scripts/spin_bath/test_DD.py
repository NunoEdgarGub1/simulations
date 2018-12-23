
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

for B in [0,5,10, 20, 50, 100, 500, 1000, 3000]:
    exp = NBath.CentralSpinExperiment()
    exp.set_thresholds (A = 500e3, sparse = 10)
    exp.set_magnetic_field (Bz=B*1e-4, Bx=0)
    print ("------- MAGNETIC FIELD: ", B, " gauss")
    exp.generate (nr_spins = 35, concentration = 0.02, single_exp = True)
    #exp.print_nuclear_spins()
    exp.FID_indep_Nspins (tau = np.linspace (0, 30e-6, 1000))

    tau_max= 10.e-3
    exp.Hahn_echo_indep_Nspins (S1=1, S0=0, tau = np.linspace (0, tau_max, 100000))
    #exp.Hahn_Echo_clus (tauarr = np.linspace (0, tau_max, 500), phi = 0, do_compare = False)
    exp.Hahn_echo (tau = np.linspace (0, tau_max, 1000), phi = 0)

#exp.Hahn_echo_indep_Nspins (S1=-0.5, S0=0.5, tau = np.linspace (0, tau_max, 100000))
#exp.Hahn_echo_indep_Nspins (S1=-1.5, S0=-0.5, tau = np.linspace (0, tau_max, 100000))


'''
t0 = 0e-6
t1 = 10e-6
exp.dynamical_decoupling_indep_Nspins (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=0.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)

t0 = 7e-6
t1 = 10e-6
exp.dynamical_decoupling_indep_Nspins (S1=1, S0=0, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=0.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=-1.5, S0=-0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
exp.dynamical_decoupling_indep_Nspins (S1=1.5, S0=0.5, tau = np.linspace (t0, t1, 10000), nr_pulses = 128)
'''
