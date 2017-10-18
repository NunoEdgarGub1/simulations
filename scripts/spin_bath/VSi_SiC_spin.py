import numpy as np
import pylab as plt
import matplotlib
from simulations.libs.spin import V1_VSi_SiC_spin_sims as V1
#from analysis_simulations.libs.spin import rabi
reload(V1)
#reload(rabi)

matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22)

t = np.arange (0, .5e-6, 2.e-9)
s = V1.V1_SiC (B_Gauss=63., t=t, T2_star = 100e-9, verbose = True)
s.set_intensity_transitions (I_m32_m12=0., I_m12_p12=0., I_p12_p32=1.)
s.set_ODMR_pars (polariz_array = [1.,0.,0.,1.], ODMR_contrast_array=[1.5,1.,1.,1.5], verbose = True)
s.set_decay (t1 = 200e-9, t2= 200e-9)
s.rabi (f=180.4e6, Omega= 2*3.47e6, do_plot = True)
#s.plot_state_occupancy()

s.rabi_sweep_drive (init_frq=160e6, end_frq=190e6, nr_steps=60, Omega=5.e6, do_plot = True, add_colorbar = True)
s.calc_fft(in_fft=1, config = 'modulo_squared')
s.v_factor = 1.
s.plot_fft(do_renorm=False, add_colorbar = False)


'''
a = rabi.rabi_analysis_VSi_SiC()
a.load_data(23)
a.v_factor =.8
#a.plot_rabi(frq='all')
a.plot_rabi_fft()

a = rabi.rabi_analysis_VSi_SiC()
a.load_data(26)
a.v_factor =.9
#a.plot_rabi(frq='all')
a.plot_rabi_fft()

a = rabi.rabi_analysis_VSi_SiC()
a.load_data(29)
a.v_factor =.6
#a.plot_rabi(frq='all')
a.plot_rabi_fft()

a = rabi.rabi_analysis_VSi_SiC()
a.load_data(32)
a.v_factor =.7
#a.plot_rabi(frq='all')
a.plot_rabi_fft()

'''