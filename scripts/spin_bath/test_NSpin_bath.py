

from simulations.libs.spin import diluted_NSpin_bath_py3 as NSpin 
from importlib import reload

reload (NSpin)


exp2 = NSpin.CentralSpinExperiment()
exp2.set_experiment(nr_lattice_sites=False, nr_nuclear_spins=7, concentration = 0.015)
exp2.print_nuclear_spins()

for i in range(25):
	exp2.Ramsey(tau = 1e-6, phi=0)
	#print (exp2._evol_dict[str(i)])

exp2.plot_bath_evolution()

#exp2.Nruns(nr_steps=10, phi_0 = 0, tau_0=1e-6, do_plot=True)
#
# Things to do:
# 1) program a module that creates a fixed number of nuclear spins compatible 
#		with the chosen concentration
