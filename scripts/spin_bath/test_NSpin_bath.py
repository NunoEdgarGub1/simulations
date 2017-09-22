

from simulations.libs.spin import diluted_NSpin_bath as NSpin 
reload (NSpin)


exp2 = NSpin.CentralSpinExperiment()
exp2.set_experiment(nr_spins=4)

for i in range(50):
	exp2.Ramsey(tau = 1e-6, phi=0)

exp2.plot_bath_evolution()

#exp2.Nruns(nr_steps=10, phi_0 = 0, tau_0=1e-6, do_plot=True)
