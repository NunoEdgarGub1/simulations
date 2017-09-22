
#
#Convert text files from Berry group into python arrays to be used in magnetometry experiments
#

import numpy as np

def extract_values(k_array=[], verbose = False, do_save = True):
	F=2
	G=5
	folder = r'D:/measuring/analysis/scripts/magnetometry/swarm_optimization/phases_G'+str(G)+'_F'+str(F)+'/'
	start_pt_0 = 7
	for k in k_array:
		print 'Analysing... K=', k
		f_name = 'K'+str(k)+'_new.txt'
		n_els = G*(k+1)+F*k*(k+1)/2

		delta_pos = 3
		start_pt_1 = start_pt_0 + n_els + delta_pos

		with open (folder+f_name) as f:
			content = f.readlines()
		f.close()

		doc = content[0]+content[1]+content[2]+content[3]+content[4]+content[5]
		incr_0 = np.zeros(n_els)
		incr_1 = np.zeros(n_els)
		tau = []
		for i in np.arange(n_els):
			d1 = content[start_pt_0+i].split()
			d0 = content[start_pt_1+i].split()

			tau.append(d0[1])
			incr_0[i] = d0[0]
			incr_1[i] = d1[0]

		if do_save:
			out_f_name = 'swarm_opt_G='+str(G)+'_F='+str(F)+'_K='+str(k)+'.npz'
			np.savez (folder+out_f_name, u0=incr_0, u1=incr_1, tau=tau, doc=doc)
		if verbose:
			print '##### K = ', k
			print 'Msmnt result - 0, phases:', incr_0*360/np.pi
			print 'Msmnt result - 1, phases:', incr_1*360/np.pi
			print 'sensing times: ', tau
			print

extract_values (k_array = np.arange(12)+1)

