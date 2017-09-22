
#
#Convert text files from Berry group into python arrays to be used in magnetometry experiments
#

import numpy as np
import re, os

def remove_commas (array):
	new_array = []
	for s in array:
		new_s = re.sub('[,]', '', s)
		new_array.append(new_s)
	return new_array



def extract_values(fid, verbose = False, do_save = True):

	fileName = r'D:/measuring/analysis/scripts/magnetometry/swarm_optimization/fid'+str(fid)+'_hossein.txt'
	save_folder = r'D:/measuring/analysis/scripts/magnetometry/swarm_optimization/incr_fid'+str(fid)+'_G5/'
	start_pt_0 = 7

	with open (fileName) as f:
		content = f.readlines()
	f.close()

	doc = ''
	for i in np.arange(15):
		doc = doc+content[i]

	go_on = True
	curr_line = 15
	start_retrieve = False
	K = -1
	while go_on:
		if not(start_retrieve):
			a = content[curr_line].split()
			if a[0] == 'K,':
				a = remove_commas (a)
				K = int(a[4])
				G = int(a[5])
				F = int(a[6])
				N = int(a[7])

				print 'New parameters: ', K, G, F, N
				#retrieve data and position
				a_mK = content[curr_line+1].split()
				if (a_mK[0] == 'NV_H_mCappa'):
					H_mK = a_mK[1]
				else:
					print 'Error! H_mK not in correct position!'
					print K, G, F, N
					print a_mK
					print 'curr_curr_line', curr_line
					go_on = False

				a_pso = content[curr_line+2].split()
				if (a_pso[0] == 'NV_H_PSO'):
					H_PSO = a_pso[1]
				else:
					print 'Error! H_PSO not in correct position!'
					print K, G, F, N
					print a_pso
					print 'curr_curr_line', curr_line
					go_on = False

				a_incr = content[curr_line+3].split()
				if not(a_incr[0]=='increments'):
					print 'Error! string INCREMENTS not in correct position!'
					print K, G, F, N
					print 'INCR: ', a_incr
					print 'mK: ', a_mK
					print 'values: ', a
					print 'curr_curr_line', curr_line
					go_on = False
				else:
					start_retrieve = True
					curr_line = curr_line+3
			curr_line = curr_line + 1

		if start_retrieve:
			n_els = G*(K+1)+F*K*(K+1)/2
			start_pt_1 = curr_line
			start_pt_0 = curr_line+n_els+1
			a = content[start_pt_1].split()
			first_incr_1 = float(a[0])
			a = content[start_pt_0].split()
			first_incr_0 = float(a[0])
			
			if not(first_incr_1==0.):
				print 'Error!! First increment for u=1 is not zero!!'
				print first_incr_1
				go_on = False
			elif not(first_incr_0==0.):
				print 'Error!! First increment for u=0 is not zero!!'
				print first_incr_0
				go_on = False				
			else:

				try:
					tau_0 = []
					tau_1 = []
					incr_0 = np.zeros(n_els)
					incr_1 = np.zeros(n_els)

					for i in np.arange(n_els):
						d1 = content[start_pt_1+i].split()
						d0 = content[start_pt_0+i].split()

						tau_0.append(int(d0[1]))
						tau_1.append(int(d1[1]))
						incr_0[i] = d0[0]
						incr_1[i] = d1[0]

					if verbose:
						print 'EXRACTED QUANTITIES: '
						print 'tau_0: ', tau_0
						print 'incr_0: ', incr_0
						print 'tau_1: ', tau_1
						print 'incr_1: ', incr_1
					curr_line = start_pt_0+n_els
					start_retrieve = False

					if do_save:
						save_name = 'incr_fid'+str(fid)+'_G'+str(G)+'F'+str(F)+'_K='+str(K)+'.npz'
						save_path = os.path.join (save_folder, save_name)
						np.savez(save_path, tau_0=tau_0, tau_1 = tau_1, u0 = incr_0, u1 = incr_1, H_PSO=H_PSO, H_mK = H_mK)
				except:
					print 'File parsing failed...'
			start_retrieve = False

		if ((K > 13) and (F == 5)):
			go_on = False

extract_values (fid=75)
