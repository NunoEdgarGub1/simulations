
import numpy as np
import random
from matplotlib import rc, cm
import os, sys
import h5py
import logging, time

from matplotlib import pyplot as plt
from scipy import signal
import lmfit
#import winsound

from analysis_simulations.libs.adaptive_sensing import adaptive_tracking_overhead as track_lib
from analysis_simulations.libs.adaptive_sensing import tracking_sequence as trackSeq

reload (track_lib)
#reload (track_libOH)
reload (trackSeq)

s = trackSeq.SequenceStatistics()
simulate = 0
analyze = 0
compare_vs_DB = 0
compare_vs_OH = 0
compare_thr = 0
plot_ratio = 0
test_B_statistics = 0
folder = r'D:/Research/WorkData/adptv_tracking_sim/'

def compare (stamp, sweep_par='dB', fom_thr=0, fill_gaps=False, do_save = False, filename = None):

	d_array = [stamp]
	a = trackSeq.Analyze(folder = folder)
	if (sweep_par == 'dB'):
		a.do_analysis(name='analysis', stamps_list=d_array, sweep_par='dB')
		a.plot_analysis (x_scale=1e-6, sweep_par_label = 'dB [MHz*sqrt(Hz)]')
	elif (sweep_par=='OH'):
		a.do_analysis(name='analysis', stamps_list=d_array, sweep_par='overhead')
		a.plot_analysis (x_scale=1e6, sweep_par_label = 'overhead [us]')

	if do_save:
		if filename:
			fname = filename
		else:
			fname = '_'+stamp
		a.save_analysis(file_name=fname)


def simulate (track, G, F, nr_reps,sweep_par = 'dB', fix_par = 1e-9, title = '', fid = 1., sweep_list = None, hack = False, setN=False):

	s = trackSeq.SequenceStatistics()
	if track:
		protocol = 'fully_adaptive'
	else:
		protocol = 'fixed_time'

	proceed = True
	if (sweep_par == 'dB'):
		if (sweep_list==None):
			sweep_list = np.linspace (0, 20, 13)
			sweep_list[0] = 0.5
		OH = fix_par
	elif (sweep_par == 'OH'):
		if (sweep_list==None):
			sweep_list = 1e-6*np.linspace (0, 300, 25)
		dB = fix_par
	else:
		print "Unknown sweep parameter!!"
		proceed = False

	if proceed:
		i = 0
		for p in sweep_list:
			if (sweep_par=='dB'):
				dB = p
			else:
				OH = p

			dB = int(dB*10)/10.
			time_interval = 500.e-3*(OH/100e-6)**1+5.e-3
			time_interval = 1e-3
			#time_interval =1000e-3
			print time_interval*1e3, ' msec'#(5./(N+0.))*1e-3
			print "####################### dB: ", dB
			s.set_sim_params (reps=nr_reps, time_interval = time_interval, overhead = OH)
			s.set_msmnt_params (G = 5, F = F, dB = dB*1e6, N=4, tau0=20e-9, T2 = 100e-6, fid0=fid, fid1=0.01)
			s.G_adptv = G
			s.nr_sensing_reps = 0
			if setN:
				s.set_msmnt_params (G = 5, F = F, dB = dB*1e6, N=setN[i], tau0=20e-9, T2 = 100e-6, fid0=fid, fid1=0.01)
			else:
				if hack:
					s.automatic_set_NplusONE(track=track, dB = dB*1e6)
					title = '_N+1_'
				else:
					s.automatic_set_N(track=track, dB = dB*1e6)
			if track:
				s.calculate_thresholds_std()
				s.fom_threshold = s.fom_array[2]
			s.generate_simulation(track = track, do_save = False, description = '', 
								title = title,
								do_plot = True, protocol = protocol,
								do_debug = False)
			i = i + 1

def simulate_alpha (G, F, nr_reps, dB, OH, N, title = 'alpha=', fid = 1.):

	s = trackSeq.SequenceStatistics()
	protocol = 'fully_adaptive'
	proceed = True

	alpha = [0.2, 0.3, 0.5, 0.75, 1.]
	for a in alpha:
		print "----------- Alpha:  ", a

		dB = int(dB*10)/10.
		time_interval = 500.e-3*(OH/100e-6)**1+5.e-3
		print time_interval*1e3, ' msec'#(5./(N+0.))*1e-3
		print "####################### dB: ", dB
		s.set_sim_params (reps=nr_reps, time_interval = time_interval, overhead = OH)
		s.set_msmnt_params (G = 5, F = F, dB = dB*1e6, N=4, tau0=20e-9, T2 = 100e-6, fid0=fid, fid1=0.01)
		s.G_adptv = G
		s.nr_sensing_reps = 0
		#s.automatic_set_N(track=True, dB = dB*1e6)
		s.calculate_thresholds_std (alpha = a)
		s.alpha = a
		s.fom_threshold = s.fom_array[2]
		s.generate_simulation(track = True, do_save = True, description = '', 
							title = title+str(a),
							do_plot = False, protocol = protocol,
							do_debug = False)



def analyze (stamp, nr_reps = 3):

	s.load (folder = folder, stamp = stamp)
	s.analysis (reps = 'all')
	s.plot_wf_estim_std_vs_rep()
	
	print "Worse case:"
	ind = np.argmax(s.wf_estim_std)
	s.plot_simulation (rep = ind)
	print "Best case:"
	ind = np.argmin(s.wf_estim_std)
	s.plot_simulation (rep = ind, color='RoyalBlue', do_save=False)

	for i in np.arange(nr_reps):
		s.plot_simulation (rep=int(s.reps*np.random.rand()))

	s.print_parameters()
	s.hist_sensing_times()
	

def beep():
	Freq = 440 # Set Frequency To 2500 Hertz
	Dur = 1000 # Set Duration To 1000 ms == 1 second
	winsound.Beep(Freq,Dur)


L = [10.]
#N = [11, 9, 8, 8, 7,7,7,7, 6,6,6,6,6,6,6,6, 5,5,5,5,5]
N = [9]

#L = [25.]
#N = [6]
#print N
#L = [2.]
#N = [9]
#compare (stamp = 'Tracking_fid100_G1F0',  sweep_par = 'dB', fom_thr = 0, fill_gaps = False, do_save = True, filename = None)
#compare (stamp = 'G5F0_OH100',  sweep_par = 'dB', fom_thr = 0, fill_gaps = False, do_save = True, filename = None)
analyze (stamp = '20170228_164657', nr_reps=3)

'''
for nn in [12,11,10,9,8,7,6,5]:
	try:
		print 'Simulating... ', nn
		simulate (track=True, sweep_par = 'dB', fix_par=10.e-9, G=1, F=0, setN = [nn], nr_reps = 250, fid = 1.00, title = 'G1F0', hack = False, sweep_list = L)
	except:
		print nn, ' did not work!'
'''

#simulate (track=True, sweep_par = 'dB', fix_par=10.e-6, G=1, F=0, setN = N, nr_reps = 1, fid = 1.00, title = 'G1F0', hack = False, sweep_list = L)


'''
try:
	simulate (track=True, sweep_par = 'dB', fix_par=100.e-6, G=1, F=0, setN = [9], nr_reps = 250, fid = 1.00, title = 'G1F0_a025', hack = False, sweep_list = L)
except:
	print "Failed!!!"
'''
#simulate_alpha (G=1, F=0, nr_reps=250, dB=2., OH=100.e-6, title = 'alpha=', fid = 1., N=9)
#beep()
