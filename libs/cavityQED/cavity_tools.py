
import h5py
from analysis.lib.tools import toolbox
import pylab as plt
from analysis.lib import fitting
from analysis.lib.fitting import fit,esr, common
import matplotlib

from matplotlib import rc, cm
from scipy import interpolate
import os
import numpy as np

from  matplotlib import animation

reload(toolbox)


def get_files (newer_than, older_than, contains, folder=None):
	a, b, folder_names = toolbox.latest_data (contains=contains, newer_than=newer_than, older_than=older_than, return_all=True, folder=folder)
	times = []
	for f in folder_names:
		times.append (int(f[9:11])*3600 + int(f[11:13])*60 + int(f[13:15]))
	return folder_names, times

def get_files_in_folder (contains, folder):
	folder_names = [f for f in os.listdir(folder) if contains in f]
	times = []
	for f in folder_names:
		times.append (int(f[0:2])*3600 + int(f[2:4])*60 + int(f[4:6]))
	times = np.array(times)
	times = times-times[0]
	return folder_names, times


def load_data (folder, timestamp, scan_type):

	all_folders = [f for f in os.listdir(folder) if timestamp in f]
	curr_fold = os.path.join(folder, all_folders[0])
	all_files =[ f for f in os.listdir(curr_fold) if os.path.isfile(os.path.join(curr_fold,f)) ]
	file_name = [f for f in all_files if '.hdf5' in f]

	if (scan_type =='piezo'):
		grp_name = '/piezo_scan'
		x_name = 'piezo_voltage'
	elif (scan_type =='lr_scan'):
		grp_name = '/lr_scan'
		x_name = 'frequency_GHz'
	
	f = h5py.File(os.path.join(curr_fold, file_name[0]),'r')
	ls_grp = f[grp_name]
	V = ls_grp[x_name].value
	PD_signal = ls_grp['PD_signal'].value
	return V, PD_signal

def load_calibration (timestamp=''):
	if (timestamp==''):
		folder_name = toolbox.latest_data (contains='laser_calib')
	else:
		folder_name = toolbox.latest_data (contains=timestamp)
	
	all_files = [f for f in os.listdir(folder_name) if timestamp in f]  
	file_name = [f for f in all_files if '.hdf5' in f]
	
	f = h5py.File(os.path.join(folder_name, file_name[0]),'r')
	ls_grp = f['/calibration']
	V = ls_grp['V'].value
	freq = ls_grp['freq_GHz'].value
	return V, freq


def fit_calibration(V, freq):
	guess_b = (freq[-1]-freq[0])/(V[-1]-V[0])
	guess_a = freq[0]
	print 'Initial guess: ', guess_a, guess_b

	a = fit.Parameter(guess_a, 'a')
	b = fit.Parameter(guess_b, 'b')
	
	p0 = [a, b]
	fitfunc_str = ''

	def fitfunc(x):
		return a()+b()*x


	fit_result = fit.fit1d(V,freq, None, p0=p0, fitfunc=fitfunc, fixed=[],
        	do_print=False, ret=True)
	a_fit = fit_result['params_dict']['a']
	b_fit = fit_result['params_dict']['b']
	print 'a= ',a_fit
	print 'b=',b_fit
	return a_fit, b_fit

def combine_piezoscans (timestamp_array):

	voltages = np.array([])
	PD_signal = np.array([])

	for k in timestamp_array:
		V, Y = load_data (k)
		voltages = np.hstack ([voltages, V])
		PD_signal = np.hstack ([PD_signal, Y])

	f = plt.figure(figsize=(12,6))
	ax = f.add_subplot (1,1,1)
	ax.plot (voltages, PD_signal, 'b', linewidth=2)
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	ax.set_xlabel ('piezo voltage [V]', fontsize= 16)
	ax.set_ylabel ('transmission [a.u.]', fontsize=16)
	plt.show()

	f = plt.figure(figsize=(12,6))
	ax = f.add_subplot (1,1,1)
	ax.plot (voltages[1300:1800], PD_signal[1300:1800], 'b', linewidth=2)
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	ax.set_xlabel ('piezo voltage [V]', fontsize= 16)
	ax.set_ylabel ('transmission [a.u.]', fontsize=16)
	plt.show()

def combine_piezoscans_2D (folder, folder_array, time_array = None, V_lims=None, do_save=False):

	voltages = np.array([])
	PD_signal = np.array([])
	i = 0
	for k in folder_array:
		V, Y = load_data (folder=folder, timestamp = k, scan_type='piezo')
		#Y = Y/float(max(Y))
		if (i==0):
			PD_signal = Y
		else:
			PD_signal = np.vstack ([PD_signal, Y])
		i = i + 1

	if time_array == None:
		time_array = np.arange(i)
	A, B = np.meshgrid (V, time_array)

	f = plt.figure(figsize=(18,6))
	ax = f.add_subplot (1,1,1)
	ax.pcolor (A, B, PD_signal)
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	if V_lims:
		ax.set_xlim(V_lims)
	ax.set_xlabel ('piezo voltage [V]', fontsize= 16)
	ax.set_ylabel ('time [s]', fontsize=16)
	if do_save:
		f.savefig (os.path.join (folder, 'piezo_scan_2Dplot.png'))
	plt.show()


def combine_piezoscans_1D (folder, folder_array, time_array, threshold = None, xlimit = None):
	#matplotlib.rc ('xtick', labelsize=20)
	#matplotlib.rc ('ytick', labelsize=20)
	colori = cm.gist_earth(np.linspace(0,0.75, len(folder_array)))
	f = plt.figure(figsize=(8, 40))
	ax = f.add_subplot(1,1,1)
	i = 0
	for k in folder_array:
		#print k
		V, Y = load_data (folder=folder, timestamp = k, scan_type='piezo')
		if threshold:
			ind  = np.where (Y<threshold)
			Y[ind] = 0
		ax.plot (V[:], 0.02*Y[:]+(1/10.)*time_array[i], '.', color=colori[i])
		ax.plot (V[:], 0.02*Y[:]+(1/10.)*time_array[i], color=colori[i])
		i= i+1
	ax.set_xlabel ('piezo voltage [V]', fontsize=20)
	ax.set_ylabel ('time [min]', fontsize=20)
	if xlimit:
		ax.set_xlim(xlimit)
	#f.savefig ('D:/measuring/low_temp_cavity.png', dpi=300)
	plt.show()


def combine_piezoscans_2D_interpolate (folder, folder_array, min_V, max_V, time_array = None, V_lims=None, do_save=False):

	voltages = np.array([])
	PD_signal = np.array([])
	i = 0
	for k in folder_array:
		V, Y = load_data (folder=folder, timestamp = k, scan_type='piezo')
		#Y = Y/float(max(Y))
		if (i==0):
			PD_signal = Y
		else:
			PD_signal = np.vstack ([PD_signal, Y])
		i = i + 1

	if time_array == None:
		time_array = np.arange(i)
	A, B = np.meshgrid (V, time_array)

	f = plt.figure(figsize=(18,6))
	ax = f.add_subplot (1,1,1)
	ax.pcolor (A, B, PD_signal)
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	for tick in ax.yaxis.get_major_ticks():
	    tick.label.set_fontsize(16) 
	if V_lims:
		ax.set_xlim(V_lims)
	ax.set_xlabel ('piezo voltage [V]', fontsize= 16)
	ax.set_ylabel ('time [s]', fontsize=16)
	if do_save:
		f.savefig (os.path.join (folder, 'piezo_scan_2Dplot.png'))
	plt.show()

def track_single_peak (folder, folder_array, time_array, conv_factor = 1, threshold = None):

	center = []
	std = []

	for k in folder_array:
		V, Y = load_data (folder=folder, timestamp = k, scan_type='piezo')
		if threshold:
			ind  = np.where (Y<threshold)
			Y[ind] = 0

		Y = Y/float(np.sum(Y))
		media = np.sum(Y*V)
		center.append(media)
		std.append ((np.sum(Y*(V-media)**2))**0.5)

	center = np.asarray (center[:390])
	std = np.asarray (std[:390])
	time_array = time_array[:390]
	f = plt.figure(figsize=(20,5))
	plt.plot (time_array, conv_factor*center, 'ob')
	#plt.fill_between (time_array, conv_factor*(center-std), conv_factor*(center+std), color='b', alpha=0.2)
	plt.errorbar (x=time_array, y=conv_factor*center, yerr = conv_factor*(std), color='RoyalBlue')
	plt.xlabel('time [seconds]', fontsize =18)
	plt.ylabel('resonance (cav length) [nm]', fontsize =18)

	plt.show()
	f = plt.figure(figsize=(20,5))
	#plt.plot (time_array, center, 'ob')
	plt.plot (time_array, conv_factor*std, 'ob')
	plt.plot (time_array, conv_factor*std, 'r')
	plt.xlabel('time [seconds]', fontsize =18)
	plt.ylabel('st.dev. (cav length) [nm]', fontsize =18)
	plt.show()

	x = (1/(time_array[2]-time_array[1]))*(np.arange(len(std))-len(std)/2)
	a = np.abs(np.fft.fftshift(np.fft.fft(std)))**2
	plt.semilogy (x[len(std)/2:],a[len(std)/2:])
	plt.show()

	#plt.plot (time_array, std)
	#plt.show()

def process_2D_scan (folder, timestamp):

	all_folders = [f for f in os.listdir(folder) if timestamp in f]
	curr_fold = os.path.join(folder, all_folders[0])
	print 'Analyzing folder: ', curr_fold
	file_name = [f for f in os.listdir(curr_fold) if '.hdf5' in f]
	print os.path.join(curr_fold, file_name[0])
	idx = 5
	done = False
	datafile = h5py.File(os.path.join(curr_fold, file_name[0]),'r')
	nr_points = 300
	plt.figure(figsize = (12, 8))

	scan_data = np.zeros((10, nr_points))
	piezo_voltage = np.zeros(10)

	while idx<10:
		print idx
		curr_grp = datafile[str(idx)]
		f = curr_grp['frq'].value
		y = curr_grp['sgnl'].value
		piezo_voltage[idx] = curr_grp.attrs['pzV']
		#clean-up wavemeter errors 
		ind = np.where (abs(f)>5000)
		f = np.delete(f, ind)
		y = np.delete(y, ind)
		#if (idx==0):
		#	frq = np.linspace (-1200, 200, nr_points)
		#interp_funct = interpolate.interp1d(f, y)
		#y_new = interp_funct (frq)
		#scan_data [idx, :] = y_new
		#plt.plot (frq, y_new+0.1*idx, label = str(piezo_voltage[idx])+'V')
		plt.plot (f,y+0.1*idx, label = str(piezo_voltage[idx])+'V')
		idx = idx + 1
		#except:
		#	done = True
	datafile.close()
	plt.xlabel (' frq [GHz]', fontsize = 15)
	plt.legend()
	plt.show()

	#FF, PZ = meshgrid (frq, piezo_voltage)
	#plt.figure (figsize = (12, int(idx/5.)))
	#plt.pcolor (FF, PZ, scan_data)
	#plt.xlabel (' frq [GHz]', fontsize = 15)
	#plt.ylabel ('piezo [V]', fontsize = 15)
	#plt.show()


