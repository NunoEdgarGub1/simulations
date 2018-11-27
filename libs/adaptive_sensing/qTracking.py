import numpy as np
import random
from matplotlib import rc, cm
import matplotlib
import os, sys
import h5py
import cmath
import logging, time, timeit
import imageio
from importlib import reload
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#sys.path.append ('/Users/dalescerri/Documents/GitHub')

import matplotlib.pyplot as plt
from scipy import signal
from simulations.libs.math import statistics as stat
from simulations.libs.adaptive_sensing import adaptive_tracking as adptvTrack
from simulations.libs.spin import nuclear_spin_bath as NSpin
from simulations.libs.math import peakdetect as pd

reload (NSpin)
reload (adptvTrack)
reload (stat)

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)


class TimeSequenceQ (adptvTrack.TimeSequence_overhead):

    def __init__ (self, time_interval, overhead, folder):
        self.time_interval = time_interval
        self._B_dict = {}
        self.kappa = None
        self.skip=False
        self.curr_fB_idx = 0
        self.OH = overhead
        self.set_fB = []
        self.est_fB = []
        self.curr_fB = 0
        self.plot_idx = 0
        self.step = 0
        self.phase_cappellaro = 0
        self.k_list = [0]
        self.m_res = 0
        self.multi_peak = False
        self.T2starlist = []
        self.timelist = []
        self.widthratlist = []
        self.Hvar = 0
        self.curr_rep = 0
        self.mse_lst = []
        self.norm_lst = []
        self.FWHM_lst = []
        self.Hvarlist = []
        self.qmax = []
        self.cmax = []
        self._flip_prob = 0
        self._save_plots = False
        self.outcomes_list = []
        self.phase_list = []
        self.tau_list = []
		
        self.BayesianMean = [0]
        self.BayesianMax = [0]
        self.QuantumMean = [0]
        self.QuantumMax = [0]
        self.k_list = [0]

        self._curr_res = 1
        self.add_phase = 0
        self.fv=False

        self.log = logging.getLogger ('qTrack')
        logging.basicConfig (level = logging.INFO)

        # The "called modules" is  a list that tracks which functions have been used
        # so that they are saved in the output hdf5 file.
        # When you look back into old data, it's nice to have some code recorded
        # to make sure you know how the data was generated 
        # (especially if you tried out diffeent things)
        self._called_modules = ['ramsey', 'bayesian_update', 'single_estimation_step']
        self.folder = folder

    def set_plot_settings (self, do_show = False, do_save = False, do_save_analysis = False):
        self._save_plots = do_show
        self._show_plots = do_show
        self._save_analysis = do_save_analysis

    def set_log_level (self, value):
        self.log.setLevel(value)

    def set_bath_validity_conditions (self, A=None, sparse=None):
        self._A_thr = A
        self._sparse_thr = sparse

    def _check_bath_validity (self):
        self.log.debug ("large A ctr: "+str(self.nbath.close_cntr)
                +"  sparse? "+str(self.nbath.sparse_distribution))
        condition = ((self.nbath.close_cntr < 1) and (not(self.nbath.sparse_distribution)))
        self.log.debug ("Valid bath? "+str(condition))
        return condition

    def generate_spin_bath (self, eng_bath_cluster, nr_spins, concentration, store_evol_dict = False):
        '''
        Generates a nuclear spin bath

        eng_bath_cluster     [array]:    for unstructured bath, set to np.zeros(nr_spins)
        nr_spins    [int]:      number of nuclear spins in the bath
        concentration [float]: concentration of spins in the lattice
        store_evol_dict [bool]: store all steps of bath evolution in a dictionary?
        
        '''
        
        valid_bath = False
        self.nbath = NSpin.FullBathDynamics()
        self.nbath.set_thresholds (A=self._A_thr, sparse=self._sparse_thr)
        
        while not(valid_bath):

            self.nbath.generate (eng_bath_cluster = eng_bath_cluster, nr_spins=nr_spins, 
                concentration = concentration, do_plot = False, eng_bath=False)
            valid_bath = self._check_bath_validity()

        if not(store_evol_dict):
            self.nbath.deactivate_evol_dict()

    def load_bath (self, nBath):
        if isinstance (nBath, NSpin.FullBathDynamics):
            self.nbath = nBath
            return True
        else:
            self.log.error ("Object is not a nuclear spin bath.")
            return False

    def return_bath (self):
        return self.nbath

    def reset (self):
        self.nbath.reset_bath_unpolarized()
        self.k_list = [0]
        self.T2star = self.nbath.T2h
        self.over_op = self.nbath._overhauser_op()

        self.T2starlist = []
        self.timelist = []
        self.T2starlist.append(self.T2star)
        self.T2_est = self.nbath.T2est
        self.timelist.append(0)
        self.BayesianMean = [0]
        self.BayesianMax = [0]
        self.QuantumMean = [0]
        self.QuantumMax = [0]
        self.k_list = [0]
        self._latest_outcome = None

        self.k=0
        self._curr_res = 1
        self.add_phase = 0
        self.step = 0

        self.outcomes_list = []
        self.phase_list = []
        self.tau_list = []

    def set_flip_prob (self, value):
        self._flip_prob = value

    def init_a_priory (self):
        pass

    def initialize (self, T2star=None, do_plot=False):
        if (T2star == None):
            T2star = self.tau0

        self._dfB0 = 1/(4*np.sqrt(2)*np.pi*T2star)

        p = np.exp(-0.5*(self.beta/self._dfB0)**2) 
        p = p/(np.sum(p))

        self.norm = 1
        self.p_k = np.fft.ifftshift(np.abs(np.fft.ifft(p, self.discr_steps))**2)
        self.renorm_p_k()
        self._initial_fwhm = self.FWHM()
        self._curr_res = 1
        self.add_phase = 0

        #if do_plot:
        #    self.plot_hyperfine_distr(tau=1e-3, theta=0)
        
    def fitting(self, p, paz, az, T2track, T2est):
        '''
        Fits the classical distribution to the quantum one
        
        Input:
        p        [array]     : classical probability array
        paz      [array]     : quantum probability array
        az       [array]     : bath frequencies
        T2track  [Boolean]   : if True, colvolves the current classical distribution with a Gaussian (spin bath diffusion)
        T2est    [float]     : T2 estimate for spin bath diffusion 
        
        '''
        
        x = self.beta*1e-3
        y = p
        f = interp1d(x, y, kind='cubic')

        def func(x, a):
            y=a*f(x)
            return y

        popt, pcov = curve_fit(func, az.real, paz.real, p0=1)
        fit = func(x, *popt)
        
        norm = popt[0]
        error = np.absolute(pcov[0][0])**0.5
        
        return norm, error
        

    def plot_hyperfine_distr(self, tau, theta, T2track = False, T2est = 1e-3):
        
        #T2echo = self.nbath.T2echo
        T2est = 1e-3#self.nbath.T2echo
        
        p, m = self.return_p_fB(T2_track = T2track, T2_est = T2est)
        p_az, az = self.nbath.get_histogram_Az(nbins = 20)
        az2, p_az2 = self.nbath.get_probability_density()
    
        #self.norm, self.error = self.fitting(p = p, paz = p_az2,
		#       az = az2, T2track = T2track, T2est = T2est)
        #self.norm_lst.append(self.norm)
        #self.mse_lst.append(self.error)
        if self.step==0:
            self.qmax.append(0)
        else:
            self.qmax.append(az2[np.argmax(p_az2)])
        f = ((1/(self.tau0))*np.angle(self.p_k[self.points-1])*1e-6)

        self.cmax.append(self.beta[np.argmax(p)]*1e-3)
        self.FWHM_lst.append(self.FWHM())		
        
        fig = plt.figure(figsize = (12,6))
        p0 = 0.5-0.5*np.cos(2*np.pi*self.beta*tau+theta)
        p1 = 0.5+0.5*np.cos(2*np.pi*self.beta*tau+theta)
        #plt.fill_between (self.beta*1e-3, 0, max(p*self.norm)*p0/max(p0), color='magenta', alpha = 0.1)
        #plt.fill_between (self.beta*1e-3, 0, max(p*self.norm)*p1/max(p1), color='cyan', alpha = 0.1)
        plt.fill_between (self.beta*1e-3, 0, p0/max(p0), color='magenta', alpha = 0.1)
        plt.fill_between (self.beta*1e-3, 0, p1/max(p1), color='cyan', alpha = 0.1)
        tarr = np.linspace(min(self.beta)*1e-3,max(self.beta)*1e-3,1000)
        T2star = 5*(self.nbath._op_sd(self.over_op[2]).real)**-1
        T2inv = T2star**-1 *1e-3
        plt.plot (az2, p_az2/max(p_az2) , '^', color='k', label = 'spin-bath')
        plt.plot (az2, p_az2/max(p_az2) , ':', color='k')

        outcome = self.outcomes_list[-1]
        curr_t2star = int(self.T2starlist[-1]*1e6)
        curr_tau = tau*1e6
        curr_phase = int(theta*180/3.14)

#        if free_evo:
#            T2gauss = np.exp(-(self.beta*1e-3 * (.5*T2echo*1e3))**2)
#            T2exp = 2*np.pi*np.exp(((self.beta*1e-3)**2 + (.5*T2echo*1e3)**-2)**-1)
#            p = np.convolve(p,T2gauss,'same')
        #plt.plot (self.beta*1e-3, p*self.norm , color='green', linewidth = 2, label = 'classical')
        plt.plot (self.beta*1e-3, p/max(p) , color='green', linewidth = 2, label = 'classical')
        plt.title ("tau = " +str(curr_tau)+ " us, phase = "+str(curr_phase)+" deg --> outcome: "+str(outcome)+" -- T2* = "+str(curr_t2star)+ " us -- free evo: "+str(self.fv) , fontsize = 18)
        plt.xlabel (' hyperfine (kHz)', fontsize=18)
        fwhm = self.FWHM()
        plt.xlim((max(-1000, m*1e-3-15*fwhm), min (1000, m*1e-3+15*fwhm)))
        plt.xlim(-1000, 1000)
        if self._save_plots:
            plt.savefig(os.path.join(self.folder+'/', 'rep_%.04d_%.04d.png'%(self.curr_rep, self.step)))
        if self._show_plots:
            plt.show()
        plt.close("all")
        self.p_az_old = p_az2/max(p_az2)
        self.step+=1
        
    def plot_distr(self):
        p, m = self.return_p_fB()
        p_az, az = self.nbath.get_histogram_Az(nbins = 20)
        az2, p_az2 = self.nbath.get_probability_density()
        f = ((1/(self.tau0))*np.angle(self.p_k[self.points-1])*1e-6)
        
        fig = plt.figure(figsize = (12,6))
        plt.plot (az2, p_az2/max(p_az2), '^', color='k', label = 'spin-bath')
        plt.plot (az2, p_az2/max(p_az2), ':', color='k')
        plt.plot (az2, self.p_az_old , 'o', color='r')
        plt.plot (az2, self.p_az_old , ':', color='r')
        plt.xlabel (' hyperfine (kHz)', fontsize=18)
        T2gauss = np.exp(-(self.beta*1e-3 * self.nbath.T2echo)**2)
        p_broad = np.convolve(p,T2gauss,'same')
        plt.xlim(min(az2), max(az2))
        plt.legend()
        plt.ticklabel_format(useOffset=False)
        plt.show()
        

    def MSE (self):
        p, m = self.return_p_fB()
        az2, p_az2 = self.nbath.get_probability_density()
        self.az=az2
        beta_mse_ind = []
        for freq in az2:
            beta_mse_ind.append(min(range(len(self.beta)), key=lambda j: abs(self.beta[j] - freq*1e3)))   
            
        return((((p[beta_mse_ind])- p_az2)**2).mean().real), beta_mse_ind
    
    def FWHM(self):

        p, m = self.return_p_fB()
        halfmax = max(p) / 2
        betamax = p.argmax()
        betaint_right = (np.abs(p[betamax:-1] - halfmax)).argmin()
        betaint_left = (np.abs(p[0:betamax] - halfmax)).argmin()

        FWHM = (self.beta[betaint_right + betamax] - self.beta[betaint_left])*1e-3
        
        return FWHM    
				
    def return_std_old (self):

        '''
        Returns:
        std_H 		standard deviation for the frequency f_B (calculated from p_{-1}).
        fom 		figure of merit	
        '''

        self.renorm_p_k()
        self.Hvar = (2*np.pi*np.abs(self.p_k[self.points-1]))**(-2)-1
        self.Hvarlist.append(self.Hvar)
        std_H = ((abs(cmath.sqrt(self.Hvar)))/(2*np.pi*self.tau0))
        self.log.debug ('Std (Holevo): {0} kHz'.format(std_H*1e-3))
        return  std_H, 0

    def return_std (self):

        '''
        Returns:
        std_H       standard deviation for the frequency f_B (calculated from p_{-1}).
        fom         figure of merit 
        '''

        p, m = self.return_p_fB()
        v = np.sum (p*self.beta**2)-m**2
        std = v**0.5
        self.log.debug ("Std: {0} kHz".format(std*1e-3))
        return  std, 0

    def reset_called_modules(self):
        self._called_modules = ['ramsey', 'bayesian_update', 'calc_acc_phase']

    def ramsey_classical (self, t=0., theta=0., T2 = None):

        A = 0.5
        B = -0.5

        if (T2 == None):
            T2 = self.T2starlist

        # we should be updating the value of T2 in the classical Ramsey
        # shouldn't we? Maybe not, since there's not really a "decay"
        # for a single measurement. The decay is the result of adding up
        # multiple measurements

        # pick a random value for f_B based on the probbaility distibution p(f_B)
        #Bp = np.exp(-(t/T2)**2)*np.cos(2*np.pi*fB*t+theta)
        p_fB, m = self.return_p_fB()
        fB = np.random.choice (a = self.beta, p = p_fB)
        Bp = np.cos(2*np.pi*fB*t+theta)
        p0 = (1-A)-B*Bp
        p1 = A+B*Bp
        np.random.seed()
        result = np.random.choice (2, 1, p=[p0, p1])
        self.log.info ("[Classical Ramsey]: curr_fB = {0} kHz -- result: {1}".format(fB*1e-3, result[0]))
        return result[0]

    def ramsey (self, t=0., theta=0., do_plot = False):

        '''
        Ramsey experiment simulation
        Calculates the probabilities p0 and p1 to get 0/1 and draws an outcome with the probabilities

        Input:
        t 		[ns]		sensing time
        theta	[rad]		Ramsey phase

        Returns:
        Ramsey outcome
        '''

        az0, pd0 = np.real(self.nbath.get_probability_density())
        m = self.nbath.Ramsey (tau=t, phi = theta, flip_prob = self._flip_prob)
        az, pd = np.real(self.nbath.get_probability_density())
        self.fliplist = self.nbath.flipArr
		
        pB = self.return_p_fB()[0]

        self.BayesianMax.append(self.beta[(pB.tolist()).index(max(pB))]*1e-3)
        self.QuantumMax.append(az[(pd.tolist()).index(max(pd))])

        BayMean = min(pB, key=lambda x:abs(x-np.mean(pB)))
        QuantMean = min(pd, key=lambda x:abs(x-np.mean(pd)))

        self.BayesianMean.append(self.beta[(pB.tolist()).index(BayMean)]*1e-3)
        self.QuantumMean.append(az[(pd.tolist()).index(QuantMean)])


        if do_plot:
            self.plot_hyperfine_distr(tau=t, theta = theta)

        return m
            
    def freeevo (self, ms_curr=0., tf=0.):

        '''
        Ramsey experiment simulation
        Calculates the probabilities p0 and p1 to get 0/1 and draws an outcome with the probabilities

        Input:
        t 		[ns]		sensing time
        theta	[rad]		Ramsey phase

        Returns:
        Ramsey outcome
        '''

        az0, pd0 = np.real(self.nbath.get_probability_density())
        m = self.nbath.FreeEvo (ms=ms_curr, tau=tf)
        az, pd = np.real(self.nbath.get_probability_density())
        
    def hahn (self, tr, pulses, tauarr):

        #self.nbath.Hahn_Echo_clus (tauarr = tauarr, phi = 0, tr = tr, pulses = pulses, do_compare=False)
        self.nbath.Hahn_Echo_clus (tauarr = tauarr, phi = 0, do_compare=False)

    def find_optimal_k (self, alpha = 1., strategy='int'):
        
        self.alpha = alpha
        width, fom = self.return_std ()

        if (strategy == 'int'):
            k = np.int(np.log(alpha*self.t2star/self.tau0)/np.log(2))
        elif (strategy == 'round'):
            k = np.round(np.log(alpha*self.t2star/self.tau0)/np.log(2))

        if k<0:
            self.log.info ('K IS NEGATIVE {0}'.format(k))
            self.k = 0
            self.log.debug ("Optimal k = {0}".format(k))

        #if (strategy == 't2star_limit'):
        #    if (2**k)*self.tau0 > 1e-3/self.FWHM():
        #        while (2**k)*self.tau0 > 1e-3/self.FWHM() and k>=0:
        #            k = k-1
        #        self.log.warning('Ramsey time exceeded 1/FWHM, reduced measurement time')
        
        return k

    def single_estimation_step (self, k, k_old=0, T2_track=False, tf = 1e-3, adptv_phase = False, pi2_phase = False, cap_method = False, free_evo = False):

        if cap_method:
            t_i = int(2**k)
            ttt = 2**(k_old)
        else:
            t_i = int(2**k)
            ttt = -2**(self.K-k)
			
        m_list = []
        t2_list = []

        activate_cappellaro_phase = 1
		
        if free_evo:
            M=1
		
        else:
            M = int(self.G + self.F*k)

        for m in range(M):
            if free_evo:
                m_res = self._latest_outcome
                ctrl_phase = 0
            else:
                if adptv_phase:
                    ctrl_phase = np.mod(activate_cappellaro_phase*0.5*np.angle (self.p_k[int(ttt+self.points)])
                    +self.add_phase, np.pi)
                else:
                    ctrl_phase = np.pi*m/M

                m_res = self.ramsey (theta=ctrl_phase, t = t_i*self.tau0, do_plot=False)
					
            self._latest_outcome = m_res
            m_list.append(m_res)
            if pi2_phase:
                if (m_res != self._curr_res):
                    self.add_phase = np.mod(self.add_phase + np.pi/2., 2*np.pi)
            self._curr_res = m_res

            if m==0:
                self.bayesian_update (m_n = m_res, phase_n = ctrl_phase, t_n = t_i, T2_track = T2_track, T2_est = self.T2_est, tf = tf*free_evo, do_plot=False)
            else:
                self.bayesian_update (m_n = m_res, phase_n = ctrl_phase, t_n = t_i, T2_track = False, T2_est = self.T2_est, tf = tf*free_evo, do_plot=False)
            
            FWHM = self.FWHM()*1e3

            self.outcomes_list.append(m_res)
            self.phase_list.append(ctrl_phase)
            self.tau_list.append (t_i*self.tau0)
            self.T2starlist.append(.5*(self.nbath._op_sd(self.over_op[2]).real)**-1)
            self._curr_T2star = self.T2starlist[-1]
            self.timelist.append(self.timelist[-1] + t_i*self.tau0)
			
            self.log.debug ("Ramsey estim: {0} / {1}".format (m, M))
            self.log.debug ("Params: tau = {0} us --- phase: {1} -- res: {2}".format(t_i*self.tau0*1e6, int (ctrl_phase*180/3.14), m_res))
            self.log.debug ("Current T2* = {0} us".format(int(self.T2starlist[-1]*1e8)/100.))

            if ((self._show_plots) or (self._save_plots)):
                if m==0:
                    self.plot_hyperfine_distr(tau=t_i*self.tau0, theta = ctrl_phase, 
                        T2track = T2_track, T2est = self.T2_est)
                else:
                    self.plot_hyperfine_distr(tau=t_i*self.tau0, theta = ctrl_phase,
                        T2track = False, T2est = self.T2_est)

        return M


    '''
    def make_gif (self, delete_plots = False):

        for i in arange(self.nr_steps):

        with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
    '''     

class BathNarrowing (TimeSequenceQ):

    def _plot_T2star_list (self):
        
        if self._save_analysis:
            plt.figure(figsize = (8, 5))
            plt.plot (np.array(self.T2starlist)*1e6, linewidth=2, color='royalblue')
            plt.plot (np.array(self.T2starlist)*1e6, 'o', color='royalblue')
            plt.xlabel ('step nr', fontsize=18)
            plt.ylabel ('T2* (us)')
            
            plt.savefig(os.path.join(self.folder+'/', 
                        'rep_%.04d_%.04d.png'%(self.curr_rep,self.step+1)))
            if self._show_plots:
                plt.show()
            plt.close ('all')

    def non_adaptive_k (self, max_nr_steps=50):

        self._called_modules.append('non_adaptive_k')

        try:
            self.t2star = self.T2starlist[-1]
        except:
            self.t2star = 0

        i = 0
        k = self.find_optimal_k ()

        while ((k<=self.K+1) and (i<max_nr_steps)):
            M = self.single_estimation_step (k=k, T2_track=False, adptv_phase = True)
            self.t2star = self.T2starlist[-1]
            i+=M
            k+=1

        self._plot_T2star_list()

    def fully_non_adaptive (self, max_nr_steps=50):

        self._called_modules.append('fully_non_adaptive')

        try:
            self.t2star = self.T2starlist[-1]
        except:
            self.t2star = 0

        i = 0
        k = self.find_optimal_k ()

        while ((k<=self.K+1) and (i<max_nr_steps)):
            M = self.single_estimation_step (k=k, T2_track=False, adptv_phase = False)
            self.t2star = self.T2starlist[-1]
            i+=M
            k+=1

        self._plot_T2star_list()
 
    def adaptive_1step_bon (self, max_nr_steps=50):

        self._called_modules.append('adaptive_1step_bon')
        p_az = self.nbath.get_probability_density()[1]
        sparsity = 0
        print('sparsity',sparsity,max(p_az))

        try:
            self.t2star = self.T2starlist[-1]
        except:
            self.t2star = self.tau0

        i = 0
        k = 0
		
        tr = 4*self.nbath.T2h
        print('T2star', self.nbath.T2h)
        pulses = 1
        tauarr = np.linspace(0,.001,1e5)
        maxT = tauarr[-1]
        #self.hahn(tr = tr, pulses = pulses, tauarr = tauarr)

        i = 0
        k = 0

        while ((k<=self.K+1) and (i<max_nr_steps) and (sparsity < .5*max(p_az))):

            k = self.find_optimal_k (strategy = "int", alpha = 1)
            M = self.single_estimation_step (k=k, T2_track=False, adptv_phase = True, pi2_phase = True, cap_method = False)
            self.t2star = self.T2starlist[-1]
            i+=M
			
            p_az = self.nbath.get_probability_density()[1]
		
            sparsity = max(abs(np.diff(p_az)))
            print('sparsity',sparsity,max(p_az))
            if sparsity >= .5*max(p_az):
                print('sparsity condition reached')

#        print('T2', self.nbath.T2echo)
#        #self.freeevo(ms_curr=self._latest_outcome, tf=.001)
#        self.fv=True
#        self.bayesian_update (m_n = self.outcomes_list[-1], phase_n = 0, t_n = 0, T2_track = False, T2_est = self.nbath.T2echo, tf=1e-3, free_evo=True)
#        pB = self.return_p_fB()[0]
#        self.plot_hyperfine_distr(tau=1, theta = 0,
#        T2track = False, T2est = self.T2_est)
#        #M = self.single_estimation_step (k=k, T2_track=False, adptv_phase = True, pi2_phase = True, cap_method = False, free_evo=True)
#        #self.t2star = self.T2starlist[-1]


    def adaptive_1step_paola (self, max_nr_steps=50):

        self._called_modules.append('adaptive_1step_capp')
        p_az = self.nbath.get_probability_density()[1]
        sparsity = 0
        print('sparsity',sparsity,max(p_az))

        try:
            self.t2star = self.T2starlist[-1]
        except:
            self.t2star = self.tau0

        i = 0
        k = 0
		
        tr = 4*self.nbath.T2h
        print('T2star', self.nbath.T2h)
        pulses = 1
        tauarr = np.linspace(0,.001,1e5)
        maxT = tauarr[-1]
        #self.hahn(tr = tr, pulses = pulses, tauarr = tauarr)


        while ((k<=self.K+1) and (i<max_nr_steps) and (sparsity <.5*max(p_az))):

            k = self.find_optimal_k (strategy = 'int', alpha =1)
            print('k=',k,', k_old =',self.k_list[-1])
            M = self.single_estimation_step (k=k, k_old = self.k_list[-1], T2_track=False, adptv_phase = True, pi2_phase = True, cap_method = True)
            self.t2star = self.T2starlist[-1]
            i+=M

            p_az = self.nbath.get_probability_density()[1]

            self.k_list.append(k)

            sparsity = max(abs(np.diff(p_az)))
            print('sparsity',sparsity,max(p_az))
            if sparsity >= .5*max(p_az):
                print('sparsity condition reached')

        #print('T2', self.nbath.T2echo)
        #self.freeevo(ms_curr=0, tf=10*self.nbath.T2echo)



