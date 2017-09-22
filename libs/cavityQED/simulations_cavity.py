"""Simulations of a cavity, based on Albrecht, ..., Becher et al.  Author: Cristian Bonato
Improvements to be made:
- parameter initialization more clear.
- save parameters with a figure.
- get the right ZPL - PSB branching ratio, lifetime data. (not for nanodiamonds)
"""


import numpy as np
import math
import scipy as sc
import pylab as plt
import matplotlib
from analysis.lib.tools import toolbox as tb
from matplotlib import rc, cm

n_diamond=2.419
c = 2.99792e8

class CavitySims ():

    def __init__(self,use='our_bulk_params'):
        if use == 'our_bulk_params':
            #experimental values taken from fit of 5 Lorentzians to spectrometer data at LT 
            # (Cavities/SpectrumData/Y_data_2_smooth.txt ; X_data_wl.tx; measurements at 10 K)
            self.freq = np.array([471.1e12,456.3e12,440.6e12,432.4e12,423.4e12])
            self.A = np.array([119.,1648.,646.,439.,488.])
            self.linewidths = np.array([0.4e12,19.8e12,10.8e12,10.6e12,9.1e12])
            #alternatively, fitting with different range, but obtaining 0.055 branching ratio:
            #self.freq = np.array([471.1,456.0,440.9,431.9,423.7])*1.e12
            #self.A = np.array([110.,722.,686.,309.,167.])
            #self.linewidths = np.array([0.39,11.3,12.5,11.0,5.2])*1.e12
            self.gamma_tot = 1/(12.e-9) #using the typical decay tiume we measure in bulk diamond 

        elif use == 'Becher_nanodiamond_params':
            #NV parameters
            #experimental values taken from PRL 110, 243602
            self.freq = np.array([469.6, 461.9,452.6, 439.7, 429.0, 416.2, 403.2, 392.4])*1.e12
            self.A = np.array([1520., 5260., 18600., 16400., 14000., 9180., 6570., 3270.]) ##branching ratios from Albrecht et al.
            self.linewidths = np.array([2.44,15.9,15.5,15.0,16.5,12.7,13.7,16.1])*1.e12
            self.gamma_tot = 34.9e6 #deduced by g2 msmnts on nanodiamonds (Albrecht),corresponds to t = 28 ns, in nanodiamond.

        else:
            print 'Please give a valid parameter set!!! '

        self.A_tot = np.sum(self.A)
        self.epsilon = self.A/self.A_tot #relative strength transitions
        self.gamma_relative = self.gamma_tot*self.epsilon
        self.dfreq = self.freq-self.freq[0]

        # print 'brancing ratio ZPL to rest:',self.epsilon[0]
        print 'lifetime excited state:',str(round(1/self.gamma_tot*1.e9,1)),'ns'

        #assume the broadening of ZPL is due to pure dephasing (gamma) only. 
        #Note that at 10 K (our_bulk_params) this approximation cannot be great. But neither is our fitting
        self.gamma_star = 2*math.pi*self.linewidths[0]
        self.gamma_i_im1 = 2*math.pi*self.linewidths - self.gamma_star

        #CAVITY parameter
        self.cavity_length = 1.1e-6 # only air length
        self.finesse=None
        self.T = None
        self.diamond_thickness = 3.e-6
        self.radius_curvature = 15.e-6

        self.freq_ZPL = self.freq[0]
        self.wavelength_ZPL = c/float(self.freq[0])

        self.optical_length  = self.calc_optical_length()

        #simulation parameters
        #self.N_sim = 50000

    def set_mirror_curvature (self, value):
        self.radius_curvature = value

    def get_mirror_curvature(self):
        return self.radius_curvature

    def set_diamond_thickness (self, value):
        self.diamond_thickness = value

    def get_diamond_thickness(self):
        return self.diamond_thickness

    def set_cavity_length (self, value):
        self.cavity_length = value
        
    def get_cavity_length (self):
        return self.cavity_length 

    def get_optical_length(self):
        return self.optical_length

    def get_waist(self):
        return self.waist

    def get_mode_volume(self):
        return self.mode_volume

    def get_gamma_tot(self):
        return self.gamma_tot

    def set_gamma_tot(self, value):
        self.gamma_tot = value


    def set_temperature(self, value):
        self.T = value
        self.gamma_star = self.T_to_dephZPL(value)

    def set_pure_dephasing (self, value):
        self.gamma_star = value       

    def set_Q (self, value):
        self.finesse = self.Q_to_finesse(value, self.FSR, self.freq[0])

    def get_finesse(self):
        return self.finesse

    def set_finesse (self, value):
        self.finesse = value

    def calc_k(self,**kw):
        """
        self.k=math.pi*c/(self.optical_length*self.finesse)
        """
        self.k=math.pi*c/(self.optical_length*self.finesse)
        return self.k

    def calc_FSR(self, **kw):
        """
        self.FSR = c/(2*(self.diamond_thickness*n_diamond+self.cavity_length)) 
        """
        self.FSR = c/(2*(self.diamond_thickness*n_diamond+self.cavity_length)) 
        return self.FSR

    def calc_optical_length(self, **kw):
        """
        self.optical_length = self.diamond_thickness*n_diamond + self.cavity_length
        """
        self.optical_length = self.diamond_thickness*n_diamond + self.cavity_length
        return self.optical_length

    def calc_waist(self,**kw):
        """
        Input:
        wavelength   -  wavelength of the light in the cavity; default: self.wavelength_ZPL
        output:
        self.waist = ((wavelength/np.pi)**0.5)*(self.optical_length*(self.radius_curvature-self.optical_length))**(1/4.)

        """
        wavelength = kw.pop('wavelength', self.wavelength_ZPL)
        self.waist = ((wavelength/np.pi)**0.5)*(self.optical_length*(self.radius_curvature-self.optical_length))**(1/4.)
        return self.waist

    def calc_mode_volume(self):
        """
        self.mode_volume = np.pi*((0.5*self.waist)**2)*self.optical_length
        """
        self.mode_volume = np.pi*((0.5*self.waist)**2)*self.optical_length
        return self.mode_volume


    def wavelength_to_freq(self, wavelength):
        frequency = c/wavelength
        return frequency

    def T_to_dephZPL(self, T):
        """
        caculate the linewidth broadening as due to the Jahn-Teller effect from temperature T.
        Formula is from Fu et al. 
        """
        self.gamma_star = 2*math.pi*(16.2+9.2e-7*(1/0.0125)*T**5)*1e6 #in Hz. 
        return self.gamma_star

    def calc_gamma(self, **kw):
        """
        calculate 'gamma', the combined decays as in Albrecht et al:
        'gamma' = kappa + gamma_tot + gamma_i_(i-1) + gamma_star
        (note: this combined parameter is never given a name in Albrecht et al. Hence here 'gamma')
        with:
        kappa (self.k) - the cavity decay rate
        gamma_tot (self.gamma_tot) - the total decay rate from the NV excited state
        gamma_i_(i-1) (=2pi*linewidth - gamma_star) - the decay rate from vibronic state i to vibronic state i-1  
        gamma_star (self.gamma_star) - the temperature broadened part of the decay rate
        """
        self.gamma=(self.k+self.gamma_tot+self.gamma_i_im1+self.gamma_star)
        return self.gamma

    def calc_coupling(self, **kw): 
        """
        Input:
        wavelength - wavelength of the light in the cavity; default: self.wavelength_ZPL
        dipole_orientation - orientation of the dipole w.r.t. cavity field, in degrees
        spatial_mismatch - the amount by which the NV is not in the cavity maximum, in lambda.
        Uses:
        gamma_tot   - the total decay rate of the excited state  (1/lifetime)
        mode volume - the mode volume of the cavity 
        output:
        self.g = np.sqrt((3.*c*(wavelength**2)*self.gamma_tot/2.)/(4.*math.pi*self.mode_volume))
        """
        dipole_orientation = kw.pop('dipole_orientation', 0.)
        spatial_mismatch = kw.pop('spatial_mismatch',0.)
        wavelength = kw.pop('wavelength', self.wavelength_ZPL)

        #albrecht et al:

        orientation_factor= (math.cos(math.pi/180.*dipole_orientation))
        spatial_overlap = math.cos(2*math.pi*spatial_mismatch)
        #print orientation_factor,spatial_overlap
        self.g = orientation_factor*spatial_overlap*np.sqrt((3.*c*(wavelength**2)*self.gamma_tot/2.)/(4.*math.pi*self.mode_volume))
        #kaupp et al (scaling laws)
        #self.g = np.sqrt((3.*math.pi*c*(wavelength**2)*self.gamma_tot)/(2.*self.mode_volume))
        #note that for an accurate treatment, we should perhaps use n_diamond in here. (mu~1/sqrt(n)()
        return self.g

    def calc_g_i(self, **kw):
        """
        self.g_i = self.g*np.sqrt(self.epsilon)
        Input:
        epsilon     - length n array with the relative amplitude of the ZPL and PSBs. ZPL = 0th entry
        g           - the cavity-NV coupling
        Output:
        self.g_i    - the relative coupling per NV ZPL/PSB transition i.
        """
        self.calc_coupling(**kw)
        self.g_i = self.g*np.sqrt(self.epsilon)
        return self.g_i


    def calc_purcell_factor_air(self, **kw):
        """
        Function that calculates the 'naive' version of the Purcell factor, 
        for the 'air modes' as in Janitz et al.,
        assuming an emitter with infinitely narrow linewidth
        Input:
        mode volume
        wavelength
        Q
        Output:
        purcell factor (for a cavity with diamond) 
        """
        Q = kw.pop('Q',self.finesse_to_Q(self.finesse,self.FSR,self.freq[0]))
        wavelength = kw.pop('wavelength', self.wavelength_ZPL)
        mode_volume = kw.pop('mode_volume', self.mode_volume)

        self.naive_FpA = 3. * Q * (wavelength)**3. / n_diamond**3. /(4.*math.pi**2 * mode_volume) 
        return self.naive_FpA
        # naive_Fp2= 4.*((g)**2)/(self.k*gamma_tot)
        # naive_Fp3 = 3. * Q * (wavelength)**3. / (4.*(math.pi**2) * mode_volume) 


    def calc_purcell_factor_diamond(self, **kw):
        """
        Function that calculates the 'naive' version of the Purcell factor,
        for the 'diamond modes' as in Janitz et al.,
        assuming an emitter with infinitely narrow linewidth
        Input:
        mode volume
        wavelength
        Q 
        Output:
        purcell factor (for a cavity with diamond) 
        """
        Q = kw.pop('Q',self.finesse_to_Q(self.finesse,self.FSR,self.freq[0]))
        wavelength = kw.pop('wavelength', self.wavelength_ZPL)
        mode_volume = kw.pop('mode_volume', self.mode_volume)

        self.naive_FpD = 3. * Q * (wavelength)**3. /(n_diamond**3 - n_diamond)  /(4.*math.pi**2 * mode_volume) 
        return self.naive_FpD
        # naive_Fp2= 4.*((g)**2)/(self.k*gamma_tot)
        # naive_Fp3 = 3. * Q * (wavelength)**3. / (4.*(math.pi**2) * mode_volume) 

    def calc_level_energies(self):
        """
        E_i = omega_i*hbar=2*pi*(x_ci-xc0) as from Albrecht et al.
        """
        self.E_i = sc.constants.hbar*2*math.pi*(self.dfreq)/ sc.constants.e*1.e3 #in meV
        return self.E_i

    def calc_longitudinal_cav_modes(self, **kw):
        """
        function that calculations the frequencies of the longitudinal cavity modes
        it uses the resonance frequency of the cavity (given by ZPL freq), or the cavity length
        and the FSR
        Input:
        reference    -   'ZPL' or 'cavity length'. default: 'ZPL'
                            If 'ZPL', it takes the cavity modes as self.freq_ZPL, and self.freq_ZPL-i*self.FSR 
                            If 'cavity length', it calculates the modes as c*N/2*self.optical_length
        stopband     -   [wl_min,wl_max], the minimum and maximum wavelengths taken into account for modes default: [600e-9,800e-9]
                            Note that the actually different mode space outside stopband is NOT taken into account in the model now
        Ouput:
        long_modes_lambda 
        long_modes_freq
        """

        reference = kw.pop('reference','ZPL')
        stopband = kw.pop('stopband',[600.e-9,800.e-9])
        wavelength = kw.pop('wavelength',self.wavelength_ZPL)
        freq = c/wavelength
        if reference == 'ZPL':
            self.long_modes_lambda =np.array([])
            self.long_modes_freq = np.array([])
            l=0
            i=0
            while ((l<stopband[1]) and (i< 1000)):
                nu = freq-i*self.FSR
                l = c/nu 
                self.long_modes_lambda = np.append(self.long_modes_lambda,l)
                i = i+1

        elif reference == 'cavity_length':
            N = np.arange(120)+1
            self.long_modes_lambda = 2*self.optical_length / N
        else:
            print 'no valid reference given to calculate longitudinal modes! '
            return 0,0
        
        self.long_modes_lambda=self.long_modes_lambda[np.where((self.long_modes_lambda>stopband[0])&(self.long_modes_lambda<stopband[1]))]
        self.long_modes_lambda = np.sort(self.long_modes_lambda)
        self.long_modes_freq = c/self.long_modes_lambda

        return self.long_modes_lambda, self.long_modes_freq



    def calculate_derived_params (self, verbose=False,**kw):
        self.calc_optical_length()    
        if (self.optical_length>self.radius_curvature-2e-6):
            print "WARNING: Cavity is unstable!", self.optical_length,'>',self.radius_curvature-2e-6 
   
        self.calc_waist(**kw)
        self.calc_mode_volume()

        self.calc_FSR()
        self.calc_longitudinal_cav_modes(**kw)

        # calculate the cavity decay rate
        self.calc_k()

        # calculate the coupling between NV and cavity
        self.calc_g_i(**kw)

        #calculate the decay of the NV into all the channels
        self.calc_gamma()

        #calculate_naive_pUrcell factors
        self.calc_purcell_factor_air()
        self.calc_purcell_factor_diamond()

        self.calculate_detuning()

        self.calc_level_energies()

        if verbose:
            print 100*'*'
            print 'diamond length',self.diamond_thickness*1.e6,'um'
            print 'mirror radius of curvature', self.radius_curvature*1.e6,'um'
            print 'optical length',self.optical_length*1.e6,'um'
            print 'mode volume ', self.mode_volume/1.e-18,'um^3'
            print 'beam waist ', self.waist/1.e-6,'um'
            print 'cavity decay rate', self.k*1.e-9/(2*math.pi),'GHz'
            print 'lambda_ZPL = ', self.wavelength_ZPL*1e9, 'nm'
            print 'pure dephasing rate = ', self.gamma_star*1.e-9, 'GHz'
            print 100*'*'

    def calculate_detuning(self):
        """
        calculate the detuning between the longitudinal modes (self.long_modes_freq)
        and the NV transitions (ZPL and PSB, as in self.freq)
        """
        self.delta = np.zeros([len(self.freq), len(self.long_modes_freq)])
        
        for i,f_line in enumerate(self.freq):
            for j,f_mode in enumerate(self.long_modes_freq):
                self.delta[i,j] = f_line - f_mode ###get delta_ij = det between NV-line i and cav-mode j
        self.delta = 2*math.pi*self.delta
        return self.delta

    def calculate_R_and_P (self, do_plot = False, do_save = False, verbose=False):
        """
        function that calculates all the rates R and coupling parameters P, 
        for the NV lines as in self.freq and the longitudinal modes as in long_modes_lambda
        """
        G_all = self.g_i #coupling between NV and cavity, for all modes i
        Gamma = self.gamma #kappa+gamma+gamma_(i,i-1)+gamma* for each transition (ZPL & PSB)

        #create a copy of these array for each longitudinal mode, to be able to calculate R_ij
        Gamma = np.tile(Gamma,(len(self.long_modes_freq),1))
        G_all = np.tile(G_all,(len(self.long_modes_freq),1))

        self.Gamma = Gamma.transpose()
        self.G_all = G_all.transpose()

        self.R = (4.*(self.G_all**2)/self.Gamma)*(1./(1+(2*self.delta/(self.Gamma))**2))

        #sum over all cavity modes. the emission rate from ZPL, PSBs in all cavity modes
        self.R_tot = np.sum(self.R, axis=1)

        #the for us relevant factor - emission from ZPL into cavity mode (R[0,0]) compared to free speac emission gamma_tot
        self.F = self.R/float(self.gamma_tot)
        self.F0 = self.R[0,0] #I don't think this factor means anything

        #probability of emission into all channels
        self.P = self.R/ float(np.sum(np.sum(self.R))+self.gamma_tot)
        #sum over all NV lines. probability of emission into all longitudinal modes
        self.P_tot = np.sum(self.P, axis=0)

        #sum over the PSB lines. Emission into the different longitudinal modes, per PSB
        self.P_PSB = np.sum(self.P[1:,:],axis=0) 
        #Emission from all PSB lines into the 'zero-order' longitudinal mode; thus the one on resonance with NV
        self.P_PSB_zero = self.P_PSB[0]
        #Emission from all PSB into all other cavity modes than the 'zero-order' one
        self.P_PSB_nonzero = np.sum(self.P_PSB[1:])
        #Emission into the different longitudinal modes for the ZPL
        self.P_ZPL = self.P[0,:]
        #emission from the ZPL into the 'zero-order' longitudinal cavity mode
        self.P_ZPL_zero  = self.P_ZPL[0]
        #Emission from the ZPL into all other cavity modes than the 'zero-order' one
        self.P_ZPL_nonzero = np.sum(self.P_ZPL[1:])    
        #the total decay of the NV both into the cavity and outside of the cavity, per NV line   
        self.R_plus_gm = np.add(self.R_tot, self.gamma_relative)
        #the lifetime is the inverse of the total decay rate.
        self.lifetime = 1./np.sum(self.R_plus_gm)

        if verbose:
            print 'Rij (kHz):'
            print self.R[:,0]
            print 'Rtot (kHz):'
            print self.R_tot*1.e-3
            print 'P (%)', self.P*100
            print 'P_tot (%)',self.P_tot*100
            print 'R_plus_gm',self.R_plus_gm
            print 'lifetime', self.lifetime


        if do_plot:
            plt.figure()
            plt.plot (self.long_modes_lambda, self.P_tot, 'ob', markersize=2)
            plt.ylabel ('probability of emission in the mode')
            plt.xlabel ('wavelength [nm]')
            plt.show()
            
    

    def emission_in_ZPL (self, sweep_param = 'Cavity length (um)', min_val=0, max_val=15, nr_points=50, xlogscale=False,**kw):
        sweep_vals = np.linspace (min_val, max_val, nr_points)
        if xlogscale:
            sweep_vals = np.logspace(np.log10(min_val), np.log10(max_val), nr_points)

        emission_prob = np.zeros (nr_points)
        emission_prob_others = np.zeros(nr_points)
        emission_prob_ZPL_in_0 = np.zeros(nr_points)
        emission_prob_PSB_in_0 = np.zeros(nr_points)
        emission_prob_ZPL_nonzero = np.zeros(nr_points)
        emission_prob_PSB_nonzero = np.zeros(nr_points)
        purcell = np.zeros(nr_points)
        purcellA = np.zeros(nr_points)
        purcellD = np.zeros(nr_points)
        ind = 0
        mode_vol = np.zeros(nr_points)
        rate_ZPL = np.zeros(nr_points)
        rate_PSB = np.zeros(nr_points)
        gamma_ZPL = np.zeros(nr_points)
        gamma_PSB = np.zeros(nr_points)
        lifetime = np.zeros(nr_points)

        for p in sweep_vals:
            if (sweep_param == 'Cavity length (um)'):
                self.set_cavity_length (p*1.e-6) 
            elif (sweep_param == 'Temperature (K)'):
                self.set_temperature (p)
            elif (sweep_param == 'dephasing rate (GHz)'):
                self.set_pure_dephasing (p*1e9/(2*math.pi))
            elif (sweep_param == 'Quality factor'):
                self.set_Q (p) #keep cavity length constant, change the finesse.
            elif (sweep_param == 'Finesse'):
                self.set_finesse (p)
            elif (sweep_param == 'cavity linewidth (GHz)'):
                self.set_finesse(self.dnu_nuFSR_to_finesse(p*1.e9,self.FSR))
            elif (sweep_param == 'Diamond thickness (um)'):
                self.set_diamond_thickness(p*1.e-6)
            else:
                print "You entered an invalid sweep parameter. stopping."
                break

            self.calculate_derived_params(**kw)
            self.calculate_R_and_P(do_plot=False,verbose=False)

            emission_prob[ind] = self.P_tot[0]
            emission_prob_others[ind] = np.sum(self.P_tot[1:])
            emission_prob_ZPL_in_0[ind] = self.P_ZPL_zero
            emission_prob_PSB_in_0[ind] = self.P_PSB_zero
            emission_prob_ZPL_nonzero[ind] = self.P_ZPL_nonzero
            emission_prob_PSB_nonzero[ind] = self.P_PSB_nonzero
            mode_vol[ind] = self.mode_volume
            purcell[ind] = self.F0
            purcellA[ind] = self.naive_FpA
            purcellD[ind] = self.naive_FpD
            rate_ZPL[ind] = self.R_plus_gm[0]
            rate_PSB[ind] = np.sum(self.R_plus_gm[1:])
            gamma_ZPL[ind] = self.gamma_relative[0]/float(np.sum(np.sum(self.R))+self.gamma_tot)
            gamma_PSB[ind] = np.sum(self.gamma_relative[1:])/float(np.sum(np.sum(self.R))+self.gamma_tot)
            lifetime[ind] = self.lifetime
            ind = ind + 1

        return sweep_vals[:ind], emission_prob[:ind], mode_vol[:ind], purcell[:ind],\
            purcellA[:ind],purcellD[:ind],emission_prob_others[:ind],emission_prob_ZPL_in_0[:ind],\
            emission_prob_PSB_in_0[:ind],emission_prob_ZPL_nonzero[:ind],emission_prob_PSB_nonzero[:ind],\
            rate_ZPL[:ind],rate_PSB[:ind], gamma_ZPL[:ind],gamma_PSB[:ind],lifetime[:ind]


    def plot_emission_spectrum (self):

        ll = np.linspace (580, 1000, 10000)*1e-9
        nu = 3e8/ll

        y = np.zeros(len(nu))

        #devo fare un for-loop sui modi e plottarli come lorentziane
        for i in np.arange(len(self.freq)):

            y = y + 2*self.A[i]*(np.pi*self.linewidths[i])/(4*(nu-self.freq[i])**2+self.linewidths[i]**2)

        y = y/np.sum(y)
        plt.figure(figsize=(10,5))
        plt.plot (ll*1e9, y, linewidth=2)
        plt.xlim ([600, 800])
        plt.show()
    
    def plot_vs_sweepparam(self, sweep_param, min_val, max_val, nr_points, xlogscale=False, plotmode='all_to_zero',**kw):
        x, y1, V, F_p, F_pA, F_pD, y2, y3, y4,y5,y6, y7, y8,y9,y10,y11 = self.emission_in_ZPL(sweep_param, min_val, max_val, nr_points, xlogscale,**kw)
            #y1 : emission all - into zero mode
            #y2 : emission all - into others modes
            #y3 : emission ZPL - into zero mode
        fig = plt.figure (figsize=(8,6))
        ax = fig.add_subplot(111)

        # print plotmode
        alternative_ylabel=False
        if plotmode == 'all_to_zero':
            ax.plot(x, y1, color = 'DarkViolet', linewidth = 4, label = 'resonant mode') #0.5 factor from losses
        elif plotmode == 'all_to_rest':
            ax.plot(x, y2, color = 'Green', linewidth = 4, label = 'all other modes') 
        elif plotmode == 'ZPL_to_zero':
            ax.plot(x, y3, color = 'RoyalBlue', linewidth = 4, label = 'ZPL to resonant mode') 
        elif plotmode == 'ZPL_PSB_inout':
            ax.plot(x,y3, color = 'DarkViolet', linewidth = 4, label = 'ZPL to resonant mode')
            ax.plot(x,y4, color = 'Green', linewidth = 4, label = 'PSB to resonant mode')
            ax.plot(x,y5,'--', color = 'Darkviolet', linewidth = 4, label = 'ZPL to other modes')
            ax.plot(x,y6, '--',color = 'Green', linewidth = 4, label = 'PSB to other modes')
            ax.plot(x,y9, '.',color = 'DarkViolet', linewidth = 4, label = 'ZPL outside cavity')
            ax.plot(x,y10, '.',color = 'Green', linewidth = 4, label = 'PSB outside cavity')
            #ax.text(x[0]+1,y3[0]+0.02,'P00='+str(round(y3[0],2)),color='DarkViolet')
            #ax.text(x[-1]-12,y3[-1]+0.08,'P00='+str(round(y3[-1],2)),color='DarkViolet')
        elif plotmode == 'rates':
            ax.plot(x, y7/1.e6, color = 'DarkViolet', linewidth = 4, label = 'Rate ZPL to all') 
            ax.plot(x, y8/1.e6, color = 'Green', linewidth = 4, label = 'Rate PSB to all') 
            alternative_ylabel=True
            ylabel = 'Rate (MHz)'
        elif plotmode == 'branching_ratio':
            branchingratio = np.divide(y7,(np.add(y7,y8)))
            ax.plot(x, branchingratio, color = 'DarkViolet', linewidth = 4, label = 'branching ratio') 
            ax.text(x[0]+4000,branchingratio[0],str(round(branchingratio[0],2)),fontsize=15,color='DarkViolet')
            ax.text(x[-1]-30000,branchingratio[-1]-0.1,str(round(branchingratio[-1],2)),fontsize=15,color='DarkViolet')
            alternative_ylabel=True
            ylabel = 'probability of emission into ZPL'
            ax.set_ylim([0,1])
        elif plotmode == 'Fpurcell':
            ax.plot(x, F_p, color = 'DarkViolet', linewidth = 4, label = '$R_{0,0}/\gamma$') 
            ax.plot(x, F_pA, color = 'Green', linewidth = 4, label = '$F_p^{(A)}$') 
            ax.plot(x, F_pD, color = 'Orange', linewidth = 4, label = '$F_p^{(D)}$') 
            alternative_ylabel=True
            ylabel = 'Purcell factor'
        elif plotmode == 'lifetime':
            ax.plot(x, y11*1.e9, color = 'Orange', linewidth = 4, label = 'lifetime') 
            alternative_ylabel=True
            ylabel = 'lifetime(ns)'
            
        if xlogscale:
            ax.set_xscale('log')
        
        ax.set_xlabel(sweep_param, fontsize = 20)
        if not alternative_ylabel:
            ax.set_ylabel ('Probability of emission into channel', fontsize = 20)
            ax.set_ylim(-0.02,1.02)
            ax.yaxis.set_ticks(np.arange(0, 1.2, 0.5))

        else:
            ax.set_ylabel (ylabel, fontsize = 20)
        # ax.text(x[-1]-8000,y3[-1]-0.1,str(round(y3[-1],2)),fontsize=15,color='DarkViolet')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(loc = 'middle left')
        ax.set_title('d = '+str(round(self.diamond_thickness*1e6,1))+' $\mu$m; L = ' +str(round(self.cavity_length*1e6,1)) +' $\mu$m; ROC = '+str(int(self.radius_curvature*1.e6))+' $\mu$m; finesse ='+str(int(self.finesse)) +'\n',fontsize=16)
        sweepparam = sweep_param.replace(" ","")
        
        ax.tick_params(axis='both', which='major',   top='off',right = 'off',labelsize=14)
        plt.tight_layout()


        try:
            fig.savefig('H://My Documents/Cavities/Simulations/'+tb.get_timestamp_from_now()+plotmode+'_vs_'+sweepparam+'.png')
        except: 
            print 'Figure not saved'
        plt.show()

         
        #for now always return x (the sweep paran), y3 (ZPL to zero mode) and y11 (lifetime)
        return x,y3, y11


    def Ltot_to_finesse(self,Ltot):
        """
        input:
        Ltot - total losses
        output:
        F - Finesse
        """
        finesse = 2*math.pi/Ltot
        return finesse

    def F_to_Ltot(self,F):
        """
        input:
        F - Finesse
        output:
        Ltot - total losses
        """
        Ltot = 2*math.pi/F
        return Ltot

    def optical_length_to_nuFSR(self,optical_length):
        """
        input:
        optical_length - optical length
        output:
        FSR - FSR in freuqnecy
        """        
        FSR = c/(2*optical_length)
        return FSR


    def dnu_nuFSR_to_finesse(self,dnu,nuFSR):
        """
        input:
        dnu - linewidth in frequency
        nuFSR - free spectral range in frequency
        output:
        F - Finesse
        """
        finesse = nuFSR/dnu 
        return finesse

    def dnu_nu_to_Q(self,dnu,nu):
        """
        input:
        dnu - linewidth in frequency
        nu - frequency
        output:
        Q - Quality factor
        """
        Q = nu / dnu
        return Q


    def Q_nu_to_dnu(self,Q,nu):
        """
        input:
        Q - Quality factor
        nu - frequency
        output:
        dnu - linewidth in frequency
        """
        dnu = nu / Q
        return dnu


    def Q_to_finesse(self,Q, nuFSR, nu):
        """
        input:
        Q - Quality factor
        nuFSR - free spectral range in frequency
        nu - frequency
        output:
        F - Finesse
        """
        finesse = nuFSR / nu * Q
        return finesse

    def finesse_to_Q(self,finesse, nuFSR, nu):
        """
        input:
        F - Finesse
        nuFSR - free spectral range in frequency
        nu - frequency
        output:
        Q - quality factor
        """
        Q = nu / nuFSR * finesse
        return Q


# plt.close('all')

# s = CavitySims()
# s._linewidth=False

# F=10000
# L=2e-6
# lambdaa=0.637e-6#0.6393e-6
# #L=9./2*lambdaa

# dnu = 10.e9
# nu = c/637.e-9
# Q = nu/dnu
# d_diamond=3.e-6
# ROC = 15.e-6

# s.set_diamond_thickness (d_diamond)
# s.set_mirror_curvature (ROC)
# s.set_cavity_length (L)
# s.set_Q (Q)

# #s.calculate_params()
# #s.simulate()

# sweep_param1='Temperature (K)'
# sweep_param2='Quality factor'
# sweep_param3='dephasing rate (GHz)'

# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 100, plotmode='ZPL_PSB_inout')
# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 100, plotmode='branching_ratio')
# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 20, plotmode='rates')
# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 20, plotmode='Fpurcell')
# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 21, plotmode='ZPL_to_zero')
# #s.plot_vs_sweepparam(sweep_param1, 0, 100, 51, plotmode='lifetime')

# s.set_temperature (5) #after sweeping the temperature, set T to 5.
# s.plot_vs_sweepparam(sweep_param2, 0, 400000, 100, plotmode='branching_ratio')
# # s.plot_vs_sweepparam(sweep_param2, 44999, 45000, 3, plotmode='lifetime')
# # print "t",s.lifetime
# # print "p",s.P_ZPL_zero
# # s.plot_vs_sweepparam(sweep_param2, 0, 100000, 100, plotmode='ZPL_PSB_inout')
# #s.plot_vs_sweepparam(sweep_param2, 0, 200000, 500, plotmode='Fpurcell')
# #s.plot_vs_sweepparam(sweep_param3, 0.00001, 10, 500, plotmode='Fpurcell',xlogscale=True)

