from analysis.lib.nv import nvlevels
def get_eta(saturation_counts, dead_time = 50e-9):
    """
    Returns the estimated detection efficiency from the maximum NV countrate. 
    keyword: dead_time, used to correct for finite counting response of the 
    combined APD/couting system.
    """
    NV_effective_lifetime = 12.9e-9 + 1/4.*462e-9 #ms0 cycling + singlet cycling

    real_countrates=saturation_counts*1/(1 - (dead_time*saturation_counts))

    eta=real_countrates*NV_effective_lifetime

    return eta

def get_SSRO_stats(pflip, eta):
    """
    Returns ms=0 fidelity, and average counts per RO, 
    as a function of the spin flip probability per cycle: pflip, 
    and the overall PSB detection efficiency.

    """
    F0 = eta/(pflip+eta-pflip*eta) # = 1-Sum[(1-eta)^i *pflip*(1-pflip)^(i-1) ,{i,1,Infinity}]; 
                                   # see also page 135 of bas' master thesis
    navg = eta/pflip
    return F0, navg

def get_pflip_Ey(strain_splitting):
    """
    Returns spin flip probability per cycle for the Ey transition
    """
    #p0 = 0.012 #spin flip probability at zero strain
    #S0 = 15.   #GHz, the spoint where Ey becomes degenerate with E12
    #pflip = p0*(S0/(S0 - strain_splitting)) #phenomological model for the spin flip probability 
    # well approximates
    singlet_decay_factor = 0.008
    pflip = (1-nvlevels.get_ms0_fraction(strain_splitting,2,theta_x=0))+singlet_decay_factor
    return pflip

def get_pflip_Ex(strain_splitting):
    """
    Returns the simulated spin flip probability per cycle for the Ex transition
    NOT EXPERIMENTALLY VERIFIED!!

    """
    singlet_decay_factor = 0.008
    return (1-nvlevels.get_ms0_fraction(strain_splitting,3,theta_x=0))+singlet_decay_factor

def get_Ey_vs_strain_saturation(strain_splitting,saturation_counts):
    """
    pflip = get_pflip_Ey(strain_splitting)
    eta   = get_eta(saturation_counts)
    returns get_SSRO_stats(plip, eta)
    """

    pflip = get_pflip_Ey(strain_splitting)
    eta = get_eta(saturation_counts)
    print 'eta, pflip:', eta, pflip
    return get_SSRO_stats(pflip, eta)

def get_SSRO_vs_strain(current_strain_splitting, current_RO_fid, target_strain_splitting):
    pflip = get_pflip_Ey(current_strain_splitting)
    t_pflip= get_pflip_Ey(target_strain_splitting)
    F0=current_RO_fid
    eta = (F0*pflip)/(1 - F0 + F0*pflip)
    return get_SSRO_stats(t_pflip, eta)