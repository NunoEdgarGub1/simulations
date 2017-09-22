
def Generate_NV(Carb_Conc=0.011,N_NV=1,N=25,do_sphere = True):
    """
    Takes Carbon Concentration, number of centres (N_NV) and gridsize (N) as inputs
    Returns a list of 2D numpy arrays with hyperfine strengths in Hz (real freq)
    If you want it to return a sphere around the NV instead of a cube, set do_sphere = True.
    Note: this reduces the size of your sample!
    """
    import numpy as np
    #Constants
    #lattice parameter
    a0 = 3.57 * 10**(-10)
    # Hyperfine related constants
    gam_el = 1.760859 *10**11 #Gyromagnetic ratio rad s-1 T-1
    gam_n = 67.262 *10**6 #rad s-1 T-1
    hbar = 1.05457173*10**(-34)
    pi = np.pi
    mu0 = 4*pi*10**(-7)

    ##Carbon Lattice Definition
    #Rotation matrix to get b along z-axis
    Rz=np.array([[np.cos(pi/4),-np.sin(pi/4),0],[np.sin(pi/4),np.cos(pi/4),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(np.arctan(np.sqrt(2))),-np.sin(np.arctan(np.sqrt(2)))],[0,np.sin(np.arctan(np.sqrt(2))),np.cos(np.arctan(np.sqrt(2)))]])
    # Basis vectors
    a = np.array([0,0,0])
    b =a0/4*np.array([1,1,1])
    b = Rx.dot(Rz).dot(b)

    print b
    # Basisvectors of Bravais lattice
    i =a0/2*np.array([0,1,1])
    i = Rx.dot(Rz).dot(i)
    j = a0/2*np.array([1,0,1])
    j=Rx.dot(Rz).dot(j)
    k =a0/2*np.array([1,1,0])
    k = Rx.dot(Rz).dot(k)


    # define position of NV in middle of the grid
    NVPos = round(N/2) *i +round(N/2)*j+round(N/2)*k

    prefactor = mu0*gam_el*gam_n/(4*pi)*hbar**2 /hbar/(2*pi) #Last /hbar/2pi is to convert from Joule to Hz
    prefactor_cc = mu0*(gam_n**2)/(4*pi)*hbar**2 /hbar/(2*pi)
    #Initialise
    L_size = 2*(N)**3-2 # minus 2 for N and V positions
    Ap = np.zeros(L_size) #parallel
    Ao = np.zeros(L_size) # perpendicular component
    r = np.zeros(L_size)
    x = np.zeros(L_size)
    y = np.zeros(L_size)
    z = np.zeros(L_size)
    o=0

    #Calculate Hyperfine strength for all gridpoints
    for n in range(N):
        for m in range(N):
            for l in range(N):
                if (n== round(N/2) and m==round(N/2) and l == round(N/2)) :#Omit the Nitrogen and the Vacancy centre in the calculations
                    o+=0
                else:
                    pos1 = n*i + m*j+l*k - NVPos
                    pos2 = pos1 + b
                    r[o] = np.sqrt(pos1.dot(pos1))
                    Ap[o] =prefactor*np.power(r[o],-3)*(3*np.power(pos1[2],2)*np.power(r[o],-2)-1)
                    Ao[o] = prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos1[0],2)+np.power(pos1[1],2))*pos1[2]*np.power(r[o],-2))
                    x[o] = pos1[0]
                    y[o] = pos1[1]
                    z[o] = pos1[2]
                    o +=1
                    r[o] = np.sqrt(pos2.dot(pos2))
                    Ap[o] =prefactor*np.power(r[o],-3)*(3*np.power(pos2[2],2)*np.power(r[o],-2)-1)
                    Ao[o] = prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos2[0],2)+np.power(pos2[1],2))*pos2[2]*np.power(r[o],-2))
                    x[o] = pos2[0]
                    y[o] = pos2[1]
                    z[o] = pos2[2]
                    o+=1
    # Generate different NV-Objects by randomly selecting which gridpoints contain a carbon.
    
    if do_sphere == True:
        zipped = zip(r,Ap,Ao,x,y,z)
        zipped.sort() # sort list as function of r
        zipped = zipped[0:len(r)/2] # only take half of the occurences
        r = np.asarray([r_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
        Ap = np.asarray([Ap_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
        Ao = np.asarray([Ao_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
        x = np.asarray([x_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
        y = np.asarray([y_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
        z = np.asarray([z_s for r_s,Ap_s,Ao_s,x_s,y_s,z_s in zipped])
    
    
    for p in range(N_NV):
        # here we choose the grid points that contain a carbon 13 spin, dependent on concentration
        Sel = np.where(np.random.rand(L_size/2)<Carb_Conc)
        Ap_NV =[ Ap[u] for u in Sel]
        Ao_NV =[ Ao[u] for u in Sel]
        r_NV = [ r[u] for u in Sel]
        x_NV = [ x[u] for u in Sel]
        y_NV = [ y[u] for u in Sel]
        z_NV = [ z[u] for u in Sel]
        
        # NV_list.append(A_NV[0]) #index 0 is to get rid of outher brackets in A_NV0


    print Ap[4]
    print Ao[4]

    print 'dd_zz interaction is   '  +str(prefactor_cc*2/(2*0.43*a0)**3) +  '  Hz'
    print 'Ap first c in the cluster  '  +str(prefactor*2/(1.e-9)**3) +  '  Hz'
    print 'Ap second c in the cluster  '  +str(prefactor*2/(1.e-9+0.43*a0)**3) +  '  Hz'
    return Ap_NV[0], Ao_NV[0] , r_NV[0], x_NV[0], y_NV[0], z_NV[0]





