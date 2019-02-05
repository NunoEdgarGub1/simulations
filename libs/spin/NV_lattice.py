
import numpy as np

def generate (N, a0, prefactor):
    pi = np.pi
    ##Carbon Lattice Definition
    #Rotation matrix to get b along z-axis
    Rz=np.array([[np.cos(pi/4),-np.sin(pi/4),0],[np.sin(pi/4),np.cos(pi/4),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(np.arctan(np.sqrt(2))),-np.sin(np.arctan(np.sqrt(2)))],[0,np.sin(np.arctan(np.sqrt(2))),np.cos(np.arctan(np.sqrt(2)))]])
    # Basis vectors
    a = np.array([0,0,0])
    b = a0/4*np.array([1,1,1])
    b = Rx.dot(Rz).dot(b)
    # Basisvectors of Bravais lattice
    i = a0/2*np.array([0,1,1])
    i = Rx.dot(Rz).dot(i)
    j = a0/2*np.array([1,0,1])
    j = Rx.dot(Rz).dot(j)
    k = a0/2*np.array([1,1,0])
    k = Rx.dot(Rz).dot(k)

    # define position of NV in middle of the grid
    NVPos = round(N/2) *i +round(N/2)*j+round(N/2)*k

    #Initialise
    L_size = 2*(N)**3-2 # minus 2 for N and V positions
    Ap = np.zeros(L_size) #parallel
    Ao = np.zeros(L_size) # perpendicular component
    Azx = np.zeros(L_size) # perpendicular component
    Azy = np.zeros(L_size) # perpendicular component
    
    #Elements for dC correction of Cnm:
    Axx = np.zeros(L_size)
    Ayy = np.zeros(L_size)
    Axy = np.zeros(L_size)
    Ayx = np.zeros(L_size)
    Axz = np.zeros(L_size)
    Ayz = np.zeros(L_size)
    r = np.zeros(L_size)
    theta = np.zeros(L_size)
    phi = np.zeros(L_size)
    x = np.zeros(L_size)
    y = np.zeros(L_size)
    z = np.zeros(L_size)

    o = 0
    for n in range(N):
        for m in range(N):
            for l in range(N):
                if (n== round(N/2) and m==round(N/2) and l == round(N/2)):
                    #Omit the Nitrogen and the Vacancy centre in the calculations
                    o+=0
                else:
                    pos1 = n*i + m*j+l*k - NVPos
                    pos2 = pos1 + b
                    r[o] = np.sqrt(pos1.dot(pos1))
                    #Ap[o] = self.prefactor*np.power(r[o],-3)*(1-3*np.power(pos1[2],2)*np.power(r[o],-2))
                    Ao[o] = prefactor*np.power(r[o],-3)*3*(np.sqrt(np.power(pos1[0],2)+np.power(pos1[1],2))*pos1[2]*np.power(r[o],-2))                            
                    x[o] = pos1[0]
                    y[o] = pos1[1]
                    z[o] = pos1[2]
                    if x[o] != 0:
                        phi[o] = np.arctan(y[o]/x[o])
                    else:
                        phi[o] = np.pi/2
                           
                    if r[o] != 0:
                        theta[o] = np.arccos(z[o]/r[o])
                    else:
                        print ('Error: nuclear spin overlapping with NV centre')
                            
                    Axx[o] = prefactor*(r[o]**(-3))*(1-3*(np.sin(theta[o])**2)*(np.cos(phi[o])**2))
                    Ayy[o] = prefactor*(r[o]**(-3))*(1-3*(np.sin(theta[o])**2)*(np.sin(phi[o])**2))
                    Axy[o] = prefactor*(r[o]**(-3))*(-1.5*(np.sin(theta[o])**2)*(np.sin(2*phi[o])))
                    Ayx[o] = Axy[o]
                    Axz[o] = prefactor*(r[o]**(-3))*(-3*np.cos(theta[o])*np.sin(theta[o])*np.cos(phi[o]))
                    Ayz[o] = prefactor*(r[o]**(-3))*(-3*np.cos(theta[o])*np.sin(theta[o])*np.sin(phi[o]))
                    Azx[o] = Axz[o]
                    Azy[o] = Ayz[o]
                    Ap[o] = prefactor*(r[o]**(-3))*(1-3*np.cos(theta[o])**2)
                    o +=1
                    r[o] = np.sqrt(pos2.dot(pos2))
                    Ao[o] = prefactor*(r[o]**(-3))*3*(np.sqrt(np.power(pos2[0],2)+np.power(pos2[1],2))*pos2[2]*np.power(r[o],-2))
                    x[o] = pos2[0]
                    y[o] = pos2[1]
                    z[o] = pos2[2]
                    if x[o] != 0:
                        phi[o] = np.arctan(y[o]/x[o])
                    else:
                        phi[o] = np.pi/2
                           
                    if r[o] != 0:
                        theta[o] = np.arccos(z[o]/r[o])
                    else:
                        print ('Error: nuclear spin overlapping with NV centre') 
                            
                    Axx[o] = prefactor*(r[o]**(-3))*(1-3*(np.sin(theta[o])**2)*(np.cos(phi[o])**2))
                    Ayy[o] = prefactor*(r[o]**(-3))*(1-3*(np.sin(theta[o])**2)*(np.sin(phi[o])**2))
                    Axy[o] = prefactor*(r[o]**(-3))*(-1.5*(np.sin(theta[o])**2)*(np.sin(2*phi[o])))
                    Ayx[o] = Axy[o]
                    Axz[o] = prefactor*(r[o]**(-3))*(-3*np.cos(theta[o])*np.sin(theta[o])*np.cos(phi[o]))
                    Ayz[o] = prefactor*(r[o]**(-3))*(-3*np.cos(theta[o])*np.sin(theta[o])*np.sin(phi[o]))
                    Azx[o] = Axz[o]
                    Azy[o] = Ayz[o]
                    Ap[o] = prefactor*(r[o]**(-3))*(1-3*np.cos(theta[o])**2)
                    o+=1

    return Ap, Ao, Azx, Azy, Axx, Ayy, Axy, Ayx, Axz, Ayz, r, theta, phi, x, y, z

