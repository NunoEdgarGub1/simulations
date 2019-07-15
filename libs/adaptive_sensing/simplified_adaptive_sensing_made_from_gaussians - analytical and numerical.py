# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:44:31 2019

@author: Karen Craigie
"""

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate

tmin=20*10**(-9)
tau=4*tmin
i = complex(0,1)

def RamsayProbN(n, mu=0,ran=25000000,phi=0):#this is for 1mT
    i = complex(0,1)
    tmin=20*10**(-9)
    tau=4*tmin/2**(n-1)
    f= np.linspace(-ran,ran,200) # array of phases
    mag_bit = 2*math.pi*f*tau+phi
    P_comp = 0.5*(1+cmath.exp(i*mu*math.pi)*np.cos(mag_bit))
    P = P_comp.real
    return f/1000000, P#(mag_bit/(2*math.pi*gamma)),P


"""Do the steps to get down to one peak"""

mu1 = 1
mu2 = 0
mu3 = 1
phi1 = 0
phi2 = mu1*np.pi/2
phi3 = (phi2- mu2*np.pi)/2

(fM,P1)=RamsayProbN(1,mu=mu1, phi=phi1)
#plt.plot(fM,P1)  

(fM,P2)=RamsayProbN(2,mu=mu2, phi=phi2)
P1and2 = P1*P2
#plt.plot(fM,P1and2)
#plt.plot(fM,P2)
(fM,P3)=RamsayProbN(3,mu=mu3, phi= phi3)
P12and3 = P1and2*P3
#plt.plot(fM,P12and3)  
#plt.plot(fM,P3)  

#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")


#plt.plot(fMtrimmed,Ptrimmed)
# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, moo, sigma = p
    return A*np.exp(-(x-moo)**2/(2.*sigma**2))


"""Expression for n = 1"""

#plt.figure()
#plt.plot(fM,P1)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

"""From left to right, there are a maximum of 5 peaks visible on the plot"""

sigma = 1/(math.sqrt(2)*np.pi*tau*10**6)# where des the factor of 2 come from? I derived it to be 2 pi instead of pi but it clearly does not fit
AmpC = 1
#print(sigma)
#coeff, var_matrix = curve_fit(gauss, fM, P1, p0=p0)

"""Peak 1"""

centre1=(-2+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak1 = [AmpC,centre1, sigma]

# Get the fitted curve
N1_P1_gauss = gauss(fM, *peak1)

#plt.plot(fM, P_fit4, label='Fitted data')

"""Peak 2"""

centre2=(-1+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak2 = [AmpC,centre2, sigma]
N1_P2_gauss = gauss(fM, *peak2)

#plt.plot(fM, P_fit1, label='Fitted data')


"""Peak 3"""
# p0 is the initial guess for the fitting coefficients (A, moo and sigma above)
centre3 = (mu1/2)/(tau*10**6)#(1/tau*10**6)*mu1
peak3 = [AmpC,centre3, sigma]

# Get the fitted curve
N1_P3_gauss = gauss(fM, *peak3)
#plt.plot(fM, P3_gauss, label='Fitted data')

"""Peak 4"""

centre4=(1+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak4 = [AmpC,centre4, sigma]

# Get the fitted curve
N1_P4_gauss = gauss(fM, *peak4)

#plt.plot(fM, P_fit2, label='Fitted data')

"""Peak 3"""

centre5=(2+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak5 = [AmpC,centre5, sigma]

# Get the fitted curve
N1_P5_gauss = gauss(fM, *peak5)

#plt.plot(fM, P_fit3, label='Fitted data')


N1_P_total = N1_P1_gauss+N1_P2_gauss+N1_P3_gauss+N1_P4_gauss+N1_P5_gauss
#plt.plot(fM, N1_P_total, label='Fitted data')
diffN1A = (abs(N1_P_total-P1))**2

"""Expression for n=2"""
#plt.figure()
#plt.plot(fM,P1and2)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

"""Peak 1"""
Amp1 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak1_N2 = [Amp1,centre1, sigma*2]
N2_P1_gauss = gauss(fM, *peak1_N2)
#plt.plot(fM, N2_P1_gauss)

"""Peak 2"""
Amp2 = AmpC*(1-0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi)))
peak2_N2 = [Amp2,centre2, sigma*2]
N2_P2_gauss = gauss(fM, *peak2_N2)
#plt.plot(fM, N2_P2_gauss)

"""Peak 3"""
Amp3 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak3_N2 = [Amp3,centre3, sigma*2]
N2_P3_gauss = gauss(fM, *peak3_N2)
#plt.plot(fM, N2_P3_gauss)

"""Peak 4"""
Amp4 = AmpC*(1-0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi)))
peak4_N2 = [Amp4,centre4, sigma*2]
N2_P4_gauss = gauss(fM, *peak4_N2)
#plt.plot(fM, N2_P4_gauss)

"""Peak 5"""
Amp5 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak5_N2 = [Amp5,centre5, sigma*2]
N2_P5_gauss = gauss(fM, *peak5_N2)
#plt.plot(fM, N2_P5_gauss)


N2_P_total = N1_P_total*(N2_P1_gauss+N2_P2_gauss+N2_P3_gauss+N2_P4_gauss+N2_P5_gauss)
#plt.plot(fM, N2_P_total, label='Fitted data')
diffN2A = (abs(N2_P_total-P1and2))**2

"""Expression for n=3"""
#plt.figure()
#plt.plot(fM,P12and3)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

if Amp1 <= 0.0001:
    """Peak 2"""
    peak2_N3 = [Amp2*0.5*abs(cmath.exp(i*mu2*math.pi)+cmath.exp(i*mu3*math.pi)),centre2, sigma*4]
    N3_P2_gauss = gauss(fM, *peak2_N3)
    #plt.plot(fM, N2_P2_gauss)

    """Peak 4"""
    peak4_N3 = [Amp4*(1-0.5*abs(cmath.exp(i*mu2*math.pi)+cmath.exp(i*mu3*math.pi))),centre4, sigma*4]
    N3_P4_gauss = gauss(fM, *peak4_N3)
    #plt.plot(fM, N2_P4_gauss)

    N3_P_total = N2_P_total*(N3_P2_gauss+N3_P4_gauss)
else:
        
    """Peak 1"""
    peak1_N3 = [Amp1*mu3,centre1, sigma*4]
    N3_P1_gauss = gauss(fM, *peak1_N3)
    #plt.plot(fM, N2_P1_gauss)

    """Peak 3"""
    peak3_N3 = [Amp3*(1-mu3),centre3, sigma*4]
    N3_P3_gauss = gauss(fM, *peak3_N3)
    #plt.plot(fM, N2_P3_gauss)

    """Peak 5"""
    peak5_N3 = [Amp5*mu3,centre5, sigma*4]
    N3_P5_gauss = gauss(fM, *peak5_N3)
    #plt.plot(fM, N2_P5_gauss)
    
    N3_P_total = N2_P_total*(N3_P1_gauss+N3_P3_gauss+N3_P5_gauss)

#plt.plot(fM, N3_P_total, label='Fitted data')

diffN3A = (abs(N3_P_total-P12and3))**2

"""Numerical gaussian"""


"""Expression for n = 1"""

#plt.figure()
#plt.plot(fM,P1)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

"""From left to right, there are a maximum of 5 peaks visible on the plot"""

"""Got the special constant by using python's curve-fitting tools to fit a 
gaussian to one of the peaks in the cosine and find how this varies with tau"""
special_C = 0.05078
sigmaN =special_C/(tmin*10**6)
AmpC = 1.0 # fit gave 1.02 but this looks better
#coeff, var_matrix = curve_fit(gauss, fM, P1, p0=p0)

"""Peak 1"""

centre1=(-2+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak1 = [AmpC,centre1, sigmaN]

# Get the fitted curve
N1_P1_gauss = gauss(fM, *peak1)

#plt.plot(fM, P_fit4, label='Fitted data')

"""Peak 2"""

centre2=(-1+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak2 = [AmpC,centre2, sigmaN]
N1_P2_gauss = gauss(fM, *peak2)

#plt.plot(fM, P_fit1, label='Fitted data')


"""Peak 3"""
# p0 is the initial guess for the fitting coefficients (A, moo and sigma above)
centre3 = (mu1/2)/(tau*10**6)#(1/tau*10**6)*mu1
peak3 = [AmpC,centre3, sigmaN]

# Get the fitted curve
N1_P3_gauss = gauss(fM, *peak3)
#plt.plot(fM, P3_gauss, label='Fitted data')

"""Peak 4"""

centre4=(1+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak4 = [AmpC,centre4, sigmaN]

# Get the fitted curve
N1_P4_gauss = gauss(fM, *peak4)

#plt.plot(fM, P_fit2, label='Fitted data')

"""Peak 3"""

centre5=(2+mu1/2)/(tau*10**6)# 10 to the 6 becuse MHz scale
peak5 = [AmpC,centre5, sigmaN]

# Get the fitted curve
N1_P5_gauss = gauss(fM, *peak5)

#plt.plot(fM, P_fit3, label='Fitted data')


N1_P_totalN = N1_P1_gauss+N1_P2_gauss+N1_P3_gauss+N1_P4_gauss+N1_P5_gauss
#plt.plot(fM, N1_P_total, label='Fitted data')
diffN1N = (abs(N1_P_totalN-P1))**2

"""Expression for n=2"""
#plt.figure()
#plt.plot(fM,P1and2)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

"""Peak 1"""
Amp1 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak1_N2 = [Amp1,centre1, sigmaN*2]
N2_P1_gauss = gauss(fM, *peak1_N2)
#plt.plot(fM, N2_P1_gauss)

"""Peak 2"""
Amp2 = AmpC*(1-0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi)))
peak2_N2 = [Amp2,centre2, sigmaN*2]
N2_P2_gauss = gauss(fM, *peak2_N2)
#plt.plot(fM, N2_P2_gauss)

"""Peak 3"""
Amp3 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak3_N2 = [Amp3,centre3, sigmaN*2]
N2_P3_gauss = gauss(fM, *peak3_N2)
#plt.plot(fM, N2_P3_gauss)

"""Peak 4"""
Amp4 = AmpC*(1-0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi)))
peak4_N2 = [Amp4,centre4, sigmaN*2]
N2_P4_gauss = gauss(fM, *peak4_N2)
#plt.plot(fM, N2_P4_gauss)

"""Peak 5"""
Amp5 = AmpC*0.5*abs(cmath.exp(i*mu1*math.pi)+cmath.exp(i*mu2*math.pi))
peak5_N2 = [Amp5,centre5, sigmaN*2]
N2_P5_gauss = gauss(fM, *peak5_N2)
#plt.plot(fM, N2_P5_gauss)


N2_P_totalN = N1_P_totalN*(N2_P1_gauss+N2_P2_gauss+N2_P3_gauss+N2_P4_gauss+N2_P5_gauss)
#plt.plot(fM, N2_P_total, label='Fitted data')

diffN2N = (abs(N2_P_totalN-P1and2))**2

"""Expression for n=3"""
#plt.figure()
#plt.plot(fM,P12and3)  
#plt.xlabel("Larmor frequency (MHz))")
#plt.ylabel("Probability density")

if Amp1 <= 0.0001:
    """Peak 2"""
    peak2_N3 = [Amp2*0.5*abs(cmath.exp(i*mu2*math.pi)+cmath.exp(i*mu3*math.pi)),centre2, sigmaN*4]
    N3_P2_gaussN = gauss(fM, *peak2_N3)
    
    """Peak 4"""
    peak4_N3 = [Amp4*(1-0.5*abs(cmath.exp(i*mu2*math.pi)+cmath.exp(i*mu3*math.pi))),centre4, sigmaN*4]
    N3_P4_gaussN = gauss(fM, *peak4_N3)
    
    N3_P_totalN = N2_P_totalN*(N3_P2_gaussN+N3_P4_gaussN)
    
else:
        
    """Peak 1"""
    peak1_N3 = [Amp1*mu3,centre1, sigmaN*4]
    N3_P1_gaussN = gauss(fM, *peak1_N3)

    """Peak 3"""
    peak3_N3 = [Amp3*(1-mu3),centre3, sigmaN*4]
    N3_P3_gaussN = N3_P3_gauss = gauss(fM, *peak3_N3)
    
    """Peak 5"""
    peak5_N3 = [Amp5*mu3,centre5, sigmaN*4]
    N3_P5_gaussN = gauss(fM, *peak5_N3)
    
    N3_P_totalN = N2_P_totalN*(N3_P1_gaussN+N3_P3_gaussN+N3_P5_gaussN)

#plt.plot(fM, N3_P_totalN, label='Fitted data')
    
diffN3N = (abs(N3_P_totalN-P12and3))**2
    
"""Plotting the three on a graph"""

"""
plt.figure()
plt.title("n=1",fontsize=20)
plt.plot(fM,P1, label='Without gaussian fit (raw)')  
plt.xlabel("Larmor frequency (MHz)",fontsize=20)
plt.ylabel("Probability density", fontsize=18)
plt.plot(fM, N1_P_total, label='Analytical fit', color="r")
plt.plot(fM, N1_P_totalN, label='Numerical fit',color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)

plt.figure()
plt.title("n=2",fontsize=20)
plt.plot(fM,P1and2, label='Without gaussian fit (raw)')  
plt.xlabel("Larmor frequency (MHz)", fontsize=20)
plt.ylabel("Probability density",fontsize=18)
plt.plot(fM, N2_P_total, label='Analytical fit',color="r")
plt.plot(fM, N2_P_totalN, label='Numerical fit',color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)

plt.figure()
plt.title("n=3", fontsize=20)
plt.plot(fM,P12and3, label='Without gaussian fit (raw)')  
plt.xlabel("Larmor frequency (MHz)", fontsize=20)
plt.ylabel("Probability density", fontsize=18)
plt.plot(fM, N3_P_total, label='Analytical fit', color="r")
plt.plot(fM, N3_P_totalN, label='Numerical fit',color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)


plt.figure()
plt.title("n=1, difference squared", fontsize=20)
plt.plot(fM,diffN1A, label = "Analytical", color="r")
plt.plot(fM,diffN1N, label = "Numerical", color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
plt.xlabel("Larmor frequency (MHz)", fontsize=20)
plt.ylabel("Probability density", fontsize=18)

difference_value_N1A = (1/(fM[2]-fM[1]))*integrate.trapz(diffN1A, fM)
print(difference_value_N1A)
difference_value_N1N = (1/(fM[2]-fM[1]))*integrate.trapz(diffN1N, fM)
print(difference_value_N1N)
print("For n=1, the analytical fit is worse by a factor of:")
print(difference_value_N1A/difference_value_N1N)

plt.figure()
plt.title("n=2, difference squared", fontsize=20)
plt.plot(fM,diffN2A, label = "Analytical",color="r")
plt.plot(fM,diffN2N, label = "Numerical",color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
plt.xlabel("Larmor frequency (MHz)", fontsize=20)
plt.ylabel("Probability density", fontsize=18)

difference_value_N2A = (1/(fM[2]-fM[1]))*integrate.trapz(diffN2A, fM)
print(difference_value_N2A)
difference_value_N2N = (1/(fM[2]-fM[1]))*integrate.trapz(diffN2N, fM)
print(difference_value_N2N)
print("For n=1, the analytical fit is worse by a factor of:")
print(difference_value_N2A/difference_value_N2N)

plt.figure()
plt.title("n=3, difference squared", fontsize=20)
plt.plot(fM,diffN3A, label = "Analytical",color="r")
plt.plot(fM,diffN3N, label = "Numerical",color="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
plt.xlabel("Larmor frequency (MHz)", fontsize=20)
plt.ylabel("Probability density", fontsize=18)

difference_value_N3A = (1/(fM[2]-fM[1]))*integrate.trapz(diffN3A, fM)
print(difference_value_N3A)
difference_value_N3N = (1/(fM[2]-fM[1]))*integrate.trapz(diffN3N, fM)
print(difference_value_N3N)
print("For n=1, the analytical fit is worse by a factor of:")
print(difference_value_N3A/difference_value_N3N)
"""