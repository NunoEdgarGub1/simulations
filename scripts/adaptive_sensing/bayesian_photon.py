
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc


'''
th = np.linspace (0, 2*np.pi, 1000)
V = 0.15
x = V*np.cos(th)
#x = np.linspace (-1,1,1000)
pm = 0.4
eta = (pm/(1-pm))
R = 1000.
r = 45.
 
#p = np.linspace (0,1,10000)
p = pm*(1+x)
q=1-p
#q = 1/pm-p
#q = 1-(pm/(1-pm))*x
mu = r/(R+0.)
print mu
s2 = 2*mu*(1-mu)/R
print s2
y_true = (p**r)*(q**(R-r))
#y_true = y_true/(max(y_true))

y_appr = np.exp(-(p-mu)**2/s2)
y_appr = y_appr/max(y_appr)
'''


def expon (p, r, R):
	m = min (r, R-r)
	print r, R-r, m
	y = np.ones(len(p))
	for i in np.arange(m):
		y = y*p*(1-p)
		y = y/max(y)
	y = y*(p**(r-m))*((1-p)**(R-r-m))
	y = y/max(y)
	return y


if 0:
	p = np.linspace (0, 1, 1000)
	R = 25
	r = 5

	y = expon (p=p, r=r, R=R)

	mu = r/(R+0.)
	s2 = 2*mu*(1-mu)/R
	print mu, s2**0.5

	y_appr = np.exp(-(p-mu)**2/s2)
	y_appr = y_appr/max(y_appr)

	plt.figure(figsize=(10, 5))
	plt.plot (p, y, 'crimson', linewidth=5)
	plt.plot (p, y_appr, 'RoyalBlue', linewidth = 2)
	plt.xlabel ('p', fontsize=20)
	plt.ylabel ('p^r * (1-p)^(R-r)', fontsize=20)
	plt.axis ([0,1,0,1])
	plt.show()

if 1:

	for r in np.linspace (250, 750, 20):
		th = np.linspace (0, 2*np.pi, 1000)
		V = 0.15
		x = V*np.cos(th)
		pm = 0.1
		R = 5000
		#r = 75

		'''
		p = (1+x)
		q = (1-(pm/(1+pm))*x)
		#plt.plot (th*180/np.pi, p, linewidth=2)
		#plt.show()
		y = (p**r)*(q**(R-r))
		y  =y/max(y)
		'''
		p = pm*(1+x)
		y = expon (p=p, r=r, R=R)

		mu = r/(R+0.)
		s2 = mu*(1-mu)/float(R)
		print mu, s2**0.5

		y_appr = np.exp(-(p-mu)**2/(2*s2))
		y_appr = y_appr/max(y_appr)


		plt.figure(figsize=(10, 5))
		plt.plot (th*180/np.pi, y, 'crimson', linewidth=5)
		plt.plot (th*180/np.pi, y_appr, 'RoyalBlue', linewidth = 2)
		plt.xlabel ('theta [deg]', fontsize=20)
		plt.ylabel ('p^r * (1-p)^(R-r)', fontsize=20)
		plt.axis ([0,360,0,1])
		plt.title ('r = '+str(int(r)), fontsize=22)
		plt.show()

		'''
		try:
			#y2a = np.exp(-r**2/(2*s2))*np.exp(-R**2/(2*s2))*np.exp(pm*r/(2*s2*R))
			th0 = np.arccos((1./V)*((r/(pm*R))-1))
			ss2 = s2*R**2/(((V*pm*R)**2)-((r-pm*R)**2))
			print th0*180/np.pi, ss2, (1./V)*((r/(pm*R))-1)
			y2 = np.exp(-(1/(2*ss2))*(th-th0)**2)+np.exp(-(1/(2*ss2))*(th+th0)**2)+np.exp(-(1/(2*ss2))*(th-(2*np.pi-th0))**2)
			y2 = y2/max(y2)

			plt.figure(figsize=(10, 5))
			plt.plot (th*180/np.pi, y2, 'c', linewidth=5)
			plt.plot (th*180/np.pi, y_appr, 'RoyalBlue', linewidth = 2)
			plt.xlabel ('theta [deg]', fontsize=20)
			plt.ylabel ('p^r * (1-p)^(R-r)', fontsize=20)
			plt.axis ([0,360,0,1])
			plt.title ('r = '+str(int(r)), fontsize=22)
			plt.show()
		except:
			pass
		'''
