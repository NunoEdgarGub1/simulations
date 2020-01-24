
import numpy as np
import pylab as plt

wl = 900e-9
n_sub = 2.5
n_sil = 1.5

NA = np.linspace (0,1,1000)
theta_no_sil = np.arcsin(NA/n_sub)
theta_sil = np.arcsin(n_sil*NA/n_sub)

#eta_nosil_hor = (3/8.)*((np.cos(theta_no_sil)**3)/12. - np.cos(theta_no_sil))
#eta_sil_hor = (3/8.)*((np.cos(theta_sil)**3)/12. - np.cos(theta_sil))
eta_nosil_ver_p = (3/8.)*((np.cos(theta_no_sil)**3)/4. - np.cos(theta_no_sil))
eta_sil_ver_p = (3/8.)*((np.cos(theta_sil)**3)/4. - np.cos(theta_sil))
eta_nosil_ver_s = (3/8.)*((np.cos(theta_no_sil)**3)/12. - np.cos(theta_no_sil))
eta_sil_ver_s = (3/8.)*((np.cos(theta_sil)**3)/12. - np.cos(theta_sil))
eta_nosil_hor_s = -(3/8.)*((np.cos(theta_no_sil)**3)/12. + 9*np.cos(theta_no_sil)/12.)
eta_sil_hor_s = -(3/8.)*((np.cos(theta_sil)**3)/12. + 9*np.cos(theta_sil)/12.)

#plt.figure()
#plt.plot (NA, theta_no_sil)
#plt.plot (NA, theta_sil)
#plt.show()

plt.figure(figsize = (8,8))
plt.plot (NA, 1-eta_nosil_ver_s/eta_nosil_ver_s[0], linewidth =2, label = 'no SIL, V')
plt.plot (NA, 1-eta_sil_ver_s/eta_sil_ver_s[0], linewidth =2, label = 'SIL, V')
plt.plot (NA, 1-eta_nosil_hor_s/eta_nosil_hor_s[0], linewidth =2, label = 'no SIL, H')
plt.plot (NA, 1-eta_sil_hor_s/eta_sil_hor_s[0], linewidth =2, label = 'SIL, H')
plt.xlabel ('numerical aperture', fontsize=15)
plt.ylabel ('collected emission', fontsize = 15)
plt.title ('Absolute collection efficiency')
plt.legend(loc=2)
plt.show()

plt.figure(figsize = (8,8))
plt.plot (NA, (1-eta_sil_ver_s/eta_sil_ver_s[0])/(1-eta_nosil_ver_s/eta_nosil_ver_s[0]), linewidth =3, label = 'V')
plt.plot (NA, (1-eta_sil_hor_s/eta_sil_hor_s[0])/(1-eta_nosil_hor_s/eta_nosil_hor_s[0]), linewidth =3, label = 'H')
plt.xlabel ('numerical aperture', fontsize=15)
plt.ylabel ('collected emission', fontsize = 15)
plt.title ('Improvement with SIL')
plt.legend(loc=2)
plt.show()







