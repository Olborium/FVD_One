# Stochastic integrator test:
# Measures the properties of the power spectrum of phi and pi
# Makes plots from Appendix B.3 

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SIZE = float(sys.argv[1])
N_x = int(sys.argv[2])
DT = float(sys.argv[3])
N_ENS = int(sys.argv[4])
N_SAMPLE = int(sys.argv[5])
TIME_SPAN = float(sys.argv[6])
DISTR = sys.argv[7]
INTEG = sys.argv[8]
TEMP = float(sys.argv[9])
ETA = float(sys.argv[10])
DX = SIZE/N_x
COUPL = float(sys.argv[11])

power_spectrum_phi = np.loadtxt('Out/Out_power_spectrum_Lang/ps_phi_'+f'{ETA:.6f}'+'_'+f'{TEMP:.6f}'+'_'+f'{COUPL:.6f}'+'.txt')
power_spectrum_chi = np.loadtxt('Out/Out_power_spectrum_Lang/ps_chi_'+f'{ETA:.6f}'+'_'+f'{TEMP:.6f}'+'_'+f'{COUPL:.6f}'+'.txt')

t = np.linspace(0, TIME_SPAN*N_SAMPLE, N_SAMPLE)
k = np.linspace(0.1*2*np.pi/SIZE, np.pi/DX, N_x//2)

name =  '_'+f'{ETA:.3f}'+\
        '_'+f'{TEMP:.2f}'+\
        '_'+f'{COUPL:.2f}'

Mth2 = 1+COUPL*3*TEMP/2/np.sqrt(1+DX**2/4)
def sk_thermal(k):
    Ok = np.sqrt(2/DX**2*(1-np.cos(DX*k))+Mth2)
    return np.sqrt(TEMP/DX)/Ok

def sk_bare(k):
    Ok = np.sqrt(2/DX**2*(1-np.cos(DX*k))+1)
    return np.sqrt(TEMP/DX)/Ok

t_min = 5/ETA    # Give it some time to thermalise
N_min = int(t_min/TIME_SPAN)

""" Effective temperature of long modes, k<2m """

def f(x, a):
    return a
k_max = int(SIZE/np.pi)
k_long_range = k[0:k_max]

Teff_l = np.zeros(N_SAMPLE)
Teff_l_e = np.zeros(N_SAMPLE)

for i in range(N_SAMPLE):
    params, cov = curve_fit(f, k_long_range, power_spectrum_chi[i][0:k_max])
    errors = np.sqrt(np.diag(cov))
    Teff_l[i] = params[0]*DX
    Teff_l_e[i] = errors[0]*DX 
(Teff_l_av, Teff_l_av_e) = (0, 0)
for i in range(N_min,N_SAMPLE):
    Teff_l_av += Teff_l[i]
    Teff_l_av_e += Teff_l_e[i]**2
Teff_l_av /= (N_SAMPLE-N_min)
Teff_l_av_e = np.sqrt(Teff_l_av_e)/(N_SAMPLE-N_min)

fig, ax = plt.subplots()
ax.errorbar(t, Teff_l, yerr=Teff_l_e, color='black',fmt='s',ms=1,elinewidth=1,capsize=1)
plt.hlines(TEMP,t[0],t[-1],color='red',linestyles='dashdot')
plt.hlines(Teff_l_av,t[0],t[-1],color='black',linestyles='dashed')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'$T_{eff}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.05)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_title(r'$k<2m$',fontsize=15)
plt.savefig('Teff_long'+name+'.pdf')

plt.clf()

""" Temperature of short modes """

k_short_range = k[k_max:]

Teff_s = np.zeros(N_SAMPLE)
Teff_s_e = np.zeros(N_SAMPLE)

for i in range(N_SAMPLE):
    params, cov = curve_fit(f, k_short_range, power_spectrum_chi[i][k_max:N_x//2])
    errors = np.sqrt(np.diag(cov))
    Teff_s[i] = params[0]*DX
    Teff_s_e[i] = errors[0]*DX 
(Teff_s_av, Teff_s_av_e) = (0, 0)
for i in range(N_min,N_SAMPLE):
    Teff_s_av += Teff_s[i]
    Teff_s_av_e += Teff_s_e[i]**2
Teff_s_av /= (N_SAMPLE-N_min)
Teff_s_av_e = np.sqrt(Teff_s_av_e)/(N_SAMPLE-N_min)

fig, ax = plt.subplots()
ax.errorbar(t, Teff_s, yerr=Teff_s_e, color='black',fmt='s',ms=1,elinewidth=1,capsize=1)
plt.hlines(TEMP,t[0],t[-1],color='red',linestyles='dashdot')
plt.hlines(Teff_s_av,t[0],t[-1],color='black',linestyles='dashed')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'$T_{eff}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.05)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_title(r'$k>2m$',fontsize=15)
plt.savefig('Teff_short'+name+'.pdf')

plt.clf()

""" Average power spectrum """

ps_phi_av = np.zeros(N_x//2)
for j in range(N_x//2):
    for i in range(N_min,N_SAMPLE):
        ps_phi_av[j] += power_spectrum_phi[i][j]
    ps_phi_av[j] /= (N_SAMPLE-N_min)

plt.plot(k, ps_phi_av[0:N_x//2], 'black')
plt.plot(k, sk_thermal(k)**2, 'red',linestyle='dashed')
plt.plot(k, sk_bare(k)**2, 'blue',linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$\langle\tilde\varphi_k\rangle^2$')
plt.savefig('ps_av'+name+'.pdf')

plt.clf()

""" Dispersion relation and thermal mass """

j_trunc = 30
def Omega2(k, mTh2):
    return mTh2 + k**2
ps_ratio = np.zeros((N_SAMPLE,j_trunc))
for i in range(N_SAMPLE):
    ps_ratio[i][0:j_trunc] = power_spectrum_chi[i][0:j_trunc]/power_spectrum_phi[i][0:j_trunc]
ps_ratio_av = np.zeros(j_trunc)
for j in range(j_trunc):
    for i in range(N_min,N_SAMPLE):
        ps_ratio_av[j] += ps_ratio[i][j]
    ps_ratio_av[j] /= (N_SAMPLE-N_min)
params, cov = curve_fit(Omega2, k[0:j_trunc], ps_ratio_av[0:j_trunc])
errors = np.sqrt(np.diag(cov))

fig, ax = plt.subplots()
ax.plot(k[0:j_trunc],ps_ratio_av[0:j_trunc],'black')
ax.plot(k[0:j_trunc],params[0]+k[0:j_trunc]**2,'black',linestyle='dashed')
ax.plot(k[0:j_trunc],Mth2+k[0:j_trunc]**2,'red',linestyle='dashdot')
ax.plot(k[0:j_trunc],1+k[0:j_trunc]**2,'blue',linestyle='dashdot')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.set_xlabel(r'$k$',fontsize=16)
ax.set_ylabel(r'$\Omega^2$',fontsize=16,rotation='horizontal')
ax.set_xlim((0,1.25))
ax.set_ylim((0.9,2.4))
ax.yaxis.set_label_coords(-0.02,1.05)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
plt.savefig('meff'+name+'.pdf')

""" Print numbers """

print('Thermal mass prediction:      ', f'{Mth2:.3f}')
print('Thermal mass measurement:     ', f'{params[0]:.3f}')
print('Measurement error:            ', f'{errors[0]:.3f}')
print('Temp. of long modes:          ', f'{Teff_l_av:.3f}')
print('Temp. of short modes:         ', f'{Teff_s_av:.3f}')