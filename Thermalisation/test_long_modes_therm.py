import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SIZE = float(sys.argv[1])
N = int(sys.argv[2])
DT = float(sys.argv[3])
N_ENS = int(sys.argv[4])
N_SAMPLE = int(sys.argv[5])
TIME_SPAN = float(sys.argv[6])
DISTR = sys.argv[7]
INTEG = sys.argv[8]
TEMP = float(sys.argv[9])
ETA = float(sys.argv[10])
DX = SIZE/(N-1)

power_spectrum_chi_12 = np.loadtxt('Out/Out_therm_Lang/therm_ps_chi_'+f'{TEMP:.6f}'+'_'+f'{ETA:.6f}'+'_12.txt')
power_spectrum_chi_22 = np.loadtxt('Out/Out_therm_Lang/therm_ps_chi_'+f'{TEMP:.6f}'+'_'+f'{ETA:.6f}'+'_22.txt')

t = np.linspace(0, TIME_SPAN*N_SAMPLE, N_SAMPLE)
k = np.linspace(0, np.pi/DX, N_x//2)

ps_chi_tot = []
ps_chi_tot += [power_spectrum_chi_12]
ps_chi_tot += [power_spectrum_chi_22]

name =  '_'+f'{TEMP:.3f}'

Teff = np.zeros((2,N_SAMPLE))
Teff_e = np.zeros((2,N_SAMPLE))

for j in range(2):

    def f(x, a):
        return a

    k_max = int(SIZE/np.pi)
    k_long_range = np.arange(k_max)

    for i in range(N_SAMPLE):
        params, cov = curve_fit(f, k_long_range, ps_chi_tot[j][i][0:k_max])
        errors = np.sqrt(np.diag(cov))
        Teff[j][i] = params[0]*DX
        Teff_e[j][i] = errors[0]*DX    

colors = ['black', 'blue']
shapes = ['o', 's']
fig, ax = plt.subplots()
plt.yscale('log')
for j in [0,1]:
    plt.errorbar(t, Teff[j], yerr=Teff_e[j], color=colors[j],fmt=shapes[j],ms=3,markerfacecolor='none',elinewidth=1,capsize=1)
ax.set_yticks([5e-2,1e-1,2e-1])
ax.set_yticklabels([r'$5\times 10^{-2}$',r'$10^{-1}$',r'$2\times 10^{-1}$'])
for tick in [5e-2,1e-1,2e-1]:
    ax.axhline(y=tick, linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
plt.xscale('log')
plt.xlim((7,5e3))
plt.ylim((0.035,0.29))
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'$T_{\rm eff}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.05)
plt.text(1e3,2e-1,r'$\eta=$'+f'{ETA:.3f}',fontsize=15)
plt.savefig('therm_Lang_2m_3.pdf')
plt.clf()