#%%
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

#%%
#SIZE = 100
#N_x = 4096
#DX = SIZE/N_x

eta = 0.1
TempList = [0.11, 0.105, 0.1, 0.095,  0.09, 0.085, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]
M = len(TempList)

decays_tot = np.zeros(M)
decays_osc = np.zeros(M)

path = 'Phi4_Dissipative'
files = os.listdir(path)

times = []
oscs = []

for i in range(M):

    TEMP = TempList[i]
    times_i_list = []
    oscs_i_list = []
    temp_str = f'{TEMP:.6f}'
    eta_str = f'{eta:.6f}'
    times_pattern = re.compile(rf'^times_{re.escape(temp_str)}_{re.escape(eta_str)}_(-?\d+)\.txt$')
    osc_pattern = re.compile(rf'^osc_counter_{re.escape(temp_str)}_{re.escape(eta_str)}_(-?\d+)\.txt$')

    osc_files = {}
    for filename in files:
        match = osc_pattern.match(filename)
        if match:
            osc_files[match.group(1)] = filename

    for filename in sorted(files):

        match = times_pattern.match(filename)
        if not match:
            continue

        seed = match.group(1)
        osc_filename = osc_files.get(seed)
        if osc_filename is None:
            print(f"Missing osc_counter file for TEMP={temp_str}, seed={seed}")
            continue

        times_filepath = os.path.join(path, filename)
        osc_filepath = os.path.join(path, osc_filename)   
        
        times_i_list.append(np.loadtxt(times_filepath))
        oscs_i_list.append(np.loadtxt(osc_filepath))

    times += [np.concatenate(times_i_list)]
    oscs += [np.concatenate(oscs_i_list)]

    decays_tot[i] = np.count_nonzero(times[i])/len(times[i])
    decays_osc[i] = np.count_nonzero(times[i][oscs[i] == 1.0])/len(times[i])

#%%
""" Jackknife estimate of errors """

vars = np.zeros(M)

for i in range(M):
    N = len(times[i])//2
    dectot = decays_tot[i]
    dec1 = np.count_nonzero(times[i][1:N])/N
    dec2 = np.count_nonzero(times[i][N:])/N
    vars[i] = np.sqrt(0.5*((dectot-dec1)**2 + (dectot-dec2)**2))  
#%%
""" Langer's turnaround """

omega = np.sqrt(3)
lmbd = eta/2 - np.sqrt(omega**2+eta**2/4)
P_minus_plus = 0.5*(1.0 - np.abs(lmbd)/omega)

#%%
""" Plot turnarounds """

Es = 4/3
TempListR = np.array(TempList)/Es
x = np.linspace(0.0005,0.13,1000)
fig, ax = plt.subplots(figsize=(6,4))
plt.errorbar(TempListR,decays_tot,vars,color='black', fmt='o', elinewidth=1, capsize=3)
plt.scatter(TempListR,decays_osc,color='red',marker='s')
plt.scatter(TempListR,decays_tot-decays_osc,color='blue',marker='^')
ax.set_xlabel(r'$T/E_s$',fontsize=18)
ax.set_ylabel(r'$P_{-+}$',fontsize=18)
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=16)
plt.xscale('log')
plt.yscale('log')
plt.hlines(P_minus_plus,0.0001,0.15,color='black',linestyles='dashed')
plt.text(0.0008,0.2,r'$\eta/m=10^{-1}$',fontsize=16)
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.grid(True, which='major', linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, which='minor', linestyle='dashed', linewidth=0.5, color='grey')
plt.xlim((0.0005,0.15/Es))
plt.ylim((0.0001,1))
#plt.savefig('returns_phi4_eta=0.1.pdf', bbox_inches='tight')
plt.show()
