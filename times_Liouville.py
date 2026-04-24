#%%
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

#%%
TempList = [2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
M = len(TempList)

decays_tot = np.zeros(M)
decays_osc = np.zeros(M)

path = 'Liouville'
files = os.listdir(path)

times = []
oscs = []
for i in range(M):
    TEMP = TempList[i]
    times_i_list = []
    oscs_i_list = []
    temp_str = f'{TEMP:.6f}'
    times_pattern = re.compile(rf'^times_{re.escape(temp_str)}_(-?\d+)\.txt$')
    osc_pattern = re.compile(rf'^osc_counter_{re.escape(temp_str)}_(-?\d+)\.txt$')

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
""" Fit of the prompt turnarounds """

def f(t, a, b):
    return a*t + b*t**2

params, cov = curve_fit(f,TempList[:-5],decays_tot[:-5]-decays_osc[:-5])

print(params[0],params[1])

#%%
""" Plot turnarounds """

Es = 24.7738 # critical bubble energy for kappa=0.05
TempListR = np.array(TempList)/Es
x = np.linspace(0.0009/Es,3,1000)
fig, ax = plt.subplots(figsize=(6,4))
plt.errorbar(TempListR,decays_tot,vars,color='black', fmt='o', elinewidth=1, capsize=3)  # total
plt.scatter(TempListR,decays_osc,color='red',marker='s')                                  # sloshing
plt.scatter(TempListR,decays_tot-decays_osc,color='blue',marker='^')                      # prompt
plt.plot(x,x*params[0]*Es+ params[1]*(x*Es)**2,color='blue',linestyle='dashed')
ax.set_xlabel(r'$T/E_s$',fontsize=18)
ax.set_ylabel(r'$R$',fontsize=18)
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=16)
plt.xscale('log')
plt.yscale('log')
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([1e-2, 2e-2, 5e-2, 1e-1])
ax.set_xticklabels([r'$10^{-2}$', r'$2\cdot 10^{-2}$', r'$5\cdot 10^{-2}$', r'$10^{-1}$'])
ax.xaxis.grid(True, which='major', linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, which='minor', linestyle='dashed', linewidth=0.5, color='grey')
plt.xlim((0.008,2.5/Es))
plt.ylim((0.01,1))
#plt.savefig('returns_L_full_2.pdf', bbox_inches='tight')
plt.show()

