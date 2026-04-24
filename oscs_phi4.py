#%%
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

#%%
#SIZE = 100
#N_x = 2048
#DX = SIZE/N_x

eta = 0

TempList = [0.05]
M = len(TempList)

decays_tot = np.zeros(M)
decays_osc = np.zeros(M)

path = 'Phi4'
files = os.listdir(path)

times = []
oscs = []

for i in range(M):

    TEMP = TempList[i]
    times_i_list = []
    oscs_i_list = []
    temp_str = f'{TEMP:.6f}'
    eta_str = f'{eta:.6f}'
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

#%%
""" Plot the distribution of half-oscillations """

J = 0
oscs_J = oscs[J][times[J]>0] / 4.0
values, counts = np.unique(oscs_J, return_counts=True)
abundance = counts / len(oscs_J)

#%%

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(values[0], abundance[0], width=0.4, color='blue', alpha=0.7)
ax.bar(values[1:], abundance[1:], width=0.4, color='red', alpha=0.7)
ax.set_xlabel(r'$J$', fontsize=14)
ax.set_ylabel(r'$n_J$', fontsize=14)
ax.grid(True, axis='y', linestyle='dashed', linewidth=0.5, color='grey')
ax.tick_params(axis='both', which='both', right=True, top=True, direction='in', labelsize=12)
plt.xlim((-0.2,7.2))
#plt.savefig('Phi4_Nosc_T=0.05.pdf', bbox_inches='tight')
plt.show()

#%%
""" Plot the time distribution of turnarounds """

J = 0
K = len(times[J])
times_osc = np.zeros(K)
times_per = np.zeros(K)
for k in range(K):
    if oscs[j][k] == 1:
        times_osc[k] = times[j][k]
    else:
        times_per[k] = times[j][k]
for k in range(K):
    if times_osc[k] == 0:
        times_osc[k] = 200
    if times_per[k] == 0:
        times_per[k] = 200

#%%

fig, ax = plt.subplots(figsize=(6,4))
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.set_xlabel(r'$mt$',fontsize=18)
ax.set_ylabel(r'$n$',fontsize=18)
ax.hist(times_osc,1000,weights=np.ones(J)/len(times[j]),alpha=0.7,color='red')
ax.hist(times_per,1000,weights=np.ones(J)/len(times[j]),alpha=0.7,color='blue')
plt.xlim((0,80))
plt.ylim((1e-4,0.01))
plt.yscale('log')
#plt.savefig('Phi4DensityT=0.05.pdf', bbox_inches='tight')
plt.show()
