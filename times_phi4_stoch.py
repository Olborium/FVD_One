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
#SIZE = 100
#N_x = 4096
#DX = SIZE/N_x

TEMP = 0.01
EtaList  = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

M = len(EtaList)

decays_tot = np.zeros(M)
returns_tot = np.zeros(M)

path = 'Phi4_Dissipative'
files = os.listdir(path)

times = []   # for P_{-+} total
ftimes = []  # for P_{+-} prompt
# oscs = []    # for P_{-+} sloshing

for i in range(M):

    ETA = EtaList[i]
    times_i_list = []
    ftimes_i_list = []
    temp_str = f'{TEMP:.6f}'
    eta_str = f'{ETA:.6f}'
    times_pattern = re.compile(rf'^times_{re.escape(temp_str)}_{re.escape(eta_str)}_(-?\d+)\.txt$')          
    ftimes_pattern = re.compile(rf'^ftimes_{re.escape(temp_str)}_{re.escape(eta_str)}_(-?\d+)\.txt$')           

    for filename in files:

        match = times_pattern.match(filename)
        if not match:
            continue

        times_filepath = os.path.join(path, filename)
        times_i_list.append(np.loadtxt(times_filepath))

     for filename in files:

        match = ftimes_pattern.match(filename)
        if not match:
            continue

        ftimes_filepath = os.path.join(path, filename)
        ftimes_i_list.append(np.loadtxt(ftimes_filepath))       

    times += [np.concatenate(times_i_list)]
    ftimes += [np.concatenate(ftimes_i_list)]

    decays_tot[i] = np.count_nonzero(times[i])/len(times[i])
    returns_tot[i] = (ftimes[i] == 0).sum()/len(ftimes[i])

#%%
""" Plot turnarounds, compare with Langer """

def Langer(eta):
    o1 = np.sqrt(3)
    return (-eta/2+np.sqrt(eta**2/4+o1**2))/o1

x = np.linspace(0.0005,100,10000)
fig, ax = plt.subplots(figsize=(6,4))
plt.scatter(EtaList,1-decays_tot-returns_tot,color='blue', marker='o')
plt.plot(x,Langer(x),color='black',linestyle='dashed')
ax.set_xlabel(r'$\eta/m$',fontsize=18)
ax.set_ylabel(r'$1-R$',fontsize=18)
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=16)
plt.xscale('log')
ax.xaxis.set_major_locator(LogLocator(base=10.0))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.grid(True, which='major', linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, which='minor', linestyle='dashed', linewidth=0.5, color='grey')
plt.xlim((0.8e-3,50))
plt.ylim((0,1))
#plt.savefig('returns_phi4_etas.pdf', bbox_inches='tight')
plt.show()
