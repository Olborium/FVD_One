import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

L = float(sys.argv[1])
N = int(sys.argv[2])
DT = float(sys.argv[3])
dx = L/(N-1)

path = 'Out/Out'

files = os.listdir(path)

Times = []

TempList = [0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]
TimeOfSimList = [15000, 10000, 10000, 10000, 10000, 7500, 7500, 7500, 7500, 7500, 7500]

M = len(TempList)

""" Read data """

for i in len(TempList):
    T = TempList[i]
    timesList = []
    NumOfSim = 0
    st = '_'+f'{DT:.3f}'+'_'+f'{L:.1f}'+'_'+f'{N}'+'_'+f'{TimeOfSimList[i]}'
    for filename in files:
        if re.match('Times_'+f'{T:.3f}',filename) and re.search(st, filename):
            filepath = os.path.join(path, filename)
            timesList += [np.loadtxt(filepath)]
    for i in range(0,len(timesList)):
        NumOfSim += len(timesList[i])
    times = np.zeros(NumOfSim)
    k = 0
    for i in range(0,len(timesList)):
        for j in range(0,len(timesList[i])):
            times[k] = timesList[i][j]
            k += 1    
    Times += [times]

""" Find survival probability """

t_points = 10000
t_l = [None]*M
Psurv = np.zeros((M,t_points))
for j in range(M):
    t_l[j] = np.linspace(0,TimeOfSimList[j]-0.1,t_points)
    for i in range(0,len(Times[j])):
        if Times[j][i] == 0:
            Times[j][i] = TimeOfSimList[j]
    Times[j].sort()
for j in range(M):
    for i in range(0,t_points):
        Psurv[j,i] = (Times[j]>t_l[j][i]).sum()/len(Times[j])

""" Split into vertical segments, approximate each segment by a line """

log_Psurv = np.log(Psurv)
Offset_t = 0                         # Exclude region t < Offset_t
#N_segments = np.zeros(M,dtype=int)
log_P_threshold = 0.16               # Max decay fraction
N_segments = 8
Gamma_t_list = [None]*M
Gamma_t_errors_list = [None]*M
time_points = [None]*M
num_of_events = [None]*M
for i in range(M):
    b = []
    b_s = []
    t_middle = []
    num_i = []
    log_P_min = -log_Psurv[i][np.where(t_l[i]>Offset_t)[0][0]]
    log_P_max = log_P_threshold
    Delta_P = (log_P_max-log_P_min)/N_segments
#    Delta_P = 0.05
#    N_segments[i] = math.floor((log_P_max-log_P_min)/Delta_P)
    for j in range(N_segments):
        ind = np.where((-log_Psurv[i]>log_P_min+j*Delta_P)&(-log_Psurv[i]<log_P_min+(j+1)*Delta_P))[0]
        num = ((Times[i]>t_l[i][ind[0]])&(Times[i]<t_l[i][ind[-1]])).sum()
        num_i += [num]        
        y1 = log_Psurv[i,ind[0]]
        y2 = log_Psurv[i,ind[-1]]
        t1 = t_l[i][ind[0]]
        t2 = t_l[i][ind[-1]]
        slope = -(y2-y1)/(t2-t1)
        t_middle += [(t1+t2)/2]
        b += [slope]
        b_s += [slope/np.sqrt(num)]
    num_of_events[i] = [num_i]    
    Gamma_t_list[i] = np.array(b)
    Gamma_t_errors_list[i] = np.array(b_s)
    time_points[i] = np.array(t_middle)  

""" Extrapolate the tilt of the lines to t=0 """

# def log_Gamma_fit(t, a, b, c):
#     return a - b*t - c*t**2
def log_Gamma_fit(t, a, b):
    return a - b*t
log_Gamma_at_zero = np.zeros(M)
slope = np.zeros(M)
#curv = np.zeros(M)
log_Gamma_at_zero_error = np.zeros(M)
for i in range(M):
    params, cov = curve_fit(log_Gamma_fit, time_points[i][1:], np.log(Gamma_t_list[i][1:]), sigma=Gamma_t_errors_list[i][1:]/Gamma_t_list[i][1:], absolute_sigma=True)
    log_Gamma_at_zero[i] = params[0]
    slope[i] = params[1]
#    curv[i] = params[2]
    log_Gamma_at_zero_error[i] = np.sqrt(cov[0,0])    

""" Fit: sphaleron energy """

rf = 5

T0_inv=TempList[rf]**(-1)

def log_Gamma_th(T_inv, Es):
    return -Es*(T_inv-T0_inv)

param, cov = curve_fit(log_Gamma_th, np.array(TempList)**(-1), log_Gamma_at_zero-log_Gamma_at_zero[rf], sigma=np.sqrt(log_Gamma_at_zero_error**2+log_Gamma_at_zero_error[rf]**2)/2, absolute_sigma=True)

Es = param[0]
Es_error = np.sqrt(cov[0,0])

chi2 = 0
for i in range(M):
    a = log_Gamma_th(TempList[i]**(-1),Es)
    b = log_Gamma_at_zero[i]-log_Gamma_at_zero[rf]
    c = np.sqrt(log_Gamma_at_zero_error[i]**2+log_Gamma_at_zero_error[rf]**2)/2
    chi2 += ((a-b)/c)**2
chi2 /= M

print('energy:')
print(Es, Es_error, chi2)

""" Fit: prefactor """

Es_th = 4/3
T_middle = 0.5*(TempList[0]+TempList[-1])
log_A_th = np.log(6*L/np.pi*np.sqrt(Es_th/2/np.pi/T_middle))

def log_Gamma_th2(T_inv, log_A):
    return log_A - Es_th*T_inv

param, cov = curve_fit(log_Gamma_th2, np.array(TempList)**(-1), log_Gamma_at_zero, sigma=log_Gamma_at_zero_error, absolute_sigma=True)

log_A = param[0]
log_A_error = np.sqrt(cov[0,0])

chi2 = 0
for i in range(M):
    a = log_Gamma_th2(TempList[i]**(-1),log_A)
    b = log_Gamma_at_zero[i]
    c = log_Gamma_at_zero_error[i]
    chi2 += ((a-b)/c)**2
chi2 /= M

print('log of Prefactor:')
print(log_A-np.log(L), log_A_error, chi2)
print('ratio:')
print(np.exp(log_A_th)/np.exp(log_A))

""" Plot Prefactor """

Tinv = np.linspace(TempList[0]**(-1),TempList[-1]**(-1),1000)
fig, ax = plt.subplots()
ax.errorbar(np.array(TempList)**(-1),log_Gamma_at_zero+Es_th*np.array(TempList)**(-1)-np.log(L),log_Gamma_at_zero_error,color='black', fmt='o', elinewidth=1, capsize=3)
ax.plot(Tinv,log_Gamma_th2(Tinv,log_A)+Es_th*Tinv-np.log(L),'black')
plt.xlabel(r'$T^{-1}$',fontsize=16)
ax.set_ylabel(r'$\ln\: \Gamma+E_sT^{-1}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(0, 1.05)
ax.tick_params(axis='both', which='both',right=True,top=True, direction='in', labelsize=13)
ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
plt.text(10,-0.55,r'$\ln \:A = -0.815\pm 0.032$',fontsize=14)
plt.savefig('prefactor.pdf')
plt.show()

""" Plot survival probability: example """

t = np.linspace(0, 5000, 1000)
fig, ax = plt.subplots()
plt.xlim((0,7500))
ax.set_xticks([1e3,2e3,3e3,4e3,5e3,6e3,7e3])
ax.set_xticklabels([None,r'$2\cdot 10^3$',None,r'$4\cdot 10^3$',None,r'$6\cdot 10^3$',None])
for tick in [2e3,4e3,6e3]:
    ax.axvline(x=tick, linestyle='dashed', linewidth=0.5, color='grey')
ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
ax.tick_params(axis='both', which='both',right=True,top=True, direction='in', labelsize=13)
for i in [0,5,10]:
    imin = np.max(np.where(t_l[i]<6000))
    imax = np.max(np.where(t_l[i]<7500))
    a = (np.log(Psurv[i,imax])-np.log(Psurv[i,imin]))/1500
    b = np.log(Psurv[i,imax]) - a*7500
    x = np.linspace(0,7500,10000)
    ax.plot(t_l[i], np.log(Psurv[i,:]),'black')
    ax.plot(x, a*x+b, 'grey',linewidth=1)
plt.xlabel(r'$t$',fontsize=14)
ax.set_ylabel(r'$\ln\: P_{surv}$',fontsize=14,rotation='horizontal')
ax.yaxis.set_label_coords(-0.05, 1.05)
plt.savefig('P_surv(t)_ex.pdf')
plt.clf()

""" Plot the decay rate extrapolation lines """

fig, ax = plt.subplots()
clr = ['black', 'blue', 'red']
shp = ['o', 'v', 's']
j = 0
for i in [0,6,10]:
    ax.errorbar(time_points[i][1:-1], np.log(Gamma_t_list[i][1:-1]), Gamma_t_errors_list[i][1:-1]/Gamma_t_list[i][1:-1], color=clr[j], fmt=shp[j], elinewidth=1,capsize=3)
    ax.errorbar([7500],[-10.5+j*0.3],[0],color=clr[j], fmt=shp[j], elinewidth=1,capsize=3)
    plt.text(8000,-10.55+j*0.3,r'$T=$'+f'{TempList[i]}',fontsize=13)
    ax.plot(t_l[i],log_Gamma_fit(t_l[i],log_Gamma_at_zero[i],slope[i]),color=clr[j],linewidth=0.5)
    j += 1
plt.xlabel(r'$t$',fontsize=16)
plt.xlim((0,11000))
plt.ylim((-11.7,-9.4))
ax.set_ylabel(r'$\ln\:\Gamma(t)$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.01, 1.03)
ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
ax.tick_params(axis='both', which='both',right=True,top=True, direction='in', labelsize=13)
plt.savefig('Gamma(t).pdf')
plt.clf()    

""" Plot the resulting fit """

Tinv = np.linspace(TempList[0]**(-1),TempList[-1]**(-1),1000)
fig, ax = plt.subplots()
ax.plot(Tinv, log_Gamma_th(Tinv,Es), 'black', linewidth=2)
ax.plot(Tinv, log_Gamma_th2(Tinv,log_A_th)-log_Gamma_th2(TempList[rf]**(-1),log_A_th), 'red', linestyle='dashed', linewidth=3)
plt.errorbar(np.array(TempList)**(-1), log_Gamma_at_zero-log_Gamma_at_zero[rf], np.sqrt(log_Gamma_at_zero_error**2+log_Gamma_at_zero_error[rf]**2)/2, color='black', fmt='o', elinewidth=1, capsize=3)
plt.xlabel(r'$T^{-1}$',fontsize=16)
ax.set_ylabel(r'$\ln\: (\Gamma(T)/\Gamma(T_*))$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(0, 1.05)
ax.tick_params(axis='both', which='both',right=True,top=True, direction='in', labelsize=13)
ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
ax.xaxis.grid(True, linestyle='dashed', linewidth=0.5, color='grey')
plt.text(10.6,0.52,r'$E_s^{(exp)}=1.329\pm 0.087$',fontsize=14)
plt.savefig('E_sph1.pdf')
plt.clf()