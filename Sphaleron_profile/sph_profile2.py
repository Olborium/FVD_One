import os
import numpy as np
import matplotlib.pyplot as plt

SIZE = float(sys.argv[1])
N = int(sys.argv[2])
DT = float(sys.argv[3])
TEMP = float(sys.argv[4])
DISTR = sys.argv[5]
INTEG = sys.argv[6]
TIME_SPAN = float(sys.argv[7])
RECTIME = float(sys.argv[8])
RECPOINTS = int(sys.argv[9])
DX = SIZE/(N-1)

Phi_stack = []
#Chi_stack = []

path = 'Out/sph_profile'
files = os.listdir(path)

for filename in files:
    if 'sph(t)_'+f'{TEMP:.6f}' in filename:
        filepath = os.path.join(path, filename)
        Phi_stack += [np.loadtxt(filepath)]
#    if 'sph_dot(t)_'+f'{TEMP:.6f}' in filename:
#        filepath = os.path.join(path, filename)
#        Chi_stack += [np.loadtxt(filepath)]         

Phi_av = np.loadtxt('Out/Phi_av_T='+f'{TEMP:.6f}'+'.txt')
E_pot_long_av = np.loadtxt('Out/E_pot_long_av_T='+f'{TEMP:.6f}'+'.txt')

def phi_s(x, x0=0):
    return np.sqrt(2)/np.cosh(x-x0-SIZE/2)

def phi_s1(x, T=TEMP, x0=0):
    mth = np.sqrt(1 - 3*T/2)
    return mth*np.sqrt(2)/np.cosh(mth*(x-x0-SIZE/2))

x = np.linspace(0, SIZE, N)
t = np.linspace(0, RECTIME, RECPOINTS+1)

i_sph = np.argmax(E_pot_long_av)

Variance = np.zeros(N)
for i in range(N):
    for j in range(len(Phi_stack)):
        Variance[i] += (Phi_stack[j][i_sph][i] - Phi_av[i_sph][i])**2
    Variance[i] = np.sqrt(Variance[i])/np.sqrt(len(Phi_stack))/np.sqrt(len(Phi_stack))

""""""""""""""""""""""""""
"""        PLOTS       """
""""""""""""""""""""""""""

""" Plot potential energy of long modes """

fig, ax = plt.subplots()
ax.plot(t, E_pot_long_av,'black')
ax.scatter(t[i_sph],E_pot_long_av[i_sph],20,'black',marker='o')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$t$',fontsize=14)
ax.set_ylabel(r'$U_{av.}$',fontsize=16,rotation='horizontal')
plt.hlines(E_pot_long_av[i_sph],0,4,'black',linestyle='dashed',linewidth=1)
plt.vlines(t[i_sph],-4,4,'black',linestyle='dashed',linewidth=1)
ax.yaxis.set_label_coords(-0.02,1.01)
plt.xlim((0,3))
plt.ylim((-2.9,3))
plt.savefig('U_av_T='+f'{TEMP:.6f}'+'.pdf')
plt.clf()

""" Plot potential energy of long modes history """

fig, ax = plt.subplots()
ax.plot(t, E_pot_long_av,'black')
ax.scatter(t[i_sph],E_pot_long_av[i_sph],50,'black',marker='o')
ax.scatter(t[i_sph+10],E_pot_long_av[i_sph+10],50,'blue',marker='s')
ax.scatter(t[i_sph-10],E_pot_long_av[i_sph-10],50,'green',marker='v')
ax.scatter(t[i_sph-17],E_pot_long_av[i_sph-17],50,'magenta',marker='D')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$t$',fontsize=14)
ax.set_ylabel(r'$U_{av.}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.01)
plt.xlim((0,3.4))
plt.ylim((-2.9,2.4))
plt.savefig('sph_profile_Ulong_av_T='+f'{TEMP:.6f}'+'.pdf')
plt.clf()

""" Plot sphaleron history """

fig, ax = plt.subplots()
ax.plot(x,Phi_av[i_sph],'black',linewidth=2)
ax.scatter(x[N//2],Phi_av[i_sph][N//2],50,'black',marker='o')
ax.plot(x,Phi_av[i_sph+10],'blue',linewidth=1)
ax.scatter(x[N//2],Phi_av[i_sph+10][N//2],50,'blue',marker='s')
ax.plot(x,Phi_av[i_sph-10],'green',linewidth=1)
ax.scatter(x[N//2],Phi_av[i_sph-10][N//2],50,'green',marker='v')
ax.plot(x,Phi_av[i_sph-17],'magenta',linewidth=1)
ax.scatter(x[N//2],Phi_av[i_sph-17][N//2],50,'magenta',marker='D')
ax.plot(x,phi_s(x),'red',linestyle='dashed')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$x$',fontsize=14)
ax.set_ylabel(r'$\varphi$',fontsize=14,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.01)
plt.ylim((-0.03,2.2))
plt.xlim((46,54))
plt.savefig('sph_profiles_T='+f'{TEMP:.6f}'+'.pdf')
plt.clf()

""" Plot sphaleron """

fig, ax = plt.subplots()
ax.plot(x-50,Phi_av[i_sph],'black',linewidth=2)
ax.plot(x-50,phi_s(x),'red',linestyle='dashed')
ax.plot(x-50,phi_s1(x,0.1),'blue',linestyle='dashdot')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$x$',fontsize=14)
ax.set_ylabel(r'$\varphi$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.04)
plt.ylim((-0.1,1.55))
plt.xlim((-5.9,5.9))
plt.savefig('sph_profile_T='+f'{TEMP:.6f}'+'.pdf')
plt.clf()

""" Plot sphaleron with the variance band """

fig, ax = plt.subplots()
#ax.set_title(r'$T=0.1$',fontsize=16)
ax.plot(x-50,Phi_av[i_sph],'black',linewidth=0.5)
ax.plot(x-50,phi_s(x),'red',linestyle='dashed')
ax.plot(x-50,Phi_av[i_sph]+Variance,'black',linewidth=0.5)
ax.plot(x-50,Phi_av[i_sph]-Variance,'black',linewidth=0.5)
ax.fill_between(x-50,Phi_av[i_sph]+Variance,Phi_av[i_sph]-Variance, color='black',alpha=1)
ax.plot(x-50,phi_s1(x,0.1),'blue',linestyle='dashdot')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$x$',fontsize=14)
ax.set_ylabel(r'$\varphi$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.04)
plt.ylim((-0.35,1.7))
plt.xlim((-5.9,5.9))
plt.savefig('sph_profile_band_Lang_T='+f'{TEMP:.6f}'+'.pdf')
plt.clf()

""" Plot theoretical sphaleron """

fig, ax = plt.subplots()
ax.plot(x-50,phi_s(x),'black')
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$mx$',fontsize=16)
ax.set_ylabel(r'$\varphi$',fontsize=18,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.04)
plt.ylim((-0.05,1.57))
plt.xlim((-5.9,5.9))
plt.savefig('sph_profile_th.pdf')
plt.clf()