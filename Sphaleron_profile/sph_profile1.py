import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft

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

def U(phi):
    En = 0
    for m in range(0,N-1):
        f_x = (phi[np.mod(m+1,N-1)]-phi[np.mod(m,N-1)])/DX
        En += DX*(0.5*f_x**2+0.5*phi[m]**2-0.25*phi[m]**4)
    return En

def Ulong(phi, K):
    Nk = N//2+1
    c = np.sqrt(N-1)
    fk = 1/c*rfft(phi,N)
    for j in range(K,Nk):
        fk[j] = 0
    phi_trunc = c*irfft(fk,N)
    return U(phi_trunc)

def synchronise_stack(Phi_set, Ulong_set):
    K = len(Ulong_set)
    iCross = np.zeros(K)
    for i in range(0,K):
        j = 0
        while Ulong_set[i][j]<0:
            j += 1
        iCross[i] = j
    (jMin, iMin) = (int(min(iCross)), np.argmin(iCross))
    Nt = len(Ulong_set[0])
    a = np.zeros(Nt)
    b = np.zeros((Nt,N))
    iShift = np.zeros(K)
    print('Synchronisation begins...')
    for i in range(0,K):
        if i != iMin:
            MinAchieved = False
            (sOld, sNew) = (1e8, 0)
            jShift = 0
            while MinAchieved == False:
                for j in range(0,jMin):
                    sNew += (Ulong_set[i][j+jShift] - Ulong_set[iMin][j])**2
                if sNew > sOld:
                    MinAchieved = True
                    for j in range(0,Nt-1-jShift):
                        a[j] = Ulong_set[i][j+jShift]
                        b[j,:] = Phi_set[i][j+jShift][:]
                    for j in range(Nt-1-jShift,Nt-1):
                        a[j] = 0
                        b[j,:] = 0
                    Ulong_set[i][:] = np.copy(a[:])
                    for j in range(0,Nt):
                        Phi_set[i][j][:] = b[j,:]
                    iShift[i] = jShift
                else:
                    sOld = sNew
                    sNew = 0
                    jShift += 1		
    print('Synchronisation complete.')
    return (Phi_set, Ulong_set)

def orient_stack(Phi_set):
    for i in range(len(Phi_set)):
        if np.abs(np.min(Phi_set[i][0])) > np.abs(np.max(Phi_set[i][0])):
            for j in range(len(Phi_set[i])):
                Phi_set[i][j][:] = - Phi_set[i][j][:]
    return Phi_set

def centralise_stack(Phi_set, Ulong_set):
    Nset = len(Phi_set)
    Nt = len(Phi_set[0])
    def mock_profile(x):
        return 2*np.exp(-0.5*(x-SIZE/2)**2)
    print('Centralisation begins...')
    for i in range(Nset):
        print('Centralising',i,'th element of the stack...')
        kCross = 0 
        while Ulong_set[i][kCross] < 0:
            kCross += 1
        jMaxIn = np.argmax(Phi_set[i][0])
        for j in range(len(Phi_set[i])):
            Phi_set[i][j] = np.roll(Phi_set[i][j],N//2-jMaxIn)
        jMaxVar = N//40
        sm = np.zeros(2*jMaxVar)
        for j in range(N//2-jMaxVar,N//2+jMaxVar):
            for l in range(N//2-jMaxVar,N//2+jMaxVar):
                sm[-N//2+jMaxVar+j] += (mock_profile(l*DX)-Phi_set[i][20][l-j+N//2])**2
        jMax = np.argmin(sm)
    print('Centralisation complete.')
    return (Phi_set, Ulong_set)

def average_stack(Phi_set):
    Nt = len(Phi_set[0])
    Nsmp = len(Phi_set)
    Phi = np.zeros((Nt,N))
    for j in range(Nt):
        for i in range(Nsmp):
            Phi[j,:] += Phi_set[i][j][:]
        Phi[j,:] /= Nsmp 
    return Phi

def pinpoint(Phi_av):
    """ Maximum of Ulong """
    iMax = 0
    while True:
        if np.max(Phi_av[iMax,:]) < 1:
            if iMax > 3:
                break
        iMax += 1
    s = np.zeros(iMax)
    for i in range(0,iMax):
        for j in range(N//2-int(5/DX),N//2+int(5/DX)):
            s[i] += (Phi_av[i,j] - Phi_av[i+1,j])**2
    iMin = np.argmin(s)	
    return (Phi_av[iMin,:], iMin)

def phi_s(x, x0=0):
    return np.sqrt(2)/np.cosh(x-x0-SIZE/2)

def phi_s1(x, T=TEMP, x0=0):
    mth = np.sqrt(1 - 3*T/2)
    return mth*np.sqrt(2)/np.cosh(mth*(x-x0-SIZE/2))

def correlate(Phi_av):
    s = np.zeros(len(Phi_av)-1)
    for i in range(0,len(Phi_av)-1):
        for j in range(N//2-int(5/DX),N//2+int(5/DX)):
            s[i] += (Phi_av[i,j] - phi_s(DX*j))**2
    iMin = np.argmin(s)
    return (iMin, np.min(s))

def mock_profile(x):
    return 2*np.exp(-0.5*(x-SIZE/2)**2)

x = np.linspace(0, SIZE, N)
t = np.linspace(0, RECTIME, RECPOINTS+1)

""" Compute staff """

Ulong_stack = []
for i in range(len(Phi_stack)):
    print('Computing Ulong for i=',i,'...')
    Epot = np.zeros(RECPOINTS+1)
    for j in range(RECPOINTS+1):
        Epot[j] = Ulong(Phi_stack[i][j],30)
    Ulong_stack += [Epot]

(Phi_stack,Ulong_stack) = synchronise_stack(Phi_stack,Ulong_stack)
Phi_stack = orient_stack(Phi_stack)
(Phi_stack,Ulong_stack) = centralise_stack(Phi_stack,Ulong_stack)
Phi_av = average_stack(Phi_stack)

E_pot_av = np.zeros(RECPOINTS+1)
for i in range(RECPOINTS+1):
    E_pot_av[i] = U(Phi_av[i])
i_sph = np.argmax(E_pot_av)

E_pot_long_av = np.zeros(RECPOINTS+1)
for i in range(RECPOINTS+1):
    E_pot_long_av[i] = Ulong(Phi_av[i], 55)
i_sph = np.argmax(E_pot_long_av)

Variance = np.zeros(N)
for i in range(N):
    for j in range(len(Phi_stack)):
        Variance[i] += (Phi_stack[j][i_sph][i] - Phi_av[i_sph][i])**2
    Variance[i] = np.sqrt(Variance[i])/np.sqrt(len(Phi_stack))/np.sqrt(len(Phi_stack))

""" Save data """

np.savetxt('Out/E_pot_long_av_T='+f'{TEMP:.6f}'+'.txt',E_pot_long_av)
np.savetxt('Out/Phi_av_T='+f'{TEMP:.6f}'+'.txt',Phi_av)

""""""""""""""""""""""""""
"""        PLOTS       """
""""""""""""""""""""""""""

""" field evolution: example, moving sphaleron """

fig, ax = plt.subplots()
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$x$',fontsize=16)
ax.set_ylabel(r'$\varphi$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.01)
plt.plot(x, Phi_stack[2][10])
plt.plot(x, Phi_stack[2][20])
plt.plot(x, Phi_stack[2][40])
plt.xlim((45,55))
#ax.set_title(r'$T=0.1, \eta=0$',fontsize=15)
plt.savefig('App_sph_example.pdf')
plt.clf()

""" i_sph (N_long) """

i_sph_list = []
N_long_list = np.arange(30,140,1)
for i in range(110):
    print(i)
    E_pot_long_av = np.zeros(RECPOINTS+1)
    for j in range(RECPOINTS+1):
        E_pot_long_av[j] = Ulong(Phi_av[j], N_long_list[i])
    i_sph_list += [np.argmax(E_pot_long_av)]
    
fig, ax = plt.subplots()
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$k_*$',fontsize=16)
ax.set_ylabel(r'$t_*$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.05)
ax.scatter(np.array(N_long_list)*2*np.pi/100, np.array(i_sph_list)*0.05, color='black')
#ax.set_title(r'$T=0.1, \eta=0$',fontsize=15)
plt.savefig('App_Nlong.pdf')
plt.clf()

""" U_long example trajectories """

fig, ax = plt.subplots()
ax.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=13)
ax.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'$U_{long}$',fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.02,1.05)
ax.plot(t,Ulong_stack[0])
ax.plot(t,Ulong_stack[1])
ax.plot(t,Ulong_stack[2])
ax.set_xlim((-0.1,2.5))
ax.set_ylim((-5.9,6.5))
#ax.set_title(r'$T=0.1, \eta=0$',fontsize=15)
plt.savefig('App_Ulong_ex.pdf')
plt.clf()