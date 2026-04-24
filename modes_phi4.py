#%%
#import os
import numpy as np
#import scipy.special as sp
#import re
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.fft import rfft, irfft
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
#from scipy.integrate import simps
#from scipy.optimize import curve_fit

#%%

SIZE=100
N=2048

#TEMP=0.1

DX = SIZE/(N-1)
x = np.linspace(-SIZE/2,SIZE/2,N)

#%%

def phi_guess(x):
    return np.sqrt(2)/np.cosh(x)

def Sph(x, guess=phi_guess):
    def bc(ya, yb):
        return np.array([ya[1],yb[0]])
    def F(x, y):
        return np.vstack([y[1],y[0]-y[0]**3])
    y = np.zeros((2,x.size))
    for i in np.arange(0,len(x),1):
        y[0,i] = guess(x[0]+(x[-1]-x[0])/len(x)*i)
    y[1] = np.gradient(y[0], x)
    return solve_bvp(F, bc, x, y, tol=1e-5).sol(x)[0] 

#%%
# zero mode

zero_mode = np.gradient(Sph(x),x)

#norm = np.sum(zero_mode**2)

#zero_mode = zero_mode/np.sqrt(norm)
#%%

def neg_mode(x):
    return 1.0/np.cosh(x)**2

#%%
M2 = np.zeros(N)

for i in range(N):
    M2[i] = 1.0 - 3.0*Sph(x)[i]**2

# for i in range(N):
#     M2[i] = 1.0 - 0.3*Sph(x)[i]**2

#%%
Omega02 = np.zeros(N-1)
for i in np.arange(1,N,1):
    k = 2*np.pi*i/SIZE    
    Omega02[i-1] = 2.0/DX/DX*(1.0-np.cos(DX*k)) + 1.0

#%%

Omega02.sort()

#%%

plt.plot(x,phi_guess(x))
plt.plot(x,Sph(x))
plt.plot(x,zero_mode)
plt.xlim((-5,5))
plt.show()

#%%

plt.plot(x,M2)
plt.show()

#%%

r=np.zeros((3*N))
c=np.zeros((3*N))
v=np.zeros((3*N))

for i in np.arange(0,N,1):
    r[i]=i
    c[i]=i
    v[i]=2.0/DX/DX + M2[i]
for i in np.arange(0,N-1,1):
    r[i+N]=i
    c[i+N]=i+1
    v[i+N]=-1.0/DX/DX
for i in np.arange(0,N-1,1):
    r[i+2*N-1]=i+1
    c[i+2*N-1]=i
    v[i+2*N-1]=-1.0/DX/DX

r[3*N-2]=0
c[3*N-2]=N-1
v[3*N-2]=-1.0/DX/DX

r[3*N-1]=N-1
c[3*N-1]=0
v[3*N-1]=-1.0/DX/DX

B = csc_matrix((v,(r,c)),shape=(N,N))

#%%

O2, P = eigsh(B,k=N-1)

#%%

plt.scatter(np.arange(0,N-1,1),O2)
plt.scatter(np.arange(0,N-1,1),Omega02)
plt.xlim((0,10))
plt.ylim((0,1))
plt.show()

# %%

plt.scatter(np.arange(1,N-1,1),O2[1:]/Omega02[1:])
#plt.xlim((2000,2046))
plt.show()

#%%
# Negative mode

plt.plot(x,P[:,0])
plt.plot(x,-0.19*neg_mode(x))
plt.xlim((-5,5))
plt.show()

#%%
# Even modes

plt.plot(x,np.sqrt(N)*P[:,2])
plt.plot(x,-np.sqrt(2)*np.cos(2*np.pi*x*(2)/2/SIZE))
#plt.xlim((-50,-49))
plt.show()

#%%

plt.plot(x,np.sqrt(N)*P[:,400])
plt.plot(x,np.sqrt(2)*np.cos(2*np.pi*x*(400)/2/SIZE))
plt.xlim((45,50))
#plt.xlim((-5,5))
plt.show()

#%%
# Odd modes

plt.plot(x,np.sqrt(N)*P[:,3])
plt.plot(x,np.sqrt(2)*np.sin(2*np.pi*x*1/SIZE))
plt.show()


#%%
# longest mode

plt.plot(x,np.sqrt(N)*P[:,1])
plt.plot(x,x**0.0)
plt.show()

#%%
# Ortho-normality

print(np.sum(P[:,400]*P[:,400]))

#%%
# Modes array

Modes = np.zeros((N,N+1))
Modes[:,:-2] = P
Modes[:,-2] = zero_mode
Modes[:,-1] = Sph(x)

#%%

plt.plot(x,Modes[:,-1])
plt.show()

#%%

np.savetxt('modes_phi4.txt',Modes,fmt='%1.6f')

#%%

np.savetxt('eigenvalues_phi4.txt',O2.reshape(1,-1),delimiter=' ',fmt='%1.6f')

#%%