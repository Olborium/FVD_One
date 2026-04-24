#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

#%%
""" Parameters of the lattice """
SIZE=100
N=4096

DX = SIZE/(N-1)
x = np.linspace(-SIZE/2,SIZE/2,N)

#%%
""" Find critical bubble """

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

#%%
""" Vacuum frequencies """
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
""" Find eigenmodes and eigenvalues of the critical bubble """

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
""" Check orthonormality """

print(np.sum(P[:,100]*P[:,100]))
print(np.sum(P[:,100]*P[:,101]))

#%%
""" Save eigenmodes and eigenvalues (including zero and negative modes) """

Modes = np.zeros((N,N+1))
Modes[:,:-2] = P
Modes[:,-2] = zero_mode
Modes[:,-1] = Sph(x)

#%%

np.savetxt('modes_phi4.txt',Modes,fmt='%1.6f')

#%%

np.savetxt('eigenvalues_phi4.txt',O2.reshape(1,-1),delimiter=' ',fmt='%1.6f')

#%%
