from numpy.core import empty, clip, zeros, exp
from numpy import linspace
import scipy.sparse.linalg.dsolve as linsolve
from scipy import sparse
import matplotlib.pyplot as plt
import time

class BS_FDM_cn:

    def __init__(self, r, sigma, T, Smin, Smax, Fl, Fu, Fp, m, n):
    
        self.r  = r
        self.sigma = sigma
        self.T  = T
    
        self.Smin = Smin
        self.Smax = Smax
        self.Fl = Fl
        self.Fu = Fu
    
        self.m  = m
        self.n  = n
        
        # Step sizes
        self.dt = float(T)/m
        self.dx = float(Smax-Smin)/(n+1)
        self.xs = Smin/self.dx
    
        self.u = empty((m+1, n))
        self.u[0,:] = Fp
    
    def build(self):
    
        A = sparse.lil_matrix((self.n, self.n))
        C = sparse.lil_matrix((self.n, self.n))
    
        for j in xrange(0, self.n):
            xd = j+1+self.xs
            ssxx = (self.sigma * xd) ** 2
        
            A[j,j] = 1.0 - 0.5*self.dt*(ssxx + self.r)
            C[j,j] = 1.0 + 0.5*self.dt*(ssxx + self.r)
            
            if j > 0:
                A[j,j-1] = 0.25*self.dt*(+ssxx - self.r*xd)
                C[j,j-1] = 0.25*self.dt*(-ssxx + self.r*xd)
            if j < self.n-1:
                A[j,j+1] = 0.25*self.dt*(+ssxx + self.r*xd)
                C[j,j+1] = 0.25*self.dt*(-ssxx - self.r*xd)
    
        self.A = A.tocsr()
        self.C = linsolve.splu(C)  # sparse LU decomposition
    
        # Buffer to store right-hand side of the linear system Cu = v
        self.v = empty((n, ))
    
    def solve(self):
        
        self.build()

        for i in xrange(0, m):

            # explicit part of time step
            self.v[:] = self.A * self.u[i,:]
        
            # Add in the two other boundary conditions
            xdl = 1+self.xs
            xdu = self.n+self.xs
            self.v[0] += self.Fl[i] * 0.25*self.dt*(+(self.sigma*xdl)**2 - self.r*xdl)
            self.v[self.n-1] += self.Fu[i] * 0.25*self.dt*(+(self.sigma*xdu)**2 + self.r*xdu)
            self.v[0] -= self.Fl[i+1] * 0.25*self.dt*(-(self.sigma*xdl)**2 + self.r*xdl)
            self.v[self.n-1] -= self.Fu[i+1] * 0.25*self.dt*(-(self.sigma*xdu)**2 - self.r*xdu)
            
            # implicit part of time step
            self.u[i+1,:] = self.C.solve(self.v)

        return self.u
    
def european_call(r, sigma, T, Smax, m, n, Smin=0.0, barrier=None):
    
    X = linspace(0.0, Smax, n+2)
    X = X[1:-1]
    
    Fp = clip(X-K, 0.0, 1e600)
    
    if barrier is None:
        Fu = Smax - K*exp(-r * linspace(0.0, T, m+1))
        Fl = zeros((m+1, ))
    elif barrier == 'up-and-out':
        Fu = Fl = zeros((m+1,))
    
    bss = BS_FDM_cn(r, sigma, T, Smin, Smax, Fl, Fu, Fp, m, n)
    return X, bss.solve()

def plot_solution(T, X, u):
  
    # Plot of price function at time 0
    plt.plot(X, u[-1,:])
    plt.xlabel('Stock price $S$', fontsize=14)
    plt.ylabel('Option value', fontsize=14)


# Parameters
r = 0.05
sigma = 0.35
K = 100.
T = 1
Smax = 200.
n = 201

plt.figure(figsize=[20,5])
for i,m in enumerate([4555, 1000, 150]):
    plt.subplot(1,3, i+1)    
    t1 = time.time()
    X, u = european_call(r, sigma, T, Smax, m, n)
    t = time.time() - t1
    plot_solution(T, X, u)
    plt.title('Time Steps: %.f, CPU Time: %.3g' % (m, t), fontsize=14)

Smax = 120
plt.figure(figsize=[20,5])
for i,m in enumerate([4555, 1000, 150]):
    plt.subplot(1,3, i+1)    
    t1 = time.time()
    X, u = european_call(r, sigma, T, Smax, m, n, barrier='up-and-out')
    t = time.time() - t1
    plot_solution(T, X, u)
    plt.title('Time Steps: %.f, CPU Time: %.3g' % (m, t), fontsize=14)

plt.show()