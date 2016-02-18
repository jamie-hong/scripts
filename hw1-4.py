import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Question (a)

def fib(n):
    i = 1
    n = int(n)
    if n < 0:
        raise ValueError('n must be a non-negative integer')
    f = np.ones(n+1)
    a,b = 1,1
    for i in range(1,n+1):
        a,b = b, a+b
        f[i]=a
    return f


def pib(n):
    c = 1 + np.sqrt(3)/100
    i = 1
    n = int(n)
    if n < 0:
        raise ValueError('n must be a non-negative integer')
    if n == 0:
        p = [1]
    elif n > 0:
        p = [1,1]
        while i < n:
            p.append(c*p[i]+p[i-1])
            i += 1
    return p

n = 100
single = 1/2**(-24)*np.ones(n+1)
double = 1/2**(-53)*np.ones(n+1)
x=np.arange(0,n+1,1)

df = pd.DataFrame(np.array([fib(n), pib(n), single, double]).T, index=x, 
                  columns=[r'$fib(n)$', r'$pib(n)$',r'$1/\epsilon$: single',r'$1/\epsilon$: double'])
ax = df.plot(logy=True,style=['.--b','.--r','.--k','.--g'],markersize=6,grid=True,figsize=(8,6))
ax.set_ylabel('Log value of the series',fontsize=18)
ax.set_xlabel(r'$n$',fontsize=18)
plt.show()

# Question (b)

def rfib(n,f):
    if n > len(f)-1:
        raise ValueError('n must be smaller than or equal to len(f)-1')
    a = f[n]
    b = f[n-1]
    while n - 1 > 0:
        tmp = a
        a = b
        b = tmp - b
        n -= 1
    return 1-b

f32 = np.float32(fib(100))
f64 = np.float64(fib(100))

nmax32 = 100
nmax64 = 100
r32 = np.arange(2,nmax32+1,1)
r64 = np.arange(2,nmax64+1,1)
fdiff32 = np.zeros(nmax32-1,dtype=np.float32)
fdiff64 = np.zeros(nmax64-1)

for i,j in enumerate(r32):
    fdiff32[i] = np.abs(rfib(j,f32))

for i,j in enumerate(r64):
    fdiff64[i] = np.abs(rfib(j,f64))

fig2 = plt.figure()
plt.plot(r32,fdiff32,'.')
plt.yscale('log')
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel(r'Difference between $f_0 = 1 $ and $\^{f_0}$',fontsize=18)
plt.title('Simple precision')
plt.grid()

fig3 = plt.figure()
plt.plot(r64,fdiff64,'.')
plt.yscale('log')
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel(r'Difference between $f_0 = 1$ and $\^{f_0}$',fontsize=18)
plt.title('Double precision')
plt.grid()

# Question (c)

def rpib(n,p):
    if n > len(p)-1:
        raise ValueError('n must be smaller than or equal to len(p)-1')
    a = p[n]
    b = p[n-1]
    while n - 1 > 0:
        tmp = a
        a = b
        b = tmp - b
        n -= 1
    return 1-b

p32 = np.float32(pib(100))
p64 = np.float64(pib(100))
pdiff32 = np.zeros(nmax32-1,dtype=np.float32)
pdiff64 = np.zeros(nmax64-1)

for i,j in enumerate(r32):
    pdiff32[i] = np.abs(rpib(j,p32))

for i,j in enumerate(r64):
    pdiff64[i] = np.abs(rpib(j,p64))

fig4 = plt.figure()
plt.plot(r32,pdiff32,'.')
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel(r'Difference between $p_0 = 1 $ and $\^{p_0}$',fontsize=18)
plt.yscale('log')
plt.title('Simple precision')
plt.grid()

fig5 = plt.figure()
plt.plot(r64,pdiff64,'.')
plt.xlabel(r'$n$',fontsize=18)
plt.ylabel(r'Difference between $p_0 = 1$ and $\^{p_0}$',fontsize=18)
plt.yscale('log')
plt.title('Double precision')
plt.grid()

plt.show()