import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(k):
    r = 1
    sig2 = 1
    t = 1
    return r*sig2/k*(np.exp(-k*t)-np.exp(-2*k*t))

def g(k, h):
    ans = np.zeros_like(k)
    r = 1
    sig2 = 1
    t = 1
    G = (k>-h) & (k<h)
    F = np.ones(len(k),dtype=bool)
    F[G] = False
    ans[F] = f(k[F])
    ans[G] = 1-1.5*r*sig2*t**2*k[G]+7/6.*r*sig2*t**3*k[G]**2
    return ans

k = np.arange(-20e-8, 20e-8, 1e-9)*1e-7
df = pd.DataFrame(np.array([f(k),g(k,5e-15)]).T, index=k, columns=['f(k)', 'Approximation'])
df.plot()

k_s = np.float32(np.arange(-1e-5,1e-5,1e-7))
df_s = pd.DataFrame(np.array([f(k_s),g(k_s,2e-6)]).T, index=k_s, columns=['f(k_s)', 'Approximation'])
df_s.plot()

plt.show()
