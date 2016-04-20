import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

lbd = 2
def f(y,t):
    return [y[1], -lbd*y[0]-(lbd+1)*y[1]]
    
y0 = [1., lbd-2]

T = 30
h = 0.05
ts = np.arange(0.0001, T, h)
ys, infodict = odeint(f, y0, ts, full_output=True, ixpr=True)

plt.figure(figsize=[14,5])
plt.subplot(121)
plt.plot(ts, ys[:,:], '.-')
plt.plot(ts, 2*np.exp(-ts)-np.exp(-lbd*ts),'r')
plt.xlabel('Time')
plt.title('$\lambda$ = %d and $h$ = %.2f' % (lbd,h))
plt.legend(['odeint $y(t)$','odeint $y\'(t)$', 'exact $y(t)$'])

lbd = 50
y0 = [1., lbd-2]
ys2, infodict2 = odeint(f, y0, ts, full_output=True, ixpr=True)

plt.subplot(122)
plt.plot(ts, ys2[:,:], '.-')
plt.plot(ts, 2*np.exp(-ts)-np.exp(-lbd*ts),'r')
plt.xlim(0,T/10)
plt.xlabel('Time')
plt.title('$\lambda$ = %d and $h$ = %.2f' % (lbd,h))
plt.legend(['odeint $y(t)$','odeint $y\'(t)$', 'exact $y(t)$'])

plt.show()
