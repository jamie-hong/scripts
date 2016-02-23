import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from swap import *
import lin


def swap2df2(par,terms,freq):
    '''Swap rates to discount factors with flat rate of return in between tenor intervals'''
    k = np.arange(1./freq, terms[-1] + 1e-6, 1./freq) # payment dates
    d = dict(zip(k,np.zeros(len(k)))) # discount factors
    s = dict(zip(terms,par))
    
    for i in range(len(terms)):
        t = terms[i]
        if i == 0:
            d[t] = opt.brentq(pv_c, 0, 1, args=(s[t], t, d, freq, 0))
            for j in np.arange(1./freq, t, 1./freq):
                d[j] = (d[t])**(j/t)
        if i > 0:
            d[t] = opt.brentq(pv_c, 0, 1, args=(s[t], t, d, freq, terms[i-1]))
            for j in np.arange(1./freq+terms[i-1], t, 1./freq):
                d[j] = (d[t])**(j/t)

    return np.array([d[x] for x in k])

def pv_c(x,s,t,d,freq,prev_term):
    if prev_term == 0:
        tmp = 0
    else:
        tmp = np.sum([d[m]/freq for m in np.arange(1./freq, prev_term+1e-6,1./freq)])
    for j in np.arange(1./freq+prev_term, t+1e-6, 1./freq):
        tmp = tmp + 1./freq*x**(j/t)
    return tmp*s+x-1


terms = np.array([1, 2, 3, 5, 7, 10, 12, 15, 20, 25])*1.
par = np.array([.042, .043, .047, .054, .057, .06, .061, .059, .056, .0555])

d = swap2df2(par, terms, 2)
plt.plot(d.keys(),d.values(),'o')
plt.show()

k = np.arange(1./2, terms[-1] + 1e-6, 1./2)

cyy = -np.log(dd)
plt.figure(figsize=[7,5])
plt.plot(k, dd,'.')
plt.plot(k, cyy,'o')
plt.legend(['discount factors'] + ['cumulative yield'], loc='best')
plt.xlabel('Time (Year)', fontsize=14)
plt.title(' Figure 2.1: $b(t)$ and $y(t)$ - Own Method', fontsize=14)
plt.show()


def plotTension(x, y, lbds, xd):
    plt.plot(x, y, 'o')

    for lbd in lbds:
        ts = tensionSpline(x, y, lbd, xd)
        plt.plot(xd, ts.value(xd))

    plt.legend(['data'] + ['$\lambda = %.f$' % l for l in lbds], loc='best')

def tensionSpline (x,y,lbd,xd):
    ts = lin.RationalTension(lbd)
    ts.build(x, y)
    return ts

def y2pv(swap, curve) :
    discf = lambda ts: np.exp(-curve(ts))
    return priceSwap(swap, discf)

# plot tension splines of y(t)
lbds = (0, 2, 10, 50)
xs = np.arange(0,25.1,.1)
plt.figure(figsize=[7,5])
plotTension(k, cyy, lbds, xs)
plt.ylabel('Cumulative Yield', fontsize=14)
plt.ylim([0, 1.4])
plt.xlabel('Time (Year)',fontsize=14)
plt.title('Figure 2.2: Tension Spline of $y(t)$', fontsize=14)
plt.show()
  
# create the benchmark instruments and print out errors
bm_swaps = {Swap(m, c, 2) : 0 for m, c in zip (terms, par)}
ts = tensionSpline(k,cyy,lbds[0],xs)
pvs = {swap.maturity : y2pv(swap, ts) for swap in bm_swaps.keys()}
print "PV by maturity, lambda = 0:"
print "\n".join(["%.2g Y : %.4g" % (m, v) for m, v in pvs.items()])

# plot error vs. lambda
def plotError(benchmark, x, y, lbds, xd):
    error = np.zeros(len(lbds))
    for i, lbd in enumerate(lbds):
        ts = tensionSpline(x, y, lbd, xd)
        error[i] = np.linalg.norm([y2pv(swap, ts) for swap in benchmark.keys()],2)
            
    plt.plot(lbds,error,'.')
    plt.xlabel('$\lambda$', fontsize=16)
    plt.ylabel('$\Vert \epsilon \Vert _2$', fontsize=16)

lbds = np.arange(0, 5, 0.01)
plt.figure(figsize=[7,5])
plotError(bm_swaps, k, cyy, lbds, xs)
plt.title('Figure 2.3: Norms of Pricing Errors',fontsize=14)
plt.show()