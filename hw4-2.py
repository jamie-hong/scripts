import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lin, inst
from swap import *

terms = np.array([1, 2, 3, 5, 7, 10, 12, 15, 20, 25])*1.
par = np.array([.042, .043, .047, .054, .057, .06, .061, .059, .056, .0555]) # par swap rate

# par -> discount factors -> cumulative yield -> interpolation

def swap2df(s,terms):
    '''Swap rates to piecewise flat discount factors'''
    d = np.zeros(len(terms))
    interval = np.diff(np.insert(terms,0,0))
    d[0] = 1./(s[0]+1)*interval[0]
    for i in range(1,len(terms),1):
        d[i] = (1-s[i]*sum(d*interval))/(1+interval[i]*s[i])
    return d

d=swap2df(par,terms)
m=dict(zip(terms,d))
m.update(dict(zip(terms[:-1] + 1e-6, d[1:])))
m[0] = d[0]
k, v = zip(*sorted(m.items()))
cy = -np.log(d)
plt.plot(k, v,'-')
plt.plot(terms,cy,'o')
plt.legend(['discount factors'] + ['cumulative yield'], loc='best')
plt.xlabel('Time (Year)')
plt.title('$y(t)$ and $b(t)$ - Own Implementation')

def plotTension(x, y, lbds, xd):
    plt.plot(x, y, 'o')
    plt.title('Tension Spline')

    for lbd in lbds:
        ts = tensionSpline(x, y, lbd, xd)
        plt.plot(xd, ts.value(xd))

    plt.legend(['data'] + ['$\lambda = %.f$' % l for l in lbds], loc='best')


def tensionSpline (x,y,lbd,xd):
    ts = lin.RationalTension(lbd)
    ts.build(x, y)
    return ts

lbds = (0, 2, 10, 50)
xs = np.arange(0,25.1,.1)
plt.figure()
plotTension(terms, cy, lbds, xs)
plt.ylabel('Cumulative Yield')
plt.xlabel('Time (Year)')


def y2pv(swap, curve) :
    discf = lambda ts: np.exp(-curve(ts))
    return priceSwap(swap, discf)

# create the benchmark instruments
bm_swaps = {Swap(m, c, 2) : 0 for m, c in zip (terms, par)}
ts = tensionSpline(terms,cy,lbds[0],xs)
pvs = {swap.maturity : y2pv(swap, ts) for swap in bm_swaps.keys()}
print "PV by maturity - lambda = 0"
print "\n".join(["%.2g Y : %.4g" % (m, v) for m, v in pvs.items()])

def plotError(benchmark, x, y, lbds, xd):
    error = np.zeros(len(lbds))
    for i, lbd in enumerate(lbds):
        ts = tensionSpline(x, y, lbd, xd)
        error[i] = np.linalg.norm([y2pv(swap, ts) for swap in benchmark.keys()],2)
            
    plt.plot(lbds,error,'.')
    plt.title('Norm of Pricing Errors')
    plt.xlabel('$\lambda$')
    plt.ylabel('$\Vert \epsilon \Vert _2$')

        

lbds = np.arange(0, 5, 0.01)
plt.figure()
plotError(bm_swaps, terms, cy, lbds, xs)

def plotboot(tsit, lbd, ax, tagsf) :
    plt.xlabel('Time')    
    
    lbd_tag = '$\\lambda=%.f$' % lbd
    df = pd.DataFrame({'$t$':xs}).set_index(['$t$'])
    
    for tag, f in tagsf.items() :
        df[tag] = f(tsit, xs) 
    
    df.plot(ax = ax, secondary_y = [tagsf.keys()[1]], title = 'Tension Spline ' + lbd_tag)
    plt.plot(terms, tsit(terms), 'o')

tagsf = {"$y(t)$" : lambda cv, xs : cv(xs), "$f(t)$": lambda cv, xs : cv.deriv(xs)}

lbds = [0, 2, 10, 100]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12, 8])

for lbd, ax in zip(lbds, axes.flatten()) :
    tsit, e = inst.iterboot(bm_swaps, y2pv, lbd=lbd, x0=0, its=1 , bds=[-1., 3.])
    plotboot(tsit, lbd, ax, tagsf)  
    
    
####


hlin = lin.PiecewiseLinear()
hlin.build(terms, terms*.01)
hlin.addKnot(0, 0)
hlin = inst.bootstrap(bm_swaps, hlin, y2pv, bds = [-2., 2.])

plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
plt.plot(xs, hlin(xs))
plt.plot(terms, hlin(terms), 'o')
plt.xlabel('Time')
plt.title('$y(t)$');

plt.subplot(1, 2, 2)
plt.plot(xs, hlin.deriv(xs))
plt.plot(terms, hlin.deriv(terms-1e-4), 'o')
plt.xlabel('Time')
plt.title('$f(t)$');
lbds = np.arange(0, 5, 0.01)
plt.figure()
plotError(bm_swaps, terms, hlin(terms), lbds, xs)

ts = tensionSpline(terms,hlin(terms),0,xs)
pvs = {swap.maturity : y2pv(swap,ts) for swap in bm_swaps.keys()}
print "PV by maturity - Piecewise flat bootstrap, lambda=0"
print "\n".join(["%.2g Y : %.4g" % (m, v) for m, v in pvs.items()])

lbds = (0, 2, 10, 50)
xs = np.arange(0,25.1,.1)
plt.figure()
plotTension(terms, hlin(terms), lbds, xs)
plt.ylabel('Cumulative Yield')
plt.xlabel('Time (Year)')
plt.title('Tension Spline')




_, e2 = inst.iterboot(bm_swaps, y2pv, lbd=0, x0=0, its=4 , bds=[-1., 3.])

err = [np.linalg.norm(e2[i,:],2)*1e4 for i in range(4)]
plt.figure(figsize=[7,5])
x = range(1,5,1)
plt.plot(x, err,'-o')
plt.xticks(x, [1,2,3,4])
plt.xlabel('Number of Iterations')
plt.ylabel('$\Vert \epsilon \Vert _2$')
plt.yscale('log')
plt.title('L-2 Norms of Errors in bps, $\lambda = 0$')
plt.annotate('$\Vert \epsilon \Vert _2 $ = 0.63 bps', xy=(2.05, 1), xytext=(2.3, 5),
            arrowprops=dict(facecolor='black', shrink=0.03))



####

def pv_lbds(bms, y2pv, lbds, x0) :
    cvs = []
    for lbd in lbds:
        cv, e = inst.iterboot(bms, y2pv, x0, lbd, mixf = 0.5, bds=[-1.,3.])
        cvs.append(cv)
    
    return cvs

tags = ['$\\lambda=%.f$' % l for l in lbds]
cv0 = pv_lbds(bm_swaps, y2pv, lbds, 0)

plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.plot(xs, np.array([cv(xs) for cv in cv0]).T);
plt.title('$y(t)$')
plt.xlabel('$t$')

plt.subplot(1, 2, 2)
plt.plot(xs, np.array([cv.deriv(xs) for cv in cv0]).T);
plt.legend(tags, loc='best');
plt.title('$f(t)$');
plt.xlabel('$t$');

def showPerts(bms, bms_ps, y2pv, lbds, x0, pertf) :
    cv0 = pv_lbds(bms, y2pv, lbds, x0=x0)
    cvp = pv_lbds(bms_ps, y2pv, lbds, x0=x0)
    
    lbd_tags = ['$\\lambda=%.f$' % lbd for lbd in lbds]

    plt.figure(figsize=[7, 5])
    plt.plot(xs, 1e4*np.array([pertf(f, g)(xs) for f, g in zip(cv0, cvp)]).T);
    plt.xlabel('Tenor')
    plt.ylabel('$\Delta f(t)$ (bps)')
    plt.title('1bps Spread Perturbation @t=%.2f' % pt)
    plt.legend(lbd_tags, loc='best');
    plt.plot(terms, 1e4*np.array([pertf(f, g)(terms) for f, g in zip(cv0, cvp)]).T, '.');
    
    
pt = 5
bms_ps = {k if k.maturity != pt else Swap(k.maturity, k.coupon-1e-4, k.freq) : v 
        for k, v in bm_swaps.items()}
    
showPerts(bm_swaps, bms_ps, y2pv, lbds, 0, lambda f, g : lambda xs : f.deriv(xs) - g.deriv(xs))

####


def f2pv(swap, curve) :
    discf = lambda ts: np.exp(-curve.integral(ts))
    return priceSwap(swap, discf)

x0 = .01
chartf = {'F(t)' : lambda cv, xs : cv.integral(xs), '$f(t)$' : lambda cv, xs : cv(xs), }

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 4])

lbds = (0, 20)
tsit0, e0 = inst.iterboot(bm_swaps, f2pv, x0, lbds[0], mixf=.5, bds=[-3.,3.])
plotboot(tsit0, lbds[0], axes[0], chartf)    

tsit1, e1 = inst.iterboot(bm_swaps, f2pv, x0, lbds[1], mixf=.5, bds=[-3.,3.])
plotboot(tsit1, lbds[1], axes[1], chartf) 

plt.show()