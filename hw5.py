import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fmt, lin, inst, time
from swap import *

cmturl = "https://raw.githubusercontent.com/yadongli/nyumath2048/master/data/cmt.csv"
cmt_rates = pd.read_csv(cmturl, parse_dates=[0], index_col=[0])*.01
'''
cmt_rates.plot(legend=False);
tenors = cmt_rates.columns.map(float)
tenorTags = ['T=%g' % m for m in tenors]
'''
t = map(float, cmt_rates.columns)
r = cmt_rates.iloc[-1, :].values

fmt.displayDF(pd.DataFrame(r, index=t, columns=['Zero Rate (%)']).T*100, fmt="4g")

# Question 1-1

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

lbds = [5]
r=np.insert(r,0,0)
t=np.insert(t,0,0)
xs = np.arange(0,20.1,.1)
plt.figure(figsize=[7,5])
plotTension(t, r*100, lbds, xs)
plt.ylabel('Zero Rate (%)', fontsize=14)
plt.ylim([0,3])
plt.xlabel('Time (Year)', fontsize=14)
plt.title('Figure 1.1: Tension Spline of $r(t)$', fontsize=14)


# Question 1-2
interp_r = tensionSpline(t,r,5,xs)
discf = lambda ts : np.exp(-interp_r(ts)*ts)
cp = [swapParSpread(m, discf, 2) for m in t[2:]]
bm_par = {Swap(m, c, 2) : 0 for m, c in zip (t[2:], cp)}
c1 = (1-discf(0.25))*4/discf(0.25) # convert zero rate to swap rate for t=0.25
bm_par[Swap(0.25, c1, 4)] = 0

cp = np.insert(cp,0,c1)
hi_names = ['3M ZCB'] + ['Swap @%gY' % ti for ti in t[2:]] # benchmark instruments
fmt.displayDF(pd.DataFrame(cp.T*100, index=hi_names, columns=['Par Spread (%)']).T, "2f")

# Question 1-3
def z2pv(swap, curve):
    discf = lambda ts: np.exp(-curve(ts)*ts)
    return priceSwap(swap, discf)

tsit, _ = inst.iterboot(bm_par, z2pv, lbd=5, x0=0, its=1 , bds=[-1., 1.])
plt.figure(figsize=[7,5])
plt.plot(xs,tsit.value(xs)*100,'r-',LineWidth = 2)
plt.plot(xs, interp_r(xs)*100, 'b--', LineWidth = 2)
plt.ylabel('Zero Rate (%)', fontsize=14)
plt.ylim([0,3])
plt.xlabel('Time (Year)', fontsize=14)
plt.title('Figure 1.2: Interpolated Zero Rate Curve $r(t)$', fontsize=14)
plt.legend(['Bootstrapped from swap rates','Tension spline of market zero rates'], loc='best')

# Question 1-4
bespoke = Swap(6,0.05,2)
notional = 100e6
v0 = priceSwap(bespoke,discf) # present value of bespoke
deltas = [] # deltas of bespoke (dv/dq)
pv01 = [] # PV01s of benchmarks (db/dq)

def swap_pv01(swap,discf_o, discf_n):
    return (priceSwap(swap, discf_n) - priceSwap(swap, discf_o))*1e4

def pertCurves(bm_sorted, pert):
    curves = []
    for s in bm_sorted:
        swap_pert = Swap(s.maturity, s.coupon + pert, s.freq)
        bm_pert = {k if k.maturity != s.maturity else swap_pert : 0 for k in bm_sorted}
        reb_r, _ = inst.iterboot(bm_pert, z2pv, lbd=5, x0=0, its=2 , bds=[-1., 1.]) # rebootstrapped zero rate curve
        curves.append(reb_r)
    return curves
    
pert = 1e-4 # 1 bp
bm_sorted=sorted(bm_par.keys(), key = lambda x : x.maturity)

tic = time.clock()
cvs = pertCurves(bm_sorted,pert)
Jac = np.zeros([len(bm_par),len(t)-1]) # Jacobian wrt quoted spread
for i,pt in enumerate(t[1:]):
    # benchmark swaps with perturbated par spread when t = pt (all pv's = 0)
    if i == 0:
        freq = 4
    else:
        freq = 2
    swap_pert = Swap(pt, cp[i]+pert, freq) # perturbed swap
    bm_p = np.array(bm_sorted)
    bm_p[i] = swap_pert
    discf_n = lambda ts : np.exp(-cvs[i](ts)*ts)
    new_v = priceSwap(bespoke, discf_n)
    deltas.append((new_v-v0)/pert)
    Jac[:,i] = [swap_pv01(s,discf, discf_n) for s in bm_p]

deltas = np.array(deltas)
pv01 = np.diag(Jac)   
hn4 = -deltas.dot(np.linalg.inv(Jac))*notional*1e-6 # hedge notional
toc = time.clock()
t_4 = toc - tic

fmt.displayDF(pd.DataFrame(np.array([pv01,deltas*1e-4*notional,hn4]).T, index=hi_names, columns=['PV01','Deltas (wrt bmk quotes)','Hedge Notional ($MM)']).T, "3f")
print("Time spent: %.3f seconds" % t_4)

# Question 1-5

deltas_r = [] # deltas wrt zero rates
J = np.zeros([len(bm_par),len(t)-1]) # Jacobian wrt zero rates
tic = time.clock()
for i in np.arange(1,len(t),1):
    rp = np.array(r) 
    rp[i] = rp[i] + pert
    interp_rp = tensionSpline(t,rp,5,xs) # perturbed zero rate curve
    discf_rp = lambda ts : np.exp(-interp_rp(ts)*ts)
    deltas_r.append((priceSwap(bespoke,discf_rp)-v0)/pert)
    J[:,i-1] = [priceSwap(s,discf_rp)/pert for s in bm_sorted]

deltas_r=np.array(deltas_r)
hn5 = -deltas_r.dot(np.linalg.inv(J))*notional*1e-6
toc = time.clock()
t_5 = toc - tic

fmt.displayDF(pd.DataFrame(np.array([deltas_r*1e-4*notional,hn5]).T, index=hi_names, columns=['Deltas (wrt zero rates)','Hedge Notional ($MM)']).T, "3f")
print("Time spent: %.3f seconds" % t_5)

# Question 2-1

Ju = np.zeros([3,len(t)-1]) # Jacobian of 3 liquid swaps wrt zero rates
bm_liq = [bm_sorted[2], bm_sorted[5], bm_sorted[7]]
for i in np.arange(1,len(t),1):
    rp = np.array(r) 
    rp[i] = rp[i] + pert
    interp_rp = tensionSpline(t,rp,5,xs) # perturbed zero rate curve 
    discf_rp = lambda ts : np.exp(-interp_rp(ts)*ts)
    Ju[:,i-1] = [priceSwap(s,discf_rp)/pert for s in bm_liq]

Ju=np.matrix(Ju)
delta_m = np.matrix(deltas_r).T
h_21 = np.linalg.inv(Ju*Ju.T)*Ju*delta_m # hedge ratio
hn21 = -h_21*notional*1e-6 # hedge notional
liq_names = ['Swap @%gY' % s.maturity for s in bm_liq]
fmt.displayDF(pd.DataFrame(hn21, index=liq_names, columns=['Hedge Notional ($MM)']).T,"3f")

# Question 2-2

dchange = cmt_rates.diff()[1:] # daily rate changes
V = np.matrix(dchange.cov())
h_22 = np.linalg.inv(Ju*V*Ju.T)*Ju*V*delta_m
hn22 = -h_22*notional*1e-6 # hedge notional
fmt.displayDF(pd.DataFrame(hn22, index=liq_names, columns=['Hedge Notional ($MM)']).T,"3f")

opt_h = [np.matrix(np.zeros(len(h_21))).T, h_21, h_22]
l2 = np.array([np.linalg.norm(Ju.T*h - delta_m,2) for h in opt_h])*1e-4 # L-2 norms
vv = [np.float64((Ju.T*h-delta_m).T*V*(Ju.T*h-delta_m)) for h in opt_h] # variance of hedged portfolio

df_opt = pd.DataFrame(np.array([l2,np.sqrt(vv)]), index=['L2 norm', 'Std dev of daily PnL'], 
                      columns=['Unhedged', 'Min residual risk', 'Min var']).T

fmt.displayDF(df_opt.T*notional*1e-6, "5g")

plt.show()