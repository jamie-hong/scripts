import pandas as pd
import numpy as np
import fmt
from scipy.stats import norm
from sobol_lib import i4_sobol_generate as sobol

n = 125
t = 5.
defProbs = 1 - np.exp(-(np.random.uniform(size=n)*.03)*t)
recovery = 0.4*np.ones(n)
w = 1./n*np.ones(n)
rho = 0.5 # correlation in Gaussian copula
discf = .9
npath = 1000 # number of simulation paths

# a list of attachements and detachements, they pair up by elements
attachements = np.array([0, .03, .07, .1, .15, .3])
detachements = np.array([.03, .07, .1, .15, .3, .6])

#portfolio expected loss
el = np.sum(w*defProbs*(1-recovery))
print "portfolio expected loss is ", el


class CDO(object) :
    def __init__(self, w, defProbs, recovery, a, d) :
        self.w = w/np.sum(w)
        self.p = defProbs
        self.rec = recovery
        self.rho = rho
        self.a = a
        self.d = d

    def drawDefaultIndicator(self, z, rho) :
        '''return a list of default indicators given common factor z, using one factor Gaussian Copula
        '''
        e = np.random.normal(size=np.shape(self.p))
        x = z*np.sqrt(self.rho) + np.sqrt(1-self.rho)*e
        return np.less(norm.cdf(x), self.p) # if cdf(x) < p, default

    def portfolioLoss(self, defIndicator) :
        '''compute portfolio loss given default indicators'''
        return np.sum(defIndicator*self.w*(1-self.rec))

    def tranchePV(self, portfLoss, discf) :
        '''compute tranche PV from portfolio loss
        Args:
            portfLoss: the total portfolio loss
            discf: discount factor
        Returns:
            tranche PVs'''
        
        sz = self.d - self.a
        return discf/sz*np.minimum(np.maximum(portfLoss - self.a, 0), sz)

    def drawPV(self, z, rho, discf) :
        ''' compute PV and portfolio Loss conditioned on a common factor z'''
        di = self.drawDefaultIndicator(z, rho)
        pfLoss = self.portfolioLoss(di)
        return self.tranchePV(pfLoss, discf), pfLoss
    
    
cdo = CDO(w, defProbs, recovery, attachements, detachements)

## price the tranches using simulation
def simCDO(cdo, rho, disc, paths) :
    zs = np.random.normal(size=[paths])
    pv = np.zeros(np.shape(cdo.a)) # np.shape(cdo.a): number of tranches
    pv2 = np.zeros(np.shape(cdo.d))
    for z in zs:
        thisPV, _ = cdo.drawPV(z, rho, discf)
        pv += thisPV
        pv2 += thisPV*thisPV
        
    v = pv/paths
    var = pv2/paths - v**2
    return pv/paths, np.sqrt(var/paths), zs

pv_0, err_0, zs = simCDO(cdo, rho, discf, npath)
df = pd.DataFrame(np.array([cdo.a, cdo.d, pv_0, err_0]), index=['Attach', 'Detach', 'PV', 'MC err'])

fmt.displayDFs(df, headers=['Plain'], fmt='4g')

# Antithetic

def simCDO_antithetic(cdo, rho, disc, paths, zs):
    zs = zs[0:paths]
    pv = np.zeros(np.shape(cdo.a))
    pv2 = np.zeros(np.shape(cdo.d))
    for z in zs:
        thisPV1, _ = cdo.drawPV(z, rho, discf)
        thisPV2, _ = cdo.drawPV(-z, rho, discf)
        thisPV = (thisPV1 + thisPV2)/2
        pv += thisPV
        pv2 += thisPV*thisPV
        
    v = pv/paths
    var = pv2/paths - v**2
    return pv/paths, np.sqrt(var/paths)


pv_1, err_1 = simCDO_antithetic(cdo, rho, discf, npath/2, zs)
vrf_1 = err_0**2*npath/(err_1**2*npath/2) # variance reduction factor
df1 = pd.DataFrame(np.array([cdo.a, cdo.d, pv_1, err_1, vrf_1]), index=['Attach', 'Detach', 'PV', 'MC err', 'VRF'])

fmt.displayDFs(df1, headers=['Antithetic'], fmt='4g')

# Importance Sampling

def simCDO_IS(cdo, rho, disc, paths, u, b):
    means = np.zeros([b, np.shape(cdo.a)[0]])
    for i in range(b):
        zs_q = np.random.normal(size=paths)
        zs_p = zs_q + u # P sample
        m = np.exp(-u*zs_p + 0.5*u*u) # R-N derivative
        qs = 1./paths*np.ones(paths) # Q weights
        ps = m*qs # P weights
        ps = ps/np.sum(ps) # normalization
            
        pv = np.zeros(np.shape(cdo.a))

        for z,p in zip(zs_p,ps):
            thisPV, _ = cdo.drawPV(z, rho, discf)
            pv += thisPV*p
        means[i,:] = pv
            
    return np.mean(means,0), np.std(means,0)

b = 20 # number of batches
pv_2, err_2 = simCDO_IS(cdo, rho, discf, npath, -1, b)
vrf_2 = err_0**2/(err_2**2) # variance reduction factor
df2 = pd.DataFrame(np.array([cdo.a, cdo.d, pv_2, err_2, vrf_2]), index=['Attach', 'Detach', 'PV', 'MC err', 'VRF'])

fmt.displayDFs(df2, headers=['Importance Sampling'], fmt='4g')

# Sobol Sequence

def simCDO_Sobol(cdo, rho, disc, paths, b):
    means = np.zeros([b, np.shape(cdo.a)[0]])
    ss = sobol(1,paths*b,0)
    for i in range(b):
        zs = norm.ppf(ss[0,i*paths:(i+1)*paths]).T
        pv = np.zeros(np.shape(cdo.a)) # np.shape(cdo.a): number of tranches
        for z in zs:
            thisPV, _ = cdo.drawPV(z, rho, discf)
            pv += thisPV
        means[i,:] = pv/paths

    return np.mean(means,0), np.std(means,0)

pv_3, err_3 = simCDO_Sobol(cdo, rho, discf, npath, b)
vrf_3 = err_0**2/(err_3**2) # variance reduction factor
df3 = pd.DataFrame(np.array([cdo.a, cdo.d, pv_3, err_3, vrf_3]), index=['Attach', 'Detach', 'PV', 'MC err', 'VRF'])

fmt.displayDFs(df3, headers=['Sobol Sequence'], fmt='4g')

# Stratified Sampling

def stratify(u, bs, shuffle) :
    b = len(bs)
    r = len(u)/b + 1
    sb = []
    
    for i in range(r) :
        if shuffle :
            np.random.shuffle(bs)
        sb = sb + bs.tolist()
            
    return [1.*(i + x)/b for x, i in zip(u, sb)]

def simCDO_SS(cdo, rho, disc, paths, nbins, b):
    means = np.zeros([b, np.shape(cdo.a)[0]])
    for i in range(b):
        u = np.random.uniform(size=paths)
        v = stratify(u, np.arange(nbins), False)
        zs = norm.ppf(v)
        pv = np.zeros(np.shape(cdo.a)) # np.shape(cdo.a): number of tranches
        for z in zs:
            thisPV, _ = cdo.drawPV(z, rho, discf)
            pv += thisPV
            
        means[i,:] = pv/paths
    return np.mean(means,0), np.std(means,0)

nbins = 500
pv_4, err_4 = simCDO_SS(cdo, rho, discf, npath, nbins, b)
vrf_4 = err_0**2/(err_4**2) # variance reduction factor
df4 = pd.DataFrame(np.array([cdo.a, cdo.d, pv_4, err_4, vrf_4]), index=['Attach', 'Detach', 'PV', 'MC err', 'VRF'])

fmt.displayDFs(df4, headers=['Stratefied Sampling'], fmt='4g')

