import numpy as np

class OU (object) :
    def __init__(self, k, u, sig):
        self.k = k
        self.u = u
        self.sig = sig

    def fit(self, logb):
        x = np.array([logb[:-1], np.ones(len(logb)-1)])
        a, b = np.linalg.lstsq(x.T, logb[1:])[0]
        e = logb[1:] - (a*logb[:-1] + b)
        sd = np.std(e)
        self.k = -math.log(a)
        self.u = b/(1-a)
        self.sig = sd*math.sqrt(-2*math.log(a)/(1-a*a))

        return e

    def draw(self, es, x0, dt) :
        '''es is assumed to have a dimension of [npath, nd]'''
        xs = []

        x = np.ones(len(es))*x0
        for e in es.T :
            x = x + self.k*(self.u-x)*dt + self.sig*e*np.sqrt(dt)
            xs.append(x)

        return np.array(xs).T

    def draw_dt(self, es, x0, dt) :
        '''es is of dimension[npath, nd]'''
        xs = []

        x = np.ones(len(es))*x0
        for e in es.T :
            x = self.cond_u(x0, dt) + self.cond_std(x0, dt)*e
            xs.append(x)

        return np.array(xs).T

    def cond_u(self, x0, t) :
        return self.u*(1. - math.exp(-self.k*t)) + x0*math.exp(-self.k*t)

    def cond_std(self, x0, t) :
        return np.sqrt(self.sig*self.sig/2./self.k*(1. - math.exp(-2*self.k*t)))

    def v2z(self, v0, vi, dt) :
        x = v0
        zs = []

        for v in vi.T:
            zs.append((v-self.cond_u(x, dt))/self.cond_std(x, dt))
            x = v

        return np.array(zs).T

    def z2v(self, v0, zs, dt) :
        x = v0
        vs = []
        for z in zs.T :
            x = z*self.cond_std(x, dt) + self.cond_u(x, dt)
            vs.append(x)

        return np.array(vs).T

    @staticmethod
    def decorr(kx, ky, dt) :
        return 2*math.sqrt(kx*ky)*(1. - math.exp(-dt*(kx+ky)))/(kx+ky)/\
               math.sqrt((1. - (math.exp(-2*kx*dt)))*(1. - math.exp(-2*ky*dt)))

    def __str__(self) :
        return "kappa = %.4f, mu = %.4f, sigma = %.4f, mean rev time = %.2f" \
               % (self.k, self.u, self.sig, 1./self.k)


class GBM(object) :
    def __init__(self, u, vol) :
        self.u = u
        self.vol = vol

    def draw(self, es, x0, dt) :
        '''es is assumed to have a dimension of [npath, nd]'''
        xs = []

        x = np.ones(len(es))*np.log(x0)
        for e in es.T :
            x = x + (self.u - .5*self.vol*self.vol)*dt + self.vol*e*np.sqrt(dt)
            xs.append(x)

        return np.exp(np.array(xs).T)

from scipy.linalg import cholesky

class MF_GBM(object) :
    def __init__(self, u, cov) :
        self.u = u
        self.l = cholesky(cov)

    def draw(self, ei, x0, dt) :
        '''es is assumed to have a dimension of [npath, nd, nf]'''
        paths = []
        for es in ei :
            xs = []
            x = x0
            for e in es :
                x = x + x*self.u*dt + x*self.l.dot(e)*np.sqrt(dt)
                xs.append(x)
            paths.append(xs)

        return np.array(paths)


class CIR(object) :
    def __init__(self, k, u, vol) :
        self.k = k 
        self.u = u
        self.vol = vol

    def draw(self, es, x0, dt) :
        '''es is assumed to have a dimension of [npath, nd]'''
        xs = []

        x = np.ones(len(es))*x0
        for e in es.T :
            x1 = x + self.k*(self.u-x)*dt + self.vol*e*np.sqrt(x*dt)
            x2 = x + self.k*(self.u-x)*dt - self.vol*e*np.sqrt(x*dt)
            flag = np.less(x1, 0.)
            x = flag*x2 + (1.-flag)*x1
            xs.append(x)

        return np.array(xs).T


    def draw_log(self, es, x0, dt) :
        ys = []
        y = np.ones(len(es))*np.log(x0)

        for e in es.T :
            x = np.exp(y)
            y = y + (self.k*(self.u/x-1.) - .5/x*self.vol*self.vol)*dt \
                    + self.vol*e*np.sqrt(dt/x)
            y = np.minimum(np.maximum(y, -200), 200)
            ys.append(y)

        return np.exp(ys).T


    def __str__(self) :
        return "mean rev = %f, mean = %f, vol = %f" % (self.k, self.u, self.vol)



