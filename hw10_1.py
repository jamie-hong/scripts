import numpy as np
from matplotlib.pyplot import *
import me
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
matplotlib.rcParams.update({'font.size': 14})

# 1 and 2
dx = 0.01
x = np.arange(-6., 6., dx)
a = np.array([x, x*x]) # first and second moments (pricing equation)
u = np.array([0., 1.]) # first and second moment values
e = np.array([0., 0.]) # error
q = np.ones(np.size(x))/len(x) # prior belief (if uniform: cross entropy = regular entropy)
dual = me.MaxEntDual(q, a, u, e)

res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")

figure(figsize=[15, 5.5])
subplot(1, 2, 1)
plot(x, dual.dist(res.x)/dx);
xlim(-6, 6)
xlabel('$x$')
title('$\mathbb{E}[x] = 0, \; \mathbb{E}[x^2] = 1,  \; x \in (-6, 6) $');

subplot(1, 2, 2)

x = np.arange(-1.5, 1.5, dx)
a = np.array([x, x*x]) # first and second moments (pricing equation)
q = np.ones(np.size(x))/len(x) # prior belief (if uniform: cross entropy = regular entropy)
dual = me.MaxEntDual(q, a, u, e)

res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")
plot(x, dual.dist(res.x)/dx)
xlabel('$x$')
title('$\mathbb{E}[x] = 0, \; \mathbb{E}[x^2] = 1, \; x \in (-1.5, 1.5)$ ');

# 3
dy = 0.01
y = np.arange(0.01, 100, dy)
a = np.array([np.log(y), np.log(y)*np.log(y)]) # first and second moments (pricing equation)
q = np.ones(np.size(y))/np.size(y) # prior belief (if uniform: cross entropy = regular entropy)
dual = me.MaxEntDual(q, a, u, e)

res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")
pdf_y = dual.dist(res.x);

figure(figsize=[21, 5.5])
subplot(1, 3, 1)
plot(y, pdf_y/dy);
xlim(0, 8)
xlabel('$y$')
title('$\mathbb{E}[\log{(y)}] = 0, \; \mathbb{E}[\log^2{(y)}] = 1,  \; y \in (0, 100) $');

subplot(1, 3, 2)
cdf_y = np.cumsum(pdf_y)
cdf_logn = lognorm.cdf(y,1)
plot(cdf_y, cdf_logn,'o')
xlabel('$F_Y(y)$')
ylabel('$F_{lognorm}(y)$')

subplot(1, 3, 3)
plot(np.log(y), dual.dist(res.x)*y/dy)
xlim(-6, 6)
xlabel('$\log(y)$')
title('$\mathbb{E}[\log{(y)}] = 0, \; \mathbb{E}[\log^2{(y)}] = 1,  \; y \in (0, 100) $');


# 5
qn = norm.pdf(np.log(y))
qn = qn/np.sum(qn)

dual = me.MaxEntDual(qn, a, u, e)
res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")

figure(figsize=[15, 5.5])
subplot(1, 2, 1)
plot(y, dual.dist(res.x)/dy);
xlim(0, 8)
xlabel('$y$')
title('$\mathbb{E}[\log{(y)}] = 0, \; \mathbb{E}[\log^2{(y)}] = 1,  \; y \in (0, 100) $');

subplot(1, 2, 2)
plot(np.log(y), dual.dist(res.x)*y/dy);
xlim(-6, 6)
xlabel('$\log(y)$')
title('$\mathbb{E}[\log{(y)}] = 0, \; \mathbb{E}[\log^2{(y)}] = 1,  \; y \in (0, 100) $');

show()
