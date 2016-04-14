import numpy as np
from matplotlib.pyplot import *
import me
from scipy.optimize import minimize
from scipy.stats import norm

matplotlib.rcParams.update({'font.size': 14})

x = np.arange(-10., 10., .01)
a = np.array([x**3, x**3]) # first and second moments (pricing equation)
u = np.array([2., -2.]) # first and second moment values
e = np.array([0., 0.]) # error
q = norm.pdf(x) # prior belief (if uniform: cross entropy = regular entropy)
dual = me.MaxEntDual(q, a, u, e)

res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")

figure(figsize=[14, 12])
subplot(2, 2, 1)
plot(x, dual.dist(res.x));
xlim(-6, 6)
xlabel('$x$')
title('$\mathbb{E}[x^3] = 2, \; \mathbb{E}[x^3] = -2,  \; \epsilon = %s $' % list(e));

e = np.array([1., 0.]) # error
dual = me.MaxEntDual(q, a, u, e)
res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")
subplot(2, 2, 2)
plot(x, dual.dist(res.x));
xlim(-6, 6)
xlabel('$x$')
title('$\mathbb{E}[x^3] = 2, \; \mathbb{E}[x^3] = -2,  \; \epsilon = %s $' % list(e));

e = np.array([0., 1.]) # error
dual = me.MaxEntDual(q, a, u, e)
res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")
subplot(2, 2, 3)
plot(x, dual.dist(res.x));
xlim(-6, 6)
xlabel('$x$')
title('$\mathbb{E}[x^3] = 2, \; \mathbb{E}[x^3] = -2,  \; \epsilon = %s $' % list(e));

e = np.array([1., 1.]) # error
dual = me.MaxEntDual(q, a, u, e)
res = minimize(dual.dual, np.zeros(len(u)), jac=dual.grad, method="BFGS")
subplot(2, 2, 4)
plot(x, dual.dist(res.x));
xlim(-6, 6)
xlabel('$x$')
title('$\mathbb{E}[x^3] = 2, \; \mathbb{E}[x^3] = -2,  \; \epsilon = %s $' % list(e));

show()