import numpy as np
import trid
import matplotlib.pyplot as plt
import time

x0 = np.array([.1, 1.,2, 3., 5., 10., 25.])
y0 = np.array([.0025, .01, .016,.02, .025, .030, .035])

ts = trid.TridiagTension(1)
ts.build(x0,y0)

M = ts.a
y = ts.b

def thomas(M,y):
    '''Thomas algorithm to solve tridiagonal matrix'''
    length = len(M)
    b = np.diag(M)
    a = np.diag(M[1:,:-1])
    c = np.diag(M[:-1,1:])
    
    N = M - np.diag(b) # checker matrix 
    N[1:,:-1] = N[1:,:-1] - np.diag(a)
    N[:-1,1:] = N[:-1,1:] - np.diag(c)
    if M.shape[0] != M.shape[1] or not np.all(N==0):
        raise ValueError('matrix is not tridiagonal')
    
    a = np.insert(a,0,0)
    c = np.append(c,0) 
    x = np.zeros(length)
    cp = np.zeros(length)
    yp = np.zeros(length)
    
    cp[0] = c[0]/b[0]
    yp[0] = y[0]/b[0]
    
    for i,m in enumerate(M):
        if i > 0 & i < length-1:
            cp[i] = c[i] / (b[i] - a[i]*cp[i-1])
            yp[i] = (y[i] - a[i]*yp[i-1])/(b[i] - a[i]*cp[i-1])
        elif i == length - 1:
            yp[i] = (y[i] - a[i]*yp[i-1])/(b[i] - a[i]*cp[i-1])
    
    k = np.arange(length-1)[::-1]
    x[length-1] = yp[length-1]
    
    for i in k:
        x[i] = yp[i] - cp[i]*x[i+1]
    
    return x

x = thomas(M,y)

x_b = np.linalg.solve(M,y)
print("Same solutions? %s" % np.allclose(x,x_b))

t = np.zeros(200)
for i in range(5,205,1):
    x0 = np.arange(0.1,(i+1)*0.1,0.1)
    y0 = np.log(x0+1)

    ts = trid.TridiagTension(1)
    ts.build(x0,y0)
    M = ts.a
    y = ts.b

    t1 = time.clock()
    ans = thomas(M,y)
    t2 = time.clock()
    t[i-5] = t2 - t1

plt.figure(figsize=(7,5))
plt.plot(range(5,205,1),t)
plt.xlabel('Matrix dimension',fontsize=14)
plt.ylabel('Computational time (sec)',fontsize=14)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
plt.show()    