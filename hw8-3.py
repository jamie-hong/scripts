import numpy as np

kx, ux, volx = .5, .05, .05
ky, uy, voly = .1, .02, .01
rho_xy = .5 # correlation between dw_x and dw_y
x0, y0 = .01, .01
dt = 0.01
T = 1
N = T/dt
M = 50000

z1 = np.random.normal(size=[N,M])
z2 = rho_xy*z1 + np.sqrt(1-rho_xy**2)*np.random.normal(size=[N,M])

z1 = z1*np.sqrt(dt)
z2 = z2*np.sqrt(dt)

x = np.zeros([N+1,M])
x[0,:] = x0
y = np.zeros([N+1,M])
y[0,:] = y0
for n in np.arange(1,N+1):
    x[n,:] = x[n-1,:] + kx*(np.tile(ux,(1,M))-x[n-1,:])*dt + volx*x[n-1,:]*z1[n-1,:]
    y[n,:] = y[n-1,:] + ky*(np.tile(uy,(1,M))-y[n-1,:])*dt + voly*y[n-1,:]*z2[n-1,:]


corr = np.corrcoef(x[-1,:],y[-1,:])[0,1]
print("The simulated correlation between x(1Y) and y(1Y) is %.3f" % corr)

corr_a = 2*np.sqrt(kx*ky)/(kx+ky)*(1-np.exp(-(kx+ky)*T))/np.sqrt((1-np.exp(-2*kx*T))*(1-np.exp(-2*ky*T)))*rho_xy