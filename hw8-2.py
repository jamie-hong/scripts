import numpy as np
import matplotlib.pyplot as plt


k = 0.1
r_bar = 0.03
sigma = 0.05
r0 = 0.01
T = 10.
r10 = r0*np.exp(-k*T)+r_bar*(1-np.exp(-k*T))

# Euler

def CIR_Euler(k, r_bar, sigma, r0, T, N=1000, M=50000, eps=[]):  
    if np.size(eps) == 0:
        eps = np.random.normal(size=[N,M])
    else:
        N = int(np.size(eps,0))
        M = int(np.size(eps,1))
    
    dt = T/N
    r = np.zeros([N+1,M])
    r[0,:] = r0
    
    eps = eps * np.sqrt(dt)
    flag =np.zeros(M)
    
    for n in np.arange(1,N+1):
        r[n,:] = r[n-1,:] + k*(np.tile(r_bar,(1,M))-r[n-1,:])*dt + sigma*np.sqrt(r[n-1,:])*eps[n-1,:]
        neg_ind = np.where(r[n,:] < 0)
        flag[neg_ind] = 1
        eps[n-1,neg_ind] = - eps[n-1,neg_ind]
        r[n,:] = r[n-1,:] + k*(np.tile(r_bar,(1,M))-r[n-1,:])*dt + sigma*np.sqrt(r[n-1,:])*eps[n-1,:]
    
    return r, flag

# Log-Euler

def CIR_LogEuler(k, r_bar, sigma, r0, T, N=1000, M=10000, eps=[], floor=-10):
    if np.size(eps) == 0:
        eps = np.random.normal(size=[N,M])
    else:
        N = int(np.size(eps,0))
        M = int(np.size(eps,1))
   
    dt = T/N
    r = np.zeros([N+1,M])
    y = np.zeros([N+1,M])
    r[0,:] = r0
    y[0,:] = np.log(r0)

    eps = eps * np.sqrt(dt)

    for n in np.arange(1,N+1):
        y[n,:] = y[n-1,:] + 1/r[n-1,:]*(k*(np.tile(r_bar,(1,M))-r[n-1,:])-sigma**2/2)*dt + sigma/np.sqrt(r[n-1,:])*eps[n-1,:]
        floor_ind = np.where(y[n,:] < floor)
        y[n,floor_ind] = floor
        r[n,:] = np.exp(y[n,:])

    return r

# Milstein

def CIR_Milstein(k, r_bar, sigma, r0, T, N=1000, M=50000, eps=[]):
    if np.size(eps) == 0:
        eps = np.random.normal(size=[N,M])
    else:
        N = int(np.size(eps,0))
        M = int(np.size(eps,1))
    
    dt = T/N
    r = np.zeros([N+1,M])
    r[0,:] = r0
    
    eps = eps * np.sqrt(dt)

    for n in np.arange(1,N+1):
        r[n,:] = r[n-1,:] + k*(np.tile(r_bar,(1,M))-r[n-1,:])*dt + sigma*np.sqrt(r[n-1,:])*eps[n-1,:] 
        + 0.25*sigma**2/np.sqrt(r[n-1,:])*(eps[n-1,:]**2-dt)
        neg_ind = np.where(r[n,:] < 0)
        eps[n-1,neg_ind] = - eps[n-1,neg_ind]
        r[n,:] = r[n-1,:] + k*(np.tile(r_bar,(1,M))-r[n-1,:])*dt + sigma*np.sqrt(r[n-1,:])*eps[n-1,:]
        + 0.25*sigma**2/np.sqrt(r[n-1,:])*(eps[n-1,:]**2-dt)

    return r

steps = np.array([20, 50, 80, 100, 200, 500, 800, 1000])
diff = np.zeros([len(steps),3])
i = 0
for n in steps:
    r_E = CIR_Euler(k, r_bar, sigma, r0, T, n, M=50000)
    r_L = CIR_LogEuler(k, r_bar, sigma, r0, T, n, M=50000)
    r_M = CIR_Milstein(k, r_bar, sigma, r0, T, n, M=50000)
    diff[i,0] = np.abs(r10-np.mean(r_E[-1,:]))
    diff[i,1] = np.abs(r10-np.mean(r_L[-1,:]))
    diff[i,2] = np.abs(r10-np.mean(r_M[-1,:]))
    i += 1

plt.figure(figsize=[7,5])
plt.plot(T/steps, diff[:,0], T/steps, diff[:,1], T/steps, diff[:,2])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sample Step Size')
plt.ylabel('Bias in $\hat{r}_{10}$')
plt.legend(['Euler','Log-Euler','Milstein'], loc='best')
plt.show()


# Question 5
z = np.random.normal(size=[200,5000])
r_E, flag = CIR_Euler(k, r_bar, sigma, r0, T, eps=z)
r_L = CIR_LogEuler(k, r_bar, sigma, r0, T, eps=z)
r_M = CIR_Milstein(k, r_bar, sigma, r0, T, eps=z)

equiv = np.nonzero(flag==0)
eq = equiv[0][0]
reflected = np.nonzero(flag==1)
ref = reflected[0][0]

plt.figure(figsize=[14,5])
plt.subplot(1,2,1)

plt.plot(r_E[:,eq],'r--',LineWidth=2)
plt.plot(r_L[:,eq])
plt.plot(r_M[:,eq])
plt.xlabel('Step')
plt.ylabel('$r_t$')
plt.title('Equivalent Paths')
plt.legend(['Euler','Log-Euler','Milstein'], loc='best')

plt.subplot(1,2,2)
plt.plot(r_E[:,ref],'r--',LineWidth=2)
plt.plot(r_L[:,ref])
plt.plot(r_M[:,ref])
plt.xlabel('Step')
plt.ylabel('$r_t$')
plt.title('Reflected Paths')
plt.legend(['Euler','Log-Euler','Milstein'], loc='best')
plt.show()