import pandas as pd
import numpy as np
from numpy.linalg import *
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import fmt


cmturl = "https://raw.githubusercontent.com/yadongli/nyumath2048/master/data/cmt.csv"
cmt_rates = pd.read_csv(cmturl, parse_dates=[0], index_col=[0])

cmt_rates.plot(legend=False);
tenors = cmt_rates.columns.map(float)
tenorTags = ['T=%g' % m for m in tenors]

V = cmt_rates.cov()
V_cond = cond(V,2)

u,s,v= svd(V)
fmt.displayDF(pd.DataFrame(s, columns=["Singular Values of V"]).T, fmt="3f")

delta = cmt_rates.diff()[1:]
V_d = delta.cov()
C_d = delta.corr()

ev_V, evec_V = eig(V_d)
ev_C, evec_C = eig(C_d)

fmt.displayDFs(pd.DataFrame(evec_V, columns=["PC %d" % x for x in range(1,10,1)]), 
               pd.DataFrame(evec_C, columns=["PC %d" % x for x in range(1,10,1)]), 
               headers=["Eigenvectors of Covariance Matrix", "EigenVectors of Correlation Matrix"],fmt="4f")

pct_V = np.cumsum(ev_V)/sum(ev_V)*100
fmt.displayDF(pd.DataFrame({'Covariance PC':range(1, len(ev_V)+1), 
                            'Eigenvalues':ev_V, 'Cumulative Var (%)': pct_V}).set_index(['Covariance PC']).T, "4f")

pct_C = np.cumsum(ev_C)/sum(ev_C)*100
fmt.displayDF(pd.DataFrame({'Correlation PC':range(1, len(ev_C)+1), 
                            'Eigenvalues':ev_C, 'Cumulative Var (%)': pct_C}).set_index(['Correlation PC']).T, "4f")

H_V = np.diag(ev_V[0:3]**0.5)
L_V = evec_V[:,:3].dot(H_V)
Z = np.random.randn(3,5000)
dt = 1
Sdw = L_V.dot(Z)*np.sqrt(dt)

df = pd.DataFrame(Sdw.T, columns=tenors)

V_ds = df.cov()
VDiff=np.mat(V_d)-np.mat(V_ds)

ending = np.array(cmt_rates.iloc[-1,:])
SimRates = np.cumsum(Sdw.T,0) + repmat(ending,5000,1)
df_s = pd.DataFrame(SimRates, columns=tenors)
ax = df_s.plot(figsize=[11,8])
ax.set_title("Simulated Interest Rates",fontsize=18)
ax.set_xlabel("Time(Year)",fontsize=18)
ax.set_xticklabels(np.arange(0,21,4))
ax.set_ylabel("Daily Rate (%)",fontsize=18)
fmt.displayDFs(pd.DataFrame({'Tenor':tenors,
                            'Mean':df_s.mean(), 
                            'Standard Deviation': df_s.std(),
                            '2% Quantile': df_s.quantile(q=0.02),
                            '98% Quantile': df_s.quantile(q=0.98)}).set_index(['Tenor']).T, 
                            headers=["Statistics of Simulated Interest Rate Levels"], fmt="4f")

Oneyear = np.zeros((4751,4))
Tenyear = np.zeros((4751,4))
for i in range(4751):
    Oneyear[i,0] = df_s.iloc[i:250+i,2].mean()
    Oneyear[i,1] = df_s.iloc[i:250+i,2].std()
    Oneyear[i,2] = df_s.iloc[i:250+i,2].quantile(q=0.02)
    Oneyear[i,3] = df_s.iloc[i:250+i,2].quantile(q=0.98)
    Tenyear[i,0] = df_s.iloc[i:250+i,7].mean()
    Tenyear[i,1] = df_s.iloc[i:250+i,7].std()
    Tenyear[i,2] = df_s.iloc[i:250+i,7].quantile(q=0.02)
    Tenyear[i,3] = df_s.iloc[i:250+i,7].quantile(q=0.98)

ax2 = pd.DataFrame(Oneyear,columns=['mean','std','2%','98%']).plot(figsize=[8,6])
ax2.set_title("One-Year Rate Statistics",fontsize=18)
ax2.set_xlabel("Time(Year)",fontsize=18)
ax2.set_xticklabels(np.arange(0,21,4))
ax2.set_ylabel("Rate (%)",fontsize=18)
ax3 = pd.DataFrame(Tenyear,columns=['mean','std','2%','98%']).plot(figsize=[8,6])
ax3.set_title("Ten-Year Rate Statistics",fontsize=18)
ax3.set_xlabel("Time(Year)",fontsize=18)
ax3.set_xticklabels(np.arange(0,21,4))
ax3.set_ylabel("Rate (%)",fontsize=18)
plt.show()