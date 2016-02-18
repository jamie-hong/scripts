import numpy as np
import pandas as pd
import fmt
import matplotlib.pyplot as plt

djiaurl = "https://raw.githubusercontent.com/yadongli/nyumath2048/master/data/djia.csv"
djia = pd.read_csv(djiaurl, index_col=[0])

def download_hist_prices(tickers) : # we download the stock prices from Yahoo!Finance
    base_url = "http://ichart.finance.yahoo.com/table.csv?s="
    closeKey = 'Adj Close'
    dfs = []

    for ticker in tickers:
        try:
            url = base_url + ticker
            dfs.append(pd.read_csv(url, parse_dates=[0], index_col=[0]).rename(columns={closeKey:ticker})[ticker])
        except Exception:
            print "error in " + ticker
            continue

    return pd.concat(dfs, axis=1, join='inner')

histprice = download_hist_prices(djia.index).sort()
histprice.plot(legend=False, title='Historical Prices')

rtn = np.log(histprice).diff()
rtn=rtn.iloc[1:]

cv=rtn.cov() # covariance matrix
cr=rtn.corr() # correlation matrix

fmt.displayDF(cv*1e4,fontsize=1)
fmt.displayDF(cr,fontsize=1)

hxr = rtn.mean()*252 - 0.03 # historical excess returns assuming 260 trading days per year

cvi = np.linalg.inv(cv)
w = cvi.dot(hxr)/hxr.T.dot(cvi).dot(hxr)
w = w/np.sum(w) # optimal portfolio using hist_xr
plt.bar(np.arange(30),w,width=1,alpha=0.4)
plt.xticks(np.arange(30)+0.5,rtn.keys(),rotation=90)
plt.title('Mean-Variance Portfolio Weights')


w2 = djia.Weights
vb = w2.dot(cv).dot(w2)
ixr = cv.dot(w2)/vb # implied annual excess returns

df_ret = pd.concat([hxr,ixr], axis = 1, keys = ['Historical','Implied'])
df_ret.plot(title='Historical and Implied Excess Returns of Dow 30',style='.--',markersize=14
           , figsize=[7, 5])

cond = np.linalg.norm(cvi)*np.linalg.norm(cv) # condition number of covariance matrix

# alternating series, not actual odd/even dates
odd = rtn.iloc[::2]
even = rtn.iloc[1::2]

hxr_odd = odd.mean()*252 - 0.03
hxr_even = even.mean()*252 - 0.03
std_odd = odd.std()*np.sqrt(252)
std_even = even.std()*np.sqrt(252)
std = rtn.std()*np.sqrt(252)

df_std=pd.concat([std, std_odd, std_even], axis = 1, keys=['full','odd','even'])
df_std.plot(title='Annualized Volatility Estimates', style='.--',markersize=14)
df_mean = pd.concat([hxr,hxr_odd,hxr_even], axis = 1,keys=['full','odd','even'])
df_mean.plot(title='Annualized Return Estimates', style='.--',markersize=14)

plt.show()