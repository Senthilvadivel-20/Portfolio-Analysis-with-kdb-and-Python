import pandas as pd
import numpy as np


def hello():
    return "Hello Senthil!"

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("./data/edhec_hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


# Compute the skewness
def skewness(r):
    mean_r = r.mean() 
    exp = ((r- mean_r)**3).mean()
    std = (r.std(ddof=0))**3
    return exp/std

# Compute kurtosis
def kurtosis(r):
    exp = ((r-r.mean())**4).mean()
    sigma = (r.std(ddof=0))**4
    return exp/sigma


def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation


# Historic value at risk

# It's kind of recursive function

def var_historic(r,level=5):
    """
    Return the VaR at specified level.
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return - np.percentile(r,level)
    else:
        raise TypeError("Function is expecting Series or Data Frame")
        

# Compute the conditional value at risk which are below VaR

def cvar_historic(r, level = 5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= - var_historic(r, level = level)
        return r[is_beyond].mean()
    else: 
        raise TypeError("Expecting Series or Data Frame")