import pandas as pd
import numpy as np


def hello():
    return "Hello Senthil!"


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("./data/edhec-hedgefundindices.csv",
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