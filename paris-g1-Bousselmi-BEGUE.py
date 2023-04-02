#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:09:52 2023

   
@author: tanguy
"""

#!/usr/bin/env python
# coding: utf-8

# # Python project - Momentum analysis

# 
# For this python project we decided to analyse the momentum strategy using the CAC40 index. Finance professionals use momentum analysis as a method to evaluate the strength and direction of an asset's price movement. It assists traders in deciding when to purchase or sell based on momentum by calculating the rate of change in the asset's price over a given time frame. This approach is based on the concept that patterns in price changes typically last for a given amount of time.This strategy prevaled in the 1990s just when the CAC40 was created. 
# 
# This project starts with the collection of CAC40 data and the calculation of returns. Then, a basic long-only strategy will be introduced as a benchmark to compare it to other strategies. Using different metrics such as Sharpe, Sortino, and Calmar ratios we will measure the performance of this strategy. 
# 
# Then we will introduce the time series momentum strategy and explore its performance by varying one parameter: the time horizon. 
# 
# Finally, we will analyse the benefits of using time series momentum strategies for their diversification role when combined in a portfolio with long-only strategies.
# 

# ### Collection of CAC40 data

# In[1]:


#If you have never used Yahoo finance#
get_ipython().system('pip install yfinance')


# In[4]:


#Instal the packages#

import pandas as pd      
import numpy as np   
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt


# In[5]:


#Import data from Yahoo finance#

data_source = yf.Ticker('^FCHI')
data = data_source.history(period='max')
data


# In[6]:


#Control the quality of the data + graph#
len(
  data[
    data.Close.isna() |
    data.Close.isnull() |
    data.Close < 1e-8
  ]
)
data.Close.plot()
plt.ylabel("Price");


# In[7]:


#Compute the returns#
def calc_returns(srs, offset=1):
    
    returns = srs / srs.shift(offset) - 1
    return returns
data["daily_returns"] = calc_returns(data["Close"])
data.head()
data["next_day_returns"] = data["daily_returns"].shift(-1)
data.head()


# In[8]:


##Lon-only strategy##

# Adapte the volatility
def rescale_to_target_volatility(srs,vol):
    return srs *  vol / srs.std() / np.sqrt(252)

def plot_captured_returns(next_day_captured, plot_with_equal_vol = None):
    
    if plot_with_equal_vol is not None:
        srs = rescale_to_target_volatility(next_day_captured.copy(),vol=plot_with_equal_vol)
    else:
        srs = next_day_captured.copy()
        
    ((srs.shift(1) + 1).cumprod() - 1).plot()
    plt.ylabel("Cumulative  returns");

#Load long-only returns and plot
captured_returns_longonly = data['next_day_returns']["1990-01-01":]
plot_captured_returns(captured_returns_longonly)

#Define performance metrics and ratios
def returns(srs, tau=252):
    return srs.mean()*tau

def volatility(srs, tau=252):
    return srs.std()*np.sqrt(tau)
    
def calc_downside_deviation(srs, tau=252):
    negative_returns = srs.apply(lambda x: x if x < 0 else np.nan).dropna() * np.sqrt(tau)
    return negative_returns.std()
    
def calc_max_drawdown(srs):
    cumulative_max = srs.cummax()
    drawdown = cumulative_max - srs
    return drawdown.max()
    
def calc_profit_and_loss_ratio(srs):
    return np.mean(srs[srs>0])/np.mean(np.abs(srs[srs<0]))
    
def lperc_positive_returns(srs):
    return len(srs[srs>0])/len(srs)

def sharpe_ratio(srs, tau=252):
    return srs.mean()/srs.std()*np.sqrt(tau)

def sortino_ratio(srs, tau=252):
    return srs.mean() / calc_downside_deviation(srs, tau) * tau

def calmar_ratio(srs, tau=252):
    return srs.mean() / calc_max_drawdown(srs) * tau

def calculate_statistics(next_day_captured, print_results=True):
    tau = 252
    
    srs = next_day_captured.shift(1)
    
#Compute performance metrics and ratios :
    returns_annualised =  returns(srs, tau)
    vol_annualised = volatility(srs, tau)
    downside_devs_annualised = calc_downside_deviation(srs, tau)
    max_drawdown = calc_max_drawdown(srs)
    pnl_ratio = calc_profit_and_loss_ratio(srs)
    perc_positive_return = lperc_positive_returns(srs)
    
    sharpe = sharpe_ratio(srs, tau)
    sortino = sortino_ratio(srs, tau)
    calmar = calmar_ratio(srs, tau)
    
    if print_results:
        print("\033[4mPerformance Metrics:\033[0m")
        print(f"Annualised Returns = {returns_annualised:.2%}")
        print(f"Annualised Volatility = {vol_annualised:.2%}")
        print(f"Downside Deviation = {downside_devs_annualised:.2%}")
        print(f"Maximum Drawdown = {max_drawdown:.2%}")
        print(f"Sharpe Ratio = {sharpe:.2f}")
        print(f"Sortino Ratio = {sortino:.2f}")
        print(f"Calmar Ratio = {calmar:.2f}")
        print(f"Percentage of positive returns = {perc_positive_return:.2%}")
        print(f"Profit/Loss ratio = {pnl_ratio:.3f}")
   
    return {
        "returns_annualised":  returns_annualised,
        "vol_annualised": vol_annualised,
        "downside_deviation_annualised": downside_devs_annualised,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "pnl_ratio": pnl_ratio,
      }

stats_longonly = calculate_statistics(captured_returns_longonly)


# In[ ]:


## TIME SERIES MOMENTUM ##


# In[11]:


#Scale volatility to improuve Time series momentum
VOL_LOOKBACK = 60 # Lookback window pour calculer la daily volatility
VOL_TARGET = 0.15 # Volatility target annualisée

def volatility_scaled_returns(daily_returns, vol_lookback = VOL_LOOKBACK, vol_target = VOL_TARGET):
    
    daily_vol = (
        daily_returns
        .ewm(span=vol_lookback, min_periods=vol_lookback).std()
        .fillna(method='bfill')
    )
    
    vol = daily_vol * np.sqrt(252)
    
    scaled_returns = vol_target * daily_returns / vol.shift(-1) # shift(-1) car ex-ante
    
    return scaled_returns

# adjusted returns
data['scaled_returns'] = volatility_scaled_returns(data["daily_returns"])
print(f"Signal annualised volatility: {data['scaled_returns'].std()*np.sqrt(252):.2%}")

data["trading_rule_signal"] = (1 + data["scaled_returns"]).cumprod()
data["scaled_next_day_returns"] = data["scaled_returns"].shift(-1)

captured_returns_volscaled_lo = data["scaled_next_day_returns"]["1990-01-01":]

# Plot returns
plot_captured_returns(captured_returns_longonly, plot_with_equal_vol = VOL_TARGET)

# Plot volatility-scaled returns
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Unscaled", "Vol. scaled"]);

print("Vol. scaled long only:")
stats_volscaled_longonly = calculate_statistics(captured_returns_volscaled_lo)
print("Unscaled long only:")
stats_longonly = calculate_statistics(captured_returns_longonly)


# In[25]:


#Compute the returns#
def calc_returns(srs, offset=1):
    
    returns = srs / srs.shift(offset) - 1
    return returns
data["annual_returns"] = calc_returns(data["Close"],252)
data.head()


# In[26]:


# strategy long-only
captured_returns_volscaled_lo = data["scaled_next_day_returns"]["1990-01-01":]

# Strategy TSMOM
captured_returns_volscaled_tsmom = (
    np.sign(data["annual_returns"])*data["scaled_next_day_returns"]
)["1990-01-01":]

# Plot returns of TSMOM and long-only
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Long Only", "TSMOM"]);


# In[27]:


#Compute performance metrics and ratios for TSMOM :
stats_volscaled_tsmom = calculate_statistics(captured_returns_volscaled_tsmom)


# In[29]:


#explore performance by varying the time horizon. 

signal_lookback = [5, 21, 63, 126, 252]

for lookback in signal_lookback:
    srs = (
        np.sign(calc_returns(data['Close'], lookback))*data['scaled_next_day_returns']
    )['1990-01-01':]
    plot_captured_returns(srs, plot_with_equal_vol=VOL_TARGET)
    
plt.legend(signal_lookback);

#combined signals to monitor better performance

ws = [.0, .25, .5, .75, 1.0]

for w in ws:
    srs = (
        w * np.sign(calc_returns(data['Close'], 21))
        * data['scaled_next_day_returns']
        + (1-w) * np.sign(calc_returns(data['Close'], 252))
        * data['scaled_next_day_returns']
    )['1990-01-01':]
    plot_captured_returns(srs, plot_with_equal_vol=VOL_TARGET)
    print(f'w = {w}')
    calculate_statistics(srs)
    print()
    
plt.legend(ws);


# ### TSMOM as a tool to diversify 

# In[30]:


#Search correlation between TSMOM and long-only
captured_returns_volscaled_lo.corr(captured_returns_volscaled_tsmom)


# In[31]:


#Analyse the returns of portfolio that has 50% TSMOM and 50% long-only
calculate_statistics(.5*(captured_returns_volscaled_lo+captured_returns_volscaled_tsmom))
plot_captured_returns(.5*(captured_returns_volscaled_lo+captured_returns_volscaled_tsmom), plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol=VOL_TARGET)

legends = ['portfolio', 'Long only', 'TSMOM']
plt.legend(legends);


# In[32]:


#Zoom the graph to get a better view
calculate_statistics(.5*(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"]+captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"]))
plot_captured_returns(.5*(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"]+captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"]), plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"], plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"], plot_with_equal_vol=VOL_TARGET)

legends = ['portfolio', 'Long only', 'TSMOM']
plt.legend(legends);


# The performance of the portfolio is better than that of long-only or TSMOM alone.
# 
# To observe the phenomenon, it is interesting to zoom in on a part of the curve. It appears that the long-only only strategy experiences considerable losses during the crisis periods of the internet bubble (2001) and the subprime crisis (2008–2009). On the other hand, the TSMOM strategy is kept and allows for the loss-compensation.

