#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:09:52 2023

@author: tanguy
"""

# packages pour travailler avec les données
import pandas as pd      
import numpy as np   
import datetime as dt
import yfinance as yf

# packages pour le visualisation
import matplotlib.pyplot as plt

data_source = yf.Ticker('^GSPC')
# On récupère tout l'historique 
data_hist = data_source.history(period='max')
# Print data history
data_hist

#Début de l'utilisation de la momentum strategie 
data = data_hist["1985-12-01":"2020-12-31"].copy() 

len(
  data[
    data.Close.isna() |
    data.Close.isnull() |
    data.Close < 1e-8])

data.Close.plot()
plt.ylabel("Price");

#calculation of the returns#


def calc_returns(srs, offset=1):
    
    returns = srs / srs.shift(offset) - 1
    return returns

data["daily_returns"] = calc_returns(data["Close"])
data.head()

data["next_day_returns"] = data["daily_returns"].shift(-1)
data.head()


#Code pour atténuer les effets des outliers 
VOL_THRESHOLD = 5  
data["srs"] = data["Close"]
SMOOTH_WINDOW = 252 
ewm = data["srs"].ewm(halflife=SMOOTH_WINDOW)

# Exponentially-weighted moving mean
means = ewm.mean()

# Exponentially-weighted moving standard deviation
stds = ewm.std()

# Upper bound (EWM_mean + 5 * EWM_std)
ub = means + VOL_THRESHOLD * stds
# On remplace les valeurs trop grandes (plus grandes que l'upper bound) par l'upper bound 
data["srs"] = np.minimum(data["srs"], ub);

# Lower bound (EWM_mean - 5 * EWM_std)
lb = means - VOL_THRESHOLD * stds

# On remplace les valeurs trop petites (plus petites que le lower bound) par le lower bound
data["srs"] = np.maximum(data["srs"], lb);

# on calcule les daily_returns à partir de la série srs (celle que nous avons winsorisée)
data["daily_returns"] = calc_returns(data["srs"],1)

plt.plot(data["daily_returns"]);

# Mise à l'échelle de la volatilité
def rescale_to_target_volatility(srs,vol):
    return srs *  vol / srs.std() / np.sqrt(252)

def plot_captured_returns(next_day_captured, plot_with_equal_vol = None):
    
    if plot_with_equal_vol is not None:
        srs = rescale_to_target_volatility(next_day_captured.copy(),vol=plot_with_equal_vol)
    else:
        srs = next_day_captured.copy()
        
    ((srs.shift(1) + 1).cumprod() - 1).plot()
    plt.ylabel("Cumulative  returns");
    
    # Load long-only returns and plot
captured_returns_longonly = data['next_day_returns']["1990-01-01":]
plot_captured_returns(captured_returns_longonly)

#3  Calculs des différentes métriques des performances 

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
    
    # Calcul des métriques anualisées :
    returns_annualised =  returns(srs, tau)
    vol_annualised = volatility(srs, tau)
    downside_devs_annualised = calc_downside_deviation(srs, tau)
    max_drawdown = calc_max_drawdown(srs)
    pnl_ratio = calc_profit_and_loss_ratio(srs)
    perc_positive_return = lperc_positive_returns(srs)
    
    # Calcul des "risk-adjusted performance metrics" :
    sharpe = sharpe_ratio(srs, tau)
    sortino = sortino_ratio(srs, tau)
    calmar = calmar_ratio(srs, tau)
    
    # Afficher les différents results 
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
   
    # Return performance metrics
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


# Calcul des métriques de performances pour une stratégie basique long-only
stats_longonly = calculate_statistics(captured_returns_longonly)

VOL_LOOKBACK = 60 # Lookback window pour calculer la daily volatility
VOL_TARGET = 0.15 # Volatility target annualisée

#Calcul de la volatilité quotidienne

def volatility_scaled_returns(daily_returns, vol_lookback = VOL_LOOKBACK, vol_target = VOL_TARGET):
    
    daily_vol = (
        daily_returns
        .ewm(span=vol_lookback, min_periods=vol_lookback).std()
        .fillna(method='bfill')
    )
    
#volatilité annualisée 
    vol = daily_vol * np.sqrt(252)
    
    scaled_returns = vol_target * daily_returns / vol.shift(-1) # shift(-1) car ex-ante
    
    return scaled_returns

# Calcul des returns ajustés de la volatility
data['scaled_returns'] = volatility_scaled_returns(data["daily_returns"])
print(f"Signal annualised volatility: {data['scaled_returns'].std()*np.sqrt(252):.2%}")

data["trading_rule_signal"] = (1 + data["scaled_returns"]).cumprod()
data["scaled_next_day_returns"] = data["scaled_returns"].shift(-1)

captured_returns_volscaled_lo = data["scaled_next_day_returns"]["1990-01-01":]

# Plot des returns
plot_captured_returns(captured_returns_longonly, plot_with_equal_vol = VOL_TARGET)

# Plot des volatility-scaled returns
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Unscaled", "Vol. scaled"]);

print("Vol. scaled long only:")
stats_volscaled_longonly = calculate_statistics(captured_returns_volscaled_lo)

print("Unscaled long only:")
stats_longonly = calculate_statistics(captured_returns_longonly)

#Utilisation time Series Momentum 

data["annual_returns"] = calc_returns(data["srs"], 252)

# Pour rappel : stratégie long-only
captured_returns_volscaled_lo = data["scaled_next_day_returns"]["1990-01-01":]

# Stratégie TSMOM, on ajuste le returns en multipliant par +1 si annual_returns > 0 et par -1 si annual_returns < 0
captured_returns_volscaled_tsmom = (
    np.sign(data["annual_returns"])*data["scaled_next_day_returns"]
)["1990-01-01":]

# Plot des returns de la stratégie time series momentum vs de la stratégie long-only
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol = VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol = VOL_TARGET)
plt.legend(["Long Only", "TSMOM"]);

# Calcul des métriques de performance
stats_volscaled_tsmom = calculate_statistics(captured_returns_volscaled_tsmom)

signal_lookback = [5, 21, 63, 126, 252]

for lookback in signal_lookback:
    srs = (
        np.sign(calc_returns(data['srs'], lookback))*data['scaled_next_day_returns']
    )['1990-01-01':]
    plot_captured_returns(srs, plot_with_equal_vol=VOL_TARGET)
    
plt.legend(signal_lookback);

ws = [.0, .25, .5, .75, 1.0]

for w in ws:
    srs = (
        w * np.sign(calc_returns(data['srs'], 21))
        * data['scaled_next_day_returns']
        + (1-w) * np.sign(calc_returns(data['srs'], 252))
        * data['scaled_next_day_returns']
    )['1990-01-01':]
    plot_captured_returns(srs, plot_with_equal_vol=VOL_TARGET)
    print(f'w = {w}')
    calculate_statistics(srs)
    print()
    
plt.legend(ws);


print('Correlation.\n') 
captured_returns_volscaled_lo.corr(captured_returns_volscaled_tsmom)

print('Portefeuille Pondéré.\n')
calculate_statistics(.5*(captured_returns_volscaled_lo+captured_returns_volscaled_tsmom))
plot_captured_returns(.5*(captured_returns_volscaled_lo+captured_returns_volscaled_tsmom), plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_lo, plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom, plot_with_equal_vol=VOL_TARGET)

legends = ['portfolio', 'Long only', 'TSMOM']
plt.legend(legends);

print('Comparaison performance.\n')


calculate_statistics(.5*(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"]+captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"]))
plot_captured_returns(.5*(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"]+captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"]), plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_lo["1990-01-02": "2010-12-31"], plot_with_equal_vol=VOL_TARGET)
plot_captured_returns(captured_returns_volscaled_tsmom["1990-01-02": "2010-12-31"], plot_with_equal_vol=VOL_TARGET)

legends = ['portfolio', 'Long only', 'TSMOM']
plt.legend(legends);

