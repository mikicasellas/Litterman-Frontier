import datetime
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.moment_helpers as mh
import sys
import json
import os
import io
import shutil
from cvxopt.solvers import qp
from cvxopt import matrix
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions, CLA

FMP_API_KEY = st.secrets['API_key']

def load_data(config):
    symbols = sorted(config['views'].keys())
    max_lookback_years = config['max_lookback_years']
    prices = load_prices(symbols, max_lookback_years, config['price_data'])
    market_prices = load_market_prices()
    mkt_caps = load_mkt_caps(symbols)
    return prices, market_prices, mkt_caps, symbols, config

def load_prices(symbols, max_lookback_years, data_source):
    "begin loading prices"
    price_data = pd.DataFrame()
    if data_source == 'fmp':
        start_date = (datetime.datetime.today()
                      - datetime.timedelta(days=365*max_lookback_years)).date()
        symbols = sorted(symbols)
        if len(symbols) > 0:
            try:
                for symbol in symbols:
                    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&apikey={FMP_API_KEY}"
                    data = requests.get(url).json()
                    df = pd.DataFrame(data['historical'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    price_data[symbol] = df['adjClose']
            except:
                print('Error downloading data from FMP')
                sys.exit(-1)
    price_data = price_data.sort_index()
    return price_data

def load_market_prices():
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/SPY?apikey={FMP_API_KEY}"
    data = requests.get(url).json()
    df = pd.DataFrame(data['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    mkt_prices = df['adjClose']
    return mkt_prices

def load_mkt_caps(symbols):
    base_url = "https://financialmodelingprep.com/api/v3/profile/"
    mcaps = {}
    try:
        for ticker in symbols:
                response = requests.get(f"{base_url}{ticker}?apikey={FMP_API_KEY}")
                data = response.json()

                # Extract the market cap and add it to the dictionary
                mcaps[ticker] = data[0]['mktCap']
    except:
        print('Error downloading market cap data from FMP')
        sys.exit(-1)
    return mcaps

def calc_omega(config, symbols):
    variances = []
    for symbol in sorted(symbols):
        view = config['views'][symbol]
        lb, ub  = view[0], view[2]
        std_dev = (ub - lb)/2
        variances.append(std_dev ** 2)
    omega = np.diag(variances)
    return omega

def load_mean_views(views, symbols):
    mu = {}
    for symbol in sorted(symbols):
        mu[symbol] = views[symbol][1]
    return mu

def plot_black_litterman_results(ret_bl, covar_bl, market_prior, mu):
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(mu)],
                           index=["Prior", "Posterior", "Views"]).T
    ax = rets_df.plot.bar(figsize=(12,8), title='Black-Litterman Expected Returns')
    rets_fig = ax.get_figure()
    covar_bl_fig = plot_heatmap(covar_bl, 'Black-Litterman Covariance', '', '')
    corr_bl = mh.cov2corr(covar_bl)
    corr_bl = pd.DataFrame(corr_bl, index=covar_bl.index, columns=covar_bl.columns)
    corr_bl_fig = plot_heatmap(corr_bl, 'Black-Litterman Correlation', '', '')

    return rets_fig, covar_bl_fig, corr_bl_fig,

def plot_heatmap(df, title, xlabel, ylabel):

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25, left=0.25)
    heatmap = ax.pcolor(df, edgecolors='w', linewidths=1)
    cbar = plt.colorbar(heatmap)
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(df.columns , rotation=45)
    ax.set_yticklabels(df.index)

    for y, idx in enumerate(df.index):
        for x, col in enumerate(df.columns):
            plt.text(x + 0.5, y + 0.5, '%.2f' % df.loc[idx, col],
                     horizontalalignment='center', verticalalignment='center',)

    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return fig

def calc_black_litterman(market_prices, mkt_caps, covar, config, symbols):
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mkt_caps, delta, covar)
    mu = load_mean_views(config['views'], symbols)
    omega = calc_omega(config, symbols)
    bl = BlackLittermanModel(covar, pi="market", market_caps=mkt_caps, risk_aversion=delta,
                             absolute_views=mu, omega=omega)
    rets_bl = bl.bl_returns()
    covar_bl = bl.bl_cov()
    rets_fig, covar_bl_fig, corr_bl_fig = plot_black_litterman_results(rets_bl, covar_bl, market_prior, mu);
    
    return rets_bl, covar_bl, rets_fig, covar_bl_fig, corr_bl_fig

def kelly_optimize(M_df, C_df, config):
    "objective function to maximize is: g(F) = r + F^T(M-R) - F^TCF/2"
    r = config['annual_risk_free_rate']
    M = M_df.to_numpy()
    C = C_df.to_numpy()

    n = M.shape[0]
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    G = matrix(0.0, (n, n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n, 1))
    try:
        max_pos_size = float(config['max_position_size'])
    except KeyError:
        max_pos_size = None
    try:
        min_pos_size = float(config['min_position_size'])
    except KeyError:
        min_pos_size = None
    if min_pos_size is not None:
        h = matrix(min_pos_size, (n, 1))

    if max_pos_size is not None:
       h_max = matrix(max_pos_size, (n,1))
       G_max = matrix(0.0, (n, n))
       G_max[::n+1] = 1.0
       G = matrix(np.vstack((G, G_max)))
       h = matrix(np.vstack((h, h_max)))

    S = matrix((1.0 / ((1 + r) ** 2)) * C)
    q = matrix((1.0 / (1 + r)) * (M - r))
    sol = qp(S, -q, G, h, A, b)
    kelly = np.array([sol['x'][i] for i in range(n)])
    kelly = pd.DataFrame(kelly, index=C_df.columns, columns=['Weights'])
    kelly = kelly.round(3) 
    kelly.columns=['Kelly']
    return kelly


def effecient_frontier(rets_bl, covar_bl, config):

    returns, sigmas, weights, deltas = [], [], [], []
    for delta in np.arange(1, 10, 1):
        ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds=(
            config['min_position_size'], config['max_position_size']))
        ef.max_quadratic_utility(delta)
        ret, sigma, __ = ef.portfolio_performance()
        weights_vec = ef.clean_weights()
        returns.append(ret)
        sigmas.append(sigma)
        deltas.append(delta)
        weights.append(weights_vec)
    
    fig, ax = plt.subplots()
    ax.plot(sigmas, returns)
    for i, delta in enumerate(deltas):
        ax.annotate(str(delta), (sigmas[i], returns[i]))
    plt.xlabel('Volatility (%) ')
    plt.ylabel('Returns (%)')
    plt.title('Efficient Frontier for Max Quadratic Utility Optimization')


def max_quad_utility_weights(rets_bl, covar_bl, config, opt_delta):
    
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds=(
        config['min_position_size'], config['max_position_size']))
    ef.max_quadratic_utility(opt_delta)
    opt_weights = ef.clean_weights()
    opt_weights = pd.DataFrame.from_dict(opt_weights, orient='index')
    opt_weights.columns = ['Max Quad Util']

    return opt_weights, ef

def min_volatility_weights(rets_bl, covar_bl, config):
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    ef.min_volatility()
    weights = ef.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['Min Vol']
    return weights, ef

def max_sharpe_weights(rets_bl, covar_bl, config):
    ef = EfficientFrontier(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    ef.max_sharpe()
    weights = ef.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['Max Sharpe']
    return weights, ef

def cla_max_sharpe_weights(rets_bl, covar_bl, config):
    cla = CLA(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    cla.max_sharpe()
    weights = cla.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['CLA Max Sharpe']
    return weights, cla

def cla_min_vol_weights(rets_bl, covar_bl, config):
    cla = CLA(rets_bl, covar_bl, weight_bounds= \
            (config['min_position_size'] ,config['max_position_size']))
    cla.min_volatility()
    weights = cla.clean_weights()
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights.columns=['CLA Min Vol']
    return weights, cla