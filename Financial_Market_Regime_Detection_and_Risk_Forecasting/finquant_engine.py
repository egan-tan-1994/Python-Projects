import os
import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as sco
import scipy.linalg as la
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.interpolate import griddata
import statsmodels.api as sm  
from statsmodels.tsa.stattools import coint
import pymc as pm
from hmmlearn import hmm
from pykalman import KalmanFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.subplots as plt
from arch import arch_model
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# Step 1: Data Ingestion
# ==========================================
def load_data(db_path, ticker, table="asset_returns", return_col="log_return"):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table} WHERE ticker = '{ticker}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df[return_col].dropna().values

# ==========================================
# Step 2: Probability Distributions & MLE
# ==========================================
def estimate_return_distribution(returns):
    mu_norm, std_norm = stats.norm.fit(returns)
    df_t, mu_t, std_t = stats.t.fit(returns)
    print(f"Normal MLE -> Mean: {mu_norm:.6f}, Vol: {std_norm:.6f}")
    print(f"Student-t MLE -> DoF: {df_t:.2f}, Mean: {mu_t:.6f}, Vol: {std_t:.6f}")
    return mu_t, std_t, df_t

# ==========================================
# Step 3: Risk Metrics (VaR & CVaR)
# ==========================================
def calculate_risk_metrics(returns, confidence_level=0.05):
    var = np.percentile(returns, confidence_level * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

# ==========================================
# Step 4: Maximum Drawdown Analysis
# ==========================================
def calculate_maximum_drawdown(returns, plot=True, output_dir="plots"):
    cumulative_returns = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    return max_drawdown, drawdowns

# ==========================================
# Step 5: Bootstrap Resampling (Standard & BCa)
# ==========================================
def bootstrap_sharpe_ratio(returns, risk_free_rate=0.0, n_iterations=10000):
    n_days = len(returns)
    sharpe_ratios = np.zeros(n_iterations)
    for i in range(n_iterations):
        sample = np.random.choice(returns, size=n_days, replace=True)
        mean_ret = np.mean(sample) * 252
        vol = np.std(sample) * np.sqrt(252)
        sharpe_ratios[i] = (mean_ret - risk_free_rate) / vol
    ci_lower, ci_upper = np.percentile(sharpe_ratios, [2.5, 97.5])
    point_estimate = (np.mean(returns) * 252 - risk_free_rate) / (np.std(returns) * np.sqrt(252))
    return point_estimate, ci_lower, ci_upper

def bootstrap_sharpe_ratio_bca(returns, risk_free_rate=0.0):
    def sharpe(rets):
        return (np.mean(rets) * 252 - risk_free_rate) / (np.std(rets) * np.sqrt(252))
    res = stats.bootstrap((returns,), sharpe, confidence_level=0.95, method='BCa', n_resamples=10000)
    return sharpe(returns), res.confidence_interval.low, res.confidence_interval.high

# ==========================================
# Step 6: Hypothesis & Permutation Testing
# ==========================================
def test_strategy_difference(returns_asset_a, returns_asset_b, verbose=True):
    t_stat, p_val = stats.ttest_ind(returns_asset_a, returns_asset_b, equal_var=False)
    return t_stat, p_val

def permutation_test_difference(returns_asset_a, returns_asset_b, n_permutations=10000):
    def diff_in_means(x, y):
        return np.mean(x) - np.mean(y)
    res = stats.permutation_test((returns_asset_a, returns_asset_b), diff_in_means, permutation_type='independent', n_resamples=n_permutations, alternative='two-sided')
    return res.pvalue

# ==========================================
# Step 7: GARCH Volatility Modeling
# ==========================================
def estimate_garch_volatility(returns, plot=True, output_dir="plots"):
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t')
    return model.fit(disp='off')

# ==========================================
# Step 8: Monte Carlo Portfolio Simulation
# ==========================================
def monte_carlo_simulation(returns, initial_portfolio=10000, days_to_simulate=252, n_simulations=5000, plot=True, output_dir="plots"):
    mu, sigma = np.mean(returns), np.std(returns)
    simulated_log_returns = np.random.normal(mu, sigma, (days_to_simulate, n_simulations))
    price_paths = initial_portfolio * np.exp(np.cumsum(simulated_log_returns, axis=0))
    final_values = price_paths[-1, :]
    return price_paths, np.median(final_values), *np.percentile(final_values, [5, 95])

# ==========================================
# Step 9: Bayesian Inference Engine
# ==========================================
def robust_linear_regression(returns_y, returns_x):
    return sm.OLS(returns_y, sm.add_constant(returns_x)).fit(cov_type='HC3')

def bayesian_linear_regression(returns_y, returns_x):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.1)
        beta = pm.Normal("beta", mu=1.0, sigma=2.0)
        sigma = pm.HalfCauchy("sigma", beta=0.1)
        y_obs = pm.Normal("y_obs", mu=alpha + beta * returns_x, sigma=sigma, observed=returns_y)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=False, cores=1)
    return trace, pm.summary(trace, var_names=["alpha", "beta", "sigma"])

def hierarchical_bayesian_model(returns_matrix, group_idx):
    n_groups = len(np.unique(group_idx))
    with pm.Model() as model:
        mu_group = pm.Normal("mu_group", mu=0.0, sigma=0.1)
        sigma_group = pm.HalfCauchy("sigma_group", beta=0.1)
        mu = pm.Normal("mu", mu=mu_group, sigma=sigma_group, shape=n_groups)
        sigma = pm.HalfCauchy("sigma", beta=0.1, shape=n_groups)
        nu = pm.Exponential("nu", 1/10.0) + 2.0
        y_obs = pm.StudentT("y_obs", nu=nu, mu=mu[group_idx], sigma=sigma[group_idx], observed=returns_matrix)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=False, cores=1)
    return trace, pm.summary(trace, var_names=["mu_group", "sigma_group", "mu", "sigma"])

# ==========================================
# Step 10: Extreme Value Theory (POT)
# ==========================================
def estimate_evt_tail_risk(returns, threshold_percentile=95):
    losses = -returns
    threshold = np.percentile(losses, threshold_percentile)
    excess = losses[losses > threshold] - threshold
    shape, loc, scale = stats.genpareto.fit(excess)
    var_99 = threshold + (scale / shape) * (((len(returns) / len(excess)) * 0.01)**(-shape) - 1)
    return shape, loc, scale, var_99

# ==========================================
# Step 11: Multi-Asset Copula Engine
# ==========================================
def estimate_copulas(returns_a, returns_b):
    u = stats.rankdata(returns_a) / (len(returns_a) + 1)
    v = stats.rankdata(returns_b) / (len(returns_b) + 1)
    rho, _ = stats.pearsonr(stats.norm.ppf(u), stats.norm.ppf(v))
    tau, _ = stats.kendalltau(returns_a, returns_b)
    clayton_theta = 2 * tau / (1 - tau) if tau > 0 else 0.001
    gumbel_theta = 1 / (1 - tau) if tau >= 0 else 1.001
    return rho, clayton_theta, gumbel_theta, u, v

# ==========================================
# Step 12: Hidden Markov Models (Regime Detection)
# ==========================================
def detect_market_regimes_hmm(returns, n_components=2):
    X = returns.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)
    return model, model.predict(X)

# ==========================================
# Step 13: Pairs Trading (Cointegration Engine)
# ==========================================
def test_pairs_cointegration(prices_a, prices_b):
    score, pvalue, _ = coint(prices_a, prices_b)
    return score, pvalue, pvalue < 0.05

# ==========================================
# Step 14: Kalman Filter Dynamic Spread Ratio
# ==========================================
def kalman_filter_pairs_trading(price_x, price_y):
    obs_mat = np.vstack([price_x, np.ones(len(price_x))]).T[:, np.newaxis]
    trans_cov = (1e-5 / (1 - 1e-5)) * np.eye(2)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.ones((2, 2)), transition_matrices=np.eye(2),
                      observation_matrices=obs_mat, observation_covariance=1.0, transition_covariance=trans_cov)
    state_means, _ = kf.filter(price_y)
    return price_y - (state_means[:, 0] * price_x + state_means[:, 1]), state_means[:, 0]

# ==========================================
# Step 15: Black-Scholes Pricing & Greeks
# ==========================================
def calculate_black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return price

# ==========================================
# Step 16: Ornstein-Uhlenbeck (OU) Process
# ==========================================
def simulate_ou_process(S0, theta, mu, sigma, dt=1/252, n_steps=252, n_simulations=50):
    paths = np.zeros((n_steps, n_simulations))
    paths[0] = S0
    for t in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_simulations)
        paths[t] = paths[t-1] + theta * (mu - paths[t-1]) * dt + sigma * dW
    return paths

# ==========================================
# Step 17: Continuous-Time Kelly Criterion
# ==========================================
def calculate_kelly_criterion(returns, risk_free_rate=0.0):
    mu = np.mean(returns) * 252
    sigma = np.std(returns) * np.sqrt(252)
    kelly_fraction = (mu - risk_free_rate) / (sigma ** 2)
    return kelly_fraction, kelly_fraction / 2.0 

# ==========================================
# Step 18: Heston Stochastic Volatility
# ==========================================
def heston_pricing_mc(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type='call', n_steps=100, n_simulations=10000):
    dt = T / n_steps
    prices, variances = np.full(n_simulations, S0), np.full(n_simulations, v0)
    for _ in range(n_steps):
        Z1, Z2 = np.random.normal(0, 1, n_simulations), np.random.normal(0, 1, n_simulations)
        W1, W2 = Z1, rho * Z1 + np.sqrt(1 - rho**2) * Z2
        variances = np.maximum(variances, 0.0) 
        variances += kappa * (theta - variances) * dt + sigma * np.sqrt(variances * dt) * W2
        variances = np.maximum(variances, 0.0)
        prices *= np.exp((r - 0.5 * variances) * dt + np.sqrt(variances * dt) * W1)
    payoffs = np.maximum(prices - K, 0) if option_type == 'call' else np.maximum(K - prices, 0)
    return np.exp(-r * T) * np.mean(payoffs)

# ==========================================
# Step 19: Fractional Brownian Motion (fBm)
# ==========================================
def simulate_fbm(n_steps, hurst, T=1.0):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    gamma = np.zeros(n_steps)
    for i in range(n_steps):
        gamma[i] = 0.5 * ((i+1)**(2*hurst) - 2*i**(2*hurst) + (abs(i-1))**(2*hurst))
    r = np.zeros(n_steps)
    r[0] = gamma[0]
    for i in range(1, n_steps):
        r[i] = gamma[i]
    Sigma = la.toeplitz(r)
    L = la.cholesky(Sigma, lower=True)
    Z = np.random.standard_normal(n_steps)
    fGn = L @ Z
    fBm = np.zeros(n_steps + 1)
    fBm[1:] = np.cumsum(fGn) * (dt ** hurst)
    return t, fBm

# ==========================================
# Step 20: Markowitz Optimization (Mean-Variance)
# ==========================================
def optimize_markowitz(expected_returns, cov_matrix, risk_free_rate=0.0):
    n = len(expected_returns)
    def neg_sharpe(weights):
        port_ret = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_ret - risk_free_rate) / port_vol
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init_guess = np.array(n * [1.0 / n])
    res = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

# ==========================================
# Step 21: Risk Parity Optimization
# ==========================================
def optimize_risk_parity(cov_matrix):
    n = len(cov_matrix)
    def risk_budget_objective(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk = np.dot(cov_matrix, weights) / port_vol
        risk_contribution = weights * marginal_risk
        target_risk = port_vol / n
        return np.sum((risk_contribution - target_risk)**2)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init_guess = np.array(n * [1.0 / n])
    res = sco.minimize(risk_budget_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

# ==========================================
# Step 22: Black-Litterman Model
# ==========================================
def black_litterman_posterior(implied_returns, cov_matrix, P, Q, omega, tau=0.025):
    tau_cov_inv = la.inv(tau * cov_matrix)
    omega_inv = la.inv(omega)
    post_cov = la.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
    part1 = np.dot(tau_cov_inv, implied_returns)
    part2 = np.dot(np.dot(P.T, omega_inv), Q)
    post_ret = np.dot(post_cov, (part1 + part2))
    return post_ret, post_cov + cov_matrix

# ==========================================
# Step 23: Hierarchical Risk Parity (HRP)
# ==========================================
def hierarchical_risk_parity(cov_matrix):
    cov = pd.DataFrame(cov_matrix)
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    link = sch.linkage(ssd.squareform(dist), 'single')
    
    link_arr = link.astype(int)
    sort_ix = pd.Series([link_arr[-1, 0], link_arr[-1, 1]])
    num_items = link_arr[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link_arr[j, 0]
        df2 = pd.Series(link_arr[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df2]).sort_index()
    sort_ix = sort_ix.tolist()
    
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0, c_items1 = c_items[i], c_items[i+1]
            cov0, cov1 = cov.iloc[c_items0, c_items0], cov.iloc[c_items1, c_items1]
            
            ivp0 = 1. / np.diag(cov0)
            ivp0 /= ivp0.sum()
            c_var0 = np.dot(np.dot(ivp0.T, cov0), ivp0)
            
            ivp1 = 1. / np.diag(cov1)
            ivp1 /= ivp1.sum()
            c_var1 = np.dot(np.dot(ivp1.T, cov1), ivp1)
            
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
            
    return w.sort_index().values

# ==========================================
# Step 24: Merton's Jump-Diffusion Model
# ==========================================
def simulate_jump_diffusion(S0, mu, sigma, lambd, mu_j, sigma_j, T=1.0, n_steps=252, n_simulations=100):
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        z1 = np.random.standard_normal(n_simulations)
        z2 = np.random.standard_normal(n_simulations)
        poisson = np.random.poisson(lambd * dt, n_simulations)
        jump_impact = np.exp(mu_j * poisson + sigma_j * np.sqrt(poisson) * z2)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z1) * jump_impact
    return paths

# ==========================================
# Step 25: Option Volatility Surface
# ==========================================
def build_volatility_surface(strikes, maturities, implied_vols):
    grid_x, grid_y = np.mgrid[min(strikes):max(strikes):50j, min(maturities):max(maturities):50j]
    grid_z = griddata((strikes, maturities), implied_vols, (grid_x, grid_y), method='cubic')
    return grid_x, grid_y, grid_z

# ==========================================
# Step 26: Local Volatility (Dupire) 
# ==========================================
def local_volatility_dupire(grid_strike, grid_mat, grid_vol, S0, r):
    """Calculates the Dupire Local Volatility surface using finite differences on option prices."""
    call_grid = np.zeros_like(grid_vol)
    for i in range(grid_vol.shape[0]):
        for j in range(grid_vol.shape[1]):
            K = grid_strike[i, j]
            T = grid_mat[i, j]
            sigma = grid_vol[i, j]
            call_grid[i, j] = calculate_black_scholes(S0, K, max(T, 1e-5), r, sigma, 'call')
            
    dK = np.gradient(grid_strike, axis=0)
    dT = np.gradient(grid_mat, axis=1)
    
    dC_dK = np.gradient(call_grid, axis=0) / dK
    d2C_dK2 = np.gradient(dC_dK, axis=0) / dK
    dC_dT = np.gradient(call_grid, axis=1) / dT
    
    var_local = (dC_dT + r * grid_strike * dC_dK) / (0.5 * grid_strike**2 * d2C_dK2 + 1e-8)
    return np.sqrt(np.maximum(var_local, 1e-8))

# ==========================================
# Step 27: Option Delta-Hedging Backtester
# ==========================================
def delta_hedged_backtester(price_path, K, r, vol, T, dt=1/252):
    """Simulates a self-financing delta-hedged portfolio for a long Call Option."""
    n_steps = len(price_path)
    shares, cash = 0.0, 0.0
    pnl = []
    
    for i in range(n_steps):
        S = price_path[i]
        tau = max(T - i*dt, 1e-6)
        
        d1 = (np.log(S/K) + (r + 0.5*vol**2)*tau) / (vol * np.sqrt(tau))
        delta = stats.norm.cdf(d1)
        opt_price = calculate_black_scholes(S, K, tau, r, vol, 'call')
        
        if i == 0:
            cash = delta * S - opt_price
            shares = -delta
        else:
            cash *= np.exp(r * dt)
            share_diff = -delta - shares
            cash -= share_diff * S
            shares = -delta
            
        portfolio_val = opt_price + shares * S + cash
        pnl.append(portfolio_val)
        
    return np.array(pnl)

# ==========================================
# Step 28: Machine Learning Alpha Generation
# ==========================================
def generate_ml_alpha(X_train, y_train, X_latest):
    """Generates next-period expected returns using Random Forest for Black-Litterman integration."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_latest)