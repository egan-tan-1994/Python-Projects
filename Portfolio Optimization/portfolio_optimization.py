import os
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import skew
from scipy.stats import kurtosis
import cvxpy as cp
from scipy.optimize import minimize
import yfinance as yf
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError
from pypfopt import objective_functions
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.discrete_allocation import (DiscreteAllocation, get_latest_prices)
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.spatial.distance import pdist

plt.style.use("default")
plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 11

OUTPUT_DIR = Path("Outputs")
FIGURE_DIR = Path("Figures")
DATA_DIR = Path("Data")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


##############################################################################
                            # Project Configuration
##############################################################################

tickers = [
    "SPY",      # State Street SPDR S&P 500 ETF Trust
    "JPM",      # JPMorgan Chase
    "^DJI",     # Dow Jones Industrial Average
    "SPGI",     # S&P Global
    "T",        # AT&T
    "DD",       # DuPont
    "^GSPC",    # S&P 500
    "^SPX",     # S&P 500 Index
    "NVDA",     # Nvidia
    "AAPL",     # Apple
    "IBM",      # IBM
    "INTC",     # Intel
    "TXN",      # Texas Instruments
    "AMAT",     # Applied Materials
    "MU",       # Micron Technology
    "MRK",      # Merck
    "JNJ",      # Johnson & Johnson
    "PG",       # Procter & Gamble
    "CL",       # Colgate-Palmolive
    "GE"        # GE Aerospace
    ]

start_date = "2000-01-01"
end_date = "2026-01-01"
initial_capital = 1000000
risk_free_rate = 0.05
rebalance_frequency = "Quarterly"
transaction_cost = 0.001
optimization_method = "max_sharpe"
return_model = "historical"
covariance_model = "ledoit_wolf"
min_weight = 0.00
max_weight = 0.30
allow_short = False
target_volatility = 0.15
target_return = 0.12
risk_aversion = 1.0

if allow_short:
    weight_bounds = (-max_weight, max_weight)
else:
    weight_bounds = (min_weight, max_weight)

# Monte Carlo Parameters

num_simulations = 1000000
lookback_window = 252
objective_metric = "Sharpe"
random_seed = 42
np.random.seed(random_seed)

# Monte Carlo Parameters

save_figures = True
save_csv = True
save_html = True


# Validate Configuration

if initial_capital <= 0:
    raise ValueError("Initial capital must be positive.")

if risk_free_rate <= 0:
    raise ValueError("Risk-free rate cannot be negative.")

if not 0 <= transaction_cost <= 1:
    raise ValueError("Transaction cost must be between 0 and 1.")

if min_weight < 0 or max_weight > 1:
    raise ValueError("Weights must be between 0 and 1")

if min_weight > max_weight:
    raise ValueError("Minimum weight cannot exceed maximum weight")

if len(tickers) < 2:
    raise ValueError("At least two assets are required.")

# Validate Configuration

print("=" * 60)
print("PROJECT CONFIGURATION")
print("=" * 60)
print(f"Assets: {len(tickers)}")
print(f"Tickers: {tickers}")
print(f"Analysis Period: {start_date} to {end_date}")
print(f"Initial Capital: $ {initial_capital:,.2f}")
print(f"Risk-free Rate: {risk_free_rate:.2%}")
print(f"Optimization Method: {optimization_method}")
print(f"Return Model: {return_model}")
print(f"Covariance Model: {covariance_model}")
print(f"Target Volatility: {target_volatility}")
print(f"Target Return: {target_return}")
print(f"Risk Aversion Coefficient: {risk_aversion}")
print(f"Rebalancing: {rebalance_frequency}")
print(f"Transaction Cost: {transaction_cost:.2%}")
print(f"Short Selling: {allow_short}")
print(f"Monte Carlo Samples: {num_simulations:,}")
print("=" * 60)


##############################################################################
        # Download Historical Raw Data + Data Cleaning and Preprocessing
##############################################################################

raw_data = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=True
    )

prices = raw_data["Close"].copy()

if prices.empty:
    raise ValueError("No market data was downloaded.")

prices = prices.dropna(axis=1, how="all")
prices = prices.ffill()
prices = prices.bfill()
prices = prices.dropna()
prices = prices[~prices.index.duplicated()]
prices = prices.sort_index()

prices.index = pd.to_datetime(prices.index)

summary = pd.DataFrame({
    "Minimum": prices.min(),
    "Maximum": prices.max(),
    "Mean": prices.mean(),
    "Standard Deviation": prices.std(),
    "Missing Values": prices.isna().sum(),
    "Observations": prices.count()
    })

print("Summary")
print("=" * 60)
print(summary)
print("=" * 60)

if save_csv:
    prices.to_csv(DATA_DIR / "Historical Prices.csv")
    summary.to_csv(DATA_DIR / "Summary.csv")

plt.figure(figsize=(12,6))
for ticker in prices.columns:
    plt.plot(prices.index, prices[ticker], label=ticker)
plt.title("Historical Adjusted Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Historical Adjusted Prices.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()


print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"Trading Days: {len(prices)}")
print(f"Assets: {len(prices.columns)}")
print(f"Start: {prices.index.min().date()}")
print(f"End: {prices.index.max().date()}")
print("=" * 60)


##############################################################################
                                # Return Calculations
##############################################################################

prices_returns = prices.copy()

if prices_returns.empty:
    raise ValueError("Price dataset is empty.")

if (prices_returns <= 0).any().any():
    raise ValueError("Price dataset contains zero or negative values.")


daily_returns = prices_returns.pct_change()
daily_returns = daily_returns.dropna()

log_returns = np.log(prices_returns / prices_returns.shift(1))
log_returns = log_returns.dropna()

initial_value = 1.0

cumulative_returns = (1 + daily_returns).cumprod()
portfolio_growth = (cumulative_returns * initial_capital)


rolling_returns_21 = cumulative_returns.pct_change(21)
rolling_returns_63 = cumulative_returns.pct_change(63)
rolling_returns_252 = cumulative_returns.pct_change(252)


mean_daily_returns = daily_returns.mean()
annual_return = mean_daily_returns * 252

daily_volatility = daily_returns.std()
annual_volatility = daily_volatility * 252

returns_skewness = daily_returns.skew()
returns_kurtosis = daily_returns.kurtosis()


plt.figure(figsize=(12,6))
for ticker in daily_returns.columns:
    plt.plot(daily_returns.index, daily_returns[ticker], label=ticker,
             linewidth=0.8)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Daily Returns.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()

for ticker in daily_returns.columns:
    plt.figure(figsize=(12,6))
    plt.plot(daily_returns.index, daily_returns[ticker])
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Daily Returns.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

plt.figure(figsize=(12,6))
for ticker in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker],
             label=ticker, linewidth=0.8)
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Cumulative Returns.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()

plt.figure(figsize=(12,6))
daily_returns.boxplot()
plt.title("Daily Returns Boxplots")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Daily Return Boxplots.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()

for ticker in daily_returns.columns:
    plt.figure(figsize=(12,6))
    plt.hist(daily_returns[ticker], bins=50)
    plt.title(f"{ticker} Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Daily Returns Distribution.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

required_data = [
    "prices",
    "daily_returns",
    "log_returns"
    ]

for variable in required_data:
    if variable not in globals():
        raise ValueError(f"{variable} is missing")


correlation_matrix = daily_returns.corr()
covariance_matrix = daily_returns.cov()

plt.figure(figsize=(12,6))
plt.imshow(
    correlation_matrix,
    interpolation="nearest"
    )
plt.colorbar()
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
    )

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
    )
plt.title("Correlation Matrix")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Correlation Matrix.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(
    covariance_matrix,
    interpolation="nearest"
    )
plt.colorbar()
plt.xticks(
    range(len(covariance_matrix.columns)),
    covariance_matrix.columns,
    rotation=45
    )

plt.yticks(
    range(len(covariance_matrix.columns)),
    covariance_matrix.columns
    )
plt.title("Covariance Matrix")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Covariance Matrix.png",
                dpi=300, bbox_inches="tight"
                )
plt.show()
plt.close()


rolling_mean_returns_21 = daily_returns.rolling(window=21).mean()
rolling_mean_returns_63 = daily_returns.rolling(window=63).mean()
rolling_mean_returns_252 = daily_returns.rolling(window=252).mean()

rolling_volatility_21 = daily_returns.rolling(window=21).std()
rolling_volatility_63 = daily_returns.rolling(window=63).std()
rolling_volatility_252 = daily_returns.rolling(window=252).std()

rolling_sharpe_21 = ((rolling_mean_returns_21 / rolling_volatility_21)*
                     np.sqrt(252))
rolling_sharpe_63 = ((rolling_mean_returns_63 / rolling_volatility_63)*
                     np.sqrt(252))
rolling_sharpe_252 = ((rolling_mean_returns_252 / rolling_volatility_252)*
                     np.sqrt(252))

for ticker in daily_returns.columns:
    plt.figure(figsize=(12,6))
    plt.plot(rolling_mean_returns_21.index, rolling_mean_returns_21[ticker])
    plt.title(f"{ticker} Monthly Rolling Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Monthly Rolling Returns.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_volatility_21.index, rolling_volatility_21[ticker])
    plt.title(f"{ticker} Monthly Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Monthly Rolling Volatility.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_sharpe_21.index, rolling_sharpe_21[ticker])
    plt.title(f"{ticker} Monthly Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Monthly Rolling Sharpe Ratio.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_mean_returns_63.index, rolling_mean_returns_63[ticker])
    plt.title(f"{ticker} Quarterly Rolling Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Quarterly Rolling Returns.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_volatility_63.index, rolling_volatility_63[ticker])
    plt.title(f"{ticker} Quarterly Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Quarterly Rolling Volatility.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_sharpe_63.index, rolling_sharpe_63[ticker])
    plt.title(f"{ticker} Quarterly Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Quarterly Rolling Sharpe Ratio.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_mean_returns_252.index, rolling_mean_returns_252[ticker])
    plt.title(f"{ticker} Annual Rolling Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Annual Rolling Returns.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_volatility_252.index, rolling_volatility_252[ticker])
    plt.title(f"{ticker} Annual Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Annual Rolling Volatility.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.plot(rolling_sharpe_252.index, rolling_sharpe_252[ticker])
    plt.title(f"{ticker} Annual Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Annual Rolling Sharpe Ratio.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


returns_summary = pd.DataFrame({
    "Simple Mean (Daily) Returns": daily_returns.mean(),
    "Log Mean (Daily) Returns": log_returns.mean(),
    "Median Returns": daily_returns.median(),
    "Variance of Returns": daily_returns.var(),
    "Standard Deviation of Returns": daily_returns.std(),
    "Minimum Returns": daily_returns.min(),
    "Maximum Returns": daily_returns.max(),
    "(Simple) Annual Returns": annual_return,
    "Daily Volatility": daily_volatility,
    "Annual Volatility": annual_volatility,
    "Skewness": returns_skewness,
    "Kurtosis": returns_kurtosis
    })

print("Returns Summary")
print("=" * 60)
print(returns_summary)
print("=" * 60)

risk_return = pd.DataFrame({
    "Annual Return": daily_returns.mean() * 252,
    "Annual Volatility": daily_returns.std() * np.sqrt(252)
    })

plt.figure(figsize=(10,7))
plt.scatter(
    risk_return["Annual Volatility"],
    risk_return["Annual Return"]
    )
for ticker in risk_return.index:

    plt.text(
        risk_return.loc[ticker, "Annual Volatility"],
        risk_return.loc[ticker, "Annual Return"],
        ticker
    )
plt.xlabel("Annual Volatility")
plt.ylabel("Annual Return")
plt.title("Risk-Return Profile")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Risk-Return Plot.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


rolling_correlation_21 = pd.DataFrame()
benchmark = "SPY"
for ticker in daily_returns.columns:
    if ticker != benchmark:
        rolling_correlation_21[ticker] = (daily_returns[benchmark].rolling(21)
                                       .corr(daily_returns[ticker]))

rolling_correlation_63 = pd.DataFrame()
benchmark = "SPY"
for ticker in daily_returns.columns:
    if ticker != benchmark:
        rolling_correlation_63[ticker] = (daily_returns[benchmark].rolling(63)
                                       .corr(daily_returns[ticker]))

rolling_correlation_252 = pd.DataFrame()
benchmark = "SPY"
for ticker in daily_returns.columns:
    if ticker != benchmark:
        rolling_correlation_252[ticker] = (daily_returns[benchmark].rolling(252)
                                       .corr(daily_returns[ticker]))

for ticker in rolling_correlation_21.columns:
    plt.figure(figsize=(12,6))
    plt.plot(rolling_correlation_21.index, rolling_correlation_21[ticker])
    plt.title(f"{ticker} Monthly Rolling Correlation with SPY")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR /
                    f"{ticker} Monthly Rolling Correlation with SPY.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

for ticker in rolling_correlation_63.columns: 
    plt.figure(figsize=(12,6))
    plt.plot(rolling_correlation_63.index, rolling_correlation_63[ticker])
    plt.title(f"{ticker} Quarterly Rolling Correlation with SPY")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR /
                    f"{ticker} Quarterly Rolling Correlation with SPY.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

for ticker in rolling_correlation_252.columns:
    plt.figure(figsize=(12,6))
    plt.plot(rolling_correlation_252.index, rolling_correlation_252[ticker])
    plt.title(f"{ticker} Annual Rolling Correlation with SPY")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR /
                    f"{ticker} Annual Rolling Correlation with SPY.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


wealth = (1 + daily_returns).cumprod()
running_max = wealth.cummax()
drawdown = (wealth / running_max) - 1

for ticker in drawdown.columns:
    plt.figure(figsize=(12,6))
    plt.plot(drawdown.index, drawdown[ticker])
    plt.title(f"{ticker} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    if save_figures:
        plt.savefig(FIGURE_DIR / f"{ticker} Drawdown.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

if save_csv:
    returns_summary.to_csv(DATA_DIR / "Returns Summary.csv")
    correlation_matrix.to_csv(DATA_DIR / "Correlation Matrix.csv")
    covariance_matrix.to_csv(DATA_DIR / "Covariance Matrix.csv")
    rolling_mean_returns_21.to_csv(DATA_DIR / "Monthly Rolling Returns.csv")
    rolling_volatility_21.to_csv(DATA_DIR / "Mothly Rolling Volatility.csv")
    rolling_sharpe_21.to_csv(DATA_DIR / "Monthly Rolling Sharpe Ratio.csv")
    rolling_mean_returns_63.to_csv(DATA_DIR / "Quarterly Rolling Returns.csv")
    rolling_volatility_63.to_csv(DATA_DIR / "Quarterly Rolling Volatility.csv")
    rolling_sharpe_63.to_csv(DATA_DIR / "Quarterly Rolling Sharpe Ratio.csv")
    rolling_mean_returns_252.to_csv(DATA_DIR / "Annual Rolling Returns.csv")
    rolling_volatility_252.to_csv(DATA_DIR / "Annual Rolling Volatility.csv")
    rolling_sharpe_252.to_csv(DATA_DIR / "Annual Rolling Sharpe Ratio.csv")
    rolling_correlation_21.to_csv(DATA_DIR /
                                  "Monthly Rolling Correlation with SPY.csv")
    rolling_correlation_63.to_csv(DATA_DIR /
                                  "Quarterly Rolling Correlation with SPY.csv")
    rolling_correlation_252.to_csv(DATA_DIR /
                                  "Annual Rolling Correlation with SPY.csv")
    drawdown.to_csv(DATA_DIR / "Asset Drawdowns.csv")


##############################################################################
                        # Expected Return Estimations
##############################################################################

required_variables = [
    "prices",
    "daily_returns"
    ]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")


market_prices = prices["SPY"]

mu_historical = expected_returns.mean_historical_return(prices)
mu_ema = expected_returns.ema_historical_return(prices, span=252)
mu_capm = expected_returns.capm_return(prices,
                                       market_prices=market_prices,
                                       risk_free_rate=risk_free_rate)


expected_return = pd.DataFrame({
    "Historical": mu_historical,
    "EMA": mu_ema,
    "CAPM": mu_capm
    })

expected_return.plot(kind="bar", figsize=(12,6))
plt.ylabel("Expected Annual Return")
plt.title("Expected Return Estimates")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Expected Return Estimates.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


historical_rank = mu_historical.sort_values(ascending=False)
ema_rank = mu_ema.sort_values(ascending=False)
capm_rank = mu_capm.sort_values(ascending=False)

if return_model == "historical":
    mu = mu_historical
elif return_model == "ema":
    mu = mu_ema
elif return_model == "capm":
    mu = mu_capm
else:
    raise ValueError("Unkown return model.")

assert all(mu.index == prices.columns)

return_summary = pd.DataFrame({
    "Expected Return": mu,
    "Rank": mu.rank(ascending=False)
    })

if save_csv:
    expected_return.to_csv(DATA_DIR / "Expected Returns.csv")
    return_summary.to_csv(DATA_DIR / "Selected Expected Returns.csv")

plt.figure(figsize=(12,6))
mu.sort_values().plot(kind="barh")
plt.xlabel("Annual Expected Returns")
plt.title("Selected Expected Returns")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Selected Expected Returns.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


##############################################################################
                            # Covariance Estimations
##############################################################################

required_variables = [
    "prices",
    "daily_returns",
    "mu"
    ]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")


sample_covariance = risk_models.sample_cov(prices)
sample_corr = risk_models.cov_to_corr(sample_covariance)
ledoit_wolf_covariance = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
lw_corr = risk_models.cov_to_corr(ledoit_wolf_covariance)
oracle_covariance = (
    risk_models.CovarianceShrinkage(prices).oracle_approximating())
oracle_corr = risk_models.cov_to_corr(oracle_covariance)
exp_covariance = risk_models.exp_cov(prices, span=252)
exp_corr = risk_models.cov_to_corr(exp_covariance)


if covariance_model == "sample":
    S = sample_covariance
elif covariance_model == "ledoit_wolf":
    S = ledoit_wolf_covariance
elif covariance_model == "oracle":
    S = oracle_covariance
elif covariance_model == "exp_cov":
    S = exp_covariance
else:
    raise ValueError("Unkown covariance model.")


variance_comparison = pd.DataFrame({
    "Sample": np.diag(sample_covariance),
    "Ledoit-Wolf": np.diag(ledoit_wolf_covariance),
    "Oracle": np.diag(oracle_covariance),
    "Exponential": np.diag(exp_covariance),
    }, index=prices.columns
    )

matrix_diagnostics = pd.DataFrame({
    "Condition Number": [
        np.linalg.cond(sample_covariance),
        np.linalg.cond(ledoit_wolf_covariance),
        np.linalg.cond(oracle_covariance),
        np.linalg.cond(exp_covariance)
        ]
    },
    index = [
        "Sample",
        "Ledoit-Wolf",
        "Oracle",
        "Exponential"
        ]
    )

covariance_comparison = pd.DataFrame({
    "Variance": np.diag(S),
    "Volatility": np.sqrt(np.diag(S))
    },index=S.index)



if save_csv:
    sample_covariance.to_csv(DATA_DIR / "Sample Covariance.csv")
    sample_corr.to_csv(DATA_DIR / "Sample Correlation.csv")
    ledoit_wolf_covariance.to_csv(DATA_DIR / "Leoit-Wolf Covariance.csv")
    lw_corr.to_csv(DATA_DIR / "Ledoit-Wolf Correlation.csv")
    oracle_covariance.to_csv(DATA_DIR / "Oracle Covariance.csv")
    oracle_corr.to_csv(DATA_DIR / "Oracle Correlation.csv")
    exp_covariance.to_csv(DATA_DIR / "Exponential Covariance.csv")
    exp_corr.to_csv(DATA_DIR / "Exponential Correlation.csv")
    variance_comparison.to_csv(DATA_DIR / "Variance Comparison.csv")
    matrix_diagnostics.to_csv(DATA_DIR / "Matrix Condition Numbers.csv")
    covariance_comparison.to_csv(DATA_DIR / "Covariance Comparison.csv")


plt.figure(figsize=(12,6))
plt.imshow(sample_covariance, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(sample_covariance.columns)),
           sample_covariance.columns, rotation=45)
plt.yticks(range(len(sample_covariance.columns)),
           sample_covariance.columns)
plt.title("Sample Covariance")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Sample Covariance.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(ledoit_wolf_covariance, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(ledoit_wolf_covariance.columns)),
           ledoit_wolf_covariance.columns, rotation=45)
plt.yticks(range(len(ledoit_wolf_covariance.columns)),
           ledoit_wolf_covariance.columns)
plt.title("Ledoit-Wolf Covariance")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Ledoit-Wolf Covariance.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(oracle_covariance, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(oracle_covariance.columns)),
           oracle_covariance.columns, rotation=45)
plt.yticks(range(len(oracle_covariance.columns)),
           oracle_covariance.columns)
plt.title("Oracle Covariance")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Oracle Covariance.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(exp_covariance, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(exp_covariance.columns)),
           exp_covariance.columns, rotation=45)
plt.yticks(range(len(exp_covariance.columns)),
           exp_covariance.columns)
plt.title("Exponential Covariance")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Exponential Covariance.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(sample_corr, interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(sample_corr.columns)),
           sample_corr.columns, rotation=45)
plt.yticks(range(len(sample_corr.columns)),
           sample_corr.columns)
plt.title("Sample Correlation")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Sample Correlation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(lw_corr, interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(lw_corr.columns)),
           lw_corr.columns, rotation=45)
plt.yticks(range(len(lw_corr.columns)),
           lw_corr.columns)
plt.title("Ledoit-Wolf Correlation")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Ledoit-Wolf Correlation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(oracle_corr, interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(oracle_corr.columns)),
           oracle_corr.columns, rotation=45)
plt.yticks(range(len(oracle_corr.columns)),
           oracle_corr.columns)
plt.title("Oracle Correlation")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Oracle Correlation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.imshow(exp_corr, interpolation="nearest", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(exp_corr.columns)),
           exp_corr.columns, rotation=45)
plt.yticks(range(len(exp_corr.columns)),
           exp_corr.columns)
plt.title("Exponential Correlation")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Exponential Correlation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


n_assets = len(S)

equal_weight = np.repeat(1/n_assets, n_assets)
portfolio_variance = (equal_weight.T
                      @ S
                      @ equal_weight
                      )
portfolio_volatility = np.sqrt(portfolio_variance)

print("=" * 60)
print("=" * 60)
print("Portfolio Risk Computation (Equal Weight)")
print("=" * 60)
print(f"Portfolio Variance: {portfolio_variance}")
print(f"Portfolio Volatility: {portfolio_volatility}")
print("=" * 60)

portfolio_risk = pd.DataFrame({
    "Portfolio Variance": [portfolio_variance],
    "Portfolio Volatility": [portfolio_volatility]
    })

if save_csv:
    portfolio_risk.to_csv(DATA_DIR /
                          "Equal-Weight Portfolio Risk Computation.csv")



##############################################################################
                            # Equal-weight Portfolio
##############################################################################

required_variables = [
    "daily_returns",
    "prices"
]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")

num_assets = len(prices.columns)
equal_weights = np.repeat(1/num_assets, num_assets)

equal_weights = pd.Series(
    equal_weights,
    index=prices.columns
    )

equal_weight_returns = daily_returns @ equal_weights
equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
equal_weight_value = initial_capital * equal_weight_cumulative

ew_return = equal_weight_returns.mean() * 252
ew_volatility = equal_weight_returns.std() * np.sqrt(252)
sharpe_ew = (ew_return - risk_free_rate) / ew_volatility
running_max_ew = equal_weight_cumulative.cummax()
dd_ew = (equal_weight_cumulative / running_max_ew) - 1
max_dd_ew = dd_ew.min()
downside_returns_ew = equal_weight_returns[equal_weight_returns < 0]
downside_deviation_ew = downside_returns_ew.std() * np.sqrt(252)
sortino_ew = (ew_return - risk_free_rate) / downside_deviation_ew
calmar_ew = ew_return / abs(max_dd_ew)
var_ew = np.percentile(equal_weight_returns, 5)
cvar_ew = (equal_weight_returns[equal_weight_returns <= var_ew]).mean()

equal_weight_metrics = pd.DataFrame({
    "Metric": [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Value at Risk (VaR, 95%)",
        "Conditional VaR (95%)",
        ],
    "Value": [
        ew_return,
        ew_volatility,
        sharpe_ew,
        sortino_ew,
        max_dd_ew,
        calmar_ew,
        var_ew,
        cvar_ew
        ]
    })

if save_csv:
    equal_weight_metrics.to_csv(DATA_DIR /
                                "Equal Weight Portfolio Metrics.csv")


plt.figure(figsize=(12,6))
plt.plot(equal_weight_value.index, equal_weight_value, linewidth=2)
plt.title("Equal-Weight Portfolio Value")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Equal-Weight Portfolio Growth.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.fill_between(dd_ew.index, dd_ew, 0, alpha=0.4)
plt.title("Equal-Weight Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Equal-Weight Portfolio Drawdown.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.hist(equal_weight_returns, bins=60)
plt.title("Equal-Weight Portolfio Return Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Equal-Weight Portfolio Return Distribution.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.pie(equal_weights, labels=equal_weights.index, autopct="%1.1f%%")
plt.title("Equal-Weight Portfolio Allocation")
if save_figures:
    plt.savefig(FIGURE_DIR / "Equal-Weight Portfolio Allocation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


print("=" * 60)
print("Portfolio Performance (Equal-Weight)")
print("=" * 60)
print(f"Expected Return: {ew_return}")
print(f"Volatility: {ew_volatility}")
print(f"Sharpe Ratio: {sharpe_ew}")
print("=" * 60)


##############################################################################
                        # Markowitz Portfolio Optimization
##############################################################################

required_variables = [
    "mu",
    "S"
]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")


ef = EfficientFrontier(expected_returns=mu, cov_matrix=S,
                       weight_bounds=weight_bounds)

if optimization_method == "max_sharpe":
    ef.max_sharpe(risk_free_rate=risk_free_rate)
elif optimization_method == "min_volatility":
    ef.min_volatility()
elif optimization_method == "efficient_risk":
    ef.efficient_risk(target_volatility)
elif optimization_method == "efficient_return":
    ef.efficient_return(target_return)
elif optimization_method == "max_quadratic_utility":
    ef.max_quadratic_utility(risk_aversion=risk_aversion)
else:
    raise ValueError("Unkown optimizer.")


clean_weights = ef.clean_weights()

markowitz_weights = pd.Series(clean_weights)
assert np.isclose(markowitz_weights.sum(), 1)

    
markowitz_return, volatility_markowitz, sharpe_markowitz = \
ef.portfolio_performance(risk_free_rate=risk_free_rate)

print("=" * 60)
print("Portfolio Performance (Markowitz Optimization)")
print("=" * 60)
print(f"Expected Return: {markowitz_return}")
print(f"Volatility: {volatility_markowitz}")
print(f"Sharpe Ratio: {sharpe_markowitz}")
print("=" * 60)


markowitz_portfolio_returns = (daily_returns @ markowitz_weights)
markowitz_portfolio_cumulative = (1 + markowitz_portfolio_returns).cumprod()
markowitz_portfolio_value = (initial_capital * markowitz_portfolio_cumulative)
running_max_markowitz = markowitz_portfolio_cumulative.cummax()
markowitz_dd = (markowitz_portfolio_cumulative / running_max_markowitz) - 1
max_dd_markowitz = markowitz_dd.min()
downside_returns_markowitz = markowitz_portfolio_returns[
    markowitz_portfolio_returns < 0]
downside_volatility_markowitz = downside_returns_markowitz.std() * np.sqrt(252)
sortino_markowitz = \
    (markowitz_return - risk_free_rate) / downside_volatility_markowitz
calmar_markowitz = markowitz_return / abs(max_dd_markowitz)
var_markowitz = np.percentile(markowitz_portfolio_returns, 5)
cvar_markowitz = (markowitz_portfolio_returns[markowitz_portfolio_returns
                                             <= var_markowitz]).mean()



markowitz_metrics = pd.DataFrame({
    "Metric": [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Value at Risk (VaR, 95%)",
        "Conditional VaR (95%)",
        ],
    "Value": [
        markowitz_return,
        volatility_markowitz,
        sharpe_markowitz,
        sortino_markowitz,
        max_dd_markowitz,
        calmar_markowitz,
        var_markowitz,
        cvar_markowitz
        ]
    })

if save_csv:
    markowitz_metrics.to_csv(DATA_DIR / "Markowitz Portfolio Metrics.csv")


plt.figure(figsize=(12,6))
plt.pie(markowitz_weights, labels=markowitz_weights.index, autopct="%1.1f%%")
plt.title("Optimized Portfolio Allocation (Markowitz Optimization)")
if save_figures:
    plt.savefig(FIGURE_DIR / "Optimized Portfolio Allocation (Markowitz).png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(12,6))
plt.plot(markowitz_portfolio_value.index, markowitz_portfolio_value)
plt.title("Optimized Portfolio Value (Markowitz Optimization)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Optimized Portfolio Value (Markowitz).png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()



metrics_comparison = pd.DataFrame({
    "Equal Weight":[
        ew_return,
        ew_volatility,
        sharpe_ew,
        sortino_ew,
        max_dd_ew,
        calmar_ew,
        var_ew,
        cvar_ew
        ],
    "Optimized (Markowitz)": [
        markowitz_return,
        volatility_markowitz,
        sharpe_markowitz,
        sortino_markowitz,
        max_dd_markowitz,
        calmar_markowitz,
        var_markowitz,
        cvar_markowitz
        ]
    },
    index = [
        "Expected Returns",
        "Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Value-at-Risk (VaR, 95%)",
        "Conditional VaR (CVaR, 95%)"
        ])

if save_csv:
    metrics_comparison.to_csv(DATA_DIR / "Portfolio Metrics Comparison.csv")

plt.figure(figsize=(12,6))
plt.plot(equal_weight_value, label="Equal Weight")
plt.plot(markowitz_portfolio_value, label="Optimized (Markovitz Model)")
plt.legend()
plt.title("Portfolio Growth Comparison")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Portfolio Growth Comparison.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()

##############################################################################
                    # Efficient Frontier Generation and Analysis
##############################################################################

required_variables = [
    "mu",
    "S",
    "markowitz_weights"
]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")



frontier_returns = []
frontier_volatility = []
frontier_sharpe = []
frontier_weights = []

min_return = mu.min()
max_return = mu.max()
target_returns = np.linspace(min_return, max_return, 100)


for target in target_returns:
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    try:
        ef.efficient_return(target_return=target)
    except (ValueError, OptimizationError) as e:
        print(f"Skipping target return."
              f"{target:.4%}: {e}")
        continue
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    frontier_returns.append(ret)
    frontier_volatility.append(vol)
    frontier_sharpe.append(sharpe)
    frontier_weights.append(ef.clean_weights())

efficient_frontier = pd.DataFrame({
    "Return": frontier_returns,
    "Volatility": frontier_volatility,
    "Sharpe": frontier_sharpe
    })

best_index = efficient_frontier["Sharpe"].idxmax()
best_return = frontier_returns[best_index]
best_volatility = frontier_volatility[best_index]
best_sharpe = frontier_sharpe[best_index]

min_index = efficient_frontier["Volatility"].idxmin()
min_return = frontier_returns[min_index]
min_volatility = frontier_volatility[min_index]


frontier_summary = pd.DataFrame({
    "Minimum Return": [min(frontier_returns)],
    "Maximum Return": [max(frontier_returns)],
    "Minimum Volatility": [min(frontier_volatility)],
    "Maximum Volatility": [max(frontier_volatility)],
    "Minimum Sharpe Ratio": [min(frontier_sharpe)],
    "Maximum Sharpe Ratio": [max(frontier_sharpe)]
    })



if save_csv:
    efficient_frontier.to_csv(DATA_DIR / "Efficient Frontier Data.csv",
                              index=False)
    frontier_summary.to_csv(DATA_DIR / "Frontier Summary.csv", index=False)



plt.figure(figsize=(12,6))
plt.plot(frontier_volatility, frontier_returns, label="Efficient Frontier")
plt.legend()
plt.xlabel("Annual Volatility")
plt.ylabel("Expected Annual Returns")
plt.title("Efficient Frontier Plot")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Efficient Frontier Plot.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


cml_volatility = np.linspace(0, max(frontier_volatility), 100)
cml_return = (risk_free_rate + best_sharpe * cml_volatility)

plt.figure(figsize=(12,6))
plt.plot(cml_volatility, cml_return,
         linestyle="--", label="Capital Market Line")
plt.scatter(ew_volatility, ew_return,
            marker="o", s=100, label="Equal Weight")
plt.scatter(volatility_markowitz, markowitz_return,
            marker="*", s=250, label="Optimized (Markowitz)")
plt.scatter(best_volatility, best_return,
            marker="X", s=200, label="Maximum Sharpe")
plt.scatter(min_volatility, min_return,
            marker="D", s=150, label="Minimum Volatility")
plt.legend()
plt.xlabel("Annual Volatility")
plt.ylabel("Expected Annual Returns")
plt.title("Annual Returns vs Volatility Scatter Plots")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Returns vs Volatility Scatter Plots.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()



##############################################################################
                        # Monte Carlo Portfolio Simulation
##############################################################################

required_variables = [
    "mu",
    "S",
    "daily_returns"
]

for variable in required_variables:
    if variable not in globals():
        raise ValueError(f"{variable} is missing.")


num_assets = len(mu)


simulation_returns = []
simulation_volatility = []
simulation_sharpe = []
simulation_weights = []


for i in range(num_simulations):
    weights = np.random.random(num_assets)
    weights /= weights.sum()
    
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(weights.T @ S @ weights)
    portfolio_sharpe = ((portfolio_return - risk_free_rate) /
                        portfolio_volatility)
    
    simulation_returns.append(portfolio_return)
    simulation_volatility.append(portfolio_volatility)
    simulation_sharpe.append(portfolio_sharpe)
    simulation_weights.append(weights.copy())
    

simulation_results = pd.DataFrame({
    "Return": simulation_returns,
    "Volatility": simulation_volatility,
    "Sharpe": simulation_sharpe
    })



if save_csv:
    simulation_results.to_csv(DATA_DIR /
                              "Monte Carlo Portfolio Simulation.csv",
                              index=False)


max_sharpe_index = simulation_results["Sharpe"].idxmax()
max_sharpe_simulation = simulation_results.loc[max_sharpe_index]
max_sharpe_weights = pd.Series(simulation_weights[max_sharpe_index],
                               index=mu.index)

min_volatility_index = simulation_results["Volatility"].idxmin()
min_volatility_simulation = simulation_results.loc[min_volatility_index]
min_volatility_weights = pd.Series(simulation_weights[min_volatility_index],
                                   index=mu.index)


plt.figure(figsize=(12,6))
scatter = plt.scatter(simulation_results["Volatility"],
                      simulation_results["Return"],
                      c=simulation_results["Sharpe"],
                      cmap="viridis", alpha=0.8, s=8)
plt.colorbar(scatter, label="Sharpe Ratio")
plt.plot(frontier_volatility, frontier_returns,
         label="Efficient Frontier", color="blue", linewidth=3)
plt.plot(cml_volatility, cml_return,
         linestyle="--", label="Capital Market Line")
plt.scatter(ew_volatility, ew_return,
            marker="o", s=100, label="Equal Weight")
plt.scatter(volatility_markowitz, markowitz_return,
            marker="*", s=250, label="Optimized (Markowitz)")
plt.scatter(max_sharpe_simulation["Volatility"],
            max_sharpe_simulation["Return"],
            marker="X", s=200, label="Best Random")
plt.scatter(min_volatility_simulation["Volatility"],
            min_volatility_simulation["Return"],
            marker="D", s=150, label="Lowest Risk")
plt.legend()
plt.xlabel("Annual Volatility")
plt.ylabel("Expected Annual Returns")
plt.title("Monte Carlo Portfolio Simulation")
plt.tight_layout()
if save_figures:
    plt.savefig(FIGURE_DIR / "Monte Carlo Portfolio Simulation.png",
                dpi=300, bbox_inches="tight")
plt.show()
plt.close()


simulation_summary = pd.DataFrame({
    "Portfolio": [
        "Equal Weight",
        "Markowitz",
        "Best Random",
        "Lowest Risk Random"
        ],
    "Return": [
        ew_return,
        markowitz_return,
        max_sharpe_simulation["Return"],
        min_volatility_simulation["Return"]
        ],
    "Volatility": [
        ew_volatility,
        volatility_markowitz,
        max_sharpe_simulation["Volatility"],
        min_volatility_simulation["Volatility"]
        ],
    "Sharpe": [
        sharpe_ew,
        sharpe_markowitz,
        max_sharpe_simulation["Sharpe"],
        min_volatility_simulation["Sharpe"]
        ]
    })


if save_csv:
    simulation_results.to_csv(DATA_DIR /
                              "Portfolio Statistics Comparison.csv",
                              index=False)

























































































































































































































































































































































































































































































































































































































































































































































