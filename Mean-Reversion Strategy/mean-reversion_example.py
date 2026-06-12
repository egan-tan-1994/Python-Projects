import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import itertools
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression

#--------------------------------------------------------------------------#
                                # Configuration #
#--------------------------------------------------------------------------#

TICKER = "SPY"

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

ROLLING_WINDOW = 20
ZSCORE_THRESHOLD = 2.0

INITIAL_CAPITAL = 100000


#--------------------------------------------------------------------------#
                                # Download Data #
#--------------------------------------------------------------------------#

df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE
    )

df.columns = df.columns.get_level_values(0)

print(df.head())


#--------------------------------------------------------------------------#
                                # Clean Data #
#--------------------------------------------------------------------------#

df.dropna(inplace=True)


#--------------------------------------------------------------------------#
                             # Calculate Returns #
#--------------------------------------------------------------------------#

df['returns'] = np.log(
    df['Close'] / df['Close'].shift(1)
)

df.dropna(inplace=True)


#--------------------------------------------------------------------------#
                             # Price Chart #
#--------------------------------------------------------------------------#

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'])
plt.title(f'{TICKER} Closing Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
                            # Returns Distribution #
#--------------------------------------------------------------------------#

plt.figure(figsize=(10,5))
plt.hist(
    df['returns'],
    bins=100
    )
plt.title('Log Returns Distribution')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
                            # Returns Statistics #
#--------------------------------------------------------------------------#

mean_return = df['returns'].mean()
volatility = df['returns'].std()
return_skew = skew(df['returns'])
return_kurtosis = kurtosis(df['returns'])

print("\nRETURN STATISTICS")
print("------------------------")

print(f"Mean Return: {mean_return:.10f}")
print(f"Volatility: {volatility:.10f}")
print(f"Skewness: {return_skew:.10f}")
print(f"Kurtosis: {return_kurtosis:.10f}")


#--------------------------------------------------------------------------#
                            # Rolling Statistics #
#--------------------------------------------------------------------------#

df['rolling_mean'] = (
    df['Close']
    .rolling(ROLLING_WINDOW)
    .mean()
    )

df['rolling_std'] = (
    df['Close']
    .rolling(ROLLING_WINDOW)
    .std()
    )


#--------------------------------------------------------------------------#
                            # Calculate Z-score #
#--------------------------------------------------------------------------#

df['zscore'] = (
    (df['Close'] - df['rolling_mean']) /
    df['rolling_std']
    )

#--------------------------------------------------------------------------#
                            # Plot Z-score #
#--------------------------------------------------------------------------#

plt.figure(figsize=(12,6))
plt.plot(df.index, df['zscore'],color='black')
plt.xlabel('Year')
plt.ylabel('Z-Score')
plt.axhline(ZSCORE_THRESHOLD, linestyle='--', color='red')
plt.axhline(-ZSCORE_THRESHOLD, linestyle='--', color='red')
plt.axhline(0, linestyle='-')
plt.title('Rolling Z-Score')
plt.legend()
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
                        # Trading Signal Generation #
#--------------------------------------------------------------------------#

df['signal'] = 0

df.loc[
       df['zscore'] < -ZSCORE_THRESHOLD,
       'signal'
       ] = 1

df.loc[
       df['zscore'] > ZSCORE_THRESHOLD,
       'signal'
       ] = -1


#--------------------------------------------------------------------------#
                        # Trading Signal Visualization #
#--------------------------------------------------------------------------#

plt.figure(figsize=(14,7))
plt.plot(df.index, df['Close'], label='Close Price', color='black')
buy_signals = df[df['signal'] == 1]
sell_signals = df[df['signal'] == -1]

plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', s=100,
            label='Long Signal', color='red')
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', s=100,
            label='Short Signal', color='blue')
plt.title(f'{TICKER} Mean-Reversion Signals')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
                        # Trading Signal Counter #
#--------------------------------------------------------------------------#

num_long = (
    df['signal'] == 1
    ).sum()

num_short = (
    df['signal'] == -1
    ).sum()

print("\nSIGNAL COUNTS")
print("---------------------------")

print(f"Long Signals: {num_long}")
print(f"Short Signals: {num_short}")


#--------------------------------------------------------------------------#
                    # Signals --> Portfolio Performance #
#--------------------------------------------------------------------------#

df['position'] = df['signal'].shift(1)
df['strategy_returns'] = df['position'] * df['returns']

transaction_cost = 0.0005 # 5 bps per trade


#--------------------------------------------------------------------------#
                            # Detect Trade Events #
#--------------------------------------------------------------------------#

df['trade'] = df['position'].diff().abs()


#--------------------------------------------------------------------------#
                                # Apply costs #
#--------------------------------------------------------------------------#

df['strategy_returns'] -= df['trade'] * transaction_cost


#--------------------------------------------------------------------------#
                        # Build and Plot Equity Curve #
#--------------------------------------------------------------------------#

df['equity_curve'] = (1 + df['strategy_returns']).cumprod()
df['buy_hold'] = (1 + df['returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['equity_curve'], label='Strategy')
plt.plot(df.index, df['buy_hold'], label='Buy and Hold')
plt.xlabel('Year')
plt.ylabel('Portfolio Value (as a ratio of Initial Capital)')
plt.title("Equity Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
 # Calculate Sharpe Ratio, Maximum Drawdown, Win Rate, and Profit Factor #
#--------------------------------------------------------------------------#

sharpe = (
    df['strategy_returns'].mean() /
    df['strategy_returns'].std()
    ) * np.sqrt(252)

rolling_max = df['equity_curve'].cummax()
drawdown = (df['equity_curve'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

win_rate = (df['strategy_returns'] > 0).mean()

gross_profit = df[df['strategy_returns'] > 0].sum()
gross_loss = abs(df[df['strategy_returns'] < 0].sum())
profit_factor = gross_profit / gross_loss


#--------------------------------------------------------------------------#
                            # Backtesting Results #
#--------------------------------------------------------------------------#

print("\nBacktesting Results")
print("----------------------------------")

print("Sharpe Ratio:", sharpe)
print("Maximum Drawdown:", max_drawdown)
print("Win Rate:", win_rate)
print("Profit Factor:", profit_factor)


#--------------------------------------------------------------------------#
                            # Calculate Volatility #
#--------------------------------------------------------------------------#

df['volatility'] = df['returns'].rolling(20).std()

low_vol = df['volatility'].quantile(0.33)
high_vol = df['volatility'].quantile(0.66)

def regime(v):
    if v < low_vol:
        return "low"
    elif v > high_vol:
        return "high"
    else:
        return "medium"

df['regime'] = df['volatility'].apply(regime)

low = df[df['regime'] == "low"]
mid = df[df['regime'] == "medium"]
high = df[df['regime'] == "high"]

def sharpe(x):
    return (x.mean() / x.std()) * np.sqrt(252)


#--------------------------------------------------------------------------#
                    # Volatility Regime Analysis Results #
#--------------------------------------------------------------------------#

print("\nRegime Analysis Results")
print("---------------------------------")

print("Low-Volatility Sharpe Ratio:", sharpe(low['strategy_returns']))
print("Medium-Volatility Sharpe Ratio:", sharpe(mid['strategy_returns']))
print("High-Volatility Sharpe Ratio:", sharpe(high['strategy_returns']))


#--------------------------------------------------------------------------#
                            # Volatility Plot #
#--------------------------------------------------------------------------#

plt.figure(figsize=(12,6))
plt.plot(df.index, df['volatility'])
plt.xlabel('Year')
plt.ylabel('Volatility')
plt.title("Realized Volatility (20 days)")
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------#
                            # ML-Based Prediction #
#--------------------------------------------------------------------------#

df['future_return'] = df['returns'].shift(-1)
df['target'] = (df['future_return'] > 0).astype(int)

df['zscore_feature'] = df['zscore']
df['vol_feature'] = df['volatility']
df['momentum'] = df['returns'].rolling(5).mean()

df = df.dropna()

split = int(len(df) * 0.7)

train = df.iloc[:split]
test = df.iloc[split:]

features = ['zscore_feature', 'vol_feature', 'momentum']

model = LogisticRegression()
model.fit(train[features], train['target'])

test['prob_up'] = model.predict_proba(test[features])[:,1]

test['ml_signal'] = 0
test.loc[test['prob_up'] > 0.55, 'ml_signal'] = 1
test.loc[test['prob_up'] < 0.45, 'ml_signal'] = -1


#--------------------------------------------------------------------------#
                            # Backtesting ML Strategy #
#--------------------------------------------------------------------------#

test['ml_position'] = test['ml_signal'].shift(1)

test['ml_returns'] = test['ml_position'] * test['returns']
test['ml_equity'] = (1 + test['ml_returns']).cumprod()


print("\nModel Comparison")
print("------------------------------")

print("Rule-based Sharpe Ratio:", sharpe(df['strategy_returns']))
print("ML Sharpe Ratio:", sharpe(test['ml_returns']))


#--------------------------------------------------------------------------#
                        # Parameter Sensitivity Analysis #
#--------------------------------------------------------------------------#

window_values = [10, 20, 30, 50, 60]            # rolling window values
threshold_values = [1.0, 1.5, 2.0, 2.5, 3.0]    # z-score values

results = []

for window, threshold in itertools.product(
        window_values,
        threshold_values
        ):
    temp_df = df.copy()
    temp_df['rolling_mean'] = (
        temp_df['Close'].rolling(window).mean()
        )
    temp_df['rolling_std'] = (
        temp_df['Close'].rolling(window).std()
        )
    temp_df['zscore'] = (
        (temp_df['Close'] - temp_df['rolling_mean']) /
        temp_df['rolling_std']
        )
    temp_df['signal'] = 0
    temp_df.loc[
        temp_df['zscore'] < -threshold,
        'signal'
        ] = 1
    temp_df.loc[
        temp_df['zscore'] > threshold,
        'signal'
        ] = -1
    temp_df['position'] = temp_df['signal'].shift(1)
    temp_df['strategy_returns'] = (
        temp_df['position'] *
        temp_df['returns']
        )
    temp_df.dropna(inplace=True)
    sharpe = (
        temp_df['strategy_returns'].mean() /
        temp_df['strategy_returns'].std()
        ) * np.sqrt(252)
    
    results.append({
        'window': window,
        'threshold': threshold,
        'sharpe': sharpe
    })

results_df = pd.DataFrame(results)

pivot_table = results_df.pivot(
    index = 'window',
    columns = 'threshold',
    values = 'sharpe'
)

plt.figure(figsize=(12,6))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".5f"
    )
plt.xlabel('Rolling Window')
plt.ylabel('Z-Score Threshold')
plt.title("Parameter Sensitivity Heatmap")
plt.show()


#--------------------------------------------------------------------------#
                            # Monte Carlo Simulation #
#--------------------------------------------------------------------------#

strategy_returns = (
    df['strategy_returns']
    .dropna()
    .values
    )

num_simulations = 1000
simulation_length = len(strategy_returns)

simulation_results = np.zeros(
    (simulation_length, num_simulations)
    )

for i in range(num_simulations):
    simulated_returns = np.random.choice(
        strategy_returns,
        size = simulation_length,
        replace=True
        )
    simulated_equity = (
        1 + simulated_returns
        ).cumprod()
    simulation_results[:,i] = simulated_equity


plt.figure(figsize=(12,6))
plt.plot(simulation_results, alpha=0.1)
plt.title('Monte Carlo Simulations of Equity Curve')
plt.xlabel('Time')
plt.ylabel('Portfolio Growth')
plt.grid(True)
plt.show()

final_values = simulation_results[-1,:]

plt.figure(figsize=(12,6))
plt.hist(final_values,
         bins=50
         )
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Equity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
   # Calculate Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) #
#--------------------------------------------------------------------------#

var_95 = np.percentile(final_values, 5)

cvar_95 = final_values[
    final_values <= var_95
    ].mean()

print("\nVaR Calculation")
print("----------------------------")

print("95% VaR:", var_95)
print("95% CVaR:", cvar_95)


#--------------------------------------------------------------------------#
                        # Walk-forward Validation #
#--------------------------------------------------------------------------#

train_size = int(len(df) * 0.6)
validation_size = int(len(df) * 0.2)
test_size = len(df) - train_size - validation_size

train_df = df.iloc[:train_size]
validation_df = df.iloc[train_size:train_size + validation_size]
test_df = df.iloc[train_size + validation_size:]

print("\nWalk-forward Validation")
print("-------------------------------------------")

print("Train size:", len(train_df))
print("Validation size:", len(validation_df))
print("Test size:", len(test_df))


#--------------------------------------------------------------------------#
                    # Parameter Optimization on Train Set #
#--------------------------------------------------------------------------#

optimization_results = []

for window in window_values:
    
    for threshold in threshold_values:
        
        temp = train_df.copy()
        
        temp['rolling_mean'] = (
            temp['Close']
            .rolling(window)
            .mean()
            )
        
        temp['rolling_std'] = (
            temp['Close']
            .rolling(window)
            .std()
            )
        
        temp['zscore'] = (
            (temp['Close'] - temp['rolling_mean']) /
            temp['rolling_std']
            )
        
        temp['signal'] = 0
        
        temp.loc[
            temp['zscore'] < -threshold,
            'signal'
            ] = 1
        
        temp.loc[
            temp['zscore'] > threshold,
            'signal'
            ] = -1
        
        temp['position'] = temp['signal'].shift(1)
        
        temp['strategy_returns'] = (
            temp['position'] *
            temp['returns']
            )
        
        sharpe = (
            temp['strategy_returns'].mean() /
            temp['strategy_returns'].std()
            ) * np.sqrt(252)
        
        optimization_results.append({
            'window': window,
            'threshold': threshold,
            'sharpe': sharpe
            })
        
        optimization_df = pd.DataFrame(optimization_results)
        
        best_row = optimization_df.loc[optimization_df['sharpe'].idxmax()]
        
        best_window = best_row['window']
        
        best_threshold = best_row['threshold']

print("\nParameter Optimization")
print("-----------------------------------")

print("Best Window:", best_window)
print("Best Threshold:", best_threshold)


#--------------------------------------------------------------------------#
                            # Validation Testing #
#--------------------------------------------------------------------------#

validation = validation_df.copy()

validation['rolling_mean'] = (
    validation['Close']
    .rolling(int(best_window))
    .mean()
    )

validation['rolling_std'] = (
    validation['Close']
    .rolling(int(best_window))
    .std()
    )

validation['zscore'] = (
    (validation['Close'] - validation['rolling_mean']) /
    validation['rolling_std']
    )

validation['signal'] = 0

validation.loc[
    validation['zscore'] < -best_threshold,
    'signal'
    ] = 1

validation.loc[
    validation['zscore'] > best_threshold,
    'signal'
    ] = -1

validation['position'] = (validation['signal'].shift(1))

validation['strategy_returns'] = (
    validation['position'] * validation['returns']
    )

validation.dropna(inplace=True)

validation_sharpe = (
    validation['strategy_returns'].mean() /
    validation['strategy_returns'].std()
    ) * np.sqrt(252)


#--------------------------------------------------------------------------#
                            # Out-of-sample Testing #
#--------------------------------------------------------------------------#

test_sharpe = (
    test_df['strategy_returns'].mean() / test_df['strategy_returns'].std()
    ) * np.sqrt(252)


#--------------------------------------------------------------------------#
                            # Cross-asset Robustness #
#--------------------------------------------------------------------------#

tickers = [
    '^GSPC',
    'QQQ',
    'GLD',
    'BTC-USD',
    'ETH-USD'
    ]

asset_results = []

for ticker in tickers:
    
    temp_df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE
        )
    
    temp_df.columns = temp_df.columns.get_level_values(0)
    temp_df.dropna(inplace=True)
    
    temp_df['returns'] = np.log(
        temp_df['Close'] / temp_df['Close'].shift(1)
        )
    temp_df.dropna(inplace=True)
    
    temp_df['rolling_mean'] = (
        temp_df['Close'].rolling(int(best_window))
        .mean()
        )
    temp_df['rolling_std'] = (
        temp_df['Close'].rolling(int(best_window))
        .std()
        )
    
    temp_df['zscore'] = (
        (temp_df['Close'] - temp_df['rolling_mean']) / temp_df['rolling_std']
        )
    
    temp_df['signal'] = 0
    
    temp_df.loc[
        temp_df['zscore'] < -best_threshold,
        'signal'
        ] = 1
    
    temp_df.loc[
        temp_df['zscore'] > best_threshold,
        'signal'
        ] = -1
    
    temp_df['position'] = (
        temp_df['signal'].shift(1)
        )
    
    temp_df['strategy_returns'] = (
        temp_df['position'] * temp_df['returns']
        )
    
    temp_df['equity_curve'] = (
        1 + temp_df['strategy_returns']
        ).cumprod()
    
    asset_sharpe = (
        temp_df['strategy_returns'].mean() / temp_df['strategy_returns'].std()
        ) * np.sqrt(252)
    
    rolling_max = (
        temp_df['equity_curve'].cummax()
        )
    
    drawdown = (
        temp_df['equity_curve'] - rolling_max
        ) / rolling_max
    
    asset_max_dd = drawdown.min()
    
    asset_return = (
        temp_df['equity_curve'].iloc[-1] - 1
        )
    
    asset_results.append({
        'ticker': ticker,
        'sharpe': asset_sharpe,
        'max_drawdown': asset_max_dd,
        'total_return': asset_return
        })

asset_results_df = pd.DataFrame(asset_results)

print(asset_results_df)

#--------------------------------------------------------------------------#
                                # Stress Testing #
#--------------------------------------------------------------------------#

# Isolate crisis period (e.g. COVID crash)

covid_period = df[
    (df.index >= '2020-02-01') & (df.index <= '2020-06-01')
    ]

# Plot crisis equity curve

plt.figure(figsize=(12,6))
plt.plot(covid_period.index, covid_period['equity_curve'])
plt.xlabel('Date')
plt.ylabel('Portfolio Growth')
plt.title('COVID Crash Performance')
plt.grid(True)
plt.show()

covid_return = (
    covid_period['equity_curve'].iloc[-1] /
    covid_period['equity_curve'].iloc[0]
    ) - 1

#--------------------------------------------------------------------------#
                                # Drawdown Analysis #
#--------------------------------------------------------------------------#


rolling_max = (
    df['equity_curve'].cummax()
    )

drawdown = (
    df['equity_curve'] - rolling_max
    ) / rolling_max


plt.figure(figsize=(12,6))
plt.plot(drawdown)
plt.title('Strategy Drawdown')
plt.xlabel('Time')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

max_drawdown = drawdown.min()
print(max_drawdown)


#--------------------------------------------------------------------------#
                        # Rolling Performance Analysis #
#--------------------------------------------------------------------------#

rolling_sharpe = (
    df['strategy_returns'].rolling(252).mean() /
    df['strategy_returns'].rolling(252).std()
    ) * np.sqrt(252)

plt.figure(figsize=(12,6))
plt.plot(rolling_sharpe)
plt.title('Rolling Sharpe Ratio')
plt.grid(True)
plt.show()


#--------------------------------------------------------------------------#
                                # Save Results #
#--------------------------------------------------------------------------#

df.to_csv(
    rf"C:\Users\Craig Egan Tan\Desktop\Physics\Finance\Quantitative Finance Simulations\Mean-reversion\{TICKER}_dataset.csv"
)

print("\nDataset saved.")