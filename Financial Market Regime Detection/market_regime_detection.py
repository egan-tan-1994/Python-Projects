import logging
import time
import numpy as np
import pandas as pd
import itertools
import joblib
import random
from sklearn.preprocessing import (StandardScaler, LabelEncoder)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


logging.basicConfig(
    filename="regime_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


##############################################################################
                            # Download Historical Data #
##############################################################################

ticker = "SPY"

df = yf.download(ticker, start="2005-01-01", end="2026-01-01", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    
    df.columns = df.columns.get_level_values(0)


df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

df.dropna(inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'])
plt.title(f"{ticker} Adjusted Closing Price")
plt.xlabel("Year")
plt.ylabel("Price (USD)")
plt.grid()
plt.show()


##############################################################################
                                # Create Returns #
##############################################################################

df['return'] = df['Close'].pct_change()

df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

df['cum_return'] = (1 + df['return']).cumprod()

annual_return = (df['return'].mean() * 252)

annual_volatility = (df['return'].std() * np.sqrt(252))

print(f"Annualized Return: {annual_return:.6f}")
print(f"Annualized Voltility: {annual_volatility:.6f}")

plt.figure(figsize=(12,6))
plt.plot(df.index, df['return'])
plt.title(f"{ticker} Daily Returns")
plt.xlabel("Year")
plt.ylabel("Returns")
plt.grid()
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df['return'].dropna(), bins=100, kde=True)
plt.xlabel("Returns")
plt.title(f"{ticker} Returns Distribution")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['cum_return'])
plt.title(f"{ticker} Cumulative Returns")
plt.xlabel("Year")
plt.ylabel("Growth of $1")
plt.grid()
plt.show()

##############################################################################
                # Create Volatility and Momemtum Features #
##############################################################################

df['volatility_10'] = (df['return'].rolling(10).std())

df['volatility_20'] = (df['return'].rolling(20).std())

df['volatility_60'] = (df['return'].rolling(60).std())


plt.figure(figsize=(12,6))
plt.plot(df.index, df['volatility_10'], label="10-day Rolling Volatility")
plt.plot(df.index, df['volatility_20'], label="20-day Rolling Volatility")
plt.plot(df.index, df['volatility_60'], label="60-day Rolling Volatility")
plt.legend()
plt.title(f"{ticker} Rolling Volatility")
plt.show()


# Average True Range (ATR)

tr1 = df["High"] - df['Low']

tr2 = abs(df['High'] - df['Close'].shift(1))

tr3 = abs(df['Low'] - df['Close'].shift(1))

tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)

df['ATR'] = tr.rolling(14).mean()


plt.figure(figsize=(12,6))
plt.plot(df.index, df['ATR'])
plt.title(f"{ticker} Average True Range")
plt.xlabel("Year")
plt.ylabel("ATR")
plt.grid()
plt.show()


# Moving Average Returns

df['ma20'] = (df['Close'].rolling(20).mean())

df['ma50'] = (df['Close'].rolling(50).mean())

df['ma200'] = (df['Close'].rolling(200).mean())


plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Closing Price")
plt.plot(df.index, df['ma20'], label="20-day Moving Average Returns")
plt.plot(df.index, df['ma50'], label="50-day Moving Average Returns")
plt.plot(df.index, df['ma200'], label="200-day Moving Average Returns")
plt.legend()
plt.title(f"{ticker} Closing Price and Moving Averages")
plt.show()


# Price Distance from Moving Average

df['dist_ma20'] = (df['Close'] / df['ma20']) - 1

df['dist_ma50'] = (df['Close'] / df['ma50']) - 1

df['dist_ma200'] = (df['Close'] / df['ma200']) - 1


plt.figure(figsize=(12,6))
plt.plot(df.index, df['dist_ma20'], label="Price Distance from 20-day MA")
plt.plot(df.index, df['dist_ma50'], label="Price Distance from 50-day MA")
plt.plot(df.index, df['dist_ma200'], label="Price Distance from 20-day MA")
plt.legend()
plt.title(f"{ticker} Price Distance from Moving Average")
plt.show()


# Momentum

df['mom_5'] = (df['Close'] / df['Close'].shift(5)) - 1

df['mom_20'] = (df['Close'] / df['Close'].shift(20)) - 1

df['mom_60'] = (df['Close'] / df['Close'].shift(60)) - 1


plt.figure(figsize=(12,6))
plt.plot(df.index, df['mom_5'], label="5-day Momentum")
plt.plot(df.index, df['mom_20'], label="20-day Momentum")
plt.plot(df.index, df['mom_60'], label="60-day Momentum")
plt.legend()
plt.title(f"{ticker} Momentum Plot")
plt.show()


##############################################################################
                        # Market Microstructure Features #
##############################################################################

# Daily Range

df['range'] = (df['High'] - df['Low']) / df['Close']

plt.figure(figsize=(12,6))
plt.plot(df.index, df['range'])
plt.title(f"{ticker} Daily Range")
plt.xlabel("Year")
plt.ylabel("Range")
plt.grid()
plt.show()


# Volume Z-score

vol_mean = (df['Volume'].rolling(20).mean())

vol_std = (df['Volume'].rolling(20).std())

df['volume_zscore'] = (df['Volume'] - vol_mean) / vol_std

plt.figure(figsize=(12,6))
plt.plot(df.index, df['volume_zscore'])
plt.title(f"{ticker} Volume Z-Score (20-day window)")
plt.xlabel("Year")
plt.ylabel("Z-Score")
plt.grid()
plt.show()


# Price Efficiency

net_move = abs(df['Close'] - df['Close'].shift(20))

path_length = abs(df['Close'].diff().rolling(20).sum())

df['efficiency'] = (net_move / path_length)

df['efficiency'] = (df['efficiency'].replace([-np.inf, np.inf], np.nan))

plt.figure(figsize=(12,6))
plt.plot(df.index, df['efficiency'])
plt.title(f"{ticker} Price Efficiency (20-day window)")
plt.xlabel("Year")
plt.ylabel("Efficiency")
plt.grid()
plt.show()



##############################################################################
                        # Data Processing and Cleaning #
##############################################################################

df.replace([-np.inf, np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,6))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title(f"{ticker} Feature Correlation Matrix")
plt.show()

feature_columns = [
    'return',
    'volatility_10',
    'volatility_20',
    'volatility_60',
    'ATR',
    'dist_ma20',
    'dist_ma50',
    'dist_ma200',
    'mom_5',
    'mom_20',
    'mom_60',
    'range',
    'volume_zscore',
    'efficiency'
    ]


##############################################################################
            # Hidden Markov Model (HMM) Dataset and Parameters #
##############################################################################


hmm_features = [
    'return',
    'volatility_20',
    'mom_20',
    'range',
    'volume_zscore'
    ]

X_hmm_raw = df[hmm_features].copy()

scaler = StandardScaler()

scaler.fit(X_hmm_raw)

X_hmm_scaled = scaler.transform(X_hmm_raw)

X_hmm_df = pd.DataFrame(X_hmm_scaled, columns=hmm_features, index=df.index)

X_hmm_df.hist(figsize=(12,6), bins=40)
plt.tight_layout()
plt.show()

feature_names = (hmm_features.copy())

X_hmm = X_hmm_scaled

n_states = 4

hmm = GaussianHMM(
    n_components=n_states,
    covariance_type='full',
    n_iter=1000,
    random_state=42)

hmm.fit(X_hmm)

regimes = hmm.predict(X_hmm)

df['regime'] = regimes


summary = df.groupby('regime').agg({
    'return':'mean',
    'volatility_20':'mean'
    })

print(summary)

means = pd.DataFrame(hmm.means_, columns=hmm_features)

print(means)

print(hmm.transmat_)

df['regime_change'] = (df['regime'] != df['regime'].shift(1))

print(df['regime_change'].sum())

df.drop(columns=['regime_change'], inplace=True)


##############################################################################
                        # Regime Mapping and Visualization #
##############################################################################

regime_stats = (df.groupby('regime').agg({
    'return':'mean',
    'volatility_20':'mean',
    'mom_20':'mean'
    }))

bull_state = (regime_stats['return'].idxmax())

bear_state = (regime_stats['return'].idxmin())

highvol_state = (regime_stats['volatility_20'].idxmax())

all_states = set(regime_stats.index)

used_states = {
    bull_state,
    bear_state,
    highvol_state
    }

sideways_state = list(all_states - used_states)[0]

mapping = {
    bull_state:'Bull',
    bear_state:'Bear',
    sideways_state:'Sideways',
    highvol_state:'HighVol'
    }

regime_mapping = {
    0: 'Bull',
    1: 'Bear',
    2: 'Sideways',
    3: 'HighVol'
    }

df['regime_label'] = (df['regime'].map(mapping))

regime_report = (df.groupby('regime_label').agg({
    'return':['mean', 'std'],
    'volatility_20':'mean',
    'mom_20':'mean',
    'Close':'count'
    }))

colors = {
    'Bull':'green',
    'Bear':'red',
    'Sideways':'blue',
    'HighVol':'orange'
    }


plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], color='black', linewidth=1)
for label in colors:
    subset = (df[df['regime_label'] == label])
    
    plt.scatter(subset.index, subset['Close'], s=5, c=colors[label], label=label)
plt.title(f"{ticker} Hidden Markov Model Market Regimes")
plt.xlabel("Year")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.scatter(df.index, df['regime'], s=2)
plt.title(f"{ticker} Hidden State Timeline")
plt.xlabel("Year")
plt.ylabel("State")
plt.show()


##############################################################################
  # Build Supervised Dataset, Train Split and XGBoost, and Predict Regimes #
##############################################################################

xgb_features = [
    'return',
    'log_return',
    'volatility_10',
    'volatility_20',
    'volatility_60',
    'mom_5',
    'mom_20',
    'mom_60',
    'dist_ma20',
    'dist_ma50',
    'dist_ma200',
    'range',
    'ATR',
    'volume_zscore',
    'efficiency'   
    ]

X = df[xgb_features].copy()

y = df['regime_label'].copy()


label_encoder = LabelEncoder()


y_encoded = (label_encoder.fit_transform(y))


dataset = df[xgb_features + ["regime_label"]].copy()

dataset = dataset.replace([np.inf, -np.inf], np.nan)

dataset = dataset.dropna()

X = dataset[xgb_features]

y = dataset["regime_label"]

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_


feature_names = (X.columns.tolist())


split_index = int(len(X) * 0.8)


X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]


tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []

fold_results = []


plt.figure(figsize=(12,6))
plt.axvspan(X_train.index.min(), X_train.index.max(), alpha=0.3, label='Train'
            , color='blue')
plt.axvspan(X_test.index.min(), X_test.index.max(), alpha=0.3, label='Test',
            color='red')
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.title(f"{ticker} Train/Test Split")
plt.legend()
plt.show()


cv_scores = []

fold_reports = []

fold_predictions = []


base_model = XGBClassifier(
    objective='multi:softmax',
    num_class = 4,
    n_estimators = 500,
    max_depth = 5,
    learning_rate = 0.03,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42    
    )

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    
    X_fold_train = (X.iloc[train_idx])
    
    X_fold_test = (X.iloc[test_idx])
    
    y_fold_train = (y_encoded[train_idx])
    
    y_fold_test = (y_encoded[test_idx])
    
    model = XGBClassifier(
        objective='multi:softmax',
        num_class = 4,
        n_estimators = 300,
        max_depth = 4,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42
        )
    
    model.fit(X_fold_train, y_fold_train)
    
    preds = model.predict(X_fold_test)
    
    acc = accuracy_score(y_fold_test, preds)
    
    cv_scores.append(acc)
    
    report = classification_report(y_fold_test, preds, output_dict=True)
    
    fold_reports.append(report)
    
    fold_predictions.append(preds)


mean_cv_accuracy = np.mean(cv_scores)

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class = 4,
    n_estimators = 300,
    max_depth = 4,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42
    )
    

xgb_model.fit(X_train, y_train)

test_preds = xgb_model.predict(X_test)

test_acc = accuracy_score(y_test, test_preds)

print("\nMean Accuracy:", mean_cv_accuracy)

print("Test Accuracy:", test_acc)



##############################################################################
                            # Generate Confusion Matrix #
##############################################################################

labels = np.arange(len(class_names))

report = classification_report(
    y_test,
    test_preds,
    labels=labels,
    target_names=class_names,
    zero_division=0)

print(report)


report_dict = classification_report(
    y_test,
    test_preds,
    labels=labels,
    target_names=class_names,
    zero_division=0,
    output_dict=True)

report_df = pd.DataFrame(report_dict).T

print(report_df)

labels = np.arange(len(class_names))

cm = confusion_matrix(y_test, test_preds, labels=labels)

print(cm)

plt.figure(figsize=(12,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
    )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{ticker} Regime Classification Matrix")
plt.tight_layout()
plt.show()

cm_norm = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis])

plt.figure(figsize=(12,6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.3f',
    xticklabels=class_names,
    yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{ticker} Normalized Confusion Matrix")
plt.show()


per_class_acc = (cm.diagonal() / cm.sum(axis=1))

acc_df = pd.DataFrame({
    'Regime': class_names,
    'Accuracy': per_class_acc})

plt.figure(figsize=(12,6))
plt.bar(acc_df['Regime'], acc_df['Accuracy'])
plt.title(f"{ticker} Regime Prediction Accuracy")
plt.ylabel("Accuracy")
plt.show()


worst_regime = (acc_df.sort_values('Accuracy').iloc[0])

best_regime = (acc_df.sort_values('Accuracy', ascending=False).iloc[0])

print("Worst Regime:", worst_regime)

print("Best Regime:", best_regime)

##############################################################################
            # Feature Importance and Regime Persistance Analysis #
##############################################################################

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
    })

importance = (importance.sort_values('Importance', ascending=False))

plt.plot(figsize=(12,6))
plt.barh(importance['Feature'], importance['Importance'])
plt.title(f"{ticker} XGBoost Feature Importance")
plt.show()

probabilities = xgb_model.predict_proba(X_test)

prob_df = pd.DataFrame(
    probabilities,
    columns=class_names,
    index=X_test.index
    )

predicted_classes = (xgb_model.predict(X_test))

predicted_labels = [
    regime_mapping[p]
    for p in predicted_classes
    ]

prob_df['Predicted Regime'] = (predicted_labels)

prob_df['Confidence'] = (probabilities.max(axis=1))

avg_confidence = (prob_df.groupby('Predicted Regime')['Confidence'].mean())

print("Average Confidence:", avg_confidence)


##############################################################################
            # Probabilistic Regime Forecasting and Confidence Scoring #
##############################################################################

plt.figure(figsize=(12,6))
plt.hist(prob_df['Confidence'], bins=30)
plt.title(f"{ticker} Prediction Confidence")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.show()

confidence_threshold = 0.80

high_confidence = (prob_df[prob_df['Confidence'] >= confidence_threshold])

high_conf_mask = (prob_df['Confidence'] >= confidence_threshold)

filtered_preds = (predicted_classes[high_conf_mask])

filtered_actuals = (y_test[high_conf_mask])

high_conf_acc = accuracy_score(filtered_actuals, filtered_preds)

print("High Confidence Accuracy:", high_conf_acc)

top_probabilities = (
    prob_df[class_names].apply(lambda row:row.sort_values(ascending=False),
                               axis=1))

sorted_probs = np.sort(probabilities, axis=1)

prob_df['Spread'] = sorted_probs[:, -1] - sorted_probs[:, -2]

prob_df['Confidence Bin'] = pd.cut(prob_df['Confidence'],
                                   bins=np.arange(0.0, 1.1, 0.1))

uncertain_cases = (prob_df.sort_values('Confidence').head(20))


plt.figure(figsize=(12,6))
plt.plot(prob_df.index, prob_df['Confidence'])
plt.title(f"{ticker} Regime Prediction Confidence")
plt.xlabel("Date")
plt.ylabel("Confidence")
plt.show()


regime_series = pd.Series(predicted_classes, index=X_test.index)

regime_prev = regime_series.shift(1)

transition_flag = (regime_series != regime_prev)

transition_flag = transition_flag.astype(int)


prob_df['Regime'] = regime_series

prob_df['Previous Regime'] = regime_prev

prob_df['Transition'] = transition_flag


total_transitions = transition_flag.sum()

print("Total Transitions:", total_transitions)


plt.figure(figsize=(12,6))
plt.plot(prob_df.index, prob_df['Transition'], drawstyle="steps-post")
plt.title(f"{ticker} Regime Transition Events")
plt.xlabel("Date")
plt.ylabel("Transition (1 = Yes)")
plt.show()


prob_df['transition_rate_20'] = (
    prob_df['Transition'].rolling(20).mean())


plt.figure(figsize=(12,6))
plt.plot(prob_df.index, prob_df['transition_rate_20'])
plt.title(f"{ticker} Regime Transition Rate (20-Day Rolling Window)")
plt.xlabel("Date")
plt.ylabel("Transition Rate")
plt.show()


eps = 1e-10

log_probs = np.log(probabilities + eps)

entropy = -np.sum(probabilities * log_probs, axis=1)

prob_df['Entropy'] = entropy


plt.figure(figsize=(12,6))
plt.plot(prob_df.index, prob_df['Entropy'])
plt.title(f"{ticker} Regime Uncertainty (Entropy)")
plt.xlabel("Date")
plt.ylabel("Entropy")
plt.show()


prob_df['Warning'] = (
    (prob_df['Entropy'] > prob_df['Entropy'].rolling(50).mean()) &
    (prob_df['transition_rate_20'] > 0.1)
    )

prob_df['Warning'] = prob_df['Warning'].astype(int)

prob_df['Close'] = df.loc[X_test.index, "Close"].values


plt.figure(figsize=(12,6))
plt.plot(prob_df.index, prob_df['Close'])
plt.scatter(
    prob_df.index[prob_df['Warning'] == 1],
    prob_df['Close'][prob_df['Warning'] == 1],
    color='red',
    label='Warning Signal',
    s=20
    )
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.title(f"{ticker} Early Regime Transition Warnings")
plt.show()


transitions = pd.DataFrame({
    'prev': regime_prev,
    'curr': regime_series
    })

transitions.dropna(inplace=True)

transition_counts = (
    transitions.groupby(['prev', 'curr']).size().sort_values(ascending=False)
    )

print("Transition Counts:", transition_counts)

transition_matrix = pd.crosstab(
    transitions['prev'],
    transitions['curr']
    )


plt.figure(figsize=(12,6))
sns.heatmap(transition_matrix, annot=True, fmt='d')
plt.xlabel("Current Regime")
plt.ylabel("Previous Regime")
plt.title(f"{ticker} Regime Transition Matrix")
plt.show()


transition_prob_matrix = (
    transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    )


strategy_df = prob_df.copy()

strategy_df['Close'] = df.loc[X_test.index, 'Close']

position_map = {
    'Bull': 1.0,
    'Bear': -1.0,
    'Sideways': 0.3,
    'HighVol': 0.0
    }

predicted_regimes = label_encoder.inverse_transform(predicted_classes)

strategy_df['Regime'] = predicted_regimes

strategy_df['Position'] = (strategy_df['Regime'].map(position_map))

strategy_df['Position'] = (strategy_df['Position'] * strategy_df['Confidence'])

strategy_df['Position'] = (strategy_df['Position'].shift(1))

strategy_df['Market Return'] = (strategy_df['Close'].pct_change())

strategy_df['Strategy Return'] = (
    strategy_df['Position'] * strategy_df['Market Return'])

strategy_df.dropna(inplace=True)

strategy_df['Cumulative Market'] = (
    1 + strategy_df['Market Return']
    ).cumprod()

strategy_df['Cumulative Strategy'] = (
    1 + strategy_df['Strategy Return']
    ).cumprod()


plt.figure(figsize=(12,6))
plt.plot(strategy_df.index, strategy_df['Cumulative Market'], label='Market')
plt.plot(strategy_df.index, strategy_df['Cumulative Strategy'], label='Strategy')
plt.legend()
plt.title(f"{ticker} Regime-Based Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.show()


market_return = (strategy_df['Cumulative Market'].iloc[-1] - 1)

strategy_return = (strategy_df['Cumulative Strategy'].iloc[-1] - 1)

sharpe = (
    strategy_df['Strategy Return'].mean() / strategy_df['Strategy Return'].std()
    ) * np.sqrt(252)

rolling_max = (strategy_df['Cumulative Strategy'].cummax())

drawdown = (strategy_df['Cumulative Strategy'] / rolling_max - 1)

max_drawdown = drawdown.min()


plt.figure(figsize=(12,6))
plt.plot(drawdown.index, drawdown)
plt.title(f"{ticker} Strategy Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.show()


print("------------------------------------------")
print("Strategy vs Market Metrics")
print("--------------------------------")
print("Market Return:", market_return)
print("Strategy Return:", strategy_return)
print("Sharpe Ratio:", sharpe)
print("Maximum Drawdown:", max_drawdown)
print("------------------------------------------")

regime_performance = strategy_df.groupby('Regime')['Strategy Return'].sum()

print("------------------------------------------")
print("Regime Contribution to Returns")
print("----------------------------------")
print("Regime Performance:", regime_performance)
print("------------------------------------------")

regime_sharpe = strategy_df.groupby(
    'Regime')['Strategy Return'].apply(
        lambda x:
            x.mean() / x.std() * np.sqrt(252))

print("------------------------------------------")
print("Regime Contribution to Sharpe Ratio")
print("----------------------------------")
print("Sharpe Ratio by Regime:", regime_sharpe)
print("------------------------------------------")

##############################################################################
   # Strategy Optimization, Hyperparameter Tuning, and Feature Refinement #
##############################################################################

def evaluate_strategy(model, X_train, y_train, X_test, y_test, df_close):
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    prob = model.predict_proba(X_test)
    
    prob_df = pd.DataFrame(
        prob,
        columns=class_names,
        index=X_test.index)
    
    prob_df['Regime'] = label_encoder.inverse_transform(preds)
    
    prob_df['Confidence'] = prob.max(axis=1)
    
    prob_df['Close'] = df_close.loc[X_test.index]
    
    position_map = {
        'Bull': 1.0,
        'Bear': -1.0,
        'Sideways': 0.3,
        'HighVol': 0.0
        }
    
    prob_df['Position'] = (
        prob_df['Regime'].map(position_map) * prob_df['Confidence']).shift(1)
    
    prob_df['Market Return'] = prob_df['Close'].pct_change()
    
    prob_df['Strategy Return'] = (
        prob_df['Position'] * prob_df['Market Return'])
    
    prob_df.dropna(inplace=True)
    
    if prob_df.empty:
        
        return np.nan, np.nan
    
    cum = (1 + prob_df['Strategy Return']).cumprod()
    
    sharpe = (
        prob_df['Strategy Return'].mean() / prob_df['Strategy Return'].std()
        ) * np.sqrt(252)
    
    max_dd = (cum / cum.cummax() - 1).min()
    
    return sharpe, max_dd

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'n_estimators': [200, 400, 600, 800, 1000],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    }

all_params = list(itertools.product(*param_grid.values()))

sampled_params = random.sample(all_params, 10)

best_score = -np.inf

best_params = None

best_model = None

results_log = []


for params in sampled_params:
    
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        max_depth=params[0],
        learning_rate=params[1],
        n_estimators=params[2],
        subsample=params[3],
        colsample_bytree=params[4],
        random_state=42
        )
    
    sharpe, max_dd = evaluate_strategy(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        df["Close"]
        )
    
    score = sharpe + (1 + max_dd)
    
    results_log.append({
        'params': params,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'score': score
        })
    
    if np.isnan(score):
        
        continue
    
    if score > best_score:
        
        best_score = score
        
        best_params = params
        
        best_model = model


print("---------------------------------------------------")
print("Initial Training Run")
print("--------------------------------")
print(f"Parameters:", params)
print(f"Sharpe Ratio: {sharpe:.5f}")
print(f"Max Drawdown: {max_dd:.5f}")
print("Best Parameters:", best_params)
print("Best Score:", best_score)
print("---------------------------------------------------")


optimized_model = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    max_depth=best_params[0],
    learning_rate=best_params[1],
    n_estimators=best_params[2],
    subsample=best_params[3],
    colsample_bytree=best_params[4],
    random_state=42
    )

optimized_model.fit(X_train, y_train)

opt_sharpe, opt_dd = evaluate_strategy(
    optimized_model,
    X_train,
    y_train,
    X_test,
    y_test,
    df["Close"]
    )


importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': optimized_model.feature_importances_
    })

importance = importance.sort_values('Importance', ascending=False)

top_features = importance.head(10)['Feature'].tolist()

X_train_top = X_train[top_features]

X_test_top = X_test[top_features]


final_model = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    max_depth=best_params[0],
    learning_rate=best_params[1],
    n_estimators=best_params[2],
    subsample=best_params[3],
    colsample_bytree=best_params[4],
    random_state=42
    )

final_model.fit(X_train_top, y_train)

final_sharpe, final_dd = evaluate_strategy(
    final_model,
    X_train_top,
    y_train,
    X_test_top,
    y_test,
    df["Close"]
    )

print("---------------------------------------------------")
print("Parameters After First and Second Optimizations")
print("--------------------------------")
print(f"Sharpe Ratio Pre-optimization: {sharpe:.5f}")
print(f"Max Drawdown Pre-optimization: {max_dd:.5f}")
print(f"Sharpe Ratio Post-first-optimization: {opt_sharpe:.5f}")
print(f"Max Drawdown Post-first-optimization: {opt_dd:.5f}")
print(f"Sharpe Ratio Post-final-optimization: {final_sharpe:.5f}")
print(f"Max Drawdown Post-final-optimization: {final_dd:.5f}")
print("---------------------------------------------------")


##############################################################################
      # Robustness Testing, Stress Testing, and Monte Carlo Simulation #
##############################################################################

stress_periods = [
    ("2008-01-01", "2009-06-01"),
    ("2020-02-01", "2020-06-01"),
    ("2022-01-01", "2022-12-01")
    ]

stress_results = []

for start, end in stress_periods:
    
    subset = strategy_df.loc[start:end].copy()
    
    if len(subset) == 0:
        
        continue
    
    stress_return = (1 + subset['Strategy Return']).prod() - 1
    
    market_return = (1 + subset['Market Return']).prod() - 1
    
    stress_results.append({
        "Period": f"{start} to {end}",
        "Strategy Return": stress_return,
        "Market Return": market_return
        })

stress_df = pd.DataFrame(stress_results)

plt.figure(figsize=(12,6))
plt.bar(stress_df['Period'], stress_df['Strategy Return'], label='Strategy')
plt.bar(stress_df['Period'], stress_df['Market Return'], alpha=0.5, label='Market')
plt.ylabel("Relative Cumulative Returns")
plt.title(f"{ticker} Stress Period Performance")
plt.legend()
plt.show()


window = 252

returns = strategy_df['Strategy Return'].dropna()

rolling_returns = []

for i in range(0, len(returns) - window, window):
    
    segment = returns.iloc[i:i+window]
    
    annual_return = (1 + segment).prod() + 1
    
    sharpe = (segment.mean() / segment.std()) * np.sqrt(252)
    
    rolling_returns.append({
        "Start": i,
        "End": i+window,
        "Return": annual_return,
        "Sharpe": sharpe
        })

wf_df = pd.DataFrame(rolling_returns)

plt.figure(figsize=(12,6))
plt.plot(wf_df['Sharpe'])
plt.xlabel("Walk-Forward Validation Fold")
plt.ylabel("Sharpe Ratio")
plt.title(f"{ticker} Walk-Forward Sharpe Stability")
plt.show()

returns = strategy_df['Strategy Return'].dropna().values

n_simulations = 1000
n_days = len(returns)

mc_results = []

for _ in range(n_simulations):
    
    sampled = np.random.choice(
        returns,
        size=n_days,
        replace=True
        )
    
    equity = (1 + sampled).cumprod()[-1] - 1
    
    mc_results.append(equity)

mc_results = np.array(mc_results)


print("-------------------------------------------------------------")
print("Monte Carlo Distribution Analysis")
print("---------------------------------------")
print("Mean Return:", mc_results.mean())
print("5% Worst Case/95% Value-at-Risk (95% VaR):", np.percentile(mc_results, 5))
print("Best Case:", np.percentile(mc_results, 95))
print("-------------------------------------------------------------")


plt.figure(figsize=(12,6))
plt.hist(mc_results, bins=30)
plt.xlabel("Relative Strategy Returns")
plt.ylabel("Counts")
plt.title(f"{ticker} Monte Carlo Strategy Return Distribution")
plt.show()

var_95 = np.percentile(mc_results, 5)


thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]

sensitivity_results = []

for t in thresholds:
    
    temp = strategy_df.copy()
    
    temp['Position'] = np.where(
        temp['Confidence'] > t,
        temp['Position'],
        0
        )
    
    temp['Strategy Return'] = (temp['Position'] * temp['Market Return'])
    
    total_return = (1 + temp['Strategy Return']).prod() - 1
    
    sensitivity_results.append({
        "Threshold": t,
        "Return": total_return
        })

sens_df = pd.DataFrame(sensitivity_results)

plt.figure(figsize=(12,6))
plt.plot(sens_df['Threshold'], sens_df['Return'])
plt.xlabel("Threshold")
plt.ylabel("Returns")
plt.title(f"{ticker} Confidence Threshold Sensitivity")
plt.show()


stability = (strategy_df['Regime'].value_counts(normalize=True))

strategy_df['rolling_vol'] = (strategy_df['Strategy Return'].rolling(20).std())


plt.figure(figsize=(12,6))
plt.plot(strategy_df['rolling_vol'])
plt.title(f"{ticker} Strategy Volatility Stability")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.show()


equity = (1 + strategy_df['Strategy Return']).cumprod()

drawdown = equity / equity.cummax() - 1

robustness_score = (mc_results.mean() - abs(var_95) - abs(drawdown.min()))


print("-----------------------------------------------------------")
print("Regime Stability and Robustness Analysis")
print("----------------------------------------")
print("Regime Stability:", stability)
print("Maximum Drawdown:", drawdown.min())
print("Robustness Score:", robustness_score)
print("-----------------------------------------------------------")


##############################################################################
                            # Production Deployment #
##############################################################################

FEATURES = [
    'return',
    'log_return',
    'volatility_10',
    'volatility_20',
    'volatility_60',
    'mom_5',
    'mom_20',
    'mom_60',
    'dist_ma20',
    'dist_ma50',
    'dist_ma200',
    'range',
    'ATR',
    'volume_zscore',
    'efficiency'
]

X = df[FEATURES]

joblib.dump(final_model, "regime_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(top_features, "features.pkl")


model = joblib.load("regime_model.pkl")

scaler = joblib.load("scaler.pkl")

features = joblib.load("features.pkl")


def fetch_latest_data(ticker="SPY", period="12mo"):
    
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open','High','Low','Close','Volume']]
    
    df.dropna(inplace=True)
    
    return df

def create_features(df):

    df = df.copy()

    # Returns
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility
    df['volatility_10'] = df['return'].rolling(10).std()
    df['volatility_20'] = df['return'].rolling(20).std()
    df['volatility_60'] = df['return'].rolling(60).std()

    # Moving averages
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['ma200'] = df['Close'].rolling(200).mean()

    # Distance from moving averages
    df['dist_ma20'] = df['Close'] / df['ma20'] - 1
    df['dist_ma50'] = df['Close'] / df['ma50'] - 1
    df['dist_ma200'] = df['Close'] / df['ma200'] - 1

    # Momentum
    df['mom_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['mom_20'] = df['Close'] / df['Close'].shift(20) - 1
    df['mom_60'] = df['Close'] / df['Close'].shift(60) - 1

    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift()).abs()
    tr3 = (df['Low'] - df['Close'].shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(14).mean()

    # Daily range
    df['range'] = (df['High'] - df['Low']) / df['Close']

    # Volume z-score
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()

    df['volume_zscore'] = (
        df['Volume'] - vol_mean
    ) / vol_std

    # Efficiency
    df['efficiency'] = (
        (df['Close'] - df['Close'].shift(20)).abs()
        /
        df['Close'].diff().abs().rolling(20).sum()
    )

    return df

def prepare_X(df):
    
    df = df.dropna()
    
    X = df[features]
    
    return X, df

def predict_regime(X):
    
    preds = model.predict(X)
    
    probs = model.predict_proba(X)
    
    return preds, probs

def build_output(df, preds, probs):
    
    out = df.copy()
    
    out['Regime'] = preds
    
    out['Confidence'] = probs.max(axis=1)
    
    return out


regime_map = {
    0: "Bull",
    1: "Bear",
    2: "Sideways",
    3: "HighVol"
    }

position_map = {
    "Bull": 1.0,
    "Bear": -1.0,
    "Sideways": 0.3,
    "HighVol": 0.0
    }


def run_pipeline():
    
    df = fetch_latest_data()
    
    df = create_features(df)
    
    X, df = prepare_X(df)
    
    preds, probs = predict_regime(X)
    
    out = build_output(df, preds, probs)
    
    out['Regime_label'] = out['Regime'].map(regime_map)
    
    out['Position'] = out['Regime_label'].map(position_map)
    
    out['Position'] = out['Position'] * out['Confidence']
    
    out['Position'] = out['Position'].shift(1)
    
    out['Market_Return'] = out['Close'].pct_change()
    
    out['Strategy_Return'] = (out['Position'] * out['Market_Return'])
    
    return out


results = run_pipeline()

latest = results.iloc[-1]


print("-------------------------------------")
print("Real-Time Signal Output")
print("-------------------------------------")
print("Regime:", latest['Regime_label'])
print("Confidence:", latest['Confidence'])
print("Position:", latest['Position'])
print("-------------------------------------")

print("-------------------------------------")
print("Monitoring Metrics")
print("-------------------------------------")
print("Mean Strategy Return:", results['Strategy_Return'].mean())
print("Cumulative Return:", (1 + results['Strategy_Return']).prod() - 1)
print("-------------------------------------")




logging.info(
    f"Regime={latest['Regime_label']}",
    f"Confidence={latest['Confidence']:.3f}",
    f"Position={latest['Position']:.3f}"
    )

logging.shutdown()