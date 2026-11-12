import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots

from finquant_engine import (
    estimate_return_distribution, calculate_risk_metrics, calculate_maximum_drawdown,
    bootstrap_sharpe_ratio_bca, permutation_test_difference, estimate_garch_volatility,
    monte_carlo_simulation, robust_linear_regression, bayesian_linear_regression,
    hierarchical_bayesian_model, estimate_evt_tail_risk, estimate_copulas,
    detect_market_regimes_hmm, test_pairs_cointegration, kalman_filter_pairs_trading,
    calculate_black_scholes, simulate_ou_process, calculate_kelly_criterion,
    heston_pricing_mc, simulate_fbm, optimize_markowitz, optimize_risk_parity, 
    black_litterman_posterior, hierarchical_risk_parity, simulate_jump_diffusion, 
    build_volatility_surface, local_volatility_dupire, delta_hedged_backtester, generate_ml_alpha
)

# Helper function to safely extract a scalar float from any NumPy array format
def to_scalar(val):
    return float(np.asarray(val).ravel()[0])

st.set_page_config(page_title="FinQuant Inference Engine", layout="wide")
st.title("📈 Statistical Inference & Financial Risk Framework")

# ----------------------------------------------------
# 0. User Input & Real Data Fetching (Yahoo Finance)
# ----------------------------------------------------
st.sidebar.header("Asset Selection")
st.sidebar.markdown("Enter Yahoo Finance tickers (e.g., AAPL, MSFT, SPY, ^GSPC, BTC-USD)")

ticker_a = st.sidebar.text_input("Asset A Ticker", "AAPL").upper()
ticker_b = st.sidebar.text_input("Asset B Ticker", "MSFT").upper()
ticker_mkt = st.sidebar.text_input("Market Benchmark", "SPY").upper()

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

@st.cache_data(show_spinner=False)
def fetch_market_data(t_a, t_b, t_m, start, end):
    df_a = yf.Ticker(t_a).history(start=start, end=end)[['Close']].rename(columns={'Close': t_a})
    df_b = yf.Ticker(t_b).history(start=start, end=end)[['Close']].rename(columns={'Close': t_b})
    df_m = yf.Ticker(t_m).history(start=start, end=end)[['Close']].rename(columns={'Close': t_m})
    
    data = df_a.join(df_b, how='inner').join(df_m, how='inner').dropna()
    data.index = data.index.tz_localize(None) 
    
    returns = data.pct_change().dropna()
    return data, returns

with st.spinner("Fetching market data..."):
    try:
        data, returns = fetch_market_data(ticker_a, ticker_b, ticker_mkt, start_date, end_date)
        
        prices_a = data[ticker_a].values
        prices_b = data[ticker_b].values
        returns_a = returns[ticker_a].values
        returns_b = returns[ticker_b].values
        returns_market = returns[ticker_mkt].values
        
        st.success(f"Data successfully loaded for {ticker_a}, {ticker_b}, and {ticker_mkt}. Running quantitative modules...")
    except Exception as e:
        st.error(f"Error fetching data: Ensure the tickers are valid on Yahoo Finance. Details: {e}")
        st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk & Stats", "Time Series & Regimes", "Pricing & Simulation", 
    "Portfolio Optimization", "Advanced Alpha & Options"
])

# ==========================================
# TAB 1: Risk, Distribution & Statistical Tests
# ==========================================
with tab1:
    st.header(f"1. Risk Metrics & Distributions ({ticker_a})")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Distribution & EVT Tail Risk")
        mu_t, std_t, df_t = estimate_return_distribution(returns_a)
        shape, loc, scale, var_99 = estimate_evt_tail_risk(returns_a)
        st.write(f"**Student-t Fit:** Mean {to_scalar(mu_t):.4f}, Vol {to_scalar(std_t):.4f}, DoF {to_scalar(df_t):.2f}")
        st.write(f"**EVT 99% VaR:** {to_scalar(var_99):.4%}")

    with col2:
        st.subheader("Standard Risk Metrics")
        ann_returns, ann_vol, ann_sharpe, var_95, cvar_95 = calculate_risk_metrics(returns_a)
        sortino, calmar, max_dd, drawdowns = calculate_maximum_drawdown(returns_a)
        st.write(f"**Annualized Returns:** {to_scalar(ann_returns):.4%}")
        st.write(f"**Annualized Volatility:** {to_scalar(ann_vol):.4%}")
        st.write(f"**Annualized Sharpe Ratio:** {to_scalar(ann_sharpe):.4f}")
        st.write(f"**Max Drawdown:** {to_scalar(max_dd):.4%}")
        st.write(f"**Annualized Sortino Ratio:** {to_scalar(sortino):.4f}")
        st.write(f"**Annualized Calmar Ratio:** {to_scalar(calmar):.4f}")
        st.write(f"**95% VaR:** {to_scalar(var_95):.4%}")
        st.write(f"**95% CVaR:** {to_scalar(cvar_95):.4%}")

    with col3:
        st.subheader("Statistical Tests")
        sr_point, sr_low, sr_high = bootstrap_sharpe_ratio_bca(returns_a)
        p_val_diff = permutation_test_difference(returns_a, returns_b)
        st.write(f"**Sharpe Ratio (BCa 95% CI):** {to_scalar(sr_point):.2f} ({to_scalar(sr_low):.2f} to {to_scalar(sr_high):.2f})")
        st.write(f"**Permutation P-Value ({ticker_a} vs {ticker_b}):** {to_scalar(p_val_diff):.4f}")

    st.divider()
    st.subheader("Copulas & Regression")
    col4, col5 = st.columns(2)
    with col4:
        st.write(f"**Dependence Structure ({ticker_a} & {ticker_b})**")
        rho, clayton, gumbel, u, v = estimate_copulas(returns_a, returns_b)
        st.write(f"**Pearson Rho:** {to_scalar(rho):.4f}, **Clayton Theta:** {to_scalar(clayton):.4f}, **Gumbel Theta:** {to_scalar(gumbel):.4f}")
        
        # Copula Scatter Plots
        fig_copula = make_subplots(rows=1, cols=2, subplot_titles=("Original Returns Scatter", "Copula Uniform Marginals (Dependence Structure)"))
        fig_copula.add_trace(go.Scatter(x=returns_a, y=returns_b, mode='markers', marker=dict(size=4, opacity=0.5)), row=1, col=1)
        fig_copula.add_trace(go.Scatter(x=u, y=v, mode='markers', marker=dict(size=4, color='purple', opacity=0.5)), row=1, col=2)
        fig_copula.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=30, b=0))
        fig_copula.update_xaxes(title_text=f"Asset A ({ticker_a})", row=1, col=1)
        fig_copula.update_yaxes(title_text=f"Asset B ({ticker_b})", row=1, col=1)
        fig_copula.update_xaxes(title_text=f"U (Asset A Rank)", row=1, col=2)
        fig_copula.update_yaxes(title_text=f"V (Asset B Rank)", row=1, col=2)
        st.plotly_chart(fig_copula, use_container_width=True)

    with col5:
        st.write(f"**Robust Regression ({ticker_a} vs {ticker_mkt})**")
        robust_res = robust_linear_regression(returns_a, returns_market)
        st.write("**Robust OLS Beta:**", to_scalar(robust_res.params[1]))

# ==========================================
# TAB 2: Time Series, Regimes & Pairs Trading
# ==========================================
with tab2:
    st.header("2. Time Series & State Space Models")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Hidden Markov Regimes ({ticker_a})")
        hmm_model, hidden_states = detect_market_regimes_hmm(returns_a)
        fig_hmm = go.Figure()
        fig_hmm.add_trace(go.Scatter(y=prices_a, mode='lines', marker=dict(color=hidden_states, colorscale='Viridis')))
        st.plotly_chart(fig_hmm, use_container_width=True)

    with col2:
        st.subheader(f"Kalman Filter Pairs Trading ({ticker_a} / {ticker_b})")
        spread, hedge_ratio = kalman_filter_pairs_trading(prices_a, prices_b)
        fig_kf = px.line(y=hedge_ratio, title="Dynamic Hedge Ratio (Kalman)")
        st.plotly_chart(fig_kf, use_container_width=True)
        
    st.divider()
    
    st.subheader(f"GARCH(1,1) Conditional Volatility ({ticker_a})")
    garch_vol = estimate_garch_volatility(returns_a)
    
    fig_garch = go.Figure()
    fig_garch.add_trace(go.Scatter(y=np.abs(returns_a), mode='lines', name='Absolute Returns', line=dict(color='lightgray', width=1)))
    fig_garch.add_trace(go.Scatter(y=garch_vol, mode='lines', name='Conditional Volatility', line=dict(color='orange', width=2)))
    fig_garch.update_layout(height=400, title="GARCH(1,1) Conditional Volatility", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_garch, use_container_width=True)

# ==========================================
# TAB 3: Pricing & Simulation
# ==========================================
with tab3:
    st.header(f"3. Stochastic Processes & Monte Carlo ({ticker_a})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Standard Monte Carlo")
        mc_paths, med_val, low_5, high_95 = monte_carlo_simulation(returns_a, initial_portfolio=prices_a[-1])
        fig_mc = px.line(mc_paths[:, :50])
        fig_mc.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_mc, use_container_width=True)
        
    with col2:
        st.subheader("Jump Diffusion Process")
        jd_paths = simulate_jump_diffusion(prices_a[-1], np.mean(returns_a)*252, np.std(returns_a)*np.sqrt(252), 5, -0.05, 0.1)
        fig_jd = px.line(jd_paths[:, :50])
        fig_jd.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_jd, use_container_width=True)

    with col3:
        st.subheader("Ornstein-Uhlenbeck (OU)")
        ou_paths = simulate_ou_process(prices_a[-1], 0.5, prices_a[-1], np.std(returns_a)*np.sqrt(252))
        fig_ou = px.line(ou_paths[:, :50])
        fig_ou.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_ou, use_container_width=True)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Fractional Brownian Motion (fBm)")
        t_fbm, fbm_path = simulate_fbm(252, hurst=0.7)
        st.plotly_chart(px.line(x=t_fbm, y=fbm_path, height=300), use_container_width=True)
        
    with col5:
        st.subheader("Options Pricing (Hypothetical ATM Call)")
        spot = prices_a[-1]
        vol = np.std(returns_a)*np.sqrt(252)
        bs_call = calculate_black_scholes(spot, spot, 1.0, 0.05, vol)
        heston_call = heston_pricing_mc(spot, spot, 1.0, 0.05, vol**2, 2.0, vol**2, 0.1, -0.7)
        st.metric("Black-Scholes Call", f"${to_scalar(bs_call):.2f}")
        st.metric("Heston MC Call", f"${to_scalar(heston_call):.2f}")

# ==========================================
# TAB 4: Portfolio Optimization
# ==========================================
with tab4:
    st.header("4. Modern Portfolio Allocation")
    
    returns_matrix = np.column_stack((returns_a, returns_b, returns_market))
    expected_rets = np.mean(returns_matrix, axis=0) * 252
    cov_mat = np.cov(returns_matrix, rowvar=False) * 252
    assets = [ticker_a, ticker_b, ticker_mkt]

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Mean-Variance (Markowitz)")
        w_mv = optimize_markowitz(expected_rets, cov_mat)
        st.plotly_chart(px.pie(values=w_mv, names=assets, height=300), use_container_width=True)

    with col2:
        st.subheader("Risk Parity")
        w_rp = optimize_risk_parity(cov_mat)
        st.plotly_chart(px.pie(values=w_rp, names=assets, height=300), use_container_width=True)

    with col3:
        st.subheader("Hierarchical Risk Parity (HRP)")
        w_hrp = hierarchical_risk_parity(cov_mat)
        st.plotly_chart(px.pie(values=w_hrp, names=assets, height=300), use_container_width=True)

# ==========================================
# TAB 5: Advanced Alpha & Options
# ==========================================
with tab5:
    st.header("5. Volatility Surfaces & ML Alpha")
    
    st.subheader("Machine Learning Alpha (Random Forest) -> Black-Litterman")
    X_train = np.random.randn(len(returns_a)-1, 3) 
    y_train = returns_matrix[1:] 
    X_latest = np.random.randn(1, 3)
    
    predicted_returns = generate_ml_alpha(X_train, y_train, X_latest)[0] * 252 
    P = np.eye(3)
    omega = np.diag([0.01, 0.01, 0.01])
    post_ret, post_cov = black_litterman_posterior(expected_rets, cov_mat, P, predicted_returns, omega)
    w_bl = optimize_markowitz(post_ret, post_cov)
    
    st.write(f"**ML Predicted Returns (Annualized):** {ticker_a}: {to_scalar(predicted_returns[0]):.2%}, {ticker_b}: {to_scalar(predicted_returns[1]):.2%}, {ticker_mkt}: {to_scalar(predicted_returns[2]):.2%}")
    st.write(f"**BL Optimized Weights:** {ticker_a}: {to_scalar(w_bl[0]):.2%}, {ticker_b}: {to_scalar(w_bl[1]):.2%}, {ticker_mkt}: {to_scalar(w_bl[2]):.2%}")

    st.divider()

    st.subheader("Volatility Surfaces: Implied vs Local (Dupire)")
    col_iv, col_lv = st.columns(2)

    spot = prices_a[-1]
    strikes = np.random.uniform(spot*0.8, spot*1.2, 150)
    maturities = np.random.uniform(0.1, 2.0, 150)
    base_vol = np.std(returns_a)*np.sqrt(252)
    implied_vols = base_vol + 0.05 * ((strikes - spot) / (spot*0.2))**2 + 0.1 * np.exp(-maturities)
    grid_strike, grid_mat, grid_vol = build_volatility_surface(strikes, maturities, implied_vols)

    with col_iv:
        fig_iv = go.Figure(data=[go.Surface(z=grid_vol.T, x=grid_strike[:,0], y=grid_mat[0,:], colorscale='Viridis')])
        fig_iv.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='IV'), height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_iv, use_container_width=True)

    with col_lv:
        local_vol_grid = local_volatility_dupire(grid_strike, grid_mat, grid_vol, spot, 0.05)
        fig_lv = go.Figure(data=[go.Surface(z=local_vol_grid.T, x=grid_strike[:,0], y=grid_mat[0,:], colorscale='Plasma')])
        fig_lv.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='Local Vol'), height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_lv, use_container_width=True)

    st.divider()

    st.subheader(f"Option Delta-Hedging Backtester ({ticker_a})")
    vol = np.std(returns_a)*np.sqrt(252)
    hedged_pnl = delta_hedged_backtester(prices_a[-252:], K=prices_a[-252], r=0.05, vol=vol, T=1.0)
    fig_hedge = px.line(y=hedged_pnl, title=f"Cumulative PnL of Delta-Hedged Portfolio (Last 252 Days of {ticker_a})")
    st.plotly_chart(fig_hedge, use_container_width=True)