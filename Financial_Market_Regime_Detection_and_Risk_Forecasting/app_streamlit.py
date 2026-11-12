import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.linalg as la
import pandas as pd

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

st.set_page_config(page_title="FinQuant Inference Engine", layout="wide")
st.title("📈 Statistical Inference & Financial Risk Framework")

# ----------------------------------------------------
# Base Dummy Data & Previous Modules Assumed Active
# ----------------------------------------------------
np.random.seed(42)
st.success("Prior models (MLE, VaR, Kalman, Heston, OU, HRP) remain active in background compute.")

# ----------------------------------------------------
# 12. Dupire Local Volatility & Implied Vol Surface
# ----------------------------------------------------
st.divider()
st.header("Volatility Surfaces: Implied vs Local (Dupire)")
col_iv, col_lv = st.columns(2)

# Dummy Volatility Surface Data (Simulating a volatility smile/skew)
S0_val, r_val = 100.0, 0.05
strikes = np.random.uniform(80, 120, 150)
maturities = np.random.uniform(0.1, 2.0, 150)
implied_vols = 0.2 + 0.05 * ((strikes - 100) / 20)**2 + 0.1 * np.exp(-maturities)
grid_strike, grid_mat, grid_vol = build_volatility_surface(strikes, maturities, implied_vols)

with col_iv:
    st.subheader("Implied Volatility Surface")
    fig_iv = go.Figure(data=[go.Surface(z=grid_vol.T, x=grid_strike[:,0], y=grid_mat[0,:], colorscale='Viridis')])
    fig_iv.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='IV'), height=350, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_iv, use_container_width=True)

with col_lv:
    st.subheader("Dupire Local Volatility Surface")
    local_vol_grid = local_volatility_dupire(grid_strike, grid_mat, grid_vol, S0_val, r_val)
    fig_lv = go.Figure(data=[go.Surface(z=local_vol_grid.T, x=grid_strike[:,0], y=grid_mat[0,:], colorscale='Plasma')])
    fig_lv.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='Local Vol'), height=350, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_lv, use_container_width=True)

# ----------------------------------------------------
# 13. Delta-Hedging Option Backtester
# ----------------------------------------------------
st.divider()
st.header("Option Delta-Hedging Backtester")
st.markdown("Tracks the cumulative PnL of a continuous delta-hedged long call position.")

# Generate synthetic price path
price_path = np.zeros(252)
price_path[0] = 100.0
for i in range(1, 252):
    price_path[i] = price_path[i-1] * np.exp((0.05 - 0.5*0.2**2)* (1/252) + 0.2*np.sqrt(1/252)*np.random.randn())

hedged_pnl = delta_hedged_backtester(price_path, K=100.0, r=0.05, vol=0.2, T=1.0)

fig_hedge = go.Figure()
fig_hedge.add_trace(go.Scatter(y=hedged_pnl, mode='lines', name='Delta-Hedged Portfolio PnL', line=dict(color='orange', width=2)))
fig_hedge.update_layout(title="Hedged Portfolio Value Over Time", height=300, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig_hedge, use_container_width=True)

# ----------------------------------------------------
# 14. ML Alpha & Black-Litterman Integration
# ----------------------------------------------------
st.divider()
st.header("Machine Learning Alpha (Random Forest) -> Black Litterman")
st.markdown("Replacing subjective human views with non-linear ML predictions.")

# Dummy ML Training Data (Features: Lagged Returns, Moving Averages)
X_train = np.random.randn(1000, 5) 
y_train = np.random.randn(1000, 3) * 0.05 + 0.02 # 3 Assets
X_latest = np.random.randn(1, 5)

predicted_returns = generate_ml_alpha(X_train, y_train, X_latest)[0]

col_ml, col_bl_ml = st.columns(2)
with col_ml:
    st.subheader("Random Forest Predicted Alpha")
    assets = ["Equities", "Bonds", "Commodities"]
    for a, r_pred in zip(assets, predicted_returns):
        st.write(f"**{a}**: {r_pred:.2%}")

with col_bl_ml:
    st.subheader("ML-Adjusted BL Allocation")
    cov_matrix = np.array([[0.04, 0.005, 0.01], [0.005, 0.01, -0.002], [0.01, -0.002, 0.06]])
    # Use ML predictions directly as the 'View' Q vector, with identity Pick matrix P
    P = np.eye(3)
    Q = predicted_returns
    omega = np.diag([0.001, 0.001, 0.001])
    
    post_ret, post_cov = black_litterman_posterior(np.array([0.05, 0.03, 0.04]), cov_matrix, P, Q, omega)
    w_bl_ml = optimize_markowitz(post_ret, post_cov)
    for a, w in zip(assets, w_bl_ml):
        st.write(f"**{a}**: {w:.2%}")