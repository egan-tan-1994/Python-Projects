import streamlit as st
import pandas as pd
from streamlit_financial_engine import (
    FinancialDrivers,
    BalanceSheetPeriod,
    IntegratedFinancialEngine,
    CorporateValuationEngine,
    LBOEngine,
    DebtTranche,
    FinancialInputs,
    CashFlowDebtEngine,
    generate_financial_models_data
)

# Page configuration
st.set_page_config(page_title="Financial Model Dashboard", layout="wide")
st.title("📊 Integrated Financial Modeling & Valuation Dashboard")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "3-Statement Forecast", 
    "DCF Valuation", 
    "LBO Returns", 
    "Debt & Cash Sweep"
])

# ----------------------------------------------------------------------------
# Tab 1: Interactive 3-Statement Model
# ----------------------------------------------------------------------------
with tab1:
    st.header("3-Statement Financial Model")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Drivers & Assumptions")
        growth = st.slider("Revenue Growth Rate", 0.0, 0.30, 0.10, 0.01)
        cogs_pct = st.slider("COGS % Revenue", 0.30, 0.80, 0.60, 0.01)
        opex_pct = st.slider("OPEX % Revenue", 0.10, 0.40, 0.20, 0.01)
        tax_rate = st.slider("Tax Rate", 0.10, 0.40, 0.25, 0.01)
        
    with col2:
        # Re-run engine with dynamic sidebar parameters
        drivers = FinancialDrivers(
            revenue_growth=growth,
            cogs_pct_rev=cogs_pct,
            opex_pct_rev=opex_pct,
            tax_rate=tax_rate
        )
        init_bs = BalanceSheetPeriod(
            cash=50.0, ar=20.0, inventory=15.0, ppe_net=100.0,
            ap=15.0, debt=40.0, common_stock=30.0, retained_earnings=100.0
        )
        engine = IntegratedFinancialEngine(initial_bs=init_bs, base_revenue=200.0, drivers=drivers)
        forecast_df = engine.run_forecast(periods=5)
        
        st.subheader("Financial Projections ($M)")
        st.dataframe(forecast_df.round(2), use_container_width=True)
        st.line_chart(forecast_df.T[["Revenue", "EBITDA", "Net Income"]])

# ----------------------------------------------------------------------------
# Tab 2: DCF Corporate Valuation
# ----------------------------------------------------------------------------
with tab2:
    st.header("Corporate Valuation (DCF)")
    dcf_inputs = {
        'base_revenue': 1000.0, 'revenue_growth': [0.12, 0.10, 0.08, 0.06, 0.05],
        'ebit_margin': 0.22, 'tax_rate': 0.21, 'capex_pct': 0.045, 'dna_pct': 0.035,
        'nwc_pct': 0.08, 'risk_free_rate': 0.042, 'beta': 1.10, 'equity_risk_premium': 0.055,
        'cost_of_debt': 0.055, 'market_cap': 5000.0, 'total_debt': 1500.0, 'cash': 500.0,
        'shares_outstanding': 100.0, 'terminal_growth_rate': 0.025, 'exit_multiple': 12.0,
    }
    
    val_engine = CorporateValuationEngine(dcf_inputs)
    val_results = val_engine.calculate_valuation()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Calculated WACC", f"{val_results['wacc']:.2%}")
    m2.metric("Enterprise Value (Gordon Growth)", f"${val_results['ev_gg']:,.2f}M")
    m3.metric("Implied Share Price", f"${val_results['share_price_gg']:,.2f}")
    
    st.subheader("Explicit Cash Flow Forecast")
    st.dataframe(val_results['projection_df'].round(2), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: LBO Returns Analysis
# ----------------------------------------------------------------------------
with tab3:
    st.header("LBO Returns & Waterfall")
    lbo_engine = LBOEngine(entry_ebitda=50.0, entry_multiple=10.0, holding_period=5)
    sources_uses, total_debt, sponsor_eq = lbo_engine.build_sources_and_uses()
    op_model = lbo_engine.forecast_operations()
    debt_sched = lbo_engine.run_debt_waterfall(op_model)
    returns = lbo_engine.calculate_returns(10.0, sponsor_eq, debt_sched, op_model)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sources & Uses")
        st.dataframe(sources_uses, use_container_width=True)
    with col2:
        st.subheader("Sponsor Returns Summary")
        st.metric("Sponsor MOIC", f"{returns['MOIC']:.2f}x")
        st.metric("Sponsor IRR", f"{returns['IRR']:.2%}")
        st.metric("Exit Equity Value", f"${returns['Exit Equity Value']:.2f}M")

# ----------------------------------------------------------------------------
# Tab 4: Cash Flow & Debt Paydown
# ----------------------------------------------------------------------------
with tab4:
    st.header("Cash Flow & Liquidity Analysis")
    # Quick execution using the helper function
    dash_data = generate_financial_models_data()
    sweep_df = dash_data["cash_flow_debt_sweep"]
    
    st.dataframe(sweep_df, use_container_width=True)
    st.bar_chart(sweep_df.set_index("Year")[["Total_Ending_Debt", "Ending_Cash"]])