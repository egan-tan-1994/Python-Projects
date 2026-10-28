import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict

# ---------------------------------------------------------
# Logging & Directory Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("xva_saccr_dashboard_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------
@dataclass
class MarketParams:
    r_ois0: float = 0.030
    r_fwd0: float = 0.035
    a_ois: float = 0.05
    sigma_ois: float = 0.010
    a_fwd: float = 0.04
    sigma_fwd: float = 0.012
    correlation: float = 0.85

@dataclass
class TradeParams:
    trade_id: str
    notional: float
    fixed_rate: float
    T: float
    freq: float = 0.5
    is_payer: bool = False

@dataclass
class CSAParams:
    has_csa: bool = True
    cpty_threshold: float = 150_000.0
    own_threshold: float = 150_000.0

@dataclass
class CreditParams:
    cpty_hazard_rate: float = 0.025
    cpty_recovery: float = 0.40
    own_hazard_rate: float = 0.010
    own_recovery: float = 0.40

@dataclass
class FundingParams:
    spread_borrow: float = 0.012
    spread_lend: float = 0.004

@dataclass
class CapitalParams:
    cost_of_capital: float = 0.10   # Cost of capital rate (10%)
    saccr_alpha: float = 1.4        # Regulatory SA-CCR Alpha multiplier
    saccr_sf: float = 0.005         # Supervisory Factor for Interest Rates (0.5%)

@dataclass
class SimParams:
    n_paths: int = 2000
    n_steps: int = 120

# ---------------------------------------------------------
# Core Multi-Curve XVA & SA-CCR Engine
# ---------------------------------------------------------
class ProductionXVASACCREngine:
    def __init__(self, mkt: MarketParams, portfolio: List[TradeParams], csa: CSAParams,
                 credit: CreditParams, fnd: FundingParams, cap: CapitalParams, sim: SimParams):
        self.mkt = mkt
        self.portfolio = portfolio
        self.csa = csa
        self.credit = credit
        self.fnd = fnd
        self.cap = cap
        self.sim = sim
        
        self.max_T = max(t.T for t in self.portfolio)
        self.dt = self.max_T / self.sim.n_steps
        self.t_grid = np.linspace(0, self.max_T, self.sim.n_steps + 1)
        
        self.ois_paths = None
        self.fwd_paths = None
        self.port_mtm_paths = None
        self.uncol_epe, self.epe, self.ene = None, None, None
        self.saccr_capital_profile = None

    def _hw_A_B(self, t: float, T: float, r0: float, a: float, sigma: float) -> Tuple[float, float]:
        tau = T - t
        if tau <= 0: return 1.0, 0.0
        B = (1.0 - np.exp(-a * tau)) / a
        A = np.exp((B - tau) * (r0 - (sigma**2) / (2 * a**2)) - (sigma**2 * B**2) / (4 * a))
        return A, B

    def calculate_saccr_addon(self, active_trades: List[TradeParams], t: float) -> float:
        """Calculates SA-CCR AddOn for active trades in the interest rate asset class."""
        if not active_trades:
            return 0.0
            
        effective_notional = 0.0
        for trade in active_trades:
            M = max(trade.T - t, 0.05)
            duration_adj = (np.exp(-0.05 * 0.0) - np.exp(-0.05 * M)) / 0.05
            mf = np.sqrt(min(M, 1.0) / 1.0)
            
            direction = 1.0 if trade.is_payer else -1.0
            effective_notional += direction * trade.notional * duration_adj * mf
            
        return self.cap.saccr_sf * abs(effective_notional)

    def run_simulation(self):
        Z_ois = np.random.standard_normal((self.sim.n_paths, self.sim.n_steps))
        Z_raw = np.random.standard_normal((self.sim.n_paths, self.sim.n_steps))
        rho = self.mkt.correlation
        Z_fwd = rho * Z_ois + np.sqrt(1.0 - rho**2) * Z_raw
        
        self.ois_paths = np.zeros((self.sim.n_paths, self.sim.n_steps + 1))
        self.fwd_paths = np.zeros((self.sim.n_paths, self.sim.n_steps + 1))
        self.ois_paths[:, 0] = self.mkt.r_ois0
        self.fwd_paths[:, 0] = self.mkt.r_fwd0
        
        for i in range(self.sim.n_steps):
            d_ois = self.mkt.a_ois * (self.mkt.r_ois0 - self.ois_paths[:, i]) * self.dt + self.mkt.sigma_ois * np.sqrt(self.dt) * Z_ois[:, i]
            d_fwd = self.mkt.a_fwd * (self.mkt.r_fwd0 - self.fwd_paths[:, i]) * self.dt + self.mkt.sigma_fwd * np.sqrt(self.dt) * Z_fwd[:, i]
            self.ois_paths[:, i+1] = self.ois_paths[:, i] + d_ois
            self.fwd_paths[:, i+1] = self.fwd_paths[:, i] + d_fwd

        self.port_mtm_paths = np.zeros_like(self.ois_paths)
        for trade in self.portfolio:
            trade_mtm = np.zeros_like(self.ois_paths)
            p_dates = np.arange(trade.freq, trade.T + trade.freq, trade.freq)
            
            for i, t in enumerate(self.t_grid):
                if t >= trade.T: continue
                rem_dates = p_dates[p_dates > t]
                if len(rem_dates) == 0: continue
                
                fixed_npv, float_npv = np.zeros(self.sim.n_paths), np.zeros(self.sim.n_paths)
                for k in range(len(rem_dates)):
                    T_pay = rem_dates[k]
                    T_start = t if k == 0 else rem_dates[k-1]
                    tau = T_pay - T_start
                    
                    A_ois, B_ois = self._hw_A_B(t, T_pay, self.mkt.r_ois0, self.mkt.a_ois, self.mkt.sigma_ois)
                    P_ois = A_ois * np.exp(-B_ois * self.ois_paths[:, i])
                    fixed_npv += trade.fixed_rate * trade.freq * P_ois * trade.notional
                    
                    A_fs, B_fs = self._hw_A_B(t, T_start, self.mkt.r_fwd0, self.mkt.a_fwd, self.mkt.sigma_fwd)
                    A_fp, B_fp = self._hw_A_B(t, T_pay, self.mkt.r_fwd0, self.mkt.a_fwd, self.mkt.sigma_fwd)
                    P_fs = A_fs * np.exp(-B_fs * self.fwd_paths[:, i])
                    P_fp = A_fp * np.exp(-B_fp * self.fwd_paths[:, i])
                    
                    fwd_rate = (P_fs / P_fp - 1.0) / tau
                    float_npv += fwd_rate * tau * P_ois * trade.notional
                    
                trade_mtm[:, i] = (float_npv - fixed_npv) if trade.is_payer else (fixed_npv - float_npv)
            self.port_mtm_paths += trade_mtm

        self.uncol_epe = np.maximum(self.port_mtm_paths, 0.0).mean(axis=0)
        if self.csa.has_csa:
            col_exp = np.minimum(np.maximum(self.port_mtm_paths, 0.0), self.csa.cpty_threshold)
            col_neg_exp = np.minimum(np.maximum(-self.port_mtm_paths, 0.0), self.csa.own_threshold)
        else:
            col_exp = np.maximum(self.port_mtm_paths, 0.0)
            col_neg_exp = np.maximum(-self.port_mtm_paths, 0.0)
            
        self.epe = col_exp.mean(axis=0)
        self.ene = col_neg_exp.mean(axis=0)

        # Compute Dynamic SA-CCR Exposure at Default (EAD) & Capital Requirement
        self.saccr_capital_profile = np.zeros(self.sim.n_steps + 1)
        for i, t in enumerate(self.t_grid):
            active_trades = [tr for tr in self.portfolio if tr.T > t]
            addon = self.calculate_saccr_addon(active_trades, t)
            rc = max(self.epe[i] - self.csa.cpty_threshold if self.csa.has_csa else self.epe[i], 0.0)
            ead = self.cap.saccr_alpha * (rc + addon)
            self.saccr_capital_profile[i] = ead * 0.08

    def compute_xva_metrics(self) -> Dict[str, float]:
        cva, dva, fca, fba, kva = 0.0, 0.0, 0.0, 0.0, 0.0
        cpty_surv = np.insert(np.exp(-np.cumsum(np.full(self.sim.n_steps, self.credit.cpty_hazard_rate) * self.dt)), 0, 1.0)
        own_surv = np.insert(np.exp(-np.cumsum(np.full(self.sim.n_steps, self.credit.own_hazard_rate) * self.dt)), 0, 1.0)
        
        lgd_cpty = 1.0 - self.credit.cpty_recovery
        lgd_own = 1.0 - self.credit.own_recovery
        
        for i in range(1, self.sim.n_steps + 1):
            df = np.exp(-self.mkt.r_ois0 * self.t_grid[i])
            epe_m = (self.epe[i] + self.epe[i-1]) / 2.0
            ene_m = (self.ene[i] + self.ene[i-1]) / 2.0
            cap_m = (self.saccr_capital_profile[i] + self.saccr_capital_profile[i-1]) / 2.0
            
            cva += lgd_cpty * df * epe_m * (cpty_surv[i-1] - cpty_surv[i])
            dva += lgd_own * df * ene_m * (own_surv[i-1] - own_surv[i])
            fca += self.fnd.spread_borrow * epe_m * df * self.dt
            fba += self.fnd.spread_lend * ene_m * df * self.dt
            kva += self.cap.cost_of_capital * cap_m * df * self.dt
            
        fva = fca - fba
        rf_mtm = self.port_mtm_paths[:, 0].mean()
        net_xva = -cva + dva - fva - kva
        return {
            'RF_MtM': rf_mtm, 'CVA': cva, 'DVA': dva, 
            'FCA': fca, 'FBA': fba, 'FVA': fva, 'KVA': kva,
            'Net_XVA': net_xva, 'Adjusted_Value': rf_mtm + net_xva
        }

# ---------------------------------------------------------
# Streamlit Dashboard UI
# ---------------------------------------------------------
st.set_page_config(page_title="XVA & SA-CCR Engine", layout="wide")
st.title("Multi-Curve XVA & SA-CCR Regulatory Capital Dashboard")

if 'sim_run' not in st.session_state:
    st.session_state.sim_run = False

st.sidebar.header("Model Configuration")
with st.sidebar.expander("Market Parameters", expanded=True):
    r_ois0 = st.slider("Initial OIS Rate", 0.0, 0.10, 0.030, 0.001)
    r_fwd0 = st.slider("Initial FWD Rate", 0.0, 0.10, 0.035, 0.001)
    correlation = st.slider("Curve Correlation (OIS/FWD)", -1.0, 1.0, 0.85, 0.05)

with st.sidebar.expander("SA-CCR Capital Parameters", expanded=False):
    saccr_alpha = st.number_input("SA-CCR Alpha", value=1.4, step=0.1)
    saccr_sf = st.number_input("Supervisory Factor (IR)", value=0.005, format="%.3f")
    coc_rate = st.slider("Cost of Capital Rate (%)", 0.0, 0.20, 0.10, 0.01)

with st.sidebar.expander("Credit & Funding", expanded=False):
    cpty_hr = st.number_input("Cpty Hazard Rate (bps)", value=250) / 10000
    own_hr = st.number_input("Own Hazard Rate (bps)", value=100) / 10000
    spread_borrow = st.number_input("Funding Borrow Spread (bps)", value=120) / 10000

with st.sidebar.expander("CSA (Collateral)", expanded=False):
    has_csa = st.checkbox("Active CSA", value=True)
    cpty_thr = st.number_input("Cpty Threshold ($)", value=200000)
    own_thr = st.number_input("Own Threshold ($)", value=200000)

sim_paths = st.sidebar.number_input("Monte Carlo Paths", 500, 10000, 2000, 500)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Simulating Multi-Curve Paths & Computing SA-CCR KVA..."):
        mkt = MarketParams(r_ois0=r_ois0, r_fwd0=r_fwd0, correlation=correlation)
        trades = [
            TradeParams("SWAP_1", notional=10_000_000, fixed_rate=0.042, T=10.0, is_payer=False),
            TradeParams("SWAP_2", notional=15_000_000, fixed_rate=0.036, T=5.0,  is_payer=True),
            TradeParams("SWAP_3", notional=5_000_000,  fixed_rate=0.038, T=3.0,  is_payer=False)
        ]
        csa = CSAParams(has_csa=has_csa, cpty_threshold=cpty_thr, own_threshold=own_thr)
        crd = CreditParams(cpty_hazard_rate=cpty_hr, own_hazard_rate=own_hr)
        fnd = FundingParams(spread_borrow=spread_borrow)
        cap = CapitalParams(cost_of_capital=coc_rate, saccr_alpha=saccr_alpha, saccr_sf=saccr_sf)
        sim = SimParams(n_paths=sim_paths)

        engine = ProductionXVASACCREngine(mkt, trades, csa, crd, fnd, cap, sim)
        engine.run_simulation()
        
        st.session_state.engine = engine
        st.session_state.results = engine.compute_xva_metrics()
        
        # Automatic CSV exports
        pd.DataFrame([asdict(mkt), asdict(csa), asdict(crd), asdict(fnd), asdict(cap)]).to_csv(OUTPUT_DIR / "dashboard_params.csv", index=False)
        pd.DataFrame([st.session_state.results]).to_csv(OUTPUT_DIR / "dashboard_base_results.csv", index=False)
        pd.DataFrame({
            'Time': engine.t_grid, 'Uncol_EPE': engine.uncol_epe, 
            'Col_EPE': engine.epe, 'Col_ENE': engine.ene, 'SACCR_Capital': engine.saccr_capital_profile
        }).to_csv(OUTPUT_DIR / "dashboard_exposure_saccr_profiles.csv", index=False)
        
        st.session_state.sim_run = True

tab1, tab2, tab3 = st.tabs(["XVA & SA-CCR Summary", "Stress Scenarios", "Model Limitations"])

if st.session_state.sim_run:
    engine = st.session_state.engine
    res = st.session_state.results

    with tab1:
        st.subheader("Bilateral XVA & SA-CCR Pricing Adjustments")
        cols = st.columns(5)
        cols[0].metric("Risk-Free MtM", f"${res['RF_MtM']:,.0f}")
        cols[1].metric("CVA (Cost)", f"${res['CVA']:,.0f}", delta="Cpty Default", delta_color="inverse")
        cols[2].metric("DVA (Benefit)", f"${res['DVA']:,.0f}", delta="Own Default", delta_color="normal")
        cols[3].metric("FVA (Net Cost)", f"${res['FVA']:,.0f}", delta="Funding Cost", delta_color="inverse")
        cols[4].metric("KVA (SA-CCR Cost)", f"${res['KVA']:,.0f}", delta="SA-CCR Capital", delta_color="inverse")
        
        st.metric("Total Adjusted Portfolio Value", f"${res['Adjusted_Value']:,.0f}", f"Net XVA: ${res['Net_XVA']:,.0f}")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(engine.t_grid, engine.epe, label="Net EPE (CVA/FCA)", color="#1f77b4")
            ax1.plot(engine.t_grid, -engine.ene, label="Net ENE (DVA/FBA)", color="#d62728")
            ax1.set_xlabel("Time (Years)")
            ax1.set_ylabel("Exposure ($)", color="black")
            
            ax2 = ax1.twinx()
            ax2.plot(engine.t_grid, engine.saccr_capital_profile, label="SA-CCR Capital Profile", color="#9467bd", ls="--")
            ax2.set_ylabel("SA-CCR Capital ($)", color="#9467bd")
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            ax1.set_title("Exposures vs SA-CCR Regulatory Capital")
            fig1.savefig(OUTPUT_DIR / "dashboard_saccr_profiles.png", dpi=300)
            st.pyplot(fig1)
            plt.close(fig1)
            
        with col_chart2:
            fig2, ax = plt.subplots(figsize=(8, 5))
            adjustments = ['RF MtM', 'CVA', 'DVA', 'FVA', 'KVA', 'Adjusted']
            values = [res['RF_MtM'], -res['CVA'], res['DVA'], -res['FVA'], -res['KVA'], res['Adjusted_Value']]
            colors = ['#7f7f7f', '#d62728', '#2ca02c', '#d62728', '#9467bd', '#1f77b4']
            bars = ax.bar(adjustments, values, color=colors)
            ax.axhline(0, color='black', lw=1)
            ax.set_title("XVA Pricing Waterfall (SA-CCR KVA)")
            for bar in bars:
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, y, f'${y:,.0f}', va='bottom' if y > 0 else 'top', ha='center', fontsize=8)
            fig2.savefig(OUTPUT_DIR / "dashboard_waterfall.png", dpi=300)
            st.pyplot(fig2)
            plt.close(fig2)

    with tab2:
        st.subheader("Scenario Analysis & Stress Testing")
        if st.button("Run Stress Scenarios"):
            with st.spinner("Executing Stresses..."):
                scenarios = {'Baseline': res}
                
                crd_str = CreditParams(cpty_hazard_rate=engine.credit.cpty_hazard_rate + 0.02, own_hazard_rate=engine.credit.own_hazard_rate)
                e_s1 = ProductionXVASACCREngine(engine.mkt, engine.portfolio, engine.csa, crd_str, engine.fnd, engine.cap, engine.sim)
                e_s1.run_simulation()
                scenarios['Credit Spike (+200bps)'] = e_s1.compute_xva_metrics()
                
                mkt_str = MarketParams(r_ois0=engine.mkt.r_ois0, r_fwd0=engine.mkt.r_fwd0, 
                                       sigma_ois=engine.mkt.sigma_ois*2, sigma_fwd=engine.mkt.sigma_fwd*2, correlation=engine.mkt.correlation)
                e_s2 = ProductionXVASACCREngine(mkt_str, engine.portfolio, engine.csa, engine.credit, engine.fnd, engine.cap, engine.sim)
                e_s2.run_simulation()
                scenarios['Rate Vol Spike (2x)'] = e_s2.compute_xva_metrics()

                df_stress = pd.DataFrame(scenarios).T[['CVA', 'DVA', 'FVA', 'KVA', 'Adjusted_Value']]
                df_stress.to_csv(OUTPUT_DIR / "dashboard_stress_results.csv")
                
                st.dataframe(df_stress.style.format("${:,.0f}"))
                
                fig3, ax = plt.subplots(figsize=(10, 4))
                df_stress[['CVA', 'FVA', 'KVA']].plot(kind='bar', ax=ax, colormap='viridis')
                ax.set_title("XVA Costs Under Stress Scenarios")
                ax.set_ylabel("Adjustment ($)")
                plt.xticks(rotation=0)
                fig3.savefig(OUTPUT_DIR / "dashboard_stress_comparison.png", dpi=300)
                st.pyplot(fig3)
                plt.close(fig3)

else:
    st.info("Click **Run Simulation** in the sidebar to load the dashboard.")

with tab3:
    st.subheader("Model Limitations & Assumptions")
    limitations = [
        {"Component": "Interest Rate Dynamics", "Limitation": "Hull-White 1-Factor model assumes normally distributed short rates, allowing unfloored negative rates."},
        {"Component": "SA-CCR AddOn Scope", "Limitation": "Implements the Interest Rate Asset Class AddOn formula; does not model cross-asset diversification or commodity/equity netting sets."},
        {"Component": "Wrong-Way Risk (WWR)", "Limitation": "WWR is integrated implicitly through curve correlation, rather than stochastic default intensity linked to counterparty asset dynamics."},
        {"Component": "Collateral Dynamics (CSA)", "Limitation": "Assumes instantaneous threshold settlement without accounting for Margin Period of Risk (MPOR) or Minimum Transfer Amounts (MTA)."}
    ]
    st.table(pd.DataFrame(limitations))
    pd.DataFrame(limitations).to_csv(OUTPUT_DIR / "dashboard_limitations.csv", index=False)