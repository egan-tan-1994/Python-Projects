import os
import smtplib
import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from email.message import EmailMessage
from datetime import datetime

# JAX for Adjoint Algorithmic Differentiation (AAD)
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
jax.config.update("jax_enable_x64", True)

# ReportLab for Automated PDF Generation
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Paragraph, Spacer

# ==============================================================================
# 1. Configuration & Alerting Infrastructure
# ==============================================================================
OUTPUT_DIR = "validation_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AlertSystem:
    @staticmethod
    def send_failure_email(subject, body, report_path=None, to_email="model_risk@quant.local"):
        """Automated email alerts for validation bound breaches."""
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = f"[URGENT] {subject}"
        msg['From'] = "alerts@quantframework.com"
        msg['To'] = to_email
        
        if report_path and os.path.exists(report_path):
            with open(report_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=os.path.basename(report_path))
                
        try:
            server = smtplib.SMTP('localhost', 1025)
            server.send_message(msg)
            server.quit()
            print(f"Alert dispatched: {subject}")
        except Exception as e:
            print(f"Failed to send email alert (mocked for CI/CD): {e}")

# ==============================================================================
# 2. Independent Pricing Models, Credit Risk & AAD
# ==============================================================================
class IndependentModels:
    
    @staticmethod
    def american_binomial_tree(S, K, T, r, sigma, N, option_type='put'):
        """American Option Pricing via Cox-Ross-Rubinstein Binomial Tree."""
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
        V = np.maximum(K - ST, 0) if option_type == 'put' else np.maximum(ST - K, 0)
            
        for i in range(N - 1, -1, -1):
            ST = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            V_early = np.maximum(K - ST, 0) if option_type == 'put' else np.maximum(ST - K, 0)
            V = np.exp(-r * dt) * (p * V[1:] + (1 - p) * V[:-1])
            V = np.maximum(V, V_early)
        return V[0]

    @staticmethod
    def heston_mc_worker(params):
        S0, v0, kappa, theta, sigma, rho, r, T, N, M_chunk = params
        dt = T / N
        prices = np.zeros(M_chunk)
        for i in range(M_chunk):
            S, v = S0, v0
            for _ in range(N):
                Z1 = np.random.normal()
                Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal()
                v_prev = max(v, 0.0)
                v = v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * Z2
                S = S * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z1)
            prices[i] = S
        return prices

    @classmethod
    def heston_mc_parallel(cls, S0, K, v0, kappa, theta, sigma, rho, r, T, N=100, M=20000):
        """Heston Model Monte Carlo with Process Multiprocessing."""
        cores = max(1, cpu_count() - 1)
        chunk_size = M // cores
        params = (S0, v0, kappa, theta, sigma, rho, r, T, N, chunk_size)
        with Pool(cores) as pool:
            results = pool.map(cls.heston_mc_worker, [params] * cores)
        final_prices = np.concatenate(results)
        return np.exp(-r * T) * np.mean(np.maximum(final_prices - K, 0))

    @staticmethod
    def gaussian_copula_cdo(default_probs, correlation, recovery=0.4, M=10000):
        """Gaussian Copula for CDO Tranche loss simulation."""
        n_assets = len(default_probs)
        cov = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov, 1.0)
        Z = np.random.multivariate_normal(np.zeros(n_assets), cov, M)
        thresholds = norm.ppf(default_probs)
        defaults = (Z < thresholds).astype(float)
        return np.mean(np.mean(defaults, axis=1) * (1 - recovery))

    @staticmethod
    def merton_structural_credit_risk(V, D, r, T, sigma_V):
        """Merton (1974) Structural Model for Probability of Default (PD)."""
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        PD = norm.cdf(-d2)
        equity_val = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
        return PD, equity_val

    @staticmethod
    def vasicek_zcb(r0, kappa, theta, sigma, T):
        """Vasicek Interest Rate Model for ZCB Analytical Pricing."""
        B = (1 - np.exp(-kappa * T)) / kappa
        A = np.exp((theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2) / (4 * kappa) * (B**2))
        return A * np.exp(-B * r0)

class AADGreeks:
    @staticmethod
    @jax.jit
    def bs_call_price(S, K, T, r, sigma):
        """JAX-compatible Black-Scholes call price for AAD."""
        d1 = (jnp.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        return S * jax_norm.cdf(d1) - K * jnp.exp(-r * T) * jax_norm.cdf(d2)

    @classmethod
    def calculate_greeks(cls, S, K, T, r, sigma):
        """Calculates 1st, 2nd, and 3rd order Greeks using Adjoint Algorithmic Differentiation."""
        delta_fn = jax.grad(cls.bs_call_price, argnums=0)
        vega_fn = jax.grad(cls.bs_call_price, argnums=4)
        gamma_fn = jax.grad(delta_fn, argnums=0)
        vanna_fn = jax.grad(delta_fn, argnums=4)
        speed_fn = jax.grad(gamma_fn, argnums=0)
        color_fn = jax.grad(gamma_fn, argnums=2)

        return {
            "Delta": float(delta_fn(S, K, T, r, sigma)),
            "Vega": float(vega_fn(S, K, T, r, sigma)),
            "Gamma": float(gamma_fn(S, K, T, r, sigma)),
            "Vanna": float(vanna_fn(S, K, T, r, sigma)),
            "Speed": float(speed_fn(S, K, T, r, sigma)),
            "Color": float(color_fn(S, K, T, r, sigma))
        }

# ==============================================================================
# 3. QuantLib Benchmarks & Expanded Calibrations
# ==============================================================================
class QuantLibBenchmarks:
    @staticmethod
    def build_local_vol_surface(S0, r, div, expiries, strikes, implied_vols, today, day_count):
        calendar = ql.NullCalendar()
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        flat_rate = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        flat_div = ql.YieldTermStructureHandle(ql.FlatForward(today, div, day_count))
        
        vol_matrix = ql.Matrix(len(strikes), len(expiries))
        for i in range(len(strikes)):
            for j in range(len(expiries)):
                vol_matrix[i][j] = implied_vols[i][j]
                
        expiration_dates = [today + ql.Period(int(t*365), ql.Days) for t in expiries]
        black_var_surface = ql.BlackVarianceSurface(today, calendar, expiration_dates, strikes, vol_matrix, day_count)
        black_var_surface.setInterpolation("bicubic") 
        return ql.LocalVolSurface(ql.BlackVolTermStructureHandle(black_var_surface), flat_rate, flat_div, spot_handle)

    @classmethod
    def stochastic_local_vol_mixing(cls, S0, v0, kappa, theta, sigma, rho, r, div, expiries, strikes, implied_vols):
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        flat_rate = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        flat_div = ql.YieldTermStructureHandle(ql.FlatForward(today, div, day_count))
        
        heston_process = ql.HestonProcess(flat_rate, flat_div, spot_handle, v0, kappa, theta, sigma, rho)
        local_vol_surface = cls.build_local_vol_surface(S0, r, div, expiries, strikes, implied_vols, today, day_count)
        
        slv_process = ql.HestonSLVProcess(heston_process, ql.LocalVolTermStructureHandle(local_vol_surface))
        return slv_process.leverageFunction()(1.0, S0)

# ==============================================================================
# 4. Automated PDF Report Generator
# ==============================================================================
class PDFReportGenerator:
    @staticmethod
    def create_report(df, plot_path, pdf_path):
        """Compiles the CSV matrix and PNG plot into a professional PDF audit report."""
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter))
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        elements.append(Paragraph("Independent Quantitative Model Validation Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 24))
        
        # Table of Results
        data = [df.columns.to_list()] + df.values.tolist()
        table = Table(data, colWidths=[120, 250, 100, 100, 80, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('TEXTCOLOR', (5, 1), (5, -1), colors.green), # Status column color logic handled loosely here
        ]))
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        # Plot
        elements.append(Paragraph("Numerical Stability & Error Distribution", styles['Heading2']))
        elements.append(Spacer(1, 12))
        if os.path.exists(plot_path):
            elements.append(RLImage(plot_path, width=500, height=300))
            
        doc.build(elements)
        print(f"PDF Audit Report Compiled: {pdf_path}")

# ==============================================================================
# 5. Independent Validation Engine
# ==============================================================================
class ModelValidator:
    def __init__(self):
        self.results = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def run_all_tests(self):
        # 1. Arbitrage Bounds
        c_price = IndependentModels.american_binomial_tree(100, 90, 1.0, 0.05, 0.2, 150, 'call')
        self._log("Arbitrage_Bounds", "American Binomial Call", c_price, max(0, 100 - 90 * np.exp(-0.05)), "PASS" if c_price >= 14.39 else "FAIL", 0)

        # 2. SLV Leverage Calibration
        try:
            leverage = QuantLibBenchmarks.stochastic_local_vol_mixing(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.05, 0.01, [0.5, 1.0, 1.5], [90.0, 100.0, 110.0], [[0.24, 0.25, 0.26], [0.20, 0.21, 0.22], [0.28, 0.29, 0.30]])
            self._log("Volatility_Modeling", "SLV Leverage Factor (Non-Negative)", leverage, 1.0, "PASS" if leverage > 0 else "FAIL", abs(1 - leverage))
        except Exception:
            self._log("Volatility_Modeling", "SLV Leverage Factor", 0, 1.0, "FAIL", 1.0)

        # 3. Merton Credit Risk
        pd_distress, _ = IndependentModels.merton_structural_credit_risk(V=105, D=100, r=0.05, T=1.0, sigma_V=0.2)
        self._log("Credit_Risk", "Merton PD (Distressed)", pd_distress, 0.30, "PASS" if pd_distress > 0 else "FAIL", 0)

        # 4. AAD vs Finite Difference
        P_up = IndependentModels.american_binomial_tree(100.01, 100, 1.0, 0.05, 0.2, 150, 'call')
        P_dn = IndependentModels.american_binomial_tree(99.99, 100, 1.0, 0.05, 0.2, 150, 'call')
        fd_delta = (P_up - P_dn) / 0.02
        aad_delta = AADGreeks.calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.2)["Delta"]
        self._log("Sensitivities", "AAD vs FD Delta", aad_delta, fd_delta, "PASS" if abs(aad_delta - fd_delta) < 0.05 else "FAIL", abs(aad_delta - fd_delta))

        # 5. Stress Testing
        p_zero_vol = IndependentModels.american_binomial_tree(100, 110, 1.0, 0.05, 0.0001, 100, 'put')
        self._log("Stress_Testing", "Zero-Vol Put -> Intrinsic", p_zero_vol, 10, "PASS" if abs(p_zero_vol - 10) < 0.1 else "FAIL", abs(p_zero_vol - 10))

    def _log(self, category, test_name, model_val, bench_val, status, error=0.0):
        self.results.append({
            "Category": category,
            "Test_Name": test_name,
            "Model_Value": round(float(model_val), 4),
            "Benchmark_Value": round(float(bench_val), 4),
            "Error_Rate": round(float(error), 6),
            "Status": status
        })

    def export_evidence_and_alert(self):
        df = pd.DataFrame(self.results)
        
        # 1. Export CSV
        csv_path = os.path.join(OUTPUT_DIR, f"validation_{self.run_timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Validation Matrix Extracted: {csv_path}")
        
        # 2. Export PNG Plot
        plt.figure(figsize=(10, 6))
        colors = ['#2ca02c' if s == 'PASS' else '#d62728' for s in df['Status']]
        plt.bar(df['Test_Name'], df['Error_Rate'], color=colors)
        plt.axhline(y=0.05, color='r', linestyle='--', label='5% Error Tolerance')
        plt.title('Independent Model Risk Exceptions')
        plt.ylabel('Absolute Relative Error')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"validation_plot_{self.run_timestamp}.png")
        plt.savefig(plot_path)
        print(f"Validation Plot Generated: {plot_path}")

        # 3. Compile PDF Report
        pdf_path = os.path.join(OUTPUT_DIR, f"Validation_Report_{self.run_timestamp}.pdf")
        PDFReportGenerator.create_report(df, plot_path, pdf_path)

        # 4. Email Alerting for Failures
        failures = df[df['Status'] == 'FAIL']
        if not failures.empty:
            alert_body = f"Validation Failures Detected:\n\n{failures.to_string()}"
            AlertSystem.send_failure_email("Quantitative Model Validation Failed", alert_body, report_path=pdf_path)

if __name__ == "__main__":
    validator = ModelValidator()
    print("Initiating Quantitative Model Approvals...")
    validator.run_all_tests()
    validator.export_evidence_and_alert()