import pandas as pd
import numpy as np
import numpy_financial as npf
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================================
# 1. 3-Statement Financial Modeling & Forecasting
# ============================================================================
@dataclass
class FinancialDrivers:
    """Core driver assumptions for projections."""
    revenue_growth: float = 0.10      # Annual Revenue Growth Rate
    cogs_pct_rev: float = 0.60        # Cost of Goods Sold (COGS) as % of Revenue
    opex_pct_rev: float = 0.20        # Operating Expenses (OPEX) as % of Revenue
    tax_rate: float = 0.25            # Corporate Tax Rate
    interest_rate: float = 0.05       # Borrowing/Debt Interest Rate
    depr_rate_ppe: float = 0.10       # Depreciation as % of Beginning PP&E
    capex_pct_rev: float = 0.05       # Capital Expenditure as % of Revenue
    dso: float = 45.0                 # Days Sales Outstanding
    dio: float = 30.0                 # Days Inventory Outstanding
    dpo: float = 35.0                 # Days Payable Outstanding
    min_cash_balance: float = 20.0    # Minimum Operational Cash Target
    div_payout_ratio: float = 0.20    # Dividends as % of Net Income


@dataclass
class BalanceSheetPeriod:
    """Balance Sheet state for a single period."""
    cash: float
    ar: float
    inventory: float
    ppe_net: float
    ap: float
    debt: float
    common_stock: float
    retained_earnings: float
    
    @property
    def total_assets(self) -> float:
        return self.cash + self.ar + self.inventory + self.ppe_net
    
    @property
    def total_liabilities(self) -> float:
        return self.ap + self.debt
    
    @property
    def total_equity(self) -> float:
        return self.common_stock + self.retained_earnings
    
    @property
    def check_balance(self) -> float:
        return self.total_assets - (self.total_liabilities + self.total_equity)


class IntegratedFinancialEngine:
    """3-Statement Modeling & Forecasting Engine with Working Capital and FCF."""
    def __init__(self, initial_bs: BalanceSheetPeriod, base_revenue: float, drivers: FinancialDrivers):
        self.initial_bs = initial_bs
        self.base_revenue = base_revenue
        self.drivers = drivers
        
    def run_forecast(self, periods: int = 5) -> pd.DataFrame:
        forecast_records = []
        prev_bs = self.initial_bs
        prev_rev = self.base_revenue
        
        for period in range(1, periods + 1):
            # 1) Income Statement
            rev = prev_rev * (1 + self.drivers.revenue_growth)
            cogs = rev * self.drivers.cogs_pct_rev
            opex = rev * self.drivers.opex_pct_rev
            gross_profit = rev - cogs
            ebitda = gross_profit - opex
            
            # 2) Balance Sheet Schedule & Working Capital
            ar = (self.drivers.dso / 365.0) * rev
            inv = (self.drivers.dio / 365.0) * cogs
            ap = (self.drivers.dpo / 365.0) * cogs
            
            delta_ar = ar - prev_bs.ar
            delta_inv = inv - prev_bs.inventory
            delta_ap = ap - prev_bs.ap
            delta_nwc = delta_ar + delta_inv - delta_ap
            
            capex = rev * self.drivers.capex_pct_rev
            depr = prev_bs.ppe_net * self.drivers.depr_rate_ppe
            ppe_net = prev_bs.ppe_net + capex - depr
            
            ebit = ebitda - depr
            
            # 3) Circular Reference Solver
            debt = prev_bs.debt
            interest_exp = 0.0
            net_income = 0.0
            cfo = 0.0
            cfi = 0.0
            div_paid = 0.0
            
            for _ in range(25):
                avg_debt = (prev_bs.debt + debt) / 2.0
                interest_exp = avg_debt * self.drivers.interest_rate
                
                ebt = ebit - interest_exp
                tax = max(0.0, ebt * self.drivers.tax_rate)
                net_income = ebt - tax
                
                cfo = net_income + depr - delta_ar - delta_inv + delta_ap
                cfi = -capex
                
                div_paid = max(0.0, net_income * self.drivers.div_payout_ratio)
                cash_before_debt = prev_bs.cash + cfo + cfi - div_paid
                
                cash_deficit = self.drivers.min_cash_balance - cash_before_debt
                if cash_deficit > 0:
                    debt = prev_bs.debt + cash_deficit
                else:
                    excess_cash = -cash_deficit
                    debt_paydown = min(prev_bs.debt, excess_cash)
                    debt = prev_bs.debt - debt_paydown
                    
            # 4) Cash Flow Statement - Financing & Ending Balances
            debt_issuance_net = debt - prev_bs.debt
            cff = debt_issuance_net - div_paid
            
            ending_cash = prev_bs.cash + cfo + cfi + cff
            retained_earnings = (prev_bs.retained_earnings + net_income - div_paid)
            
            current_bs = BalanceSheetPeriod(
                cash=ending_cash, ar=ar, inventory=inv, ppe_net=ppe_net,
                ap=ap, debt=debt, common_stock=prev_bs.common_stock,
                retained_earnings=retained_earnings
            )
            
            # 5) FCF Metrics
            nopat = ebit * (1.0 - self.drivers.tax_rate)
            fcff = nopat + depr - capex - delta_nwc
            fcfe = net_income + depr - capex - delta_nwc + debt_issuance_net
            
            forecast_records.append({
                "Period": f"Year {period}", "Revenue": rev, "EBITDA": ebitda, "EBIT": ebit,
                "Interest Expense": interest_exp, "Net Income": net_income,
                "CFO": cfo, "CFI": cfi, "CFF": cff, "Net Cash Flow": cfo + cfi + cff,
                "Cash": ending_cash, "Accounts Receivable": ar, "Debt": debt,
                "Retained Earnings": retained_earnings, "Total Assets": current_bs.total_assets,
                "Total Liabilities": current_bs.total_liabilities, "Total Equity": current_bs.total_equity,
                "Total Liabilities and Equity": (current_bs.total_liabilities + current_bs.total_equity),
                "Balance Sheet (BS) Imbalance Check": current_bs.check_balance,
                "Unlevered Free Cash Flow (FCFF)": fcff, "Levered Free Cash Flow (FCFE)": fcfe
            })
            
            prev_bs = current_bs
            prev_rev = rev
            
        return pd.DataFrame(forecast_records).set_index("Period").T


# ============================================================================
# 2. DCF and Corporate Valuation Engine
# ============================================================================
class CorporateValuationEngine:
    """Production-grade DCF Corporate Valuation Engine."""
    def __init__(self, inputs: dict):
        self.inputs = inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        required_keys = [
            'base_revenue', 'revenue_growth', 'ebit_margin', 'tax_rate', 'capex_pct',
            'dna_pct', 'nwc_pct', 'risk_free_rate', 'beta', 'equity_risk_premium',
            'cost_of_debt', 'market_cap', 'total_debt', 'cash', 'shares_outstanding',
            'terminal_growth_rate', 'exit_multiple',
        ]
        missing = [k for k in required_keys if k not in self.inputs]
        if missing:
            raise ValueError(f"Missing required input key(s): {missing}")
            
    def calculate_wacc(self) -> dict:
        rf = self.inputs['risk_free_rate']
        beta = self.inputs['beta']
        erp = self.inputs['equity_risk_premium']
        tax = self.inputs['tax_rate']
        
        cost_of_equity = rf + (beta * erp)
        after_tax_cost_of_debt = self.inputs['cost_of_debt'] * (1 - tax)
        
        e = self.inputs['market_cap']
        d = self.inputs['total_debt']
        v = e + d
        
        w_e = e / v
        w_d = d / v
        
        wacc = (w_e * cost_of_equity) + (w_d * after_tax_cost_of_debt)
        
        return {
            'cost_of_equity': cost_of_equity, 'after_tax_cost_of_debt': after_tax_cost_of_debt,
            'weight_equity': w_e, 'weight_debt': w_d, 'wacc': wacc,
        }
    
    def forecast_cash_flows(self) -> pd.DataFrame:
        growth_rates = self.inputs['revenue_growth']
        years = len(growth_rates)
        periods = [f'Year {i+1}' for i in range(years)]
        df = pd.DataFrame(index=periods)
        
        revenues = []
        current_rev = self.inputs['base_revenue']
        for g in growth_rates:
            current_rev *= 1 + g
            revenues.append(current_rev)
            
        df['Revenue'] = revenues
        df['EBIT'] = df['Revenue'] * self.inputs['ebit_margin']
        df['NOPAT'] = df['EBIT'] * (1 - self.inputs['tax_rate'])
        df['D&A'] = df['Revenue'] * self.inputs['dna_pct']
        df['CapEx'] = df['Revenue'] * self.inputs['capex_pct']
        
        nwc = df['Revenue'] * self.inputs['nwc_pct']
        base_nwc = self.inputs['base_revenue'] * self.inputs['nwc_pct']
        df['NWC'] = nwc
        df['Delta_NWC'] = df['NWC'] - df['NWC'].shift(1).fillna(base_nwc)
        
        df['FCFF'] = df['NOPAT'] + df['D&A'] - df['CapEx'] - df['Delta_NWC']
        return df
    
    def calculate_valuation(self, override_wacc: float = None, override_g: float = None,
                            override_exit_mult: float = None) -> dict:
        wacc_dict = self.calculate_wacc()
        wacc = override_wacc if override_wacc is not None else wacc_dict['wacc']
        g = override_g if override_g is not None else self.inputs['terminal_growth_rate']
        exit_mult = override_exit_mult if override_exit_mult is not None else self.inputs['exit_multiple']
        
        df = self.forecast_cash_flows()
        n = len(df)
        
        df['Discount_Period'] = np.arange(1, n+1)
        df['Discount_Factor'] = 1 / ((1 + wacc) ** df['Discount_Period'])
        df['PV_FCFF'] = df['FCFF'] * df['Discount_Factor']
        pv_explicit_fcff = df['PV_FCFF'].sum()
        
        terminal_fcff = df['FCFF'].iloc[-1] * (1 + g)
        if wacc <= g:
            raise ValueError("WACC must be strictly greater than Terminal Growth Rate (g)")
            
        tv_gg = terminal_fcff / (wacc - g)
        pv_tv_gg = tv_gg / ((1 + wacc) ** n)
        
        terminal_ebitda = df['EBIT'].iloc[-1] + df['D&A'].iloc[-1]
        tv_em = terminal_ebitda * exit_mult
        pv_tv_em = tv_em / ((1 + wacc) ** n)
        
        ev_gg = pv_explicit_fcff + pv_tv_gg
        eq_val_gg = ev_gg + self.inputs['cash'] - self.inputs['total_debt']
        share_price_gg = eq_val_gg / self.inputs['shares_outstanding']
        
        ev_em = pv_explicit_fcff + pv_tv_em
        eq_val_em = ev_em + self.inputs['cash'] - self.inputs['total_debt']
        share_price_em = eq_val_em / self.inputs['shares_outstanding']
        
        return {
            'wacc': wacc, 'g_rate': g, 'exit_multiple': exit_mult,
            'pv_explicit_fcff': pv_explicit_fcff, 'tv_gg': tv_gg, 'pv_tv_gg': pv_tv_gg,
            'ev_gg': ev_gg, 'equity_value_gg': eq_val_gg, 'share_price_gg': share_price_gg,
            'tv_em': tv_em, 'pv_tv_em': pv_tv_em, 'ev_em': ev_em,
            'equity_value_em': eq_val_em, 'share_price_em': share_price_em, 'projection_df': df,
        }
    
    def generate_sensitivity_matrix(self, wacc_steps: np.ndarray, g_steps: np.ndarray) -> pd.DataFrame:
        matrix = np.zeros((len(wacc_steps), len(g_steps)))
        
        for i, w in enumerate(wacc_steps):
            for j, g in enumerate(g_steps):
                try:
                    res = self.calculate_valuation(override_wacc=w, override_g=g)
                    matrix[i, j] = res['share_price_gg']
                except ValueError:
                    matrix[i, j] = np.nan
                
        index_labels = [f'{w:.2%}' for w in wacc_steps]
        col_labels = [f'{g:.2%}' for g in g_steps]
        
        sensitivity_df = pd.DataFrame(matrix, index=index_labels, columns=col_labels)
        sensitivity_df.index.name = 'WACC \ Growth'
        return sensitivity_df


# ============================================================================
# 3. LBO Modeling and Returns
# ============================================================================
class LBOEngine:
    def __init__(self, entry_ebitda=50.0, entry_multiple=10.0, holding_period=5):
        self.entry_ebitda = entry_ebitda
        self.entry_multiple = entry_multiple
        self.holding_period = holding_period
        self.ev = self.entry_ebitda * self.entry_multiple
        
        self.term_loan_a_multiple = 3.0
        self.term_loan_b_multiple = 2.0
        self.mezz_debt_multiple = 1.0
        
        self.tla_rate = 0.06
        self.tlb_rate = 0.075
        self.mezz_rate = 0.10
        self.tla_amort_rate = 0.10
        
        self.transaction_fee_pct = 0.02
        self.min_cash = 5.0
        
    def build_sources_and_uses(self):
        tla_amount = self.entry_ebitda * self.term_loan_a_multiple
        tlb_amount = self.entry_ebitda * self.term_loan_b_multiple
        mezz_amount = self.entry_ebitda * self.mezz_debt_multiple
        total_debt = tla_amount + tlb_amount + mezz_amount
            
        transaction_fees = self.ev * self.transaction_fee_pct
        total_uses = self.ev + transaction_fees + self.min_cash
        sponsor_equity = total_uses - total_debt
            
        sources_uses = pd.DataFrame({
            "Sources/Uses Item": ["Term Loan A", "Term Loan B", "Mezzanine Debt", "Sponsor Equity", "Total Sources/Uses"],
            "Amount ($M)": [tla_amount, tlb_amount, mezz_amount, sponsor_equity, total_uses],
            "% Structure": [tla_amount / total_uses, tlb_amount / total_uses, mezz_amount / total_uses, sponsor_equity / total_uses, total_uses / total_uses],
        })
        return sources_uses, total_debt, sponsor_equity
        
    def forecast_operations(self, rev_growth=0.06, ebitda_margin=0.25, capex_pct=0.035, nwc_pct=0.015):
        years = list(range(1, self.holding_period + 1))
        base_revenue = self.entry_ebitda / ebitda_margin
        
        op_model = pd.DataFrame(index=years)
        op_model["Revenue"] = [base_revenue * ((1 + rev_growth) ** y) for y in years]
        op_model["EBITDA"] = op_model["Revenue"] * ebitda_margin
        op_model["D&A"] = op_model["Revenue"] * 0.03
        op_model["CapEx"] = op_model["Revenue"] * capex_pct
        
        rev_prev = pd.Series([base_revenue] + list(op_model["Revenue"][:-1]), index=years)
        op_model["NWC Change"] = (op_model["Revenue"] - rev_prev) * nwc_pct
        
        return op_model
        
    def run_debt_waterfall(self, op_model):
        years = list(range(1, self.holding_period + 1))
        
        tla_beg = self.entry_ebitda * self.term_loan_a_multiple
        tlb_beg = self.entry_ebitda * self.term_loan_b_multiple
        mezz_beg = self.entry_ebitda * self.mezz_debt_multiple
        
        prev_cash = self.min_cash
        records = []
        
        for y in years:
            ebitda = op_model.loc[y, "EBITDA"]
            da = op_model.loc[y, "D&A"]
            capex = op_model.loc[y, "CapEx"]
            nwc_change = op_model.loc[y, "NWC Change"]
            
            tla_int = tla_beg * self.tla_rate
            tlb_int = tlb_beg * self.tlb_rate
            mezz_int = mezz_beg * self.mezz_rate
            total_int = tla_int + tlb_int + mezz_int
            
            ebt = ebitda - da - total_int
            tax = max(0.0, ebt * 0.25)
            cfads = ebitda - capex - nwc_change - tax
            cash_avail = cfads - total_int
            
            mandatory_tla_amort = (self.entry_ebitda * self.term_loan_a_multiple) * self.tla_amort_rate
            tla_amort = min(tla_beg, mandatory_tla_amort, max(0.0, cash_avail))
            
            cash_after_amort = max(0.0, cash_avail - tla_amort)
            
            tla_sweep = min(tla_beg - tla_amort, cash_after_amort)
            cash_after_tla = cash_after_amort - tla_sweep
            
            tlb_sweep = min(tlb_beg, cash_after_tla)
            cash_after_tlb = cash_after_tla - tlb_sweep
            
            mezz_sweep = min(mezz_beg, cash_after_tlb)
            cash_remaining = cash_after_tlb - mezz_sweep
            
            tla_end = tla_beg - tla_amort - tla_sweep
            tlb_end = tlb_beg - tlb_sweep
            mezz_end = mezz_beg - mezz_sweep
            total_ending_debt = tla_end + tlb_end + mezz_end
            
            ending_cash = prev_cash + cash_remaining
            
            records.append({
                "Year": y, "EBITDA": ebitda, "CFADS": cfads, "Interest": total_int,
                "TLA End": tla_end, "TLB End": tlb_end, "Mezz End": mezz_end,
                "Total Debt End": total_ending_debt, "Ending Cash": ending_cash,
            })
            
            tla_beg, tlb_beg, mezz_beg = tla_end, tlb_end, mezz_end
            prev_cash = ending_cash
            
        return pd.DataFrame(records).set_index("Year")
        
    def calculate_returns(self, exit_multiple, sponsor_equity, debt_sched, op_model):
        final_y = self.holding_period
        exit_ebitda = op_model.loc[final_y, "EBITDA"]
        exit_ev = exit_ebitda * exit_multiple
        
        ending_debt = debt_sched.loc[final_y, "Total Debt End"]
        ending_cash = debt_sched.loc[final_y, "Ending Cash"]
        net_debt = ending_debt - ending_cash
        
        exit_equity = exit_ev - net_debt
        moic = exit_equity / sponsor_equity
        
        cfs = [-sponsor_equity] + [0] * (self.holding_period - 1) + [exit_equity]
        irr = npf.irr(cfs)
        
        return {
            "Exit EBITDA": exit_ebitda, "Exit EV": exit_ev, "Ending Net Debt": net_debt,
            "Exit Equity Value": exit_equity, "MOIC": moic, "IRR": irr,
        }
        
    def sensitivity_matrix(self, entry_multiples, exit_multiples, rev_growth=0.06, ebitda_margin=0.25):
        matrix_irr = pd.DataFrame(index=[f"Entry {m:.1f}x" for m in entry_multiples], columns=[f"Exit {m:.1f}x" for m in exit_multiples])
        matrix_moic = pd.DataFrame(index=[f"Entry {m:.1f}x" for m in entry_multiples], columns=[f"Exit {m:.1f}x" for m in exit_multiples])
        
        for en_m in entry_multiples:
            for ex_m in exit_multiples:
                model = LBOEngine(entry_ebitda=self.entry_ebitda, entry_multiple=en_m, holding_period=self.holding_period)
                _, _, eq = model.build_sources_and_uses()
                op_mod = model.forecast_operations(rev_growth=rev_growth, ebitda_margin=ebitda_margin)
                d_sched = model.run_debt_waterfall(op_mod)
                ret = model.calculate_returns(ex_m, eq, d_sched, op_mod)
                
                row_label, col_label = f"Entry {en_m:.1f}x", f"Exit {ex_m:.1f}x"
                matrix_irr.loc[row_label, col_label] = f"{ret['IRR']:.1%}"
                matrix_moic.loc[row_label, col_label] = f"{ret['MOIC']:.2f}x"
                
        return matrix_irr, matrix_moic


# ============================================================================
# 4. Cash Flow, Debt Paydown, and Liquidity Engine
# ============================================================================
@dataclass
class DebtTranche:
    name: str
    beginning_balance: float
    interest_rate: float
    mandatory_amort_pct: float
    priority: int
    is_sweepable: bool = True
    
    def __post_init__(self):
        self.initial_balance = self.beginning_balance
        self.current_balance = self.beginning_balance


@dataclass
class FinancialInputs:
    years: int = 5
    ebitda: List[float] = field(default_factory=lambda: [100.0, 115.0, 130.0, 145.0, 160.0])
    capex_pct_ebitda: float = 0.15
    nwc_change: List[float] = field(default_factory=lambda: [-5.0, -6.0, -4.0, -5.0, -5.0])
    da: List[float] = field(default_factory=lambda: [20.0, 22.0, 24.0, 25.0, 26.0])
    tax_rate: float = 0.25
    initial_cash: float = 30.0
    min_cash_balance: float = 20.0
    revolver_capacity: float = 50.0
    revolver_rate: float = 0.065
    sweep_pct: float = 0.80
    target_max_leverage: float = 3.5
    target_min_dscr: float = 1.35


class CashFlowDebtEngine:
    """Engine integrating FCF modeling, cash sweep debt paydown, liquidity monitoring."""
    def __init__(self, inputs: FinancialInputs, tranches: List[DebtTranche]):
        self.inputs = inputs
        self.tranches = sorted(tranches, key=lambda x: x.priority)
        
    def run_projection(self) -> pd.DataFrame:
        records = []
        current_cash = self.inputs.initial_cash
        active_tranches = {
            t.name: {"balance": t.beginning_balance, "rate": t.interest_rate, "amort_pct": t.mandatory_amort_pct, "initial": t.initial_balance, "priority": t.priority, "is_sweepable": t.is_sweepable}
            for t in self.tranches
        }
        revolver_balance = 0.0
        
        for yr in range(self.inputs.years):
            ebitda, da, dnwc = self.inputs.ebitda[yr], self.inputs.da[yr], self.inputs.nwc_change[yr]
            ebit = ebitda - da
            capex = ebitda * self.inputs.capex_pct_ebitda
            beg_cash = current_cash
            
            interest_detail = {}
            total_interest = 0.0
            for name, tranche in active_tranches.items():
                interest = tranche["balance"] * tranche["rate"]
                interest_detail[f"interest_{name}"] = interest
                total_interest += interest
            
            revolver_interest = revolver_balance * self.inputs.revolver_rate
            total_interest += revolver_interest
                
            taxable_income = max(0.0, ebit - total_interest)
            taxes = taxable_income * self.inputs.tax_rate
            cfads = ebitda - capex - dnwc - taxes
            fcff = ebit * (1 - self.inputs.tax_rate) + da - capex - dnwc
            
            mandatory_amort_detail = {}
            total_mandatory_amort = 0.0
            for name, tranche in active_tranches.items():
                amort_due = tranche["initial"] * tranche["amort_pct"]
                actual_amort = min(tranche["balance"], amort_due)
                mandatory_amort_detail[f"mand_amort_{name}"] = actual_amort
                total_mandatory_amort += actual_amort
                tranche["balance"] -= actual_amort
                
            total_mandatory_debt_service = total_interest + total_mandatory_amort
            fcf_after_mandatory = cfads - total_mandatory_debt_service
            cash_before_sweep = beg_cash + fcf_after_mandatory
            
            excess_cash = max(0.0, cash_before_sweep - self.inputs.min_cash_balance)
            sweep_budget = excess_cash * self.inputs.sweep_pct
            
            sweep_detail = {}
            remaining_sweep_funds = sweep_budget
            revolver_sweep = min(remaining_sweep_funds, revolver_balance)
            revolver_balance -= revolver_sweep
            remaining_sweep_funds -= revolver_sweep
            sweep_detail["sweep_Revolver"] = revolver_sweep
            
            for name, tranche in sorted(active_tranches.items(), key=lambda x: x[1]["priority"]):
                if tranche["is_sweepable"] and remaining_sweep_funds > 0 and tranche["balance"] > 0:
                    sweep_paid = min(remaining_sweep_funds, tranche["balance"])
                    tranche["balance"] -= sweep_paid
                    remaining_sweep_funds -= sweep_paid
                    sweep_detail[f"sweep_{name}"] = sweep_paid
                else:
                    sweep_detail[f"sweep_{name}"] = 0.0
                    
            total_sweep_paid = sum(sweep_detail.values())
            ending_cash = cash_before_sweep - total_sweep_paid
            total_ending_debt = (sum(t["balance"] for t in active_tranches.values()) + revolver_balance)
            net_debt = total_ending_debt - ending_cash
            
            revolver_drawn = 0.0
            if ending_cash < self.inputs.min_cash_balance:
                cash_deficit = self.inputs.min_cash_balance - ending_cash
                remaining_capacity = self.inputs.revolver_capacity - revolver_balance
                revolver_drawn = max(0.0, min(cash_deficit, remaining_capacity))
                revolver_balance += revolver_drawn
                ending_cash += revolver_drawn
                total_ending_debt += revolver_drawn
                net_debt = total_ending_debt - ending_cash
                
            total_liquidity = ending_cash + (self.inputs.revolver_capacity - revolver_balance)
            liquidity_headroom = ending_cash - self.inputs.min_cash_balance
            
            dscr = cfads / total_mandatory_debt_service if total_mandatory_debt_service > 0 else np.nan
            interest_coverage = ebitda / total_interest if total_interest > 0 else np.nan
            gross_leverage = total_ending_debt / ebitda
            net_leverage = net_debt / ebitda
            
            max_debt_leverage = ((ebitda * self.inputs.target_max_leverage) + ending_cash)
            max_debt_service_dscr = cfads / self.inputs.target_min_dscr
            debt_service_headroom = (max_debt_service_dscr - total_mandatory_debt_service)
            
            row = {
                "Year": yr + 1, "EBITDA": ebitda, "CapEx": capex, "NWC_Change": dnwc,
                "Taxes": taxes, "FCFF": fcff, "CFADS": cfads, "Beg_Cash": beg_cash,
                "Total_Interest": total_interest, "Total_Mandatory_Amort": total_mandatory_amort,
                "FCF_Post_Mandatory_Debt_Service": fcf_after_mandatory, "Cash_Before_Sweep": cash_before_sweep,
                "Total_Sweep_Paid": total_sweep_paid, "Ending_Cash": ending_cash, "Revolver_Drawn": revolver_drawn,
                "Total_Ending_Debt": total_ending_debt, "Net_Debt": net_debt, "Total_Liquidity": total_liquidity,
                "Liquidity_Headroom": liquidity_headroom, "DSCR": dscr, "Interest_Coverage": interest_coverage,
                "Gross_Leverage": gross_leverage, "Net_Leverage": net_leverage, "Max_Debt_Cap_Leverage": max_debt_leverage,
                "Max_Debt_Service_DSCR": max_debt_service_dscr, "Debt_Service_Headroom": debt_service_headroom,
                "Revolver_Interest": revolver_interest, "Revolver_Balance": revolver_balance,
            }
            row.update(interest_detail)
            row.update(mandatory_amort_detail)
            row.update(sweep_detail)
            for name, tranche in active_tranches.items():
                row[f"end_bal_{name}"] = tranche["balance"]
                
            records.append(row)
            current_cash = ending_cash
            
        return pd.DataFrame(records)


# ============================================================================
# 5. Dashboard Wrapper Execution Function
# ============================================================================
def generate_financial_models_data() -> Dict[str, pd.DataFrame]:
    """
    Executes all financial models and returns a dictionary of output DataFrames
    and metrics. This function is ideal for importing into a Streamlit dashboard.
    """
    output_data = {}

    # --- 3-Statement Model ---
    y0_balance_sheet = BalanceSheetPeriod(
        cash=50.0, ar=20.0, inventory=15.0, ppe_net=100.0, ap=15.0, debt=40.0,
        common_stock=30.0, retained_earnings=100.0
    )
    fs_drivers = FinancialDrivers()
    fs_engine = IntegratedFinancialEngine(initial_bs=y0_balance_sheet, base_revenue=200.0, drivers=fs_drivers)
    output_data["3_statement_forecast"] = fs_engine.run_forecast(periods=5)

    # --- DCF Valuation Engine ---
    model_inputs = {
        'base_revenue': 1000.0, 'revenue_growth': [0.12, 0.10, 0.08, 0.06, 0.05], 'ebit_margin': 0.22,
        'tax_rate': 0.21, 'capex_pct': 0.045, 'dna_pct': 0.035, 'nwc_pct': 0.08, 'risk_free_rate': 0.042,
        'beta': 1.10, 'equity_risk_premium': 0.055, 'cost_of_debt': 0.055, 'market_cap': 5000.0,
        'total_debt': 1500.0, 'cash': 500.0, 'shares_outstanding': 100.0, 'terminal_growth_rate': 0.025,
        'exit_multiple': 12.0,
    }
    dcf_engine = CorporateValuationEngine(model_inputs)
    val = dcf_engine.calculate_valuation()
    
    # Store key scalar metrics and the projection dataframe
    output_data["dcf_valuation_metrics"] = {
        "WACC": val['wacc'],
        "Enterprise Value (Gordon Growth)": val['ev_gg'],
        "Implied Share Price": val['share_price_gg']
    }
    output_data["dcf_projection_df"] = val['projection_df']

    # --- LBO Model ---
    lbo = LBOEngine(entry_ebitda=50.0, entry_multiple=10.0, holding_period=5)
    sources_uses, total_debt, sponsor_equity = lbo.build_sources_and_uses()
    op_model = lbo.forecast_operations(rev_growth=0.06, ebitda_margin=0.25)
    debt_sched = lbo.run_debt_waterfall(op_model)
    returns = lbo.calculate_returns(exit_multiple=10.0, sponsor_equity=sponsor_equity, debt_sched=debt_sched, op_model=op_model)
    
    output_data["lbo_sources_uses"] = sources_uses
    output_data["lbo_debt_schedule"] = debt_sched
    output_data["lbo_returns"] = returns

    # --- Cash Flow & Debt Sweep ---
    cf_inputs = FinancialInputs()
    debt_tranches = [
        DebtTranche(name="Senior_Term_Loan_A", beginning_balance=150.0, interest_rate=0.065, mandatory_amort_pct=0.10, priority=1),
        DebtTranche(name="Senior_Term_Loan_B", beginning_balance=100.0, interest_rate=0.080, mandatory_amort_pct=0.05, priority=2),
        DebtTranche(name="Mezzanine_Note", beginning_balance=50.0, interest_rate=0.110, mandatory_amort_pct=0.00, priority=3, is_sweepable=False)
    ]
    cf_engine = CashFlowDebtEngine(inputs=cf_inputs, tranches=debt_tranches)
    output_data["cash_flow_debt_sweep"] = cf_engine.run_projection()

    return output_data

if __name__ == "__main__":
    # If run as a standalone script, fetch the data and verify everything runs correctly
    print("Running financial engines...")
    data = generate_financial_models_data()
    print("Execution complete. Ready for Streamlit import. Available Datasets:")
    for key in data.keys():
        print(f" - {key}")
