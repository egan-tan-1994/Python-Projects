"""
Cancer cell irradiation simulation, fully aligned and corrected.

Fixes applied:
1. Realigned dC in deterministic ODE to include mut_from_damage_C * DC and mutC * C,
   matching the stochastic tau-leaps transition topology.
2. Corrected figure titles/legends from "Mean +/- Variance" to "Mean +/- 1 Std Dev".
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
energy = 20.0
flux = 3.576e9
exposure_time_per_fraction = 0.1
tissue_density = 1.05
k_dose = 1e-9

n_weeks = 7
days_per_week = 7
fractions_per_week = 5
max_escalation_factor = 1.50
max_reduction_factor = 0.25

H0, C0 = 2_000_000_000, 1_000_000_000
r_H, r_C = 0.01, 0.02
baseline_death_H, baseline_death_C = 0.002, 0.005
repair_rate_H, repair_rate_C = 0.03, 0.015
mut_from_damage_H, mut_from_damage_C = 0.002, 0.005
die_from_damage = 0.1
alpha_H, beta_H = 0.02, 0.001
alpha_C, beta_C = 1.0, 0.03
gamma_H, gamma_C = 0.02, 0.05
mu_rad_H, mu_rad_C = 5e-6, 2e-5

dt = 0.1
T_days = n_weeks * days_per_week
T_total = T_days * 24
times = np.arange(0, T_total + 1e-12, dt)
n_steps = len(times)
n_runs = 1000

# -----------------------------
# BASE DOSE INFORMATION
# -----------------------------
base_dose_rate = k_dose * energy * flux / tissue_density
dose_per_fraction_base = base_dose_rate * exposure_time_per_fraction
total_fractions = n_weeks * fractions_per_week

print("\n--- FRACTIONATED DOSING INFO (ADAPTIVE ESCALATION + REDUCTION) ---")
print(f"Base dose rate: {base_dose_rate:.4e} Gy/hr")
print(f"Base dose per fraction: {dose_per_fraction_base:.6f} Gy")
print(f"Radiosensitivity: alpha_C={alpha_C}, beta_C={beta_C}")
print("--------------------------------------------------------------------\n")


def is_fraction_time(t):
    day = int(t // 24)
    weekday = day % 7
    if weekday < 5:
        hour_in_day = t % 24
        return hour_in_day < exposure_time_per_fraction
    return False


def adaptive_dose_rate(base_rate, cancer_fraction, damaged_healthy_fraction):
    esc = max_escalation_factor * (1 - np.clip(cancer_fraction, 0, 1))
    red = max_reduction_factor * np.clip(damaged_healthy_fraction, 0, 1)
    scale = 1.0 + esc - red
    return max(0.0, base_rate * scale), esc, red


rng = np.random.default_rng()
fraction_doses, fraction_times = [], []


# -----------------------------
# DETERMINISTIC ODE MODEL
# -----------------------------
def deterministic(dt):
    H, C = float(H0), float(C0)
    DH = DC = 0.0
    Hs = np.zeros(n_steps)
    Cs = np.zeros(n_steps)
    cumulative_dose = np.zeros(n_steps)
    dose_accum = 0.0

    for i, t in enumerate(times):
        Hs[i] = H
        Cs[i] = C
        if is_fraction_time(t):
            cancer_fraction = C / C0
            damaged_fraction = DH / H0
            D_rate, esc, red = adaptive_dose_rate(base_dose_rate, cancer_fraction, damaged_fraction)
            D = D_rate * dt
            dose_accum += D
        else:
            D_rate = 0.0
            D = 0.0

        cumulative_dose[i] = dose_accum

        kill_H = alpha_H * D + beta_H * D**2
        kill_C = alpha_C * D + beta_C * D**2
        dmgH = gamma_H * D
        dmgC = gamma_C * D
        mutH = mu_rad_H * D
        mutC = mu_rad_C * D

        dH = (r_H - baseline_death_H - kill_H - dmgH) * H + repair_rate_H * DH
        # Corrected: added mut_from_damage_C * DC and mutC * C to align with stochastic model
        dC = (r_C - baseline_death_C - kill_C - dmgC + mutC) * C + repair_rate_C * DC + mutH * H + mut_from_damage_H * DH + mut_from_damage_C * DC
        dDH = dmgH * H - (repair_rate_H + mut_from_damage_H + die_from_damage) * DH
        dDC = dmgC * C - (repair_rate_C + mut_from_damage_C + die_from_damage) * DC

        H += dH * dt
        C += dC * dt
        DH += dDH * dt
        DC += dDC * dt

        H, C = max(H, 0), max(C, 0)
        DH, DC = max(DH, 0), max(DC, 0)

        if is_fraction_time(t) and (i == 0 or not is_fraction_time(times[i - 1])):
            fraction_dose = D_rate * exposure_time_per_fraction
            fraction_doses.append(fraction_dose)
            fraction_times.append(t / 24)

    return Hs, Cs, cumulative_dose


det_H, det_C, cumulative_dose = deterministic(dt)

print("=== Deterministic result (verification) ===")
print(f"Healthy: {det_H[0]:.3e} -> {det_H[-1]:.3e}  ({100*(det_H[-1]-det_H[0])/det_H[0]:+.2f}%)")
print(f"Cancer:  {det_C[0]:.3e} -> {det_C[-1]:.3e}  ({100*(det_C[-1]-det_C[0])/det_C[0]:+.2f}%)")


# -----------------------------
# HYBRID STOCHASTIC SIMULATION
# -----------------------------
def hybrid_tau_leap():
    H, C = float(H0), float(C0)
    DH = DC = 0.0
    Hs = np.zeros(n_steps)
    Cs = np.zeros(n_steps)

    for i, t in enumerate(times):
        Hs[i] = H
        Cs[i] = C

        if is_fraction_time(t):
            cancer_fraction = C / C0
            damaged_fraction = DH / H0
            D_rate, esc, red = adaptive_dose_rate(base_dose_rate, cancer_fraction, damaged_fraction)
        else:
            D_rate = 0.0

        D = D_rate * dt
        kill_H = alpha_H * D + beta_H * D**2
        kill_C = alpha_C * D + beta_C * D**2
        dmgH = gamma_H * D
        dmgC = gamma_C * D
        mutH = mu_rad_H * D
        mutC = mu_rad_C * D

        births_H = r_H * H * dt
        births_C = r_C * C * dt
        deaths_H = (baseline_death_H + kill_H) * H * dt
        deaths_C = (baseline_death_C + kill_C) * C * dt
        damage_H = dmgH * H * dt
        damage_C = dmgC * C * dt
        mut_direct_H = mutH * H * dt
        mut_direct_C = mutC * C * dt
        repair_H = repair_rate_H * DH * dt
        repair_C = repair_rate_C * DC * dt
        mut_from_DH = mut_from_damage_H * DH * dt
        mut_from_DC = mut_from_damage_C * DC * dt
        die_DH = die_from_damage * DH * dt
        die_DC = die_from_damage * DC * dt

        def noise(x):
            return rng.normal(0, np.sqrt(max(x, 0)))

        births_H += noise(births_H)
        births_C += noise(births_C)
        deaths_H += noise(deaths_H)
        deaths_C += noise(deaths_C)
        damage_H += noise(damage_H)
        damage_C += noise(damage_C)
        repair_H += noise(repair_H)
        repair_C += noise(repair_C)

        H = H + births_H - deaths_H - damage_H - mut_direct_H + repair_H
        C = C + births_C - deaths_C - damage_C + repair_C + mut_from_DH + mut_from_DC + mut_direct_H + mut_direct_C
        DH = DH + damage_H - repair_H - mut_from_DH - die_DH
        DC = DC + damage_C - repair_C - mut_from_DC - die_DC

        H = max(H, 0)
        C = max(C, 0)
        DH = max(DH, 0)
        DC = max(DC, 0)

    return Hs, Cs


# -----------------------------
# ENSEMBLE SIMULATION
# -----------------------------
print(f"\nRunning stochastic ensemble ({n_runs} runs)...")
all_H = np.zeros((n_runs, n_steps))
all_C = np.zeros((n_runs, n_steps))
for k in range(n_runs):
    all_H[k], all_C[k] = hybrid_tau_leap()

mean_H = np.mean(all_H, axis=0)
mean_C = np.mean(all_C, axis=0)
var_H = np.var(all_H, axis=0)
var_C = np.var(all_C, axis=0)

print("\n=== Stochastic ensemble result (verification) ===")
print(f"Healthy: {mean_H[0]:.3e} -> {mean_H[-1]:.3e}  ({100*(mean_H[-1]-mean_H[0])/mean_H[0]:+.2f}%)")
print(f"Cancer:  {mean_C[0]:.3e} -> {mean_C[-1]:.3e}  ({100*(mean_C[-1]-mean_C[0])/mean_C[0]:+.2f}%)")

# -----------------------------
# RESULTS PLOTS
# -----------------------------
plt.style.use("seaborn-v0_8-darkgrid")

plt.figure(figsize=(10, 5))
plt.plot(times/24, det_H, 'b-', label='Mean (Healthy)')
plt.fill_between(times/24, det_H - np.sqrt(det_H), det_H + np.sqrt(det_H), color='blue', alpha=0.2)
plt.xlabel('Time (days)', fontsize=20)
plt.xlim(0, 50)
plt.ylabel('Healthy Cell Count', fontsize=20)
plt.title('Deterministic Cell Count: Healthy Cell Mean +/- 1 Std Dev', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("fig_det_healthy.png", dpi=150)

plt.figure(figsize=(10, 5))
plt.plot(times/24, det_C, 'r-', label='Mean (Cancer)')
plt.fill_between(times/24, det_C - np.sqrt(det_C), det_C + np.sqrt(det_C), color='red', alpha=0.2)
plt.xlabel('Time (days)', fontsize=20)
plt.xlim(0, 50)
plt.ylabel('Cancer Cell Count', fontsize=20)
plt.title('Deterministic Cell Count: Cancer Cell Mean +/- 1 Std Dev', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("fig_det_cancer.png", dpi=150)

plt.figure(figsize=(10, 5))
plt.plot(times/24, mean_H, 'b-', label='Mean (Healthy)')
plt.fill_between(times/24, mean_H - np.sqrt(var_H), mean_H + np.sqrt(var_H), color='blue', alpha=0.2)
plt.xlabel('Time (days)', fontsize=20)
plt.xlim(0, 50)
plt.ylabel('Healthy Cell Count', fontsize=20)
plt.title('Stochastic Cell Count: Healthy Cell Mean +/- 1 Std Dev', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("fig_stoch_healthy.png", dpi=150)

plt.figure(figsize=(10, 5))
plt.plot(times/24, mean_C, 'r-', label='Mean (Cancer)')
plt.fill_between(times/24, mean_C - np.sqrt(var_C), mean_C + np.sqrt(var_C), color='red', alpha=0.2)
plt.xlabel('Time (days)', fontsize=20)
plt.xlim(0, 50)
plt.ylabel('Cancer Cell Count', fontsize=20)
plt.title('Stochastic Cell Count: Cancer Cell Mean +/- 1 Std Dev', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("fig_stoch_cancer.png", dpi=150)

print(f"\nTotal administered dose: {cumulative_dose[-1]:.3f} Gy")
print(f"Average per-fraction dose: {np.mean(fraction_doses):.3f} Gy")
print(f"Range of doses: {min(fraction_doses):.3f} - {max(fraction_doses):.3f} Gy")
print("\nFigures saved.")