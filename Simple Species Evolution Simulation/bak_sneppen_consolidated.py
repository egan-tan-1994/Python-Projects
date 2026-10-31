"""
Consolidated Bak-Sneppen Model Analysis
--------------------------------------
Simulates self-organized criticality in the 1D Bak-Sneppen evolutionary model,
including distance distributions P(x), barrier probability densities P(B),
punctuated equilibrium dynamics, small-world shortcut effects, and power-law
exponent fitting for pi (distance) and tau (avalanche size).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# Global Parameters
# -----------------------
np.random.seed(12345)
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# System sizes
N_large = 4096
N_medium = 512

# Time settings
TRANSIENT = 200000
SAMPLE_STEPS = 1000000
T0 = 0.01  # temperature for real-time scaling
H_THRESHOLD = 0.66  # just below B_c ~= 0.67

# Theoretical reference exponents (1D Bak-Sneppen)
PI_THEORY = 3.23    # distance distribution: C(x) ~ x^-pi
TAU_THEORY = 1.1    # avalanche size distribution: P(s) ~ s^-tau


# -----------------------
# Bak-Sneppen Model Engine
# -----------------------
def run_bak_sneppen(
    N, transient, steps,
    record_positions=False,
    record_barriers=False,
    record_min_series=False,
    record_local_activity=None,
    record_barrier_for_real_time=False
):
    """Simulate 1D Bak-Sneppen dynamics with ring topology."""
    B = np.random.rand(N)

    # Transient phase (self-organized criticality setup)
    for _ in range(transient):
        imin = np.argmin(B)
        B[imin] = np.random.rand()
        B[(imin - 1) % N] = np.random.rand()
        B[(imin + 1) % N] = np.random.rand()

    start, seg_len = (record_local_activity or (None, None))
    results = dict(
        mut_pos=[] if record_positions else None,
        barriers_samples=[] if record_barriers else None,
        min_series=[] if record_min_series else None,
        local_activity=[] if record_local_activity else None,
        barrier_series=[] if record_barrier_for_real_time else None,
    )

    # Sampling phase
    for step in range(steps):
        imin = np.argmin(B)

        if record_positions:
            results["mut_pos"].append(imin)
        if record_min_series:
            results["min_series"].append(B[imin])
        if record_barriers and step % 100 == 0:
            results["barriers_samples"].append(B.copy())
        if record_local_activity is not None:
            results["local_activity"].append(int(start <= imin < start + seg_len))
        if record_barrier_for_real_time:
            results["barrier_series"].append(B[imin])

        # Update active site and nearest neighbors
        B[imin] = np.random.rand()
        B[(imin - 1) % N] = np.random.rand()
        B[(imin + 1) % N] = np.random.rand()

    for k, v in results.items():
        if v is not None:
            results[k] = np.array(v)
    return results


def bak_sneppen_shortcuts(N=1024, transient=50000, steps=300000, p=0.01):
    """Bak-Sneppen model with small-world long-range shortcut updates."""
    B = np.random.rand(N)

    # Transient phase
    for _ in range(transient):
        i_min = np.argmin(B)
        update_sites = [i_min, (i_min - 1) % N, (i_min + 1) % N]
        if np.random.rand() < p:
            update_sites.append(np.random.randint(0, N))
        B[update_sites] = np.random.rand(len(update_sites))

    # Sampling phase
    min_series = []
    for _ in range(steps):
        i_min = np.argmin(B)
        min_series.append(B[i_min])

        update_sites = [i_min, (i_min - 1) % N, (i_min + 1) % N]
        if np.random.rand() < p:
            update_sites.append(np.random.randint(0, N))

        B[update_sites] = np.random.rand(len(update_sites))

    return np.array(min_series)


# -----------------------
# Power-Law Fitting Helper
# -----------------------
def fit_power_law(x, y, xmin=None, xmax=None):
    """Fit y ~ x^-alpha via ordinary least squares in log-log space."""
    mask = (x > 0) & (y > 0)
    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)
    if mask.sum() < 3:
        raise ValueError("Not enough points in the specified fitting range.")

    logx = np.log10(x[mask])
    logy = np.log10(y[mask])
    slope, intercept = np.polyfit(logx, logy, 1)
    alpha = -slope

    y_pred = slope * logx + intercept
    ss_res = np.sum((logy - y_pred) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return alpha, r_squared


# -----------------------
# FIGURE 1: Distance Distribution P(x)
# -----------------------
print("Fig. 1: Distance distribution P(x)...")
res1 = run_bak_sneppen(N_large, TRANSIENT, SAMPLE_STEPS // 2, record_positions=True)
pos = res1["mut_pos"]
d = np.abs(np.diff(pos))
d = np.minimum(d, N_large - d)

bins = np.logspace(0, np.log10(N_large // 2), 60)
hist1, edges1 = np.histogram(d, bins=bins, density=True)
centers1 = np.sqrt(edges1[:-1] * edges1[1:])

plt.figure(figsize=(6, 4.5))
plt.loglog(centers1[hist1 > 0], hist1[hist1 > 0], 'o')
plt.xlabel("Distance x between successive mutations")
plt.ylabel("P(x)")
plt.title("Fig. 1 - Distance Distribution P(x)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_distance_distribution.png"), dpi=200)


# -----------------------
# FIGURE 2: Barrier Probability Density P(B)
# -----------------------
print("Fig. 2: Barrier probability density P(B)...")
res2 = run_bak_sneppen(N_large, TRANSIENT, SAMPLE_STEPS // 10,
                       record_barriers=True, record_min_series=True)
barriers = np.concatenate(res2["barriers_samples"])
mins = res2["min_series"]

plt.figure(figsize=(6, 4.5))
plt.hist(barriers, bins=150, density=True, alpha=0.6, label="All barriers")
plt.hist(mins, bins=150, density=True, alpha=0.6, label="Min barriers")
plt.axvline(0.67, color="k", ls="--", label="Bc ~= 0.67")
plt.xlabel("Barrier B")
plt.ylabel("P(B)")
plt.legend()
plt.title("Fig. 2 - Barrier Probability Density P(B)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_barrier_PB.png"), dpi=200)


# -----------------------
# FIGURE 3: Mutation Activity vs Count Time and Real Time
# -----------------------
print("Fig. 3: Mutation activity vs count time and real time...")
res3 = run_bak_sneppen(
    N_medium, TRANSIENT, SAMPLE_STEPS // 10,
    record_positions=True, record_barrier_for_real_time=True
)
mutpos = res3["mut_pos"]
barriers3 = res3["barrier_series"]

# Dynamically center local segment around active mutation zone
segment_start = (mutpos[0] - 5) % N_medium
segment_len = 10
window = 1000
num_bins = len(mutpos) // window

local_counts = []
for i in range(num_bins):
    seg = mutpos[i * window:(i + 1) * window]
    if segment_start + segment_len <= N_medium:
        in_seg = (seg >= segment_start) & (seg < segment_start + segment_len)
    else:
        in_seg = (seg >= segment_start) | (seg < (segment_start + segment_len) % N_medium)
    local_counts.append(np.sum(in_seg))

local_counts = np.array(local_counts)
count_time = np.arange(num_bins) * window

# Real-time transformation with safe exponent scaling
log_durations = barriers3 / T0
log_durations -= np.max(log_durations) - 200.0  # Guard against overflow
durations = np.exp(log_durations)
real_time = np.cumsum(durations)

num_rt_bins = 1500
rt_bins = np.linspace(real_time.min(), real_time.max(), num_rt_bins)
rt_counts, _ = np.histogram(real_time, bins=rt_bins)
rt_centers = 0.5 * (rt_bins[:-1] + rt_bins[1:])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(count_time, local_counts, drawstyle="steps-mid")
plt.xlabel("Count time (mutation steps)")
plt.ylabel("Mutation activity (local count)")
plt.title("Fig. 3(a) - Activity vs Count Time")
plt.grid(True, ls=":")

plt.subplot(1, 2, 2)
plt.plot(rt_centers, rt_counts, drawstyle="steps-mid")
plt.xlabel("Accumulated real time")
plt.ylabel("Mutation activity (per time bin)")
plt.title("Fig. 3(b) - Activity vs Real Time")
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_mutation_activity.png"), dpi=200)


# -----------------------
# FIGURE 4: Local Activity (Punctuated Equilibrium)
# -----------------------
print("Fig. 4: Local activity (punctuated equilibrium)...")
res4 = run_bak_sneppen(N_medium, TRANSIENT, SAMPLE_STEPS // 50,
                       record_local_activity=(0, 10))
activity = res4["local_activity"]
win = 50
smoothed = np.convolve(activity, np.ones(win), mode="same")

plt.figure(figsize=(6, 4.5))
plt.plot(smoothed, drawstyle="steps-mid")
plt.xlabel("Mutation steps")
plt.ylabel(f"Activity in 10-site region (window={win})")
plt.title("Fig. 4 - Punctuated Equilibrium (Local Activity)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_local_activity.png"), dpi=200)


# -----------------------
# FIGURE 5: Avalanche Size Probability Distribution P(s)
# -----------------------
print("Fig. 5: Avalanche size probability distribution P(s)...")
LONG_SAMPLE = 5_000_000
LONG_TRANSIENT = 500_000
res5 = run_bak_sneppen(N_large, LONG_TRANSIENT, LONG_SAMPLE,
                       record_min_series=True)
min_series = res5["min_series"]

active = (min_series < H_THRESHOLD).astype(int)
sizes, count = [], 0
for v in active:
    if v:
        count += 1
    elif count > 0:
        sizes.append(count)
        count = 0
if count > 0:
    sizes.append(count)
sizes = np.array(sizes)
print(f"Total avalanches recorded: {len(sizes)}")

hist5, centers5 = None, None
if len(sizes) > 0:
    bins5 = np.logspace(np.log10(1), np.log10(max(sizes) + 1), 60)
    hist5, edges5 = np.histogram(sizes, bins=bins5, density=True)
    centers5 = np.sqrt(edges5[:-1] * edges5[1:])

    plt.figure(figsize=(6, 4.5))
    plt.loglog(centers5[hist5 > 0], hist5[hist5 > 0], 'o')
    plt.xlabel("Avalanche size s")
    plt.ylabel("P(s)")
    plt.title(f"Fig. 5 - Avalanche Size Distribution (B={H_THRESHOLD})")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_avalanche_probability.png"), dpi=200)


# -----------------------
# FIGURE 6: Effect of Small-World Shortcuts on P(B)
# -----------------------
print("Fig. 6: Effect of shortcuts on P(B)...")
N_shortcut = 1024
steps_shortcut = 300000
p_values_shortcut = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

plt.figure(figsize=(6.5, 4.5))
for p in p_values_shortcut:
    min_series_sc = bak_sneppen_shortcuts(N_shortcut, transient=50000, steps=steps_shortcut, p=p)
    plt.hist(min_series_sc, bins=100, density=True, histtype='step', label=f"p={p}")

plt.axvline(0.67, color='k', ls='--', label="B_c (1D)")
plt.xlabel("Barrier B")
plt.ylabel("P(B)")
plt.title("Fig. 6 - Effect of Shortcuts on Barrier Distribution P(B)")
plt.legend(fontsize=7)
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_shortcut_effect.png"), dpi=200)


# -----------------------
# Power-Law Exponent Fits
# -----------------------
print("\n=== Power-Law Exponent Fits vs. Theoretical Exponents ===")

pi_fit, pi_r2 = fit_power_law(centers1, hist1, xmin=5, xmax=N_large // 8)
pi_pct_diff = 100 * abs(pi_fit - PI_THEORY) / PI_THEORY
print(f"Distance distribution exponent pi = {pi_fit:.3f} (R^2 = {pi_r2:.3f})")
print(f"  Theoretical (1D): pi ~= {PI_THEORY} | Diff: {pi_pct_diff:.1f}%")

if centers5 is not None:
    try:
        tau_xmax = max(sizes) // 20 if max(sizes) // 20 >= 3 else None
        tau_fit, tau_r2 = fit_power_law(centers5, hist5, xmin=2, xmax=tau_xmax)
        tau_pct_diff = 100 * abs(tau_fit - TAU_THEORY) / TAU_THEORY
        print(f"\nAvalanche-size distribution exponent tau = {tau_fit:.3f} (R^2 = {tau_r2:.3f})")
        print(f"  Theoretical (1D): tau ~= {TAU_THEORY} | Diff: {tau_pct_diff:.1f}%")
    except ValueError as e:
        print(f"\nAvalanche-size distribution: fit failed ({e}).")

print("\nAll output figures successfully generated and saved in:", os.path.abspath(OUTPUT_DIR))
plt.show()