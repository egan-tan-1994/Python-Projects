"""
Consolidated linear birth-death process analysis.

Combines three independent methods for solving the SAME per-capita
(linear) birth-death process:
  1. Gillespie stochastic simulation (exact, step-evaluated event-driven Monte Carlo)
  2. Analytical solution via eigendecomposition of the closed 2x2 moment ODE system
  3. Numerical solution via forward-Euler finite-difference integration of the
     truncated master equation (N_max = 400)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Shared parameters
# ============================================================
b = 0.01          # per-capita birth rate
c = 0.005         # per-capita death rate
N0 = 100          # initial population
t_max = 100       # max simulated time
num_sim = 100     # number of Gillespie trajectories
num_time_points = 500
N_max = 400       # truncation bound for numerical state space

common_time = np.linspace(0, t_max, num_time_points)
states = np.arange(N_max + 1)

# ============================================================
# 1. Gillespie stochastic simulation (Step-Evaluated)
# ============================================================
def gillespie_birthdeath(num_sim, t_max, b, c, N0):
    population_trajs = []
    for _ in range(num_sim):
        N = N0
        t = 0.0
        population = [N]
        time = [t]
        while t < t_max:
            birth_rate = b * N
            death_rate = c * N
            total_rate = birth_rate + death_rate
            if total_rate == 0:
                break
            tau = np.random.exponential(1 / total_rate)
            t += tau
            if np.random.rand() < birth_rate / total_rate:
                N += 1
            else:
                N -= 1
            population.append(N)
            time.append(t)
        population_trajs.append((time, population))
    return population_trajs

print("Running Gillespie simulation...")
population_trajs = gillespie_birthdeath(num_sim, t_max, b, c, N0)

gillespie_mean = np.zeros(num_time_points)
gillespie_variance = np.zeros(num_time_points)

# Step-function evaluation of discrete jump process
step_trajectories = []
for time_vec, pop_vec in population_trajs:
    idx = np.searchsorted(time_vec, common_time, side='right') - 1
    idx = np.clip(idx, 0, len(pop_vec) - 1)
    step_pop = np.array(pop_vec)[idx]
    step_trajectories.append(step_pop)
    gillespie_mean += step_pop
    gillespie_variance += step_pop ** 2

gillespie_mean /= num_sim
gillespie_variance = (gillespie_variance / num_sim) - gillespie_mean ** 2

# ============================================================
# 2. Analytical solution via eigendecomposition (Vectorized)
# ============================================================
print("Solving analytically via eigendecomposition of the moment ODE system...")
r = b - c
M = np.array([[r, 0.0],
              [b + c, 2 * r]])
eigvals_m, eigvecs_m = np.linalg.eig(M)
x0 = np.array([N0, N0 ** 2], dtype=float)   # [mean(0), second moment(0)]
coeffs_m = np.linalg.solve(eigvecs_m, x0)

# Vectorized exact moment calculation
x_t = np.real(eigvecs_m @ (coeffs_m[:, None] * np.exp(eigvals_m[:, None] * common_time)))
analytical_mean = x_t[0, :]
analytical_variance = x_t[1, :] - analytical_mean ** 2

# ============================================================
# 3. Finite-difference numerical solution of the master equation
# ============================================================
print("Solving numerically via finite differences...")
dt = 0.02
n_steps = int(t_max / dt)

birth_rates = b * states.astype(float)
birth_rates[-1] = 0.0  # boundary condition at N_max
death_rates = c * states.astype(float)

P = np.zeros(N_max + 1)
P[N0] = 1.0

record_every = max(1, n_steps // num_time_points)
numerical_time_series = []
numerical_mean_series = []
numerical_variance_series = []

for step in range(n_steps + 1):
    if step % record_every == 0:
        m = np.sum(states * P)
        var = np.sum(states ** 2 * P) - m ** 2
        numerical_time_series.append(step * dt)
        numerical_mean_series.append(m)
        numerical_variance_series.append(var)

    inflow_birth = np.zeros(N_max + 1)
    inflow_birth[1:] = birth_rates[:-1] * P[:-1]
    inflow_death = np.zeros(N_max + 1)
    inflow_death[:-1] = death_rates[1:] * P[1:]
    outflow = (birth_rates + death_rates) * P
    dP = inflow_birth + inflow_death - outflow

    P = P + dt * dP
    P = np.clip(P, 0, None)
    P /= P.sum()

numerical_mean = np.interp(common_time, numerical_time_series, numerical_mean_series)
numerical_variance = np.interp(common_time, numerical_time_series, numerical_variance_series)

# ============================================================
# 4. Error metrics
# ============================================================
def rmse(a, ref):
    return np.sqrt(np.mean((a - ref) ** 2))

def pct_error(a, ref):
    return 100 * np.abs(a - ref) / np.abs(ref)

mean_rmse_analytical = rmse(gillespie_mean, analytical_mean)
mean_rmse_numerical = rmse(gillespie_mean, numerical_mean)
var_rmse_analytical = rmse(gillespie_variance, analytical_variance)
var_rmse_numerical = rmse(gillespie_variance, numerical_variance)
mean_rmse_analytical_vs_numerical = rmse(numerical_mean, analytical_mean)
var_rmse_analytical_vs_numerical = rmse(numerical_variance, analytical_variance)

final_mean_pct_err_analytical = pct_error(gillespie_mean[-1], analytical_mean[-1])
final_mean_pct_err_numerical = pct_error(gillespie_mean[-1], numerical_mean[-1])

print("\n=== Error metrics ===")
print(f"Mean population, RMSE:")
print(f"  Gillespie vs analytical : {mean_rmse_analytical:.4f}")
print(f"  Gillespie vs numerical  : {mean_rmse_numerical:.4f}")
print(f"  Analytical vs numerical : {mean_rmse_analytical_vs_numerical:.6f}")
print(f"\nVariance, RMSE:")
print(f"  Gillespie vs analytical : {var_rmse_analytical:.4f}")
print(f"  Gillespie vs numerical  : {var_rmse_numerical:.4f}")
print(f"  Analytical vs numerical : {var_rmse_analytical_vs_numerical:.6f}")
print(f"\nFinal-time (t={t_max}) mean, % error vs Gillespie:")
print(f"  vs analytical : {final_mean_pct_err_analytical:.2f}%")
print(f"  vs numerical  : {final_mean_pct_err_numerical:.2f}%")

# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(12, 7))
for step_pop in step_trajectories:
    plt.plot(common_time, step_pop, color='red', alpha=0.08, drawstyle='steps-post')
plt.xlabel("Time (t)")
plt.ylabel("Population Size N(t)")
plt.title("Gillespie Trajectories (Step-Evaluated)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig1_gillespie_trajectories.png", dpi=150)

plt.figure(figsize=(12, 7))
plt.plot(common_time, gillespie_mean, label="Gillespie (empirical)", color='red')
plt.plot(common_time, analytical_mean, label="Analytical (eigendecomposition)", color='blue', ls='--')
plt.plot(common_time, numerical_mean, label="Numerical (finite difference)", color='green', ls=':')
plt.xlabel("Time (t)")
plt.ylabel("Mean Population Size")
plt.title("Mean Population Size: Method Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig2_mean_comparison.png", dpi=150)

plt.figure(figsize=(12, 7))
plt.plot(common_time, gillespie_variance, label="Gillespie (empirical)", color='red')
plt.plot(common_time, analytical_variance, label="Analytical (eigendecomposition)", color='blue', ls='--')
plt.plot(common_time, numerical_variance, label="Numerical (finite difference)", color='green', ls=':')
plt.xlabel("Time (t)")
plt.ylabel("Population Size Variance")
plt.title("Population Variance: Method Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig3_variance_comparison.png", dpi=150)

print("\nFigures saved successfully.")
plt.show()