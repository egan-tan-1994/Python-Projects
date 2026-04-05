import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# Global parameters
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
T0 = 0.01  # temperature for real time scaling
H_THRESHOLD = 0.66  # just below B_c ≈ 0.67

# -----------------------
# Bak–Sneppen model
# -----------------------
def run_bak_sneppen(
    N, transient, steps,
    record_positions=False,
    record_barriers=False,
    record_min_series=False,
    record_local_activity=None,
    record_barrier_for_real_time=False
):
    """Simulate Bak–Sneppen dynamics."""
    B = np.random.rand(N)

    def left(i): return (i - 1) % N
    def right(i): return (i + 1) % N

    # --- transient phase (allow system to self-organize)
    for _ in range(transient):
        imin = np.argmin(B)
        B[imin] = np.random.rand()
        B[left(imin)] = np.random.rand()
        B[right(imin)] = np.random.rand()

    start, seg_len = (record_local_activity or (None, None))
    results = dict(
        mut_pos=[] if record_positions else None,
        barriers_samples=[] if record_barriers else None,
        min_series=[] if record_min_series else None,
        local_activity=[] if record_local_activity else None,
        barrier_series=[] if record_barrier_for_real_time else None,
    )

    # --- sampling
    for step in range(steps):
        imin = np.argmin(B)

        if record_positions:
            results["mut_pos"].append(imin)
        if record_min_series:
            results["min_series"].append(B[imin])
        if record_barriers and step % 100 == 0:
            results["barriers_samples"].append(B.copy())
        if record_local_activity:
            results["local_activity"].append(int(start <= imin < start + seg_len))
        if record_barrier_for_real_time:
            results["barrier_series"].append(B[imin])

        # update
        B[imin] = np.random.rand()
        B[left(imin)] = np.random.rand()
        B[right(imin)] = np.random.rand()

    for k, v in results.items():
        if v is not None:
            results[k] = np.array(v)
    return results


# -----------------------
# FIGURE 1: Distance distribution P(x)
# -----------------------
print("Fig. 1: Distance distribution P(x)…")
res1 = run_bak_sneppen(N_large, TRANSIENT, SAMPLE_STEPS // 2, record_positions=True)
pos = res1["mut_pos"]
d = np.abs(np.diff(pos))
d = np.minimum(d, N_large - d)

bins = np.logspace(0, np.log10(N_large // 2), 60)
hist, edges = np.histogram(d, bins=bins, density=True)
centers = np.sqrt(edges[:-1] * edges[1:])

plt.figure()
plt.loglog(centers, hist, 'o')
plt.xlabel("Distance x between successive mutations")
plt.ylabel("P(x)")
plt.title("Fig. 1 – Distance distribution P(x)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_distance_distribution.png"), dpi=200)
plt.show()


# -----------------------
# FIGURE 2: Barrier probability density P(B)
# -----------------------
print("Fig. 2: Barrier probability density P(B)…")
res2 = run_bak_sneppen(N_large, TRANSIENT, SAMPLE_STEPS // 10,
                       record_barriers=True, record_min_series=True)
barriers = np.concatenate(res2["barriers_samples"])
mins = res2["min_series"]

plt.figure()
plt.hist(barriers, bins=150, density=True, alpha=0.6, label="All barriers")
plt.hist(mins, bins=150, density=True, alpha=0.6, label="Min barriers")
plt.axvline(0.67, color="k", ls="--", label="Bc ≈ 0.67")
plt.xlabel("Barrier B")
plt.ylabel("P(B)")
plt.legend()
plt.title("Fig. 2 – Barrier probability density P(B)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_barrier_PB.png"), dpi=200)
plt.show()


# -----------------------
# FIGURE 3: Mutation activity vs count time and real time
# -----------------------
print("Fig. 3: Mutation activity vs count time and real time…")
res3 = run_bak_sneppen(
    N_medium, TRANSIENT, SAMPLE_STEPS // 10,
    record_positions=True, record_barrier_for_real_time=True
)
mutpos = res3["mut_pos"]
barriers = res3["barrier_series"]

# --- Fig. 3(a): activity vs count time ---
window = 1000
num_bins = len(mutpos) // window
segment_start = N_medium // 2 - 5
segment_len = 10
local_counts = []
for i in range(num_bins):
    seg = mutpos[i * window:(i + 1) * window]
    local_counts.append(np.sum((seg >= segment_start) & (seg < segment_start + segment_len)))
local_counts = np.array(local_counts)
count_time = np.arange(num_bins) * window

# --- Fig. 3(b): activity vs accumulated real time ---
durations = np.exp(barriers / T0)
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
plt.title("Fig. 3(a) – Mutation activity vs count time")
plt.grid(True, ls=":")

plt.subplot(1, 2, 2)
plt.plot(rt_centers, rt_counts, drawstyle="steps-mid")
plt.xlabel("Accumulated real time (Σ e^{B/T₀})")
plt.ylabel("Mutation activity (per time bin)")
plt.title("Fig. 3(b) – Mutation activity vs real time")
plt.grid(True, ls=":")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_mutation_activity.png"), dpi=200)
plt.show()


# -----------------------
# FIGURE 4: Local activity (punctuated equilibrium)
# -----------------------
print("Fig. 4: Local activity (punctuated equilibrium)…")
res4 = run_bak_sneppen(N_medium, TRANSIENT, SAMPLE_STEPS // 50,
                       record_local_activity=(0, 10))
activity = res4["local_activity"]
win = 50
smoothed = np.convolve(activity, np.ones(win), mode="same")

plt.figure()
plt.plot(smoothed, drawstyle="steps-mid")
plt.xlabel("Mutation steps")
plt.ylabel(f"Activity in 10-site region (window={win})")
plt.title("Fig. 4 – Punctuated equilibrium (local activity)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_local_activity.png"), dpi=200)
plt.show()


# -----------------------
# FIGURE 5: Avalanche size probability distribution P(s)
# -----------------------
print("Fig. 5: Avalanche size probability distribution P(s)…")

LONG_SAMPLE = 5_000_000
LONG_TRANSIENT = 500_000
res5 = run_bak_sneppen(N_large, LONG_TRANSIENT, LONG_SAMPLE,
                       record_min_series=True)
min_series = res5["min_series"]

# --- Identify avalanches ---
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

if len(sizes) == 0:
    print("⚠ No avalanches detected. Adjust H_THRESHOLD or increase SAMPLE_STEPS.")
else:
    bins = np.logspace(np.log10(1), np.log10(max(sizes) + 1), 60)
    hist, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])

    plt.figure(figsize=(6, 4))
    plt.loglog(centers, hist, 'o')
    plt.xlabel("Avalanche size s")
    plt.ylabel("P(s)")
    plt.title(f"Fig. 5 – Avalanche size probability distribution (B={H_THRESHOLD})")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_avalanche_probability.png"), dpi=200)
    plt.show()

print("\n✅ All five figures generated in:", os.path.abspath(OUTPUT_DIR))
