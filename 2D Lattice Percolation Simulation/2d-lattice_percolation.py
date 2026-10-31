"""
Consolidated 2D Lattice Percolation & Phase Transition Simulation Engine
-----------------------------------------------------------------------
Combines C-accelerated finite-lattice monte carlo sweeps (scipy.ndimage.label)
with infinite-lattice Leath cluster growth sampling and Clauset MLE tail fitting.
"""

import math
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import label as scipy_label
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# SIMULATION PARAMETERS
# -----------------------------------------------------------------------------
# Finite-Lattice Phase Transition Parameters
L_FINITE = 128                  # Linear lattice dimension L (NxN = L^2 sites)
N_REALIZATIONS = 100            # Realizations per p-value
P_VALUES = np.linspace(0.45, 0.75, 31)

# Infinite-Lattice Leath Sampling Parameters (Critical Exponent tau)
N_LEATH_SAMPLES = 25000         # Number of Leath growth realizations
P_CRITICAL = 0.59274            # Theoretical 2D site percolation threshold p_c
MAX_CLUSTER_SIZE = 250000       # Cluster cutoff limit (giant cluster cap)
LOGBIN_BINS = 35                # Number of logarithmic bins for PDF

RNG = np.random.default_rng(12345)


# -----------------------------------------------------------------------------
# LEATH ALGORITHM (INFINITE LATTICE CLUSTER GROWTH)
# -----------------------------------------------------------------------------
def leath_cluster(p, rng, max_sites=MAX_CLUSTER_SIZE):
    """
    Grows a percolation cluster from the origin on an infinite 2D lattice.
    Returns (cluster_size: int, hit_cutoff: bool).
    """
    visited = {(0, 0): True}
    frontier = deque([(0, 0)])
    size = 1

    while frontier:
        x, y = frontier.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb not in visited:
                if rng.random() < p:
                    visited[nb] = True
                    frontier.append(nb)
                    size += 1
                    if size >= max_sites:
                        return size, True
                else:
                    visited[nb] = False
    return size, False


def sample_leath_clusters(p, n_samples, rng, max_sites=MAX_CLUSTER_SIZE):
    """Executes ensemble sampling of Leath clusters at target probability p."""
    sizes = []
    giants = 0
    t0 = time.time()
    
    for i in range(n_samples):
        size, hit_cutoff = leath_cluster(p, rng, max_sites=max_sites)
        if hit_cutoff:
            giants += 1
            sizes.append(max_sites)
        else:
            sizes.append(size)
            
    print(f"Leath Growth: {n_samples} samples generated in {time.time()-t0:.2f}s "
          f"({giants} giant clusters capped).")
    return np.array(sizes, dtype=int), giants


# -----------------------------------------------------------------------------
# CLASET-STYLE MAXIMUM LIKELIHOOD ESTIMATOR (MLE) FOR POWER-LAW TAILS
# -----------------------------------------------------------------------------
def mle_continuous_tail_alpha(data, smin_candidates=None, min_tail=50):
    """
    Fits continuous power-law tail p(s) ~ s^(-alpha) using Clauset MLE and KS-test.
    Returns (best_smin, alpha, ks_distance, n_tail).
    """
    data = np.array(data, dtype=np.float64)
    data = data[data >= 1]
    if data.size == 0:
        return None, None, None, 0

    if smin_candidates is None:
        uniq = np.unique(data)
        cutoff = np.percentile(data, 95)
        smin_candidates = uniq[uniq < cutoff]

    best = (None, np.inf, None, 0)
    for smin in np.unique(smin_candidates):
        tail = data[data >= smin]
        n = tail.size
        if n < min_tail:
            continue
        denom = np.sum(np.log(tail / smin))
        if denom <= 0:
            continue
        alpha = 1.0 + n / denom
        if alpha <= 1.0:
            continue
            
        tail_sorted = np.sort(tail)
        model_cdf = 1.0 - (tail_sorted / smin) ** (1.0 - alpha)
        emp_cdf = np.arange(1, n + 1) / n
        ks = np.max(np.abs(emp_cdf - model_cdf))
        
        if ks < best[1]:
            best = (int(smin), ks, float(alpha), int(n))
            
    smin, ks, alpha, n = best
    return smin, alpha, ks, n


# -----------------------------------------------------------------------------
# FINITE-LATTICE ANALYSIS (HOSHEN-KOPELMAN / SCIPY)
# -----------------------------------------------------------------------------
def check_percolation(labels):
    """Checks for vertical or horizontal wrapping across lattice boundaries."""
    top, bot = np.unique(labels[0, :]), np.unique(labels[-1, :])
    left, right = np.unique(labels[:, 0]), np.unique(labels[:, -1])
    
    vert = np.intersect1d(top[top > 0], bot[bot > 0])
    hori = np.intersect1d(left[left > 0], right[right > 0])
    return (len(vert) > 0) or (len(hori) > 0)


def run_finite_lattice_sweeps(L, p_values, realizations, rng):
    """Sweeps occupation probability p on an LxL grid to monitor order parameters."""
    mean_largest_frac = []
    mean_second_frac = []
    percolation_prob = []
    all_nonlargest_sizes = []

    t0 = time.time()
    for p in p_values:
        largest_list = []
        second_list = []
        perc_count = 0
        p_nonlargest = []

        for _ in range(realizations):
            occ = rng.random((L, L)) < p
            labels, num_features = scipy_label(occ)
            
            if num_features == 0:
                largest_list.append(0)
                second_list.append(0)
                continue
                
            sizes = np.bincount(labels.ravel())[1:]
            largest = sizes.max()
            nonlargest = sizes[sizes != largest]
            
            largest_list.append(largest)
            second_list.append(nonlargest.mean() if nonlargest.size > 0 else 0)
            p_nonlargest.extend(nonlargest.tolist())
            
            if check_percolation(labels):
                perc_count += 1

        mean_largest_frac.append(np.mean(largest_list) / (L * L))
        mean_second_frac.append(np.mean(second_list) / (L * L))
        percolation_prob.append(perc_count / realizations)
        all_nonlargest_sizes.append(np.array(p_nonlargest))

    print(f"Finite-Lattice Sweeps ({L}x{L}): Completed in {time.time()-t0:.2f}s.")
    return (np.array(mean_largest_frac), np.array(mean_second_frac),
            np.array(percolation_prob), all_nonlargest_sizes)


def log_bin(data, bins=LOGBIN_BINS):
    """Computes log-binned probability density function (PDF)."""
    if len(data) == 0:
        return np.array([]), np.array([])
    smin, smax = max(1, data.min()), data.max()
    if smax <= smin:
        return np.array([smin]), np.array([1.0])
    edges = np.logspace(math.log10(smin), math.log10(smax), bins)
    hist, ed = np.histogram(data, bins=edges, density=True)
    centers = np.sqrt(ed[:-1] * ed[1:])
    return centers, hist


# -----------------------------------------------------------------------------
# MAIN ENGINE EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=================================================================")
    print("  2D LATTICE PERCOLATION & PHASE TRANSITION SIMULATION ENGINE")
    print("=================================================================\n")

    # 1) Run Infinite-Lattice Leath Cluster Growth Sampling
    sizes_seed, giants = sample_leath_clusters(P_CRITICAL, N_LEATH_SAMPLES, RNG)
    sizes_uncapped = sizes_seed[sizes_seed < MAX_CLUSTER_SIZE]

    # 2) Fit Critical Exponent tau via MLE
    smin_cands = np.unique(np.percentile(sizes_uncapped, np.linspace(30, 90, 20)).astype(int))
    smin, alpha_seed, ks, n_tail = mle_continuous_tail_alpha(sizes_uncapped, smin_cands)
    
    tau_est = alpha_seed + 1.0 if alpha_seed else None
    print(f"\n[LEATH MLE FIT RESULTS]")
    print(f" -> Fitted Tail Exponent alpha = {alpha_seed:.4f} (KS = {ks:.4f})")
    print(f" -> Estimated Fisher Exponent tau = alpha + 1 = {tau_est:.4f} (Theoretical 2D tau = 187/91 ~ 2.0549)\n")

    # 3) Run Finite-Lattice Phase Transition Sweeps
    largest_frac, second_frac, perc_prob, all_nl_sizes = run_finite_lattice_sweeps(
        L_FINITE, P_VALUES, N_REALIZATIONS, RNG
    )

    # 4) Estimate Critical Threshold p_c via Interpolation
    pc_idx = np.argmin(np.abs(perc_prob - 0.5))
    p_c_est = P_VALUES[pc_idx]
    print(f"[PHASE TRANSITION METRICS]")
    print(f" -> Estimated Threshold p_c (P_perc = 0.5) ~ {p_c_est:.4f} (Theoretical p_c ~ 0.5927)")

    # -------------------------------------------------------------------------
    # PLOTTING AND VISUALIZATION
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))

    # Subplot 1: Leath Cluster Size PDF (Log-Log Scale)
    ax1 = fig.add_subplot(2, 2, 1)
    centers, hist = log_bin(sizes_uncapped)
    valid = hist > 0
    ax1.loglog(centers[valid], hist[valid], 'ob', markersize=5, label='Empirical Leath PDF')
    
    if alpha_seed and smin:
        fit_x = centers[centers >= smin]
        A_norm = (alpha_seed - 1.0) * (smin ** (alpha_seed - 1.0))
        ax1.loglog(fit_x, A_norm * fit_x ** (-alpha_seed), 'r--', linewidth=2,
                   label=f'MLE Fit: $s^{{-{alpha_seed:.3f}}}$ ($\Rightarrow \\tau={tau_est:.3f}$)')
                   
    ax1.set_xlabel("Cluster Size $s$")
    ax1.set_ylabel("Probability Density $P(s)$")
    ax1.set_title("Critical Cluster Size Distribution $n_s(p_c)$")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # Subplot 2: Order Parameter & Mean Cluster Size vs p
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(P_VALUES, largest_frac, 'o-b', label='Inf. Cluster Fraction $P_\\infty(p)$')
    ax2.set_xlabel("Occupation Probability $p$")
    ax2.set_ylabel("Largest Cluster Fraction $S_{max} / L^2$", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(P_VALUES, second_frac, 's-g', label='Mean Finite Cluster Fraction')
    ax2_twin.set_ylabel("Mean Non-Largest Size / $L^2$", color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    
    ax2.axvline(p_c_est, color='r', linestyle='--', label=f'$p_c \\approx {p_c_est:.4f}$')
    ax2.set_title("Phase Transition Order Parameter & Susceptibility")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Percolation Probability Curve P_perc(p)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(P_VALUES, perc_prob, 'd-r', linewidth=2)
    ax3.axvline(p_c_est, color='k', linestyle='--', label=f'$p_c \\approx {p_c_est:.4f}$')
    ax3.axhline(0.5, color='gray', linestyle=':')
    ax3.set_xlabel("Occupation Probability $p$")
    ax3.set_ylabel("Percolation Probability $P_{perc}(p)$")
    ax3.set_title("Finite-Size Sigmoidal Percolation Response")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Critical Lattice Intensity Map
    ax4 = fig.add_subplot(2, 2, 4)
    occ_map = RNG.random((L_FINITE, L_FINITE)) < p_c_est
    labels_map, _ = scipy_label(occ_map)
    im = ax4.imshow(labels_map % 30, cmap='tab20', interpolation='nearest')
    ax4.set_title(f"2D Lattice Domain Structure at $p \\approx {p_c_est:.4f}$")
    ax4.axis('off')
    
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    output_filename = "2d_percolation_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[EXPORT] Analysis plot saved to '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    main()