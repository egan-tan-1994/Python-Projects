import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import math
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# optional SciPy imports (required)
from scipy.optimize import curve_fit
try:
    from scipy.ndimage import label as scipy_label
except Exception as e:
    raise ImportError("scipy.ndimage.label is required for this script. Install scipy and try again.") from e

# ---------------------------
# USER PARAMETERS (tweak)
# ---------------------------
# Leath sampling (for power-law)
n_leath = 20000                # number of cluster-growth samples (seed-based). Increase for better tails.
p_for_pdf = 0.5927             # probability to sample around (expected p_c)
max_cluster_size = 200000      # cap : if cluster grows past this, treat as "giant" and stop
logbin_bins = 40               # number of log bins for plotting PDF

# Small finite-lattice simulations (for visuals & mean curves)
L_snap = 16384                  # lattice size for snapshots & percolation probability curves
n_realizations_small = 1000     # realizations per p for finite-lattice statistics
p_values_small = np.linspace(0.58, 0.60, 25)  # narrow sweep around p_c

# Misc
rng = np.random.default_rng(123456)
show_plots = True

# ---------------------------
# Leath cluster-growth algorithm
# ---------------------------
def leath_cluster(p, rng, max_sites=max_cluster_size):
    """
    Grow a cluster from origin (assumed occupied) on an infinite lattice.
    Returns (size:int, hit_cutoff:bool).
    """
    visited = {}             # coord -> True (occupied) or False (explored empty)
    frontier = deque()
    origin = (0, 0)
    visited[origin] = True
    frontier.append(origin)
    size = 1

    while frontier:
        x, y = frontier.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (x+dx, y+dy)
            if nb in visited:
                continue
            # decide occupancy
            if rng.random() < p:
                visited[nb] = True
                frontier.append(nb)
                size += 1
                if size >= max_sites:
                    return size, True
            else:
                visited[nb] = False
    return size, False

# ---------------------------
# Continuous-tail MLE (Clauset-style)
# ---------------------------
def mle_continuous_tail_alpha(data, smin_candidates=None, min_tail=50):
    """
    Fit continuous power-law tail p(s) ~ s^{-alpha} using Clauset-style MLE and KS.
    Returns (best_smin, alpha, ks, n_tail) or (None, None, None, 0) on failure.
    """
    data = np.array(data, dtype=np.float64)
    data = data[data >= 1]
    if data.size == 0:
        return None, None, None, 0

    if smin_candidates is None:
        uniq = np.unique(data)
        # choose candidate smins up to 95th percentile
        try:
            cutoff = np.percentile(data, 95)
            smin_candidates = uniq[uniq < cutoff]
        except Exception:
            smin_candidates = uniq
        if smin_candidates.size == 0:
            smin_candidates = uniq

    best = (None, np.inf, None, 0)  # smin, ks, alpha, n
    for smin in np.unique(smin_candidates):
        tail = data[data >= smin]
        n = tail.size
        if n < min_tail:
            continue
        # MLE
        denom = np.sum(np.log(tail / smin))
        if denom <= 0:
            continue
        alpha = 1.0 + n / denom
        if alpha <= 1.0:
            continue
        tail_sorted = np.sort(tail)
        model_cdf = 1.0 - (tail_sorted / smin)**(1.0 - alpha)
        emp_cdf = np.arange(1, n+1) / n
        ks = np.max(np.abs(emp_cdf - model_cdf))
        if ks < best[1]:
            best = (int(smin), ks, float(alpha), int(n))
    smin, ks, alpha, n = best
    return smin, alpha, ks, n

# ---------------------------
# Leath sampling wrapper
# ---------------------------
def sample_leath_clusters(p, n_samples, rng, max_sites=max_cluster_size, progress_step=5000):
    """
    Sample n_samples Leath clusters (conditional on seed occupied).
    Returns numpy array of sizes and count of capped (giant) clusters.
    """
    sizes = []
    giants = 0
    t0 = time.time()
    for i in range(n_samples):
        # ensure seed is occupied (we want conditional on occupied seed)
        # this loops expected 1/p trials — fine if p ~ p_c ~ 0.59
        while rng.random() >= p:
            pass
        size, hit_cutoff = leath_cluster(p, rng, max_sites=max_sites)
        if hit_cutoff:
            giants += 1
            sizes.append(max_sites)
        else:
            sizes.append(size)
        if progress_step and ((i+1) % progress_step == 0):
            print(f"Leath: {i+1}/{n_samples} samples, elapsed {time.time()-t0:.1f}s")
    total = time.time() - t0
    print(f"Leath sampling finished: {n_samples} samples in {total:.1f}s; giants (capped) = {giants}")
    return np.array(sizes, dtype=int), giants

# ---------------------------
# Log-binning helper
# ---------------------------
def log_bin(data, bins=logbin_bins):
    if len(data) == 0:
        return np.array([]), np.array([])
    smin = max(1, data.min())
    smax = data.max()
    if smax <= smin:
        return np.array([smin]), np.array([1.0])
    edges = np.logspace(math.log10(smin), math.log10(smax), bins)
    hist, ed = np.histogram(data, bins=edges, density=True)
    centers = np.sqrt(ed[:-1] * ed[1:])
    return centers, hist

# ---------------------------
# Finite-lattice stats (small L)
# ---------------------------
def get_percolation_status(labels):
    top = np.unique(labels[0, :])
    bottom = np.unique(labels[-1, :])
    left = np.unique(labels[:, 0])
    right = np.unique(labels[:, -1])
    vert = np.intersect1d(top, bottom)
    hori = np.intersect1d(left, right)
    vert = vert[vert != 0]
    hori = hori[hori != 0]
    return (len(vert) > 0) or (len(hori) > 0)

def finite_lattice_stats(L, p_values, n_realizations, rng):
    mean_largest = []
    mean_nonlargest = []
    percolation_prob = []
    for p in p_values:
        largest_list = []
        nonlargest_avgs = []
        perc_count = 0
        for _ in range(n_realizations):
            lat = rng.random((L, L)) < p
            labels, num = scipy_label(lat)
            if num == 0:
                largest_list.append(0)
                nonlargest_avgs.append(0)
                continue
            sizes = np.bincount(labels.ravel())[1:]
            largest = sizes.max()
            nonlargest = sizes[sizes != largest]
            largest_list.append(largest)
            nonlargest_avgs.append(np.mean(nonlargest) if nonlargest.size > 0 else 0)
            if get_percolation_status(labels):
                perc_count += 1
        mean_largest.append(np.mean(largest_list))
        mean_nonlargest.append(np.mean(nonlargest_avgs))
        percolation_prob.append(perc_count / n_realizations)
    return np.array(mean_largest), np.array(mean_nonlargest), np.array(percolation_prob)

# ---------------------------
# Power-law fit helper (for plotting)
# ---------------------------
def powerlaw_model(x, A, alpha):
    return A * x**(-alpha)

# ---------------------------
# Main workflow
# ---------------------------
def main():
    print("Starting Leath-based large-L power-law estimation")
    print(f"Parameters: n_leath={n_leath}, p_for_pdf={p_for_pdf}, max_cluster_size={max_cluster_size}")
    t_start = time.time()

    # 1) Leath sampling
    sizes_seed, giants = sample_leath_clusters(p_for_pdf, n_leath, rng, max_sites=max_cluster_size)

    # 2) Prepare data for fit
    sizes_uncapped = sizes_seed[sizes_seed < max_cluster_size]
    print(f"Uncapped clusters: {sizes_uncapped.size}, capped: {giants}")

    tau_est = None
    alpha_seed = None
    smin = None
    if sizes_uncapped.size >= 100:
        centers, hist = log_bin(sizes_uncapped, bins=logbin_bins)
        # select candidate smin percentiles
        try:
            smin_cands = np.unique(np.percentile(sizes_uncapped, np.linspace(40, 95, 20)).astype(int))
        except Exception:
            smin_cands = np.unique(sizes_uncapped)
        smin_try, alpha_try, ks_try, n_tail = mle_continuous_tail_alpha(sizes_uncapped, smin_cands, min_tail=50)
        if smin_try is None:
            # fallback: attempt curve_fit on upper log-bins
            mask = centers > np.percentile(centers, 60) if centers.size > 0 else np.array([], dtype=bool)
            if mask.sum() >= 5 and np.any(hist[mask] > 0):
                try:
                    popt, _ = curve_fit(powerlaw_model, centers[mask], hist[mask], p0=[1.0, 1.5])
                    alpha_try = popt[1]
                    smin_try = int(centers[mask].min())
                    n_tail = sizes_uncapped[sizes_uncapped >= smin_try].size
                    print("Fallback curve_fit succeeded.")
                except Exception as e:
                    print("Fallback curve_fit failed:", e)
                    alpha_try = None
        if alpha_try is not None:
            alpha_seed = alpha_try
            smin = smin_try
            tau_est = alpha_seed + 1.0
            print(f"alpha_seed = {alpha_seed:.4f}, smin = {smin}, n_tail = {n_tail}, tau_est = {tau_est:.4f}")
        else:
            print("Could not obtain a stable tail fit; consider increasing n_leath or max_cluster_size.")
    else:
        print("Not enough uncapped clusters for tail fitting. Increase n_leath or max_cluster_size.")

    t_mid = time.time()
    print(f"Leath sampling + tail analysis time: {t_mid - t_start:.1f}s")

    # 3) Small finite-lattice stats for snapshots & mean curves
    print("Running small finite-lattice simulations for snapshots & metrics...")
    mean_largest, mean_nonlargest, perc_prob = finite_lattice_stats(L_snap, p_values_small, n_realizations_small, rng)
    t_small = time.time()
    print(f"Small-lattice work time: {t_small - t_mid:.1f}s")

    # 4) Estimate p_c by locating P_perc = 0.5 via interpolation
    p_c_est = None
    try:
        perc_arr = np.array(perc_prob)
        p_vals = np.array(p_values_small)
        # check crossing
        if np.any(perc_arr <= 0.5) and np.any(perc_arr >= 0.5):
            # linear interpolation of crossing
            # find interval where sign changes relative to 0.5
            idx = np.where(np.diff(np.sign(perc_arr - 0.5)) != 0)[0]
            if idx.size > 0:
                i = idx[0]
                p_c_est = p_vals[i] + (0.5 - perc_arr[i])*(p_vals[i+1]-p_vals[i])/(perc_arr[i+1]-perc_arr[i])
            else:
                # fallback: pick p with percolation prob closest to 0.5
                p_c_est = p_vals[np.argmin(np.abs(perc_arr - 0.5))]
        else:
            p_c_est = p_vals[np.argmin(np.abs(perc_arr - 0.5))]
    except Exception as e:
        print("Could not compute p_c estimate:", e)
        p_c_est = None

    if p_c_est is not None:
        print(f"\nEstimated p_c (from small-L percolation prob) ≈ {p_c_est:.6f}\n")
    else:
        print("\nEstimated p_c could not be determined from the small-L data.\n")

    # 5) Plot results (if requested)
    if show_plots:
        # log-binned PDF & fit
        if sizes_uncapped.size > 0:
            centers, hist = log_bin(sizes_uncapped, bins=logbin_bins)
            plt.figure(figsize=(7,5))
            plt.loglog(centers, hist, 'o', label='log-binned empirical (seed samples)')
            if alpha_seed is not None and smin is not None:
                fit_x = centers[centers >= smin]
                # normalization approx
                A_norm = (alpha_seed - 1.0) * (smin**(alpha_seed - 1.0)) if alpha_seed > 1 else 1.0
                plt.loglog(fit_x, A_norm * fit_x**(-alpha_seed), '-', label=f'fit seed s^(-{alpha_seed:.3f}) -> tau={tau_est:.3f}')
            plt.xlabel('Cluster size s')
            plt.ylabel('PDF (seed-sampled)')
            plt.title(f'Cluster-size PDF (Leath) at p={p_for_pdf:.4f}')
            plt.legend(); plt.grid(True, which='both', ls='--', alpha=0.4); plt.tight_layout(); plt.show()

        # snapshots
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        p_snap = [p_for_pdf - 0.02, p_for_pdf, p_for_pdf + 0.02]
        titles = ['Subcritical', 'Near-critical', 'Supercritical']
        for ax, p, ttl in zip(axs, p_snap, titles):
            lat = rng.random((L_snap, L_snap)) < p
            labels, _ = scipy_label(lat)
            im = ax.imshow(labels % 20, cmap='tab20', interpolation='nearest')
            ax.set_title(f"{ttl} p={p:.4f}")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("x (site location)"); ax.set_ylabel("y (site location)")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax)
        plt.tight_layout(); plt.show()

        # mean metrics
        plt.figure(figsize=(7,5))
        plt.plot(p_values_small, mean_largest/(L_snap*L_snap), 'o-', label='Mean largest cluster fraction (small L)')
        plt.plot(p_values_small, mean_nonlargest/(L_snap*L_snap), 's-', label='Mean non-largest cluster fraction (small L)')
        if p_c_est is not None:
            plt.axvline(p_c_est, color='r', linestyle='--', label=f"Estimated p_c={p_c_est:.6f}")
        plt.xlabel('p'); plt.ylabel('Fraction of lattice'); plt.title('Mean cluster metrics vs p (small lattice)')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        # percolation probability
        plt.figure(figsize=(6,4))
        plt.plot(p_values_small, perc_prob, 'o-', label='P_perc (small L)')
        if p_c_est is not None:
            plt.axvline(p_c_est, color='r', linestyle='--', label=f"Estimated p_c={p_c_est:.6f}")
        plt.xlabel('p'); plt.ylabel('Percolation probability'); plt.title('Percolation probability (small lattice)')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    t_end = time.time()
    print(f"Total script time: {t_end - t_start:.1f}s")

    # return diagnostics
    out = {
        'sizes_seed': sizes_seed,
        'sizes_uncapped': sizes_uncapped if 'sizes_uncapped' in locals() else None,
        'alpha_seed': alpha_seed,
        'smin': smin,
        'tau_est': tau_est,
        'mean_largest_small': mean_largest,
        'mean_nonlargest_small': mean_nonlargest,
        'percolation_prob_small': perc_prob,
        'p_c_est': p_c_est
    }
    return out

# ---------------------------
# Run as script
# ---------------------------
if __name__ == '__main__':
    results = main()
    # print brief summary
    print("\nSUMMARY:")
    print("tau_est =", results.get('tau_est'))
    print("p_c_est =", results.get('p_c_est'))
