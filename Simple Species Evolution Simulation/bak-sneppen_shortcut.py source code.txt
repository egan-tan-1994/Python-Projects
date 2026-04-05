import numpy as np
import matplotlib.pyplot as plt

def bak_sneppen_shortcuts(N=1024, steps=100000, p=0.01):
    """Bak–Sneppen model with shortcuts."""
    B = np.random.rand(N)
    min_series = []

    for _ in range(steps):
        i_min = np.argmin(B)
        min_series.append(B[i_min])

        # Local neighbors
        neighbors = [(i_min - 1) % N, (i_min + 1) % N]

        # Random shortcuts with probability p
        if np.random.rand() < p:
            neighbors.append(np.random.randint(0, N))

        # Update min site + neighbors
        update_sites = [i_min] + neighbors
        B[update_sites] = np.random.rand(len(update_sites))

    return np.array(min_series)

# Example: vary shortcut probability p
N = 1024
steps = 300000
for p in [0.0, 0.001, 0.01, 0.05, 0.08, 0.1, 0.5, 1, 1.5, 2.0]:
    min_series = bak_sneppen_shortcuts(N, steps, p)
    plt.hist(min_series, bins=100, density=True, histtype='step', label=f"p={p}")

plt.axvline(0.67, color='k', ls='--', label="B_c (1D)")
plt.xlabel("Barrier B")
plt.ylabel("P(B)")
plt.title("Effect of shortcuts on P(B)")
plt.legend()
plt.grid(True, ls=":")
plt.show()
