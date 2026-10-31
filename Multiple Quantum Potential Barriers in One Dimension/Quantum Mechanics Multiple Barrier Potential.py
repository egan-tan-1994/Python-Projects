import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

# Physical constants
hbar = 1.054571817e-34     # J s
h = 6.62607015e-34        # J s
c = 2.99792458e8          # m/s
e = 1.602176634e-19       # C
m_e = 9.10938356e-31      # kg
nm = 1e-9                 # m conversion factor

# ---------------------------
# Wave number (quantum barrier model)
# ---------------------------
def k(E_eV, V_eV, m_eff):
    E_eV = np.asarray(E_eV, dtype=float)
    eps = 1e-12
    # Safeguard against exact E = V numerical singularity (k = 0)
    E_eV_adj = np.where(np.abs(E_eV - V_eV) < eps, V_eV + eps, E_eV)
    
    E = np.maximum(E_eV_adj, 1e-12) * e
    V = V_eV * e
    m = m_eff * m_e

    k_above = np.sqrt(2 * m * np.maximum(E - V, 0.0) + 0j) / hbar
    k_below = 1j * np.sqrt(2 * m * np.maximum(V - E, 0.0) + 0j) / hbar
    
    res = np.where(E >= V, k_above, k_below)
    return res.item() if res.ndim == 0 else res

# ---------------------------
# Transfer Matrix Components
# ---------------------------
def interface_matrix(k1, k2):
    return 0.5 * np.array([[1 + k2/k1, 1 - k2/k1],
                           [1 - k2/k1, 1 + k2/k1]])

def propagation_matrix(k_val, d_m):
    return np.array([[np.exp(1j * k_val * d_m), 0],
                     [0, np.exp(-1j * k_val * d_m)]])

# ---------------------------
# Total Transfer Matrix
# ---------------------------
def total_transfer_matrix(E, V0, a_nm, b_nm, N, m_eff):
    a = a_nm * nm
    b = b_nm * nm

    k_left = k(E, 0, m_eff)
    k_barrier = k(E, V0, m_eff)
    k_well = k(E, 0, m_eff)
    k_right = k(E, 0, m_eff)

    # First barrier interface and propagation
    M = interface_matrix(k_left, k_barrier)
    M = propagation_matrix(k_barrier, a) @ M

    # Loop through remaining N-1 (Well + Barrier) pairs
    for _ in range(N - 1):
        M = interface_matrix(k_barrier, k_well) @ M
        M = propagation_matrix(k_well, b) @ M
        M = interface_matrix(k_well, k_barrier) @ M
        M = propagation_matrix(k_barrier, a) @ M

    # Exit last barrier into right region
    M = interface_matrix(k_barrier, k_right) @ M

    return M

# ---------------------------
# Transmission / Reflection
# ---------------------------
def transmission_reflection(E, V0, a_nm, b_nm, N, m_eff):
    M = total_transfer_matrix(E, V0, a_nm, b_nm, N, m_eff)

    t = 1.0 / M[0, 0]
    r = M[1, 0] / M[0, 0]

    T = np.abs(t)**2
    R = np.abs(r)**2

    return T, R

# ---------------------------
# Electron Energy <-> de Broglie Wavelength
# ---------------------------
def energy_to_debroglie_wavelength_nm(E_eV, m_eff):
    # Fixed: Replaced built-in max() with np.maximum() to support vector array operations
    E_eV = np.maximum(E_eV, 1e-12)
    E = E_eV * e
    m = m_eff * m_e
    p = np.sqrt(2 * m * E)
    wavelength = h / p
    return wavelength / nm

def debroglie_wavelength_nm_to_energy_eV(lambda_nm, m_eff):
    # Fixed: Replaced built-in max() with np.maximum() to support vector array operations
    lambda_nm = np.maximum(lambda_nm, 1e-12)
    lam = lambda_nm * nm
    m = m_eff * m_e
    p = h / lam
    E = (p ** 2) / (2 * m)
    return E / e

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    try:
        if sys.stdin.isatty():
            V0 = float(input("Enter barrier height V0 (eV): "))
            a_nm = float(input("Enter barrier width a (nm): "))
            b_nm = float(input("Enter well width b (nm): "))
            N = int(input("Enter number of barriers N: "))
            m_eff = float(input("Enter effective mass (m*/m_e): "))
        else:
            V0, a_nm, b_nm, N, m_eff = 1.0, 1.0, 1.0, 2, 1.0
            print(f"Non-interactive run using default parameters: V0={V0}eV, a={a_nm}nm, b={b_nm}nm, N={N}, m_eff={m_eff}")
    except ValueError:
        print("Invalid input.")
        return

    # Energy sweep
    E_values = np.linspace(0.5, 5.0, 1000)

    T_E = []
    R_E = []

    for E in E_values:
        T, R = transmission_reflection(E, V0, a_nm, b_nm, N, m_eff)
        T_E.append(T)
        R_E.append(R)

    # De Broglie wavelength mapping
    wavelengths_nm = energy_to_debroglie_wavelength_nm(E_values, m_eff)

    # Write raw data
    filename = "transmission_reflection_data.txt"
    with open(filename, "w") as f:
        f.write("=== Transmission / Reflection vs Energy ===\n")
        f.write(f"{'Energy (eV)':>12} {'Transmission':>15} {'Reflection':>15}\n")
        for E, T, R in zip(E_values, T_E, R_E):
            f.write(f"{E:12.4f} {T:15.6f} {R:15.6f}\n")

        f.write("\n=== Transmission / Reflection vs Electron de Broglie Wavelength ===\n")
        f.write(f"{'Wavelength (nm)':>16} {'Transmission':>15} {'Reflection':>15}\n")
        for lam, T, R in zip(wavelengths_nm, T_E, R_E):
            f.write(f"{lam:16.4f} {T:15.6f} {R:15.6f}\n")

    print(f"\nRaw data successfully saved to '{filename}'")

    # Plots
    # T & R vs Energy
    plt.figure(figsize=(10, 7))
    plt.plot(E_values, T_E, label="Transmission Coefficient")
    plt.plot(E_values, R_E, label="Reflection Coefficient")
    plt.axvline(V0, linestyle="--", color="r", label=f"Barrier Height $V_0$ = {V0:.3f} eV")
    plt.xlabel("Electron Energy (eV)")
    plt.ylabel("Transmission or Reflection Probability")
    plt.title("T and R vs Electron Energy")
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.25))
    plt.legend()
    plt.grid(True)
    plt.savefig("fig_T_R_vs_energy.png", dpi=150)
    plt.close()

    # T & R vs Wavelength
    plt.figure(figsize=(10, 7))
    plt.plot(wavelengths_nm, T_E, label="Transmission Coefficient")
    plt.plot(wavelengths_nm, R_E, label="Reflection Coefficient")
    plt.xlabel("Electron de Broglie Wavelength (nm)")
    plt.ylabel("Transmission or Reflection Probability")
    plt.title("T and R vs Electron de Broglie Wavelength")
    plt.legend()
    plt.grid(True)
    plt.savefig("fig_T_R_vs_wavelength.png", dpi=150)
    plt.close()

    # Wavelength vs Energy
    plt.figure(figsize=(10, 7))
    plt.plot(E_values, wavelengths_nm)
    plt.xlabel("Electron Energy (eV)")
    plt.ylabel("de Broglie Wavelength (nm)")
    plt.title("Electron de Broglie Wavelength vs Energy")
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.25))
    plt.grid(True)
    plt.savefig("fig_wavelength_vs_energy.png", dpi=150)
    plt.close()

    print("Figures saved: fig_T_R_vs_energy.png, fig_T_R_vs_wavelength.png, fig_wavelength_vs_energy.png")

if __name__ == "__main__":
    main()