"""
Simulates a non-relativistic particle in a 1D Infinite Potential Well.
Calculates energy eigenvalues, wavefunctions, probability densities,
and Gauss-Legendre quadrature expectation values/uncertainties for quantum states n=1..1000.
"""

from functools import lru_cache
import sys
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

# Physical constants
hbar = 1.054571817e-34      # Reduced Planck constant (J s)
e = 1.602176634e-19         # Elementary charge (C)
m_e = 9.10938356e-31        # Electron rest mass (kg)
nm = 1e-9                   # Conversion factor from nm to m


# Gaussian Quadrature

@lru_cache(maxsize=None)
def legendre_nodes_weights(order):
    """Returns cached Gauss-Legendre nodes and weights."""
    return roots_legendre(order)


def gauss_integrate(func, lower, upper, args=(), order=100):
    """Integrates a vectorized function using Gauss-Legendre quadrature."""
    nodes, weights = legendre_nodes_weights(order)
    x = ((upper - lower) / 2 * nodes + (upper + lower) / 2)
    values = func(x, *args)
    return ((upper - lower) / 2 * np.sum(weights * values))


def find_optimal_nodes(func, lower, upper, exact_value, args=(), scale=1.0,
                       rtol=1e-12, atol=0, n_min=2, n_max=10000, consecutive=3):
    """Finds minimal Gauss-Legendre order achieving target tolerance."""
    reference_scale = max(abs(exact_value), abs(scale), np.finfo(float).tiny)
    tolerance = atol + rtol * reference_scale
    
    successful_orders = []
    
    for order in range(n_min, n_max + 1):
        result = gauss_integrate(func, lower, upper, args=args, order=order)
        absolute_error = abs(result - exact_value)
        
        if np.isfinite(absolute_error) and absolute_error <= tolerance:
            successful_orders.append((order, result, absolute_error))
            if len(successful_orders) >= consecutive:
                optimal_order, optimal_result, error = successful_orders[-consecutive]
                scaled_error = error / reference_scale
                return (optimal_result, optimal_order, error, scaled_error)
        else:
            successful_orders.clear()
            
    raise RuntimeError(
        f"Tolerance not achieved for orders {n_min} through {n_max}. "
        "Increase n_max or relax tolerance."
    )


def calculate_variance(second_moment, mean, scale_squared, rtol=1e-12):
    """Computes variance with numerical threshold clamping."""
    second_moment = float(np.real(second_moment))
    variance = second_moment - abs(mean)**2

    numerical_tolerance = (
        20 * rtol * max(
            abs(second_moment),
            abs(mean)**2,
            abs(scale_squared),
            np.finfo(float).tiny
        )
    )

    if abs(variance) <= numerical_tolerance:
        return 0.0

    if variance < 0:
        raise ValueError(f"Significantly negative variance: {variance}")

    return variance


# Wave number & Energy Eigenvalues
def k_n(n, a):
    return (n * np.pi) / a


def E_n(n, a, m_eff):
    k = k_n(n, a)
    return ((k * hbar) ** 2) / (2 * m_eff * e)


# Energy Eigenstates & Derivatives (Normalized internally)
def psi_n(n, x, a):
    A = np.sqrt(2 / a)
    k = k_n(n, a)
    return A * np.sin(k * x)


def d_psi_n(n, x, a):
    A = np.sqrt(2 / a)
    k = k_n(n, a)
    return (A * k) * np.cos(k * x)


def d2_psi_n(n, x, a):
    A = np.sqrt(2 / a)
    k = k_n(n, a)
    return (-A) * (k ** 2) * np.sin(k * x)


def d4_psi_n(n, x, a):
    A = np.sqrt(2 / a)
    k = k_n(n, a)
    return A * (k ** 4) * np.sin(k * x)


def phi_n(n, x, a):
    psi = psi_n(n, x, a)
    return np.conjugate(psi) * psi


# Observables
def avg_pos(x, n, a):
    psi = psi_n(n, x, a)
    return x * np.conjugate(psi) * psi


def avg_square_pos(x, n, a):
    psi = psi_n(n, x, a)
    return x**2 * np.conjugate(psi) * psi


def avg_mom(x, n, a):
    psi = psi_n(n, x, a)
    d_psi = d_psi_n(n, x, a)
    coeff = hbar / 1j
    return coeff * np.conjugate(psi) * d_psi


def avg_square_mom(x, n, a):
    psi = psi_n(n, x, a)
    d2_psi = d2_psi_n(n, x, a)
    coeff = hbar / 1j
    return coeff**2 * np.conjugate(psi) * d2_psi


def avg_Hamil(x, n, a, m_eff):
    psi = psi_n(n, x, a)
    d2_psi = d2_psi_n(n, x, a)
    coeff = -hbar**2 / (2 * m_eff)
    return coeff * np.conjugate(psi) * d2_psi


def avg_square_Hamil(x, n, a, m_eff):
    psi = psi_n(n, x, a)
    d4_psi = d4_psi_n(n, x, a)
    coeff = (hbar**2 / (2 * m_eff))**2
    return coeff * np.conjugate(psi) * d4_psi


def main():
    try:
        if sys.stdin.isatty():
            m_input = float(input("Enter particle effective mass (m_eff/m_e): "))
            L_input = float(input("Enter well width (in nm): "))
        else:
            m_input, L_input = 1.0, 1.0
            print(f"Non-interactive run detected. Using defaults: m_eff/m_e={m_input}, L={L_input} nm")

        if m_input <= 0 or L_input <= 0:
            raise ValueError("Mass and well width must be positive numbers greater than 0.")
    except ValueError as err:
        print(f"Invalid input: {err}")
        return

    m_eff = m_input * m_e
    a = L_input * nm
    A = np.sqrt(2 / a)
    x_plot = np.linspace(0, a, 1000)

    # 1. Wave Function Plot
    plt.figure(figsize=(12, 6))
    for n in range(1, 5):
        psi = psi_n(n, x_plot, a)
        plt.plot(x_plot / nm, psi / A, label=f"n={n}")
    plt.xlabel("Particle Position (nm)")
    plt.ylabel(r"Wave Function $\psi_n(x)$ / A")
    plt.title("Infinite Quantum Well Wavefunction Plots")
    plt.legend()
    plt.grid(True)
    plt.savefig("Wave_Function_Plots.png", dpi=300)
    plt.close()

    # 2. Probability Density Plot
    plt.figure(figsize=(12, 6))
    for n in range(1, 5):
        phi = phi_n(n, x_plot, a)
        plt.plot(x_plot / nm, phi / A**2, label=f"n={n}")
    plt.xlabel("Particle Position (nm)")
    plt.ylabel(r"Probability Density Function $\phi_n(x)$ / $A^2$")
    plt.title("Infinite Quantum Well Probability Density Function Plots")
    plt.legend()
    plt.grid(True)
    plt.savefig("Probability_Density_Function_Plots.png", dpi=300)
    plt.close()

    # 3. Energy Eigenvalues Scatter Plot
    plt.figure(figsize=(12, 6))
    for n in range(1, 11):
        E = E_n(n, a, m_eff)
        plt.scatter(n, E, label=f"n={n}")
    plt.xlabel("Quantum Number n")
    plt.ylabel("Particle Energy (eV)")
    plt.title("Energy Eigenvalues Scatter Plot")
    plt.grid(True)
    plt.savefig("Energy_Eigenvalues.png", dpi=300)
    plt.close()

    # Summary table generation
    n_E = np.arange(1, 1001)
    rtol = 1e-12
    n_max = 10000
    summary = []

    for n in n_E:
        k = k_n(n, a)
        E = (hbar * k) ** 2 / (2 * m_eff)

        x_avg_exact = a / 2
        x_avg_squared_exact = a ** 2 * (1 / 3 - 1 / (2 * (n * np.pi) ** 2))
        p_avg_exact = 0.0
        p_avg_squared_exact = (hbar * k) ** 2
        H_avg_exact = E
        H_avg_squared_exact = E ** 2

        x_avg, Nx, err_x, rel_x = find_optimal_nodes(avg_pos, 0, a, x_avg_exact, args=(n, a), scale=a, rtol=rtol, n_max=n_max)
        x_avg = float(np.real(x_avg))

        x_avg_squared, Nx2, err_x2, rel_x2 = find_optimal_nodes(avg_square_pos, 0, a, x_avg_squared_exact, args=(n, a), scale=a**2, rtol=rtol, n_max=n_max)
        x_avg_squared = float(np.real(x_avg_squared))

        p_avg, Np, err_p, rel_p = find_optimal_nodes(avg_mom, 0, a, p_avg_exact, args=(n, a), scale=hbar/a, atol=1e-10 * hbar/a, rtol=rtol, n_max=n_max)
        p_avg = float(np.real(p_avg))

        p_avg_squared, Np2, err_p2, rel_p2 = find_optimal_nodes(avg_square_mom, 0, a, p_avg_squared_exact, args=(n, a), scale=(hbar/a)**2, rtol=rtol, n_max=n_max)
        p_avg_squared = float(np.real(p_avg_squared))

        v_avg_squared = p_avg_squared / (m_eff ** 2)
        v_rms = np.sqrt(v_avg_squared)

        H_avg, NH, err_H, rel_H = find_optimal_nodes(avg_Hamil, 0, a, H_avg_exact, args=(n, a, m_eff), scale=E, rtol=rtol, n_max=n_max)
        H_avg = float(np.real(H_avg))

        H_avg_squared, NH2, err_H2, rel_H2 = find_optimal_nodes(avg_square_Hamil, 0, a, H_avg_squared_exact, args=(n, a, m_eff), scale=E**2, rtol=rtol, n_max=n_max)
        H_avg_squared = float(np.real(H_avg_squared))

        x_std = np.sqrt(calculate_variance(x_avg_squared, x_avg, a**2, rtol))
        p_std = np.sqrt(calculate_variance(p_avg_squared, p_avg, (hbar/a)**2, rtol))
        H_std = np.sqrt(calculate_variance(H_avg_squared, H_avg, E**2, rtol))

        summary.append({
            "n": n,
            "Mean position <x> (nm)": x_avg / nm,
            "Nodes for <x>": Nx,
            "Scaled error for <x>": rel_x,
            "Mean-square position <x^2> (nm^2)": x_avg_squared / nm ** 2,
            "Nodes for <x^2>": Nx2,
            "Scaled error for <x^2>": rel_x2,
            "Position Uncertainty (nm)": x_std / nm,
            "Nodes for Delta x": max(Nx, Nx2),
            "Mean momentum <p> (kg m/s)": p_avg,
            "Nodes for <p>": Np,
            "Scaled error for <p>": rel_p,
            "Mean-square momentum <p^2> (kg^2 m^2/s^2)": p_avg_squared,
            "Nodes for <p^2>": Np2,
            "Scaled error for <p^2>": rel_p2,
            "Momentum Uncertainty (kg m/s)": p_std,
            "Nodes for Delta p": max(Np, Np2),
            "Delta x times Delta p (kg m^2/s)": x_std * p_std,
            "Hbar / 2": hbar / 2,
            "Mean-square velocity <v^2> (m^2/s^2)": v_avg_squared,
            "Root-mean-square (RMS) velocity (m/s)": v_rms,
            "Mean Hamiltonian <H> (eV)": H_avg / e,
            "Nodes for <H>": NH,
            "Scaled error for <H>": rel_H,
            "Mean-square Hamiltonian <H^2> (eV^2)": H_avg_squared / e ** 2,
            "Nodes for <H^2>": NH2,
            "Scaled error for <H^2>": rel_H2,
            "Hamiltonian uncertainty (eV)": H_std / e,
            "Nodes for Delta H": max(NH, NH2),
        })

    pd.DataFrame({"Effective mass m_eff/m_e": [m_input], "Well width (nm)": [L_input]}).to_csv("Parameters Used.csv", index=False)
    pd.DataFrame(summary).to_csv("One-dimensional Infinite Quantum Well Summary.csv", index=False)
    print("Simulation complete. Files saved.")


if __name__ == '__main__':
    main()