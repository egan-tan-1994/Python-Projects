"""
This project simulates the behavior of a non-relativistic particle confined in a
one-dimensional infinite quantum potential well of well width a. In this system,
the particle has a well-defined wave function (psi_n(x)), with its square-modulus
being the particle's probability density function (phi_n(x)), which provides 
information containing the particle's most-likely location inside the well. The 
plots of the wave function and the probability density function for the first
four energy levels, as well as a scatter plot of the energy levels (in eV) for
the first ten energy levels were all illustrated in this project. This also
demonstrates the discrete nature of the particle's energy values (E_n), which is
dependent on the particle's quantum state, inicated by the quantum number "n".
The average, average-square, and uncertainty values of different observables,
such as position, momentum, and Hamiltonian for the first 1000 energy levels,
were also obtained and stored in a CSV file for easy viewing.
"""


import pandas as pd
import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre
import matplotlib.pyplot as plt


# Physical constants
hbar = 1.054571817e-34      # Reduced Planck constant
e = 1.602176634e-19         # Elementary charge
m_e = 9.10938356e-31        # Electron rest mass
nm = 1e-9                   # Conversion factor from nm to m


# User Input

try:
    m = float(input("Enter particle effective mass(m_eff/m_e): "))
    L = float(input("Enter well width (in nm): "))
    
    if m <= 0:
        raise ValueError("Effective mass must be a positive number greater"
                         "than 0.")
    
    if L <= 0:
        raise ValueError("Well width must be a positive number greater"
                         "than 0")
    
except ValueError as err:
    print(f"Invalid input: {err}")
    exit()



m_eff = m * m_e                   # Effective mass
a = L * nm                        # Well width converted to m
A = np.sqrt(2/a)                  # Normalization constant
x = np.linspace(0, a, 1000001)    # Particle position in the well



# Gaussian Quadrature

@lru_cache(maxsize=None)

def legendre_nodes_weights(order):
    """
    This returns cached Gauss-Legendre nodes and weights.

    """
    return roots_legendre(order)


def gauss_integrate(func, lower, upper, args=(), order=100):
    """
    This integrates a vectorized function using Gauss-Legendre quadrature, which
    supports both real-valued and complex-valued integrands.

    """
    nodes, weights = legendre_nodes_weights(order)
    
    x = ((upper - lower) / 2 * nodes + (upper + lower) / 2)
    
    values = func(x, *args)
    
    return ((upper - lower) / 2 * np.sum(weights * values))


def find_optimal_nodes(func, lower, upper, exact_value, args=(), scale=1.0,
                       rtol=1e-12, atol=0, n_min=2, n_max=10000, consecutive=3):
    
    reference_scale = max(abs(exact_value), abs(scale), np.finfo(float).tiny)
    tolerance = atol + rtol * reference_scale
    
    successful_orders = []
    
    for order in range(n_min, n_max + 1):
        
        result = gauss_integrate(func, lower, upper, args=args, order=order)
        absolute_error = abs(result - exact_value)
        
        if np.isfinite(absolute_error) and absolute_error <= tolerance:
            
            successful_orders.append((order, result, absolute_error))
            
            if len(successful_orders) >= consecutive:
                
                optimal_order, optimal_result, error = (
                    successful_orders[-consecutive])
                
                scaled_error = error / reference_scale
                
                return (optimal_result, optimal_order, error, scaled_error)
            
        else:
            
            successful_orders.clear()
            
    raise RuntimeError(
        f"Tolerance not achieved for orders. {n_min} through {n_max}."
        f"Increase n_max or relax the tolerance.")


def calculate_variance(
    second_moment,
    mean,
    scale_squared,
    rtol=1e-12
):

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
        raise ValueError(
            f"Significantly negative variance: {variance}"
        )

    return variance


# Wave number

def k_n(n, a):
    return (n * np.pi) / a


# Energy Eigenvalues in eV

def E_n(n, a, m_eff):
    k = k_n(n,a)
    return ((k * hbar) ** 2) / (2 * m_eff * e)


# Energy Eigenstates / Wave Functions

def psi_n(n, x, a):
    k = k_n(n,a)
    return A * (np.sin(k * x))

def d_psi_n(n, x, a):
    k = k_n(n,a)
    return (A * k) * (np.cos(k * x))

def d2_psi_n(n, x, a):
    k = k_n(n,a)
    return (-A) * (k **2) * (np.sin(k * x))

def d4_psi_n(n, x, a):
    k = k_n(n, a)
    return A * (k ** 4) * (np.sin(k * x))


# Probability Density Function

def phi_n(n, x, a):
    psi = psi_n(n, x, a)
    return np.conjugate(psi) * psi
    

# Mean and Mean-Square Positions, Momenta, and Hamiltonian



# --------------------------------------------------
# Mean and Mean-Square Position
# --------------------------------------------------

def avg_pos(x, n, a):
    psi = psi_n(n, x, a)

    return x * np.conjugate(psi) * psi


def avg_square_pos(x, n, a):
    psi = psi_n(n, x, a)

    return x**2 * np.conjugate(psi) * psi


# --------------------------------------------------
# Mean and Mean-Square Momentum
# --------------------------------------------------

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


# --------------------------------------------------
# Mean and Mean-Square Hamiltonian
# --------------------------------------------------

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


# Plot the Wave Functions and Probability Density Functions

plt.figure(figsize=(12,6))
for n in range(1, 5):
    psi = psi_n(n, x, a)
    plt.plot(x / nm, psi / A, label=f"n={n}")
plt.xlabel("Particle Position (nm)")
plt.ylabel(r"Wave Function $\psi_n(x)$ / A")
plt.title("Infinite Quantum Well Wavefunction Plots")
plt.legend()
plt.grid()
plt.savefig("Wave Function Plots", dpi=300)
plt.close()

plt.figure(figsize=(12,6))
for n in range(1, 5):
    phi = phi_n(n, x, a)
    plt.plot(x / nm, phi / A **2, label=f"n={n}")
plt.xlabel("Particle Position (nm)")
plt.ylabel(r"Probability Density Function $\phi_n(x)$ / $A^2$")
plt.title("Infinite Quantum Well Probability Density Function Plots")
plt.legend()
plt.grid()
plt.savefig("Probability Density Function Plots", dpi=300)
plt.close()

plt.figure(figsize=(12,6))
for n in range(1, 11):
    E = E_n(n, a, m_eff)
    plt.scatter(n, E, label=f"n={n}")
plt.xlabel("Quantum Number n")
plt.ylabel("Particle Energy (eV)")
plt.title("Energy Eigenvalues Scatter Plot")
plt.grid()
plt.savefig("Energy Eigenvalues", dpi=300)
plt.close()


# Calculate the expectation values and uncertainties, store the values, and save 
# in a CSV file.

n_E = np.arange(1,1001)
rtol = 1e-12
n_max = 10000

summary = []

for n in n_E:
    
    k = k_n(n,a)
    
    E = (hbar * k) ** 2 / (2 * m_eff)
    
    
    # Analytical expectation values of position, momentum, and Hamiltonian
    
    x_avg_exact = a/2
    x_avg_squared_exact = a ** 2 * (1 / 3 - 1 / (2 * (n * np.pi) ** 2))
    p_avg_exact = 0
    p_avg_squared_exact = (hbar * k) ** 2
    H_avg_exact = E
    H_avg_squared_exact = E ** 2
    
    
    # Mean position <x> calculation
    
    x_avg, Nx, err_x, rel_x = find_optimal_nodes(
        avg_pos,
        0, a,
        x_avg_exact,
        args=(n, a),
        scale=a,
        rtol=rtol,
        n_max=n_max
        )
    
    x_avg = float(np.real(x_avg))
    
    # Mean-square position <x^2> calculation
    
    x_avg_squared, Nx2, err_x2, rel_x2 = find_optimal_nodes(
        avg_square_pos,
        0, a,
        x_avg_squared_exact,
        args=(n, a),
        scale=a**2,
        rtol=rtol,
        n_max=n_max
        )
    
    x_avg_squared = float(np.real(x_avg_squared))
    
    # Mean momentum <p> calculation
    
    p_avg, Np, err_p, rel_p = find_optimal_nodes(
        avg_mom,
        0, a,
        p_avg_exact,
        args=(n, a),
        scale=hbar/a,
        atol = 1e-10 * hbar/a,
        rtol=rtol,
        n_max=n_max
        )
    
    p_avg = float(np.real(p_avg))
    
    # Mean-square momentum <p^2> calculation
    
    p_avg_squared, Np2, err_p2, rel_p2 = find_optimal_nodes(
        avg_square_mom,
        0, a,
        p_avg_squared_exact,
        args=(n, a),
        scale=(hbar/a)**2,
        rtol=rtol,
        n_max=n_max
        )
    
    p_avg_squared = float(np.real(p_avg_squared))
    
    # Mean-square velocity <v^2> calculation
    
    v_avg_squared = p_avg_squared / (m_eff ** 2)
    
    # Root-mean-square (RMS) velocity calculation
    
    v_rms = np.sqrt(v_avg_squared)
    
    # Mean Hamiltonian <H> calculation
    
    H_avg, NH, err_H, rel_H = find_optimal_nodes(
        avg_Hamil,
        0, a,
        H_avg_exact,
        args=(n, a, m_eff),
        scale=E,
        rtol=rtol,
        n_max=n_max
        )
    
    H_avg = float(np.real(H_avg))
    
    # Mean-square Hamiltonian <H^2> calculation
    
    H_avg_squared, NH2, err_H2, rel_H2 = find_optimal_nodes(
        avg_square_Hamil,
        0, a,
        H_avg_squared_exact,
        args=(n, a, m_eff),
        scale=E**2,
        rtol=rtol,
        n_max=n_max
        )
    
    H_avg_squared = float(np.real(H_avg_squared))
    
    
    # Position uncertainty calculation
    
    x_variance = calculate_variance(
        x_avg_squared,
        x_avg,
        a**2,
        rtol
        )
    
    x_std = np.sqrt(x_variance)
    
    # Momentum uncertainty calculation
    
    p_variance = calculate_variance(
        p_avg_squared,
        p_avg,
        (hbar/a)**2,
        rtol
        )
    
    p_std = np.sqrt(p_variance)
    
    # Hamiltonian uncertainty calculation
    
    H_variance = calculate_variance(
        H_avg_squared,
        H_avg,
        E**2,
        rtol
        )
    
    H_std = np.sqrt(H_variance)
    
    
    
    # Store the results
    
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
        
        "Mean momentum <p> (kg m/s)": p_avg.real,
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



parameters = pd.DataFrame({
    "Effective mass m_eff/m_e": [m],
    "Well width (nm)": [L],
    })

summary_df = pd.DataFrame(summary)

parameters.to_csv(f"Parameters Used.csv", index=False)
summary_df.to_csv(f"One-dimensional Infinite Quantum Well Summary.csv",
                  index=False)