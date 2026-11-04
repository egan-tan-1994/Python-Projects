import streamlit as st
import numpy as np


"""
This project calculates the Bragg angle of Gallium arsenide (GaAs) and Silicon (Si)
given the user-inputted Miller indices. This will be useful for aligning GaAs(hkl)
and Si(hkl) peaks in HRXRD experiments.
"""

#-------X-ray Wavelength and GaAs and Si Lattice Parameters in Angstroms------#
lambda_xray = 1.540598
a_gaas = 5.65325
a_si = 5.431020511
a_inas = 6.0583
x_range = np.arange(0, 1 + 0.05, 0.05)
#-----------------------------------------------------------------------------#

def reflection():
    
    print("=" * 100)
    print("Enter Miller indices")
    print("=" * 100)
    
    h = int(input("h = "))
    k = int(input("k = "))
    l = int(input("l = "))
    
    print("=" * 100)
    
    return (h, k, l)

h, k, l = reflection()

def a_algaas(x):
    
    return a_gaas + (0.0078 * x)

def a_ingaas(x):
    
    return a_inas - (0.405 * x)

def bragg_gaas(a_gaas):
    
    d_gaas = a_gaas / np.sqrt(h ** 2 + k ** 2 + l ** 2)
    
    theta_gaas = np.arcsin((lambda_xray / (2 * d_gaas)))
    
    return np.degrees(theta_gaas)

def bragg_si(a_si):
    
    d_si = a_si / np.sqrt(h ** 2 + k ** 2 + l ** 2)
    
    theta_si = np.arcsin((lambda_xray / (2 * d_si)))
    
    return np.degrees(theta_si)

def bragg_algaas(lattice_algaas):
    
    d_algaas = lattice_algaas / np.sqrt(h ** 2 + k ** 2 + l ** 2)
    
    theta_algaas = np.arcsin((lambda_xray / (2 * d_algaas)))
    
    return np.degrees(theta_algaas)

def bragg_ingaas(lattice_ingaas):
    
    d_inas = lattice_ingaas / np.sqrt(h ** 2 + k ** 2 + l ** 2)
    
    theta_inas = np.arcsin((lambda_xray / (2* d_inas)))
    
    return np.degrees(theta_inas)
    
bragg_gaas = bragg_gaas(a_gaas)
bragg_si = bragg_si(a_si)


print("=" * 100)
print("GaAs and Si Bragg Angles")
print("=" * 100)
print(f"GaAs Bragg Angle: {bragg_gaas:.10f} degrees")
print(f"Si Bragg Angle: {bragg_si:.10f} degrees")
print("=" * 100)

print("=" * 100)
print("AlGaAs Bragg Angles")
print("=" * 100)
for x in x_range:
    lattice_algaas = a_algaas(x)
    angle_algaas = bragg_algaas(lattice_algaas)
    print(f"Al({x:.2f})Ga({1-x:.2f})As Bragg Angle: {angle_algaas:.10f} degrees")
print("=" * 100)

print("=" * 100)
print("InGaAs Bragg Angles")
print("=" * 100)
for x in x_range:
    lattice_ingaas = a_ingaas(x)
    angle_ingaas = bragg_ingaas(lattice_ingaas)
    print(f"Ga({x:.2f})In({1-x:.2f})As Bragg Angle: {angle_ingaas} degrees")
print("=" * 100)

st.set_page_config(page_title="XRD Bragg Angle Calculator", layout="wide")
st.title("GaAs, Al(x)Ga(1-x)As, Ga(x)In(1-x)As, and Si Bragg Angle Calculator")

col1, col2 = st.columns(2)

with col1:
    
    st.subheader("Enter X-ray Wavelength (Angstroms)")
    wavelength = st.sidebar.number_input("X-ray Wavelength")
    
    st.subheader("Reflection (Miller Indices)")
    h = st.sidebar.number_input("h", for h > 0 and int = True)
    k = st.sidebar.number_input("k", for k > 0 and int = True)
    l = st.sidebar.number_input("l", for l > 0 and int = True)
    
    st.subheader("Al(x)Ga(1-x)As Mole Fraction")
    Al_mf = st.slider("Al Mole Fraction", 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                      0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                      0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
                      0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
                      0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
                      0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
                      0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67,
                      0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
                      0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                      0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                      0.98, 0.99, 1.00)
    
    st.subheader("Ga(x)In(1-x)As Mole Fraction")
    Ga_mf = st.slider("Ga Mole Fraction", 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                      0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                      0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
                      0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
                      0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
                      0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
                      0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67,
                      0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
                      0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                      0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                      0.98, 0.99, 1.00)
    
with col2:
    
    a_gaas = 5.65325
    a_si = 5.431020511
    a_inas = 6.0583
    
    lattice_algaas = a_algaas(a_gaas)
    lattice_inas = a_ingaas(a_inas)
    
    