"""
This project calculates the Bragg angle of Gallium arsenide (GaAs) and Silicon (Si)
given the user-inputted Miller indices. This will be useful for aligning GaAs(hkl)
and Si(hkl) peaks in HRXRD experiments.
"""

import numpy as np

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