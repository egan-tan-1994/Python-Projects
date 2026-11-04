import streamlit as st
import numpy as np
from XRD GaAs and Si Bragg Angle Calculator import (
    reflection,
    a_algaas,
    a_ingaas,
    bragg_gaas,
    bragg_si,
    bragg_algaas,
    bragg_ingaas
    )


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
    
    