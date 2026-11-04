import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(page_title="XRD Bragg Angle Calculator", layout="wide")
st.title("XRD Bragg Angle Calculator")
st.markdown("""
This app calculates the Bragg angle of Silicon (Si), Gallium arsenide (GaAs), 
Al(x)Ga(1-x)As, and Ga(x)In(1-x)As given the user-inputted Miller indices and mole fractions.
""")

# Constants (Lattice Parameters in Angstroms)
a_gaas = 5.65325
a_si = 5.431020511
a_inas = 6.0583

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    
    # 1. Manually-inputted X-ray wavelength
    lambda_xray = st.number_input("Enter X-ray Wavelength (Angstroms)", value=1.540598, format="%.6f")
    
    # 2. Manually-inputted Reflection (Miller Indices)
    st.markdown("**Reflection (Miller Indices)**")
    col_h, col_k, col_l = st.columns(3)
    with col_h:
        h = st.number_input("h", min_value=0, value=4, step=1)
    with col_k:
        k = st.number_input("k", min_value=0, value=0, step=1)
    with col_l:
        l = st.number_input("l", min_value=0, value=0, step=1)
        
    # 3 & 4. Slider-adjustable mole fractions
    st.markdown("**Alloy Mole Fractions**")
    Al_mf = st.slider("Al Mole Fraction for Al(x)Ga(1-x)As", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    Ga_mf = st.slider("Ga Mole Fraction for Ga(x)In(1-x)As", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

with col2:
    st.subheader("Calculated Bragg Angles")
    
    if h == 0 and k == 0 and l == 0:
        st.error("Miller indices (h, k, l) cannot all be zero.")
    else:
        # Calculate lattice parameters based on Vegard's law approximations from the original script
        a_algaas = a_gaas + (0.0078 * Al_mf)
        a_ingaas = a_inas - (0.405 * Ga_mf)
        
        # Universal Bragg Angle Function
        def calc_bragg_angle(lattice_param):
            d_spacing = lattice_param / np.sqrt(h**2 + k**2 + l**2)
            sin_theta = lambda_xray / (2 * d_spacing)
            
            # Check for physically impossible reflections
            if sin_theta > 1 or sin_theta < -1:
                return None
            
            return np.degrees(np.arcsin(sin_theta))

        # Perform calculations
        bragg_si = calc_bragg_angle(a_si)
        bragg_gaas = calc_bragg_angle(a_gaas)
        bragg_algaas = calc_bragg_angle(a_algaas)
        bragg_ingaas = calc_bragg_angle(a_ingaas)
        
        # 5. Output format function to 30 decimal places
        def display_angle(material_name, angle):
            if angle is None:
                st.write(f"**{material_name}:** *Reflection not possible (sinθ > 1)*")
            else:
                st.write(f"**{material_name}:** {angle:.30f} degrees")

        # Display outputs
        display_angle("Si", bragg_si)
        display_angle("GaAs", bragg_gaas)
        display_angle(f"Al({Al_mf:.2f})Ga({1-Al_mf:.2f})As", bragg_algaas)
        display_angle(f"Ga({Ga_mf:.2f})In({1-Ga_mf:.2f})As", bragg_ingaas)