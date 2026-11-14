import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q13.31 Liquid Isomerization", layout="wide")

st.title("Q13.31 — Liquid-Phase Isomerization")
st.markdown("**A(l) ⇌ B(l) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

T_sys = st.sidebar.number_input("Temperature, T (K)", value=298.15, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")
dG_298 = st.sidebar.number_input("ΔG°₂₉₈ (J/mol)", value=-1000.0, format="%.1f")
A_margules = st.sidebar.number_input("Margules Parameter (A)", value=0.10, format="%.3f")

st.sidebar.button("Calculate Equilibrium")

# ==========================================
# 3. MAIN CALCULATIONS
# ==========================================
val_rhs = abs(dG_298) / (R * T_sys)
ratio = np.exp(val_rhs)

xB_calc = 1.0 / (1.0 + ratio)
xA_calc = 1.0 - xB_calc

# ==========================================
# 4. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Equilibrium Mole Fractions")

c1, c2 = st.columns(2)
c1.metric("Mole Fraction xA", f"{xA_calc:.4f}")
c2.metric("Mole Fraction xB", f"{xB_calc:.4f}")

st.markdown("---")

# ==========================================
# 5. COMPOSITION SUMMARY TABLE
# ==========================================
st.subheader("Composition Table")

df_results = pd.DataFrame({
    "Component": ["Component A", "Component B", "TOTAL"],
    "Mole Fraction (x)": [xA_calc, xB_calc, xA_calc + xB_calc]
})

st.dataframe(df_results.style.format({"Mole Fraction (x)": "{:.4f}"}), use_container_width=True)

csv_data = df_results.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Composition (CSV)", data=csv_data, file_name='Q13_31_Isomerization.csv',
                   mime='text/csv')

# ==========================================
# 6. STEP-BY-STEP EXPLANATION EXPANDER
# ==========================================
with st.expander("Show Step-by-Step Mathematical Solution"):
    st.markdown(f"""
    **1. Excess Gibbs Energy Equation**
    * Gᴱ / RT = {A_margules} · xA · xB

    **2. Equilibrium Calculation**
    * {abs(dG_298):.2f} = {R} · {T_sys} · ln(xA / xB)
    * ln(xA / xB) = {abs(dG_298):.2f} / ({R} · {T_sys}) = {val_rhs:.4f}
    * xA / xB = exp({val_rhs:.4f}) = {ratio:.4f}
    * xA = {ratio:.4f} xB

    **3. Material Balance Closure**
    * xA + xB = 1
    * {ratio:.4f} xB + xB = 1
    * {(ratio + 1):.4f} xB = 1
    * xB = {xB_calc:.4f}
    * xA = {xA_calc:.4f}
    """)