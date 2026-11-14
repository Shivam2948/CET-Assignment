import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q13.14 Equilibrium Calculator", layout="wide")

st.title("Q13.14 — Gas-Phase Reaction Equilibrium")
st.markdown("**Styrene(g) + H₂(g) ⇌ Ethylbenzene(g) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Operating Conditions
T_sys = st.sidebar.number_input("Temperature, T (K)", value=923.15, format="%.2f")
P_sys = st.sidebar.number_input("Pressure, P (bar)", value=1.01325, format="%.5f")
P_std = 1.0  # Standard state pressure (bar)

st.sidebar.subheader("Initial Feed (moles)")
n_sty_0 = st.sidebar.number_input("Styrene (initial)", value=1.0, format="%.2f")
n_h2_0 = st.sidebar.number_input("Hydrogen (initial)", value=1.5, format="%.2f")
n_eb_0 = st.sidebar.number_input("Ethylbenzene (initial)", value=0.0, format="%.2f")

# Thermodynamic Constants
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

with st.sidebar.expander("Heat Capacity & Standard Properties"):
    st.markdown("**Styrene**")
    A_sty = st.number_input("Styrene A", value=2.05, format="%.4f")
    B_sty_table = st.number_input("Styrene B (table × 10³)", value=50.192, format="%.3f")
    C_sty_table = st.number_input("Styrene C (table × 10⁶)", value=-16.662, format="%.3f")
    D_sty_table = st.number_input("Styrene D (table × 10⁻⁵)", value=0.0, format="%.3f")
    Hf_sty = st.number_input("Styrene ΔH°f,298 (J/mol)", value=147360.0)
    Gf_sty = st.number_input("Styrene ΔG°f,298 (J/mol)", value=213900.0)

    st.markdown("**Hydrogen**")
    A_h2 = st.number_input("H₂ A", value=3.249, format="%.4f")
    B_h2_table = st.number_input("H₂ B (table × 10³)", value=0.422, format="%.3f")
    C_h2_table = st.number_input("H₂ C (table × 10⁶)", value=0.0, format="%.3f")
    D_h2_table = st.number_input("H₂ D (table × 10⁻⁵)", value=0.083, format="%.3f")
    Hf_h2 = st.number_input("H₂ ΔH°f,298 (J/mol)", value=0.0)
    Gf_h2 = st.number_input("H₂ ΔG°f,298 (J/mol)", value=0.0)

    st.markdown("**Ethylbenzene**")
    A_eb = st.number_input("EB A", value=1.124, format="%.4f")
    B_eb_table = st.number_input("EB B (table × 10³)", value=55.38, format="%.3f")
    C_eb_table = st.number_input("EB C (table × 10⁶)", value=-18.476, format="%.3f")
    D_eb_table = st.number_input("EB D (table × 10⁻⁵)", value=0.0, format="%.3f")
    Hf_eb = st.number_input("EB ΔH°f,298 (J/mol)", value=29920.0)
    Gf_eb = st.number_input("EB ΔG°f,298 (J/mol)", value=130890.0)

with st.sidebar.expander("Critical Properties (Tc, Pc, ω)"):
    Tc_sty = st.number_input("Styrene Tc (K)", value=636.0)
    Pc_sty = st.number_input("Styrene Pc (bar)", value=38.4)
    w_sty = st.number_input("Styrene ω", value=0.297)

    Tc_h2 = st.number_input("H₂ Tc (K)", value=33.19)
    Pc_h2 = st.number_input("H₂ Pc (bar)", value=13.13)
    w_h2 = st.number_input("H₂ ω", value=-0.216)

    Tc_eb = st.number_input("EB Tc (K)", value=617.2)
    Pc_eb = st.number_input("EB Pc (bar)", value=36.06)
    w_eb = st.number_input("EB ω", value=0.303)

st.sidebar.button("Calculate Equilibrium")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_fugacity_coeff(T, P, Tc, Pc, w):
    """Calculates fugacity coefficient using Generalized Pitzer/Virial Correlation."""
    Tr = T / Tc
    Pr = P / Pc
    B0 = 0.083 - (0.422 / (Tr ** 1.6))
    B1 = 0.139 - (0.172 / (Tr ** 4.2))
    ln_phi = (Pr / Tr) * (B0 + w * B1)
    return np.exp(ln_phi)


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
T0 = 298.15

# Step A: Standard Reaction Properties at 298.15 K
delta_Hf_298 = Hf_eb - Hf_sty - Hf_h2
delta_Gf_298 = Gf_eb - Gf_sty - Gf_h2
K_298 = np.exp(-delta_Gf_298 / (R * T0))

# Convert table coefficients to actual values based on standard header conventions
B_sty, B_h2, B_eb = B_sty_table * 1e-3, B_h2_table * 1e-3, B_eb_table * 1e-3
C_sty, C_h2, C_eb = C_sty_table * 1e-6, C_h2_table * 1e-6, C_eb_table * 1e-6
D_sty, D_h2, D_eb = D_sty_table * 1e5, D_h2_table * 1e5, D_eb_table * 1e5

# Step B: Heat Capacity Differences (Products - Reactants)
dA = A_eb - A_sty - A_h2
dB = B_eb - B_sty - B_h2
dC = C_eb - C_sty - C_h2
dD = D_eb - D_sty - D_h2

# Step C: Thermodynamically Correct Equilibrium Constant (K) Integration
term_H = (delta_Hf_298 / R) * (1 / T0 - 1 / T_sys)
term_A = dA * (np.log(T_sys / T0) - (T_sys - T0) / T_sys)
term_B = (dB / 2) * (2 * (T_sys - T0) - (T_sys ** 2 - T0 ** 2) / T_sys)
term_C = (dC / 6) * (3 * (T_sys ** 2 - T0 ** 2) - 2 * (T_sys ** 3 - T0 ** 3) / T_sys)
term_D = (dD / 2) * (-1 * (1 / T_sys ** 2 - 1 / T0 ** 2) + 2 * (1 / T_sys - 1 / T0) / T_sys)

# The correct mathematical derivation uses POSITIVE signs for the integrated Cp effects
ln_K_ratio = term_H + term_A + term_B + term_C + term_D
ln_K = np.log(K_298) + ln_K_ratio
K_T = np.exp(ln_K)

# Step D: Fugacity Coefficients
phi_sty = calc_fugacity_coeff(T_sys, P_sys, Tc_sty, Pc_sty, w_sty)
phi_h2 = calc_fugacity_coeff(T_sys, P_sys, Tc_h2, Pc_h2, w_h2)
phi_eb = calc_fugacity_coeff(T_sys, P_sys, Tc_eb, Pc_eb, w_eb)
K_phi = phi_eb / (phi_sty * phi_h2)

# Step E: Solve for Reaction Extent (epsilon)
# K = (y_EB * phi_EB) / (y_Sty*phi_Sty * y_H2*phi_H2) * (P/P_std)^(-1)
P_ratio_term = (P_sys / P_std) ** (-1)
K_apparent = K_T / (K_phi * P_ratio_term)


def equilibrium_objective(eps):
    n_sty = n_sty_0 - eps
    n_h2 = n_h2_0 - eps
    n_eb = n_eb_0 + eps
    n_tot = n_sty + n_h2 + n_eb

    y_sty = n_sty / n_tot
    y_h2 = n_h2 / n_tot
    y_eb = n_eb / n_tot

    return (y_eb / (y_sty * y_h2)) - K_apparent


# Safely bound the extent: it cannot reach the absolute limit mathematically
eps_max = min(n_sty_0, n_h2_0)
sol = root_scalar(equilibrium_objective, bracket=[0.0, eps_max - 1e-6], method='brentq')
epsilon = sol.root
residual = equilibrium_objective(epsilon)

# Final material balance
n_sty = n_sty_0 - epsilon
n_h2 = n_h2_0 - epsilon
n_eb = n_eb_0 + epsilon
n_tot = n_sty + n_h2 + n_eb

y_sty = n_sty / n_tot
y_h2 = n_h2 / n_tot
y_eb = n_eb / n_tot

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ΔH° reaction (298 K)", f"{delta_Hf_298:,.0f} J/mol")
c2.metric("ΔG° reaction (298 K)", f"{delta_Gf_298:,.0f} J/mol")
c3.metric("K° (298 K)", f"{K_298:.3e}")
c4.metric(f"K (at {T_sys} K)", f"{K_T:.4f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Styrene φ", f"{phi_sty:.4f}")
c6.metric("Hydrogen φ", f"{phi_h2:.4f}")
c7.metric("Ethylbenzene φ", f"{phi_eb:.4f}")
c8.metric("Reaction Extent (ε)", f"{epsilon:.4f} mol")

st.markdown("---")

# ==========================================
# 6. COMPOSITION TABLE & DOWNLOADS
# ==========================================
st.subheader("Equilibrium Composition Table")

df_results = pd.DataFrame({
    "Component": ["Styrene", "Hydrogen", "Ethylbenzene", "TOTAL"],
    "Initial (mol)": [n_sty_0, n_h2_0, n_eb_0, n_sty_0 + n_h2_0 + n_eb_0],
    "Change (mol)": [-epsilon, -epsilon, +epsilon, -epsilon],
    "Equilibrium (mol)": [n_sty, n_h2, n_eb, n_tot],
    "Mole Fraction (y)": [y_sty, y_h2, y_eb, y_sty + y_h2 + y_eb]
})

st.dataframe(df_results.style.format({
    "Initial (mol)": "{:.4f}", "Change (mol)": "{:.4f}",
    "Equilibrium (mol)": "{:.4f}", "Mole Fraction (y)": "{:.4f}"
}), use_container_width=True)

st.caption(f"**Mole Fraction Sum Check:** {y_sty + y_h2 + y_eb:.5f} | **Root Solver Residual:** {residual:.2e}")

csv_data = df_results.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Composition (CSV)", data=csv_data, file_name='Q13_14_Equilibrium.csv',
                   mime='text/csv')

# ==========================================
# 7. STEP-BY-STEP EXPLANATION EXPANDER
# ==========================================
with st.expander("Show Step-by-Step Mathematical Solution"):
    st.markdown(f"""
    **1. Reaction Heat Capacity Changes**
    * ΔA = {A_eb} - {A_sty} - {A_h2} = {dA:.4f}
    * ΔB = ({B_eb_table} - {B_sty_table} - {B_h2_table}) × 10⁻³ = {dB:.4e}
    * ΔC = ({C_eb_table} - {C_sty_table} - {C_h2_table}) × 10⁻⁶ = {dC:.4e}
    * ΔD = ({D_eb_table} - {D_sty_table} - {D_h2_table}) × 10⁵ = {dD:.2f}

    **2. Equilibrium Constant Evaluation**
    * Term H: (ΔH°₂₉₈ / R) · (1/T₀ - 1/T) = {term_H:.4f}
    * Term A: ΔA · [ln(T/T₀) - (T-T₀)/T] = {term_A:.4f}
    * Term B: (ΔB/2) · [2(T-T₀) - (T²-T₀²)/T] = {term_B:.4f}
    * Term C: (ΔC/6) · [3(T²-T₀²) - 2(T³-T₀³)/T] = {term_C:.4f}
    * Term D: (ΔD/2) · [-1(1/T² - 1/T₀²) + 2(1/T - 1/T₀)/T] = {term_D:.4f}
    * ln(K/K₂₉₈) = Term H + Term A + Term B + Term C + Term D = {ln_K_ratio:.4f}
    * K = exp(ln(K₂₉₈) + {ln_K_ratio:.4f}) = {K_T:.4f}

    **3. Non-Ideal Gas Fugacity Evaluation (Pitzer)**
    * K_φ = φ_EB / (φ_Sty · φ_H2) = {K_phi:.4f}

    **4. Reaction Root Solver**
    * K = [y_EB / (y_Sty · y_H2)] · K_φ · (P/P°)⁻¹
    * {K_T:.4f} = [ε(2.5 - ε) / ((1-ε)(1.5-ε))] · {K_phi:.4f} · {P_ratio_term:.4f}
    * Finding the root yields ε = {epsilon:.4f}.
    """)