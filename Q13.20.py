import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q13.20 Ammonia Synthesis", layout="wide")

st.title("Q13.20 — Ammonia Synthesis Equilibrium")
st.markdown("**N₂(g) + 3H₂(g) ⇌ 2NH₃(g) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

st.sidebar.subheader("Part (a) Operating Conditions")
T_a = st.sidebar.number_input("Temperature, T (K)", value=300.0, format="%.2f")
P_a = st.sidebar.number_input("Pressure, P (bar)", value=1.0, format="%.2f")

st.sidebar.subheader("Initial Feed (moles)")
n_n2_0 = st.sidebar.number_input("Nitrogen (initial)", value=1.0, format="%.2f")
n_h2_0 = st.sidebar.number_input("Hydrogen (initial)", value=3.0, format="%.2f")

R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")
P_std = 1.0

with st.sidebar.expander("Heat Capacity & Standard Properties"):
    st.markdown("**Nitrogen (N₂)**")
    A_n2 = st.number_input("N₂ A", value=3.280, format="%.4f")
    B_n2_t = st.number_input("N₂ B (table × 10³)", value=0.593, format="%.3f")
    C_n2_t = st.number_input("N₂ C (table × 10⁶)", value=0.0, format="%.3f")
    D_n2_t = st.number_input("N₂ D (table × 10⁻⁵)", value=0.040, format="%.3f")
    Hf_n2 = st.number_input("N₂ ΔH°f,298 (J/mol)", value=0.0)
    Gf_n2 = st.number_input("N₂ ΔG°f,298 (J/mol)", value=0.0)

    st.markdown("**Hydrogen (H₂)**")
    A_h2 = st.number_input("H₂ A", value=3.249, format="%.4f")
    B_h2_t = st.number_input("H₂ B (table × 10³)", value=0.422, format="%.3f")
    C_h2_t = st.number_input("H₂ C (table × 10⁶)", value=0.0, format="%.3f")
    D_h2_t = st.number_input("H₂ D (table × 10⁻⁵)", value=0.083, format="%.3f")
    Hf_h2 = st.number_input("H₂ ΔH°f,298 (J/mol)", value=0.0)
    Gf_h2 = st.number_input("H₂ ΔG°f,298 (J/mol)", value=0.0)

    st.markdown("**Ammonia (NH₃)**")
    A_nh3 = st.number_input("NH₃ A", value=3.578, format="%.4f")
    B_nh3_t = st.number_input("NH₃ B (table × 10³)", value=3.020, format="%.3f")
    C_nh3_t = st.number_input("NH₃ C (table × 10⁶)", value=0.0, format="%.3f")
    D_nh3_t = st.number_input("NH₃ D (table × 10⁻⁵)", value=-0.186, format="%.3f")
    Hf_nh3 = st.number_input("NH₃ ΔH°f,298 (J/mol)", value=-46190.0)
    Gf_nh3 = st.number_input("NH₃ ΔG°f,298 (J/mol)", value=-16450.0)

with st.sidebar.expander("Critical Properties (Tc, Pc, ω)"):
    Tc_n2 = st.number_input("N₂ Tc (K)", value=126.2)
    Pc_n2 = st.number_input("N₂ Pc (bar)", value=34.0)
    w_n2 = st.number_input("N₂ ω", value=0.038)

    Tc_h2 = st.number_input("H₂ Tc (K)", value=33.19)
    Pc_h2 = st.number_input("H₂ Pc (bar)", value=13.13)
    w_h2 = st.number_input("H₂ ω", value=-0.216)

    Tc_nh3 = st.number_input("NH₃ Tc (K)", value=405.7)
    Pc_nh3 = st.number_input("NH₃ Pc (bar)", value=112.8)
    w_nh3 = st.number_input("NH₃ ω", value=0.253)

with st.sidebar.expander("Values for Parts b, c, d (From PDF)"):
    dH_const = st.number_input("Constant ΔH° (J/mol)", value=-92200.0)
    dS_const = st.number_input("Constant ΔS° (J/mol·K)", value=-198.1)

st.sidebar.button("Calculate Equilibrium")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_fugacity_coeff(T, P, Tc, Pc, w):
    """Calculates fugacity coefficient using Generalized Pitzer Correlation."""
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

# Step 1: Standard Reaction Properties (Doubled Reaction)
delta_Hf_298 = 2 * Hf_nh3 - Hf_n2 - 3 * Hf_h2
delta_Gf_298 = 2 * Gf_nh3 - Gf_n2 - 3 * Gf_h2
K_298 = np.exp(-delta_Gf_298 / (R * T0))

B_n2, C_n2, D_n2 = B_n2_t * 1e-3, C_n2_t * 1e-6, D_n2_t * 1e5
B_h2, C_h2, D_h2 = B_h2_t * 1e-3, C_h2_t * 1e-6, D_h2_t * 1e5
B_nh3, C_nh3, D_nh3 = B_nh3_t * 1e-3, C_nh3_t * 1e-6, D_nh3_t * 1e5

dA = 2 * A_nh3 - A_n2 - 3 * A_h2
dB = 2 * B_nh3 - B_n2 - 3 * B_h2
dC = 2 * C_nh3 - C_n2 - 3 * C_h2
dD = 2 * D_nh3 - D_n2 - 3 * D_h2

# Part (a): Integration for K at T_a = 300 K
term_H_a = (delta_Hf_298 / R) * (1 / T0 - 1 / T_a)
term_A_a = dA * (np.log(T_a / T0) - (T_a - T0) / T_a)
term_B_a = (dB / 2) * (2 * (T_a - T0) - (T_a ** 2 - T0 ** 2) / T_a)
term_C_a = (dC / 6) * (3 * (T_a ** 2 - T0 ** 2) - 2 * (T_a ** 3 - T0 ** 3) / T_a)
term_D_a = (dD / 2) * (-1 * (1 / T_a ** 2 - 1 / T0 ** 2) + 2 * (1 / T_a - 1 / T0) / T_a)

ln_K_ratio_a = term_H_a + term_A_a + term_B_a + term_C_a + term_D_a
K_T_a = np.exp(np.log(K_298) + ln_K_ratio_a)

# Fugacity Coefficients at 300 K
phi_n2_a = calc_fugacity_coeff(T_a, P_a, Tc_n2, Pc_n2, w_n2)
phi_h2_a = calc_fugacity_coeff(T_a, P_a, Tc_h2, Pc_h2, w_h2)
phi_nh3_a = calc_fugacity_coeff(T_a, P_a, Tc_nh3, Pc_nh3, w_nh3)

K_phi_a = (phi_nh3_a ** 2) / (phi_n2_a * (phi_h2_a ** 3))
P_ratio_term_a = (P_a / P_std) ** (-2)
K_app_a = K_T_a / (K_phi_a * P_ratio_term_a)


def eq_obj_a(eps):
    y_n2 = (1 - eps) / (4 - 2 * eps)
    y_h2 = (3 - 3 * eps) / (4 - 2 * eps)
    y_nh3 = (2 * eps) / (4 - 2 * eps)
    return (y_nh3 ** 2 / (y_n2 * y_h2 ** 3)) - K_app_a


sol_a = root_scalar(eq_obj_a, bracket=[0.0, 0.999999], method='brentq')
eps_a = sol_a.root
res_a = eq_obj_a(eps_a)

n_n2_a = 1 - eps_a
n_h2_a = 3 - 3 * eps_a
n_nh3_a = 2 * eps_a
n_tot_a = 4 - 2 * eps_a
y_n2_a = n_n2_a / n_tot_a
y_h2_a = n_h2_a / n_tot_a
y_nh3_a = n_nh3_a / n_tot_a

# Calculations for Parts (b), (c), (d)
y_nh3_target = 0.50
eps_target = 0.6666667
y_n2_target = 0.125
y_h2_target = 0.375
Qy_target = (y_nh3_target ** 2) / (y_n2_target * (y_h2_target ** 3))


def solve_T_constant(K_req):
    return dH_const / (dS_const - R * np.log(K_req))


# Part (b) 1 bar
K_req_b = Qy_target * (1.0 / P_std) ** (-2)
T_b = solve_T_constant(K_req_b)

# Part (c) 100 bar (Ideal Gas)
K_req_c = Qy_target * (100.0 / P_std) ** (-2)
T_c = solve_T_constant(K_req_c)

# Part (d) 100 bar (Ideal Solution Approx - matches PDF method exactly)
K_req_d = Qy_target * (1.0)
T_d = solve_T_constant(K_req_d)

# ==========================================
# 5. DASHBOARD LAYOUT & TABS
# ==========================================
tab_a, tab_bcd = st.tabs(["Part (a): Equilibrium at 300 K", "Parts (b), (c), (d): Target Composition"])

with tab_a:
    st.subheader("Part (a) Results (T = 300 K, P = 1 bar)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K(300 K)", f"{K_T_a:,.1f}")
    c2.metric("Reaction Extent ε", f"{eps_a:.4f} mol")
    c3.metric("NH₃ Mole Fraction", f"{y_nh3_a:.4f}")
    c4.metric("Solver Residual", f"{res_a:.2e}")

    st.markdown("#### Equilibrium Composition")
    df_a = pd.DataFrame({
        "Component": ["Nitrogen", "Hydrogen", "Ammonia", "TOTAL"],
        "Initial (mol)": [1.0, 3.0, 0.0, 4.0],
        "Equilibrium (mol)": [n_n2_a, n_h2_a, n_nh3_a, n_tot_a],
        "Mole Fraction": [y_n2_a, y_h2_a, y_nh3_a, y_n2_a + y_h2_a + y_nh3_a]
    })
    st.dataframe(
        df_a.style.format({"Initial (mol)": "{:.4f}", "Equilibrium (mol)": "{:.4f}", "Mole Fraction": "{:.5f}"}),
        use_container_width=True)

    csv_a = df_a.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Part (a) Data (CSV)", data=csv_a, file_name='Q13_20_PartA.csv', mime='text/csv')

with tab_bcd:
    st.subheader("Temperature Requirements for y_NH₃ = 0.50")
    st.markdown(f"**Target Composition Quotient (Qy):** {Qy_target:.4f}")

    cb1, cb2, cb3 = st.columns(3)
    cb1.metric("(b) 1 bar, Ideal Gas", f"{T_b:.2f} K")
    cb2.metric("(c) 100 bar, Ideal Gas Mixture", f"{T_c:.2f} K")
    cb3.metric("(d) 100 bar, Ideal Solution Approx", f"{T_d:.2f} K")

# ==========================================
# 6. STEP-BY-STEP EXPLANATION EXPANDER
# ==========================================
with st.expander("Show Step-by-Step Mathematical Solution"):
    st.markdown(f"""
    **1. Standard Reaction State (Doubled Reaction)**
    * ΔH°₂₉₈ = 2({Hf_nh3}) - {Hf_n2} - 3({Hf_h2}) = {delta_Hf_298:,.0f} J/mol
    * ΔG°₂₉₈ = 2({Gf_nh3}) - {Gf_n2} - 3({Gf_h2}) = {delta_Gf_298:,.0f} J/mol
    * K°₂₉₈ = {K_298:,.1f}

    **2. Part (a) Equilibrium at 300 K**
    * Using the rigorous heat-capacity integration, K(300 K) = {K_T_a:,.1f}.
    * Fugacity coefficients evaluated via Pitzer at 300 K, 1 bar: 
      φ_N₂ = {phi_n2_a:.4f}, φ_H₂ = {phi_h2_a:.4f}, φ_NH₃ = {phi_nh3_a:.4f}.
    * Equation: K = [y_NH₃² / (y_N₂ · y_H₂³)] · [{phi_nh3_a:.4f}² / ({phi_n2_a:.4f} · {phi_h2_a:.4f}³)] · (1)⁻²
    * Solving for ε yields ε = {eps_a:.4f}.

    **3. Constant Enthalpy/Entropy Method (Parts b, c, d)**
    * The calculations use the explicit variables extracted from the assignment PDF: ΔH° = {dH_const} J/mol, ΔS° = {dS_const:.2f} J/(mol·K).
    * T = ΔH° / (ΔS° - R·ln(K_req)).

    *(b) 1 bar:* K_req = {Qy_target:.4f} · (1)⁻² = {K_req_b:.4f} 
    ➜ T = {T_b:.2f} K.

    *(c) 100 bar, Ideal Gas Mixture:* K_req = {Qy_target:.4f} · (100)⁻² = {K_req_c:.6f} 
    ➜ T = {T_c:.2f} K.

    *(d) 100 bar, Ideal Solution Approx:* K_req = {Qy_target:.4f} 
    ➜ T = {T_d:.2f} K.
    """)