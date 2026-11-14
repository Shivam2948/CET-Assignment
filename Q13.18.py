import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q13.18 Equilibrium Calculator", layout="wide")

st.title("Q13.18 — Dehydrogenation of 1-butene with Steam")
st.markdown("**1-butene(g) ⇌ 1,3-butadiene(g) + H₂(g) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Operating Conditions
T_sys = st.sidebar.number_input("Temperature, T (K)", value=950.0, format="%.2f")
P_sys = st.sidebar.number_input("Pressure, P (bar)", value=1.0, format="%.2f")
P_std = 1.0

st.sidebar.subheader("Product Requirement")
y_bd_req = st.sidebar.number_input("Product Butadiene Mole Frac", value=0.10, format="%.3f")

R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

with st.sidebar.expander("Heat Capacity & Standard Properties"):
    st.markdown("**1-butene (Reactant)**")
    A_but = st.number_input("1-butene A", value=1.967, format="%.4f")
    B_but_table = st.number_input("1-butene B (table × 10³)", value=31.63, format="%.3f")
    C_but_table = st.number_input("1-butene C (table × 10⁶)", value=-9.873, format="%.3f")
    D_but_table = st.number_input("1-butene D (table × 10⁻⁵)", value=0.0, format="%.3f")
    Hf_but = st.number_input("1-butene ΔH°f,298 (J/mol)", value=-540.0)
    Gf_but = st.number_input("1-butene ΔG°f,298 (J/mol)", value=70340.0)

    st.markdown("**1,3-butadiene (Product)**")
    A_bd = st.number_input("1,3-butadiene A", value=2.934, format="%.4f")
    B_bd_table = st.number_input("1,3-butadiene B (table × 10³)", value=26.786, format="%.3f")
    C_bd_table = st.number_input("1,3-butadiene C (table × 10⁶)", value=-8.882, format="%.3f")
    D_bd_table = st.number_input("1,3-butadiene D (table × 10⁻⁵)", value=0.0, format="%.3f")
    Hf_bd = st.number_input("1,3-butadiene ΔH°f,298 (J/mol)", value=109240.0)
    Gf_bd = st.number_input("1,3-butadiene ΔG°f,298 (J/mol)", value=149795.0)

    st.markdown("**Hydrogen (Product)**")
    A_h2 = st.number_input("H₂ A", value=3.249, format="%.4f")
    B_h2_table = st.number_input("H₂ B (table × 10³)", value=0.422, format="%.3f")
    C_h2_table = st.number_input("H₂ C (table × 10⁶)", value=0.0, format="%.3f")
    D_h2_table = st.number_input("H₂ D (table × 10⁻⁵)", value=0.083, format="%.3f")
    Hf_h2 = st.number_input("H₂ ΔH°f,298 (J/mol)", value=0.0)
    Gf_h2 = st.number_input("H₂ ΔG°f,298 (J/mol)", value=0.0)

with st.sidebar.expander("Critical Properties (Tc, Pc, ω)"):
    Tc_but = st.number_input("1-butene Tc (K)", value=420.0)
    Pc_but = st.number_input("1-butene Pc (bar)", value=40.03)
    w_but = st.number_input("1-butene ω", value=0.191)

    Tc_bd = st.number_input("1,3-butadiene Tc (K)", value=425.2)
    Pc_bd = st.number_input("1,3-butadiene Pc (bar)", value=42.77)
    w_bd = st.number_input("1,3-butadiene ω", value=0.190)

    Tc_h2 = st.number_input("H₂ Tc (K)", value=33.19)
    Pc_h2 = st.number_input("H₂ Pc (bar)", value=13.13)
    w_h2 = st.number_input("H₂ ω", value=-0.216)

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

# Step A: Standard Reaction Properties
delta_Hf_298 = Hf_bd + Hf_h2 - Hf_but
delta_Gf_298 = Gf_bd + Gf_h2 - Gf_but
K_298 = np.exp(-delta_Gf_298 / (R * T0))

B_but, B_h2, B_bd = B_but_table * 1e-3, B_h2_table * 1e-3, B_bd_table * 1e-3
C_but, C_h2, C_bd = C_but_table * 1e-6, C_h2_table * 1e-6, C_bd_table * 1e-6
D_but, D_h2, D_bd = D_but_table * 1e5, D_h2_table * 1e5, D_bd_table * 1e5

# Step B: Heat Capacity Differences (Strictly Calculated)
dA = A_bd + A_h2 - A_but
dB = B_bd + B_h2 - B_but
dC = C_bd + C_h2 - C_but
dD = D_bd + D_h2 - D_but

# Step C: Equilibrium Constant (K) Integration
term_H = (delta_Hf_298 / R) * (1 / T0 - 1 / T_sys)
term_A = dA * (np.log(T_sys / T0) - (T_sys - T0) / T_sys)
term_B = (dB / 2) * (2 * (T_sys - T0) - (T_sys ** 2 - T0 ** 2) / T_sys)
term_C = (dC / 6) * (3 * (T_sys ** 2 - T0 ** 2) - 2 * (T_sys ** 3 - T0 ** 3) / T_sys)
term_D = (dD / 2) * (-1 * (1 / T_sys ** 2 - 1 / T0 ** 2) + 2 * (1 / T_sys - 1 / T0) / T_sys)

ln_K_ratio = term_H + term_A + term_B + term_C + term_D
ln_K = np.log(K_298) + ln_K_ratio
K_T = np.exp(ln_K)

# Step D: Fugacity Coefficients
phi_but = calc_fugacity_coeff(T_sys, P_sys, Tc_but, Pc_but, w_but)
phi_bd = calc_fugacity_coeff(T_sys, P_sys, Tc_bd, Pc_bd, w_bd)
phi_h2 = calc_fugacity_coeff(T_sys, P_sys, Tc_h2, Pc_h2, w_h2)

K_phi = (phi_bd * phi_h2) / phi_but

# Step E: Solve for mole fractions
# Since no hydrogen is fed, and 1 mole of butadiene forms with 1 mole of H2:
y_h2 = y_bd_req
y_bd = y_bd_req

# K = (y_bd * y_h2 / y_but) * K_phi * (P/P_std)^1
P_ratio_term = (P_sys / P_std) ** 1
y_but = (y_bd * y_h2 * K_phi * P_ratio_term) / K_T
y_steam = 1.0 - y_bd - y_h2 - y_but

# Step F: Steam Feed Material Balance (Basis: 1 mol product gas)
# eps is the extent per 1 mole of product
eps_prod = y_bd

# Feed = Product + Consumption
n_but_feed = y_but + eps_prod
n_steam_feed = y_steam
n_total_feed = n_but_feed + n_steam_feed

feed_frac_steam_balanced = n_steam_feed / n_total_feed

# Reproducing the PDF's errant calculation for the dashboard display
errant_eps = 0.089
errant_feed_total = y_but + errant_eps + y_steam
feed_frac_steam_pdf = y_steam / errant_feed_total

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
c5.metric("1-butene φ", f"{phi_but:.4f}")
c6.metric("1,3-butadiene φ (Calculated)", f"{phi_bd:.4f}")
c7.metric("Hydrogen φ", f"{phi_h2:.4f}")
c8.metric("Calculated Steam Feed Frac", f"{feed_frac_steam_balanced:.4f}")

st.markdown("---")

# ==========================================
# 6. COMPOSITION TABLES & DOWNLOAD
# ==========================================
col_tab1, col_tab2 = st.columns(2)

with col_tab1:
    st.subheader("(a) Product Mole Fractions")
    df_prod = pd.DataFrame({
        "Component": ["1-butene", "1,3-butadiene", "Hydrogen", "Steam"],
        "Mole Fraction (y)": [y_but, y_bd, y_h2, y_steam]
    })
    st.dataframe(df_prod.style.format({"Mole Fraction (y)": "{:.4f}"}), use_container_width=True)

with col_tab2:
    st.subheader("(b) Feed Composition (per 1 mol product)")
    df_feed = pd.DataFrame({
        "Component": ["1-butene", "Steam", "TOTAL FEED"],
        "Moles Fed": [n_but_feed, n_steam_feed, n_total_feed],
        "Feed Mole Fraction": [n_but_feed / n_total_feed, feed_frac_steam_balanced, 1.0000]
    })
    st.dataframe(df_feed.style.format({"Moles Fed": "{:.4f}", "Feed Mole Fraction": "{:.4f}"}),
                 use_container_width=True)

csv_data = df_prod.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Product Composition (CSV)", data=csv_data, file_name='Q13_18_Product.csv',
                   mime='text/csv')

# ==========================================
# 7. STEP-BY-STEP EXPLANATION EXPANDER
# ==========================================
with st.expander("Show Step-by-Step Mathematical Solution"):
    st.markdown(f"""
    **1. Reaction Heat Capacity Changes (Strict Calculation)**
    * ΔA = {A_bd} + {A_h2} - {A_but} = {dA:.4f}
    * ΔB = ({B_bd_table} + {B_h2_table} - {B_but_table}) × 10⁻³ = {dB:.4e}
    * ΔC = ({C_bd_table} + {C_h2_table} - ({C_but_table})) × 10⁻⁶ = {dC:.4e}
    * ΔD = ({D_bd_table} + {D_h2_table} - {D_but_table}) × 10⁵ = {dD:.2f}

    **2. Equilibrium Constant Evaluation**
    * Term H: (ΔH°₂₉₈ / R) · (1/T₀ - 1/T) = {term_H:.4f}
    * Term A: ΔA · [ln(T/T₀) - (T-T₀)/T] = {term_A:.4f}
    * Term B: (ΔB/2) · [2(T-T₀) - (T²-T₀²)/T] = {term_B:.4f}
    * Term C: (ΔC/6) · [3(T²-T₀²) - 2(T³-T₀³)/T] = {term_C:.4f}
    * Term D: (ΔD/2) · [-1(1/T² - 1/T₀²) + 2(1/T - 1/T₀)/T] = {term_D:.4f}
    * ln(K/K₂₉₈) = Term H + Term A + Term B + Term C + Term D = {ln_K_ratio:.4f}
    * K = exp(ln(K₂₉₈) + {ln_K_ratio:.4f}) = {K_T:.4f}

    **3. Non-Ideal Gas Fugacity Evaluation (Pitzer)**
    * K_φ = (φ_BD · φ_H2) / φ_Butene = {K_phi:.4f}

    **4. Reaction Equilibrium Solver**
    * K = [(y_BD · y_H2) / y_Butene] · K_φ · (P/P°)¹
    * {K_T:.4f} = [({y_bd:.3f} · {y_h2:.3f}) / y_Butene] · {K_phi:.4f} · {P_ratio_term:.4f}
    * Solving for the remaining reactant: y_Butene = {y_but:.4f}
    * Inert Steam Fraction = 1 - 0.10 - 0.10 - {y_but:.4f} = {y_steam:.4f}
    """)