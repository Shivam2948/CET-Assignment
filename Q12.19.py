import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.19 NRTL VLE Calculator", layout="wide")

st.title("Q12.19 — NRTL Isobaric VLE (BUBL T, DEW T, FLASH)")
st.markdown("**Methanol (1) / Acetonitrile (2) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Main system inputs
P_sys = st.sidebar.number_input("System Pressure, P (kPa)", value=101.33, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

st.sidebar.subheader("Target Compositions")
x1_target = st.sidebar.number_input("Liquid Fraction for BUBL T (x₁)", value=0.30, min_value=0.0, max_value=1.0)
y1_target = st.sidebar.number_input("Vapor Fraction for DEW T (y₁)", value=0.30, min_value=0.0, max_value=1.0)
z1_target = st.sidebar.number_input("Feed Fraction for FLASH (z₁)", value=0.30, min_value=0.0, max_value=1.0)

st.sidebar.subheader("NRTL Parameters")
b12 = st.sidebar.number_input("NRTL b₁₂ (J/mol)", value=1439.0, format="%.1f")
b21 = st.sidebar.number_input("NRTL b₂₁ (J/mol)", value=1317.1, format="%.1f")
alpha = st.sidebar.number_input("NRTL α (Alpha)", value=0.2981, format="%.4f")

with st.sidebar.expander("Antoine Constants (Default)"):
    st.markdown("`ln(Psat in kPa) = A - B/(T_C + C)`")
    A1 = st.number_input("Methanol A₁", value=16.5785, format="%.4f")
    B1 = st.number_input("Methanol B₁", value=3638.27, format="%.2f")
    C1 = st.number_input("Methanol C₁", value=239.500, format="%.3f")

    A2 = st.number_input("Acetonitrile A₂", value=14.8950, format="%.4f")
    B2 = st.number_input("Acetonitrile B₂", value=3413.10, format="%.2f")
    C2 = st.number_input("Acetonitrile C₂", value=250.523, format="%.3f")

st.sidebar.button("Calculate VLE")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_tsat(P_kPa, A, B, C):
    """Calculates pure component boiling point (°C) using inverted Antoine eq."""
    return (B / (A - np.log(P_kPa))) - C


def calc_psat(T_C, A, B, C):
    """Calculates saturation pressure in kPa."""
    return np.exp(A - (B / (T_C + C)))


def calc_nrtl(x1, x2, T_K, b12, b21, alpha, R):
    """Calculates NRTL activity coefficients."""
    tau12 = b12 / (R * T_K)
    tau21 = b21 / (R * T_K)
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    denom1 = x1 + x2 * G21
    denom2 = x2 + x1 * G12

    term1 = tau21 * (G21 / denom1) ** 2 if denom1 != 0 else 0
    term2 = (G12 * tau12) / denom2 ** 2 if denom2 != 0 else 0
    ln_gamma1 = (x2 ** 2) * (term1 + term2)

    term3 = tau12 * (G12 / denom2) ** 2 if denom2 != 0 else 0
    term4 = (G21 * tau21) / denom1 ** 2 if denom1 != 0 else 0
    ln_gamma2 = (x1 ** 2) * (term3 + term4)

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    return gamma1, gamma2, tau12, tau21, G12, G21


def bubble_obj(T_C, x1_val, x2_val):
    """Objective function for finding Bubble T at constant P"""
    T_K = T_C + 273.15
    g1, g2, _, _, _, _ = calc_nrtl(x1_val, x2_val, T_K, b12, b21, alpha, R)
    P1_s = calc_psat(T_C, A1, B1, C1)
    P2_s = calc_psat(T_C, A2, B2, C2)
    P_calc = (x1_val * g1 * P1_s) + (x2_val * g2 * P2_s)
    return P_calc - P_sys


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
T1_sat_sys = calc_tsat(P_sys, A1, B1, C1)
T2_sat_sys = calc_tsat(P_sys, A2, B2, C2)
T_min_g = min(T1_sat_sys, T2_sat_sys) - 25.0
T_max_g = max(T1_sat_sys, T2_sat_sys) + 25.0

# --- (a) BUBL T Calculation ---
x2_bubl = 1.0 - x1_target
sol_bubl = root_scalar(bubble_obj, args=(x1_target, x2_bubl), bracket=[T_min_g, T_max_g], method='brentq')
T_bubl = sol_bubl.root
g1_b, g2_b, _, _, _, _ = calc_nrtl(x1_target, x2_bubl, T_bubl + 273.15, b12, b21, alpha, R)
y1_bubl = (x1_target * g1_b * calc_psat(T_bubl, A1, B1, C1)) / P_sys

# --- (b) DEW T Calculation (Iterative Substitution Loop) ---
y2_dew = 1.0 - y1_target
# Initialization
t_dew_guess = 0.5 * (T1_sat_sys + T2_sat_sys)
x1_dew_guess = y1_target
dew_iterations = []

for i in range(1, 21):
    x2_dew_guess = 1.0 - x1_dew_guess
    g1_dew, g2_dew, _, _, _, _ = calc_nrtl(x1_dew_guess, x2_dew_guess, t_dew_guess + 273.15, b12, b21, alpha, R)

    P1_sat_old = calc_psat(t_dew_guess, A1, B1, C1)
    P2_sat_old = calc_psat(t_dew_guess, A2, B2, C2)
    alpha12 = P2_sat_old / P1_sat_old

    # Thermodynamic pressure constraint to find new Temperature
    P1_sat_new = P_sys * ((y1_target / g1_dew) + (y2_dew / (g2_dew * alpha12)))
    t_dew_new = calc_tsat(P1_sat_new, A1, B1, C1)

    # Extract unnormalized x1
    x1_dew_new = (y1_target * P_sys) / (g1_dew * P1_sat_new)

    error = abs(t_dew_new - t_dew_guess)
    dew_iterations.append({
        "Iteration": i, "T_guess (°C)": t_dew_guess, "x1": x1_dew_guess,
        "gamma1": g1_dew, "gamma2": g2_dew, "T_new (°C)": t_dew_new, "Error": error
    })

    if error < 1e-5:
        T_dew = t_dew_new
        x1_dew = x1_dew_new
        break
    t_dew_guess = t_dew_new
    x1_dew_guess = x1_dew_new

# --- (c) P-T FLASH Calculation (Iterative) ---
T_flash = 0.5 * (T_bubl + T_dew)
P1_sat_f = calc_psat(T_flash, A1, B1, C1)
P2_sat_f = calc_psat(T_flash, A2, B2, C2)
z2_target = 1.0 - z1_target

x1_flash_guess = z1_target
flash_iterations = []

for i in range(1, 31):
    x2_flash_guess = 1.0 - x1_flash_guess
    g1_f, g2_f, _, _, _, _ = calc_nrtl(x1_flash_guess, x2_flash_guess, T_flash + 273.15, b12, b21, alpha, R)

    K1 = (g1_f * P1_sat_f) / P_sys
    K2 = (g2_f * P2_sat_f) / P_sys


    def rr_obj(V):
        return (z1_target * (K1 - 1)) / (1 + V * (K1 - 1)) + (z2_target * (K2 - 1)) / (1 + V * (K2 - 1))


    sol = root_scalar(rr_obj, bracket=[0.0, 1.0], method='brentq')
    V_calc = sol.root

    # Extract unnormalized x1 directly from flash definition
    x1_flash_new = z1_target / (1 + V_calc * (K1 - 1))

    error = abs(x1_flash_new - x1_flash_guess)
    flash_iterations.append({
        "Iteration": i, "x1": x1_flash_guess, "V_frac": V_calc,
        "gamma1": g1_f, "gamma2": g2_f, "K1": K1, "K2": K2, "Error": error
    })

    if error < 1e-6:
        x1_flash = x1_flash_new
        V_flash = V_calc
        y1_flash = K1 * x1_flash
        break
    x1_flash_guess = x1_flash_new


# --- (d) Azeotrope Calculation ---
def azeo_root_obj(x1_guess):
    """At an azeotrope (constant P), find Bubble T, then compute difference (y1 - x1)"""
    sol_az = root_scalar(bubble_obj, args=(x1_guess, 1.0 - x1_guess), bracket=[T_min_g, T_max_g], method='brentq')
    T_az_guess = sol_az.root
    g1_az, _, _, _, _, _ = calc_nrtl(x1_guess, 1.0 - x1_guess, T_az_guess + 273.15, b12, b21, alpha, R)
    y1_az_guess = (x1_guess * g1_az * calc_psat(T_az_guess, A1, B1, C1)) / P_sys
    return y1_az_guess - x1_guess


# Find Azeotropic root
azeo_sol = root_scalar(azeo_root_obj, bracket=[0.001, 0.999], method='brentq')
x1_az = azeo_sol.root
sol_az_final = root_scalar(bubble_obj, args=(x1_az, 1.0 - x1_az), bracket=[T_min_g, T_max_g], method='brentq')
T_az = sol_az_final.root

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated VLE Results (NRTL)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("(a) BUBL Temperature", f"{T_bubl:.2f} °C")
c2.metric("(b) DEW Temperature", f"{T_dew:.2f} °C")
c3.metric("(c) FLASH Temperature", f"{T_flash:.2f} °C")
c4.metric("(d) AZEO Temperature", f"{T_az:.2f} °C")

c5, c6, c7, c8 = st.columns(4)
c5.metric(f"Vapor y₁ (at x₁={x1_target})", f"{y1_bubl:.4f}")
c6.metric(f"Liquid x₁ (at y₁={y1_target})", f"{x1_dew:.4f}")
c7.metric(f"Vapor Frac V (at z₁={z1_target})", f"{V_flash:.4f}")
c8.metric("Azeotrope Comp (x₁=y₁)", f"{x1_az:.4f}")

st.markdown("### P-T Flash Verification & Conditions")
c9, c10, c11, c12 = st.columns(4)
c9.info(f"**T1_sat:** {T1_sat_sys:.2f} °C")
c10.info(f"**T2_sat:** {T2_sat_sys:.2f} °C")
c11.success(f"**FLASH x₁:** {x1_flash:.4f}")
c12.success(f"**FLASH y₁:** {y1_flash:.4f}")

# Mass balance verification
sum_mass = (V_flash * y1_flash) + ((1 - V_flash) * x1_flash)
st.caption(f"*Material Balance Check: (V * y₁) + ((1-V) * x₁) = {sum_mass:.5f} (Target z₁ = {z1_target})*")

st.markdown("---")

# ==========================================
# 6. ITERATION TABLES & DOWNLOADS
# ==========================================
st.subheader("Convergence Data Tracking")

tab1, tab2 = st.tabs(["Dew Temperature Iterations", "Flash Iterations"])
df_dew = pd.DataFrame(dew_iterations)
df_flash = pd.DataFrame(flash_iterations)

with tab1:
    st.dataframe(df_dew.style.format({col: "{:.4f}" for col in df_dew.columns if col != "Iteration"}),
                 use_container_width=True)
with tab2:
    st.dataframe(df_flash.style.format({col: "{:.4f}" for col in df_flash.columns if col != "Iteration"}),
                 use_container_width=True)

csv_flash = df_flash.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Flash Data (CSV)", data=csv_flash, file_name='Q12_19_Flash.csv', mime='text/csv')