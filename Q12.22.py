import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, fsolve

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.22 Ternary VLE Calculator", layout="wide")

st.title("Q12.22 — Ternary Wilson Isobaric VLE (BUBL T, DEW T, FLASH)")
st.markdown("**Acetone (1) / Methanol (2) / Water (3) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

P_sys = st.sidebar.number_input("System Pressure, P (kPa)", value=101.33, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

st.sidebar.subheader("Target Compositions")
x1_target = st.sidebar.number_input("Liquid x₁ (BUBL T)", value=0.30, min_value=0.0, max_value=1.0)
x2_target = st.sidebar.number_input("Liquid x₂ (BUBL T)", value=0.40, min_value=0.0, max_value=1.0)

y1_target = st.sidebar.number_input("Vapor y₁ (DEW T)", value=0.30, min_value=0.0, max_value=1.0)
y2_target = st.sidebar.number_input("Vapor y₂ (DEW T)", value=0.40, min_value=0.0, max_value=1.0)

z1_target = st.sidebar.number_input("Feed z₁ (FLASH)", value=0.30, min_value=0.0, max_value=1.0)
z2_target = st.sidebar.number_input("Feed z₂ (FLASH)", value=0.40, min_value=0.0, max_value=1.0)

st.sidebar.subheader("Wilson Parameters (J/mol)")
V1 = st.sidebar.number_input("Molar volume V₁", value=74.05, format="%.2f")
V2 = st.sidebar.number_input("Molar volume V₂", value=40.73, format="%.2f")
V3 = st.sidebar.number_input("Molar volume V₃", value=18.07, format="%.2f")

a12 = st.sidebar.number_input("Wilson a₁₂", value=-677.8, format="%.1f")
a21 = st.sidebar.number_input("Wilson a₂₁", value=2441.4, format="%.1f")
a23 = st.sidebar.number_input("Wilson a₂₃", value=449.6, format="%.1f")
a32 = st.sidebar.number_input("Wilson a₃₂", value=1965.9, format="%.1f")
a13 = st.sidebar.number_input("Wilson a₁₃", value=1219.5, format="%.1f")
a31 = st.sidebar.number_input("Wilson a₃₁", value=6062.5, format="%.1f")

with st.sidebar.expander("Antoine Constants (Default)"):
    st.markdown("`ln(Psat in kPa) = A - B/(T_C + C)`")
    st.markdown("**Acetone (1)**")
    A1 = st.number_input("A₁", value=14.3145, format="%.4f")
    B1 = st.number_input("B₁", value=2756.22, format="%.2f")
    C1 = st.number_input("C₁", value=228.060, format="%.3f")

    st.markdown("**Methanol (2)**")
    A2 = st.number_input("A₂", value=16.5785, format="%.4f")
    B2 = st.number_input("B₂", value=3638.27, format="%.2f")
    C2 = st.number_input("C₂", value=239.500, format="%.3f")

    st.markdown("**Water (3)**")
    A3 = st.number_input("A₃", value=16.3872, format="%.4f")
    B3 = st.number_input("B₃", value=3885.70, format="%.2f")
    C3 = st.number_input("C₃", value=230.170, format="%.3f")

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


def calc_wilson_ternary(x1, x2, x3, T_K):
    """Calculates Ternary Wilson Activity Coefficients"""
    RT = R * T_K

    # Binary Lambda interactions
    L12 = (V2 / V1) * np.exp(-a12 / RT)
    L21 = (V1 / V2) * np.exp(-a21 / RT)
    L23 = (V3 / V2) * np.exp(-a23 / RT)
    L32 = (V2 / V3) * np.exp(-a32 / RT)
    L13 = (V3 / V1) * np.exp(-a13 / RT)
    L31 = (V1 / V3) * np.exp(-a31 / RT)

    # Core Denominators
    D1 = x1 + x2 * L12 + x3 * L13
    D2 = x1 * L21 + x2 + x3 * L23
    D3 = x1 * L31 + x2 * L32 + x3

    # Calculate Activity Coefficients safely
    D1 = max(D1, 1e-12);
    D2 = max(D2, 1e-12);
    D3 = max(D3, 1e-12)
    ln_gamma1 = 1 - np.log(D1) - (x1 / D1 + (x2 * L21) / D2 + (x3 * L31) / D3)
    ln_gamma2 = 1 - np.log(D2) - ((x1 * L12) / D1 + x2 / D2 + (x3 * L32) / D3)
    ln_gamma3 = 1 - np.log(D3) - ((x1 * L13) / D1 + (x2 * L23) / D2 + x3 / D3)

    return np.exp(ln_gamma1), np.exp(ln_gamma2), np.exp(ln_gamma3)


def bubble_obj(T_C, x1_val, x2_val, x3_val):
    """Objective function for finding Bubble T at constant P"""
    g1, g2, g3 = calc_wilson_ternary(x1_val, x2_val, x3_val, T_C + 273.15)
    P1_s = calc_psat(T_C, A1, B1, C1)
    P2_s = calc_psat(T_C, A2, B2, C2)
    P3_s = calc_psat(T_C, A3, B3, C3)
    P_calc = (x1_val * g1 * P1_s) + (x2_val * g2 * P2_s) + (x3_val * g3 * P3_s)
    return P_calc - P_sys


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
T1_sat_sys = calc_tsat(P_sys, A1, B1, C1)
T2_sat_sys = calc_tsat(P_sys, A2, B2, C2)
T3_sat_sys = calc_tsat(P_sys, A3, B3, C3)

# Search brackets for temperature
T_min_g = min(T1_sat_sys, T2_sat_sys, T3_sat_sys) - 20.0
T_max_g = max(T1_sat_sys, T2_sat_sys, T3_sat_sys) + 20.0

# --- (a) BUBL T Calculation ---
x3_bubl = 1.0 - x1_target - x2_target
sol_bubl = root_scalar(bubble_obj, args=(x1_target, x2_target, x3_bubl), bracket=[T_min_g, T_max_g], method='brentq')
T_bubl = sol_bubl.root

g1_b, g2_b, g3_b = calc_wilson_ternary(x1_target, x2_target, x3_bubl, T_bubl + 273.15)
y1_bubl = (x1_target * g1_b * calc_psat(T_bubl, A1, B1, C1)) / P_sys
y2_bubl = (x2_target * g2_b * calc_psat(T_bubl, A2, B2, C2)) / P_sys
y3_bubl = (x3_bubl * g3_b * calc_psat(T_bubl, A3, B3, C3)) / P_sys

# --- (b) DEW T Calculation ---
y3_dew = 1.0 - y1_target - y2_target
dew_iterations = []

# Initial guesses
t_dew_guess = (y1_target * T1_sat_sys + y2_target * T2_sat_sys + y3_dew * T3_sat_sys)
x1_d_guess, x2_d_guess = y1_target, y2_target

for i in range(1, 31):
    x3_d_guess = 1.0 - x1_d_guess - x2_d_guess
    g1_d, g2_d, g3_d = calc_wilson_ternary(x1_d_guess, x2_d_guess, x3_d_guess, t_dew_guess + 273.15)

    P1_s = calc_psat(t_dew_guess, A1, B1, C1)
    P2_s = calc_psat(t_dew_guess, A2, B2, C2)
    P3_s = calc_psat(t_dew_guess, A3, B3, C3)

    alpha11 = 1.0
    alpha21 = P2_s / P1_s
    alpha31 = P3_s / P1_s

    # Thermodynamic pressure constraint to find new Temperature
    P1_sat_new = P_sys * ((y1_target / g1_d) + (y2_target / (g2_d * alpha21)) + (y3_dew / (g3_d * alpha31)))
    t_dew_new = calc_tsat(P1_sat_new, A1, B1, C1)

    # Extract unnormalized mole fractions explicitly
    x1_d_new = (y1_target * P_sys) / (g1_d * P1_sat_new)
    x2_d_new = (y2_target * P_sys) / (g2_d * calc_psat(t_dew_new, A2, B2, C2))

    error = abs(t_dew_new - t_dew_guess) + abs(x1_d_new - x1_d_guess)
    dew_iterations.append({
        "Iter": i, "T_guess (°C)": t_dew_guess, "x1": x1_d_guess, "x2": x2_d_guess,
        "gamma1": g1_d, "gamma2": g2_d, "gamma3": g3_d,
        "T_new (°C)": t_dew_new, "Error": error
    })

    if error < 1e-6:
        T_dew = t_dew_new
        x1_d, x2_d, x3_d = x1_d_new, x2_d_new, 1.0 - x1_d_new - x2_d_new
        break

    t_dew_guess = t_dew_new
    x1_d_guess, x2_d_guess = x1_d_new, x2_d_new

# --- (c) P-T FLASH Calculation (Coupled Root Solver) ---
T_flash = 0.5 * (T_bubl + T_dew)
z3_target = 1.0 - z1_target - z2_target

P1_sat_f = calc_psat(T_flash, A1, B1, C1)
P2_sat_f = calc_psat(T_flash, A2, B2, C2)
P3_sat_f = calc_psat(T_flash, A3, B3, C3)


# Solve for 3 unknowns: V, x1, x2 simultaneously
def flash_equations(vars):
    V_guess, x1_g, x2_g = vars
    x3_g = 1.0 - x1_g - x2_g

    g1, g2, g3 = calc_wilson_ternary(x1_g, x2_g, x3_g, T_flash + 273.15)
    K1 = (g1 * P1_sat_f) / P_sys
    K2 = (g2 * P2_sat_f) / P_sys
    K3 = (g3 * P3_sat_f) / P_sys

    # Rachford-Rice
    eq1 = (z1_target * (K1 - 1)) / (1 + V_guess * (K1 - 1)) + \
          (z2_target * (K2 - 1)) / (1 + V_guess * (K2 - 1)) + \
          (z3_target * (K3 - 1)) / (1 + V_guess * (K3 - 1))

    # Component mass balance definitions
    eq2 = x1_g - (z1_target / (1 + V_guess * (K1 - 1)))
    eq3 = x2_g - (z2_target / (1 + V_guess * (K2 - 1)))

    return [eq1, eq2, eq3]


sol_f = fsolve(flash_equations, [0.5, z1_target, z2_target])
V_flash = sol_f[0]
x1_f = sol_f[1]
x2_f = sol_f[2]
x3_f = 1.0 - x1_f - x2_f

g1_f, g2_f, g3_f = calc_wilson_ternary(x1_f, x2_f, x3_f, T_flash + 273.15)
y1_f = ((g1_f * P1_sat_f) / P_sys) * x1_f
y2_f = ((g2_f * P2_sat_f) / P_sys) * x2_f
y3_f = ((g3_f * P3_sat_f) / P_sys) * x3_f

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Ternary Isobaric VLE Results")

c1, c2, c3 = st.columns(3)
c1.metric("(a) BUBL Temperature", f"{T_bubl:.2f} °C")
c2.metric("(b) DEW Temperature", f"{T_dew:.2f} °C")
c3.metric("(c) FLASH Temperature", f"{T_flash:.2f} °C")

st.markdown("#### (a) Bubble Point Details")
c4, c5, c6 = st.columns(3)
c4.info(f"**Vapor y₁:** {y1_bubl:.4f}")
c5.info(f"**Vapor y₂:** {y2_bubl:.4f}")
c6.info(f"**Vapor y₃:** {y3_bubl:.4f}")

st.markdown("#### (b) Dew Point Details")
c7, c8, c9 = st.columns(3)
c7.warning(f"**Liquid x₁:** {x1_d:.4f}")
c8.warning(f"**Liquid x₂:** {x2_d:.4f}")
c9.warning(f"**Liquid x₃:** {x3_d:.4f}")

st.markdown("#### (c) P-T Flash Details")
st.success(f"**Vapor Fraction (V):** {V_flash:.4f}")
c10, c11, c12 = st.columns(3)
c10.success(f"**Liquid x₁:** {x1_f:.4f}  \n**Vapor y₁:** {y1_f:.4f}")
c11.success(f"**Liquid x₂:** {x2_f:.4f}  \n**Vapor y₂:** {y2_f:.4f}")
c12.success(f"**Liquid x₃:** {x3_f:.4f}  \n**Vapor y₃:** {y3_f:.4f}")

st.markdown("#### Thermodynamic Material Balance Verification")
mb1 = (V_flash * y1_f) + ((1 - V_flash) * x1_f)
mb2 = (V_flash * y2_f) + ((1 - V_flash) * x2_f)
mb3 = (V_flash * y3_f) + ((1 - V_flash) * x3_f)
st.caption(f"*Check 1 (Acetone): V*y₁ + (1-V)*x₁ = {mb1:.5f} (Target z₁ = {z1_target})*")
st.caption(f"*Check 2 (Methanol): V*y₂ + (1-V)*x₂ = {mb2:.5f} (Target z₂ = {z2_target})*")
st.caption(f"*Check 3 (Water): V*y₃ + (1-V)*x₃ = {mb3:.5f} (Target z₃ = {z3_target:.5f})*")

st.markdown("---")

# ==========================================
# 6. ITERATION TABLES & DOWNLOADS
# ==========================================
st.subheader("Convergence Data Tracking")

tab1, tab2 = st.tabs(["Dew Temperature Iterations", "Flash Results"])
df_dew = pd.DataFrame(dew_iterations)

with tab1:
    st.dataframe(df_dew.style.format({col: "{:.4f}" for col in df_dew.columns if col != "Iter"}),
                 use_container_width=True)
    csv_dew = df_dew.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Dew Data (CSV)", data=csv_dew, file_name='Q12_22_Dew.csv', mime='text/csv')

with tab2:
    st.write(
        "The Ternary Isobaric P-T Flash was solved using SciPy's non-linear `fsolve` function to guarantee closure without artificial normalization.")
    res = flash_equations([V_flash, x1_f, x2_f])
    st.write(f"**Final Residuals [f(V)=0, x1_balance, x2_balance]:** `[{res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}]`")