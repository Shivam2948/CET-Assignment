import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, fsolve

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.23 Ternary NRTL Isobaric VLE", layout="wide")

st.title("Q12.23 — Ternary NRTL Isobaric VLE (BUBL T, DEW T, FLASH)")
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

# Note: z2 target is 0.2 as per assignment instructions for Q12.23
z1_target = st.sidebar.number_input("Feed z₁ (FLASH)", value=0.30, min_value=0.0, max_value=1.0)
z2_target = st.sidebar.number_input("Feed z₂ (FLASH)", value=0.20, min_value=0.0, max_value=1.0)

st.sidebar.subheader("NRTL Parameters (J/mol)")
st.markdown("**Acetone(1) - Methanol(2)**")
b12 = st.sidebar.number_input("NRTL b₁₂", value=773.3, format="%.1f")
b21 = st.sidebar.number_input("NRTL b₂₁", value=932.1, format="%.1f")
a12 = st.sidebar.number_input("NRTL α₁₂", value=0.3084, format="%.4f")

st.markdown("**Acetone(1) - Water(3)**")
b13 = st.sidebar.number_input("NRTL b₁₃", value=2642.1, format="%.1f")
b31 = st.sidebar.number_input("NRTL b₃₁", value=5013.3, format="%.1f")
a13 = st.sidebar.number_input("NRTL α₁₃", value=0.5343, format="%.4f")

st.markdown("**Methanol(2) - Water(3)**")
b23 = st.sidebar.number_input("NRTL b₂₃", value=-1062.9, format="%.1f")
b32 = st.sidebar.number_input("NRTL b₃₂", value=3538.7, format="%.1f")
a23 = st.sidebar.number_input("NRTL α₂₃", value=0.2994, format="%.4f")

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


def calc_nrtl_ternary(x1, x2, x3, T_K):
    """Calculates Ternary NRTL Activity Coefficients dynamically."""
    RT = R * T_K

    # Dimensionless interaction parameters (tau_ij)
    t11, t22, t33 = 0.0, 0.0, 0.0
    t12, t21 = b12 / RT, b21 / RT
    t13, t31 = b13 / RT, b31 / RT
    t23, t32 = b23 / RT, b32 / RT

    # G_ij parameters (Symmetric alpha: a21=a12, a31=a13, a32=a23)
    G11, G22, G33 = 1.0, 1.0, 1.0
    G12 = np.exp(-a12 * t12);
    G21 = np.exp(-a12 * t21)
    G13 = np.exp(-a13 * t13);
    G31 = np.exp(-a13 * t31)
    G23 = np.exp(-a23 * t23);
    G32 = np.exp(-a23 * t32)

    x = np.array([x1, x2, x3])
    tau = np.array([[t11, t12, t13], [t21, t22, t23], [t31, t32, t33]])
    G = np.array([[G11, G12, G13], [G21, G22, G23], [G31, G32, G33]])

    # Denominators for each component i: sum_k(x_k * G_ki)
    denom = np.zeros(3)
    for i in range(3):
        for k in range(3):
            denom[i] += x[k] * G[k, i]

    # Activity Coefficients
    lng = np.zeros(3)
    for i in range(3):
        term1_num = 0.0
        for j in range(3):
            term1_num += x[j] * tau[j, i] * G[j, i]
        term1 = term1_num / denom[i] if denom[i] != 0 else 0.0

        term2 = 0.0
        for j in range(3):
            sub_num1 = x[j] * G[i, j]
            if denom[j] == 0: continue
            sub_sum2 = sum(x[k] * tau[k, j] * G[k, j] for k in range(3))
            term2 += (sub_num1 / denom[j]) * (tau[i, j] - (sub_sum2 / denom[j]))

        lng[i] = term1 + term2

    return np.exp(lng[0]), np.exp(lng[1]), np.exp(lng[2])


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
# Pure Component Boiling Points
T1_sat_sys = calc_tsat(P_sys, A1, B1, C1)
T2_sat_sys = calc_tsat(P_sys, A2, B2, C2)
T3_sat_sys = calc_tsat(P_sys, A3, B3, C3)
T_min_g = min(T1_sat_sys, T2_sat_sys, T3_sat_sys) - 20.0
T_max_g = max(T1_sat_sys, T2_sat_sys, T3_sat_sys) + 20.0

# --- (a) BUBL T Calculation ---
x3_bubl = 1.0 - x1_target - x2_target


def bubble_obj(T_C):
    g1, g2, g3 = calc_nrtl_ternary(x1_target, x2_target, x3_bubl, T_C + 273.15)
    P1_s = calc_psat(T_C, A1, B1, C1)
    P2_s = calc_psat(T_C, A2, B2, C2)
    P3_s = calc_psat(T_C, A3, B3, C3)
    return (x1_target * g1 * P1_s) + (x2_target * g2 * P2_s) + (x3_bubl * g3 * P3_s) - P_sys


sol_bubl = root_scalar(bubble_obj, bracket=[T_min_g, T_max_g], method='brentq')
T_bubl = sol_bubl.root

g1_b, g2_b, g3_b = calc_nrtl_ternary(x1_target, x2_target, x3_bubl, T_bubl + 273.15)
y1_bubl = (x1_target * g1_b * calc_psat(T_bubl, A1, B1, C1)) / P_sys
y2_bubl = (x2_target * g2_b * calc_psat(T_bubl, A2, B2, C2)) / P_sys
y3_bubl = (x3_bubl * g3_b * calc_psat(T_bubl, A3, B3, C3)) / P_sys

# --- (b) DEW T Calculation (Coupled Root Solver) ---
y3_dew = 1.0 - y1_target - y2_target


# Solve 3 unknowns: Temperature (K), x1, x2
def dew_equations(vars):
    T_K_g, x1_g, x2_g = vars
    x3_g = 1.0 - x1_g - x2_g
    g1, g2, g3 = calc_nrtl_ternary(x1_g, x2_g, x3_g, T_K_g)

    P1_s = calc_psat(T_K_g - 273.15, A1, B1, C1)
    P2_s = calc_psat(T_K_g - 273.15, A2, B2, C2)
    P3_s = calc_psat(T_K_g - 273.15, A3, B3, C3)

    # 1. Total phase equilibrium constraint (defines T)
    eq1 = P_sys - 1.0 / ((y1_target / (g1 * P1_s)) + (y2_target / (g2 * P2_s)) + (y3_dew / (g3 * P3_s)))
    # 2 & 3. Specific component definitions
    eq2 = x1_g - (y1_target * P_sys) / (g1 * P1_s)
    eq3 = x2_g - (y2_target * P_sys) / (g2 * P2_s)
    return [eq1, eq2, eq3]


t_dew_init = (y1_target * T1_sat_sys + y2_target * T2_sat_sys + y3_dew * T3_sat_sys) + 273.15
sol_d = fsolve(dew_equations, [t_dew_init, y1_target, y2_target])
T_dew = sol_d[0] - 273.15
x1_d, x2_d = sol_d[1], sol_d[2]
x3_d = 1.0 - x1_d - x2_d

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

    g1, g2, g3 = calc_nrtl_ternary(x1_g, x2_g, x3_g, T_flash + 273.15)
    K1 = (g1 * P1_sat_f) / P_sys
    K2 = (g2 * P2_sat_f) / P_sys
    K3 = (g3 * P3_sat_f) / P_sys

    # Rachford-Rice
    eq1 = (z1_target * (K1 - 1)) / (1 + V_guess * (K1 - 1)) + \
          (z2_target * (K2 - 1)) / (1 + V_guess * (K2 - 1)) + \
          (z3_target * (K3 - 1)) / (1 + V_guess * (K3 - 1))

    # Component definitions
    eq2 = x1_g - (z1_target / (1 + V_guess * (K1 - 1)))
    eq3 = x2_g - (z2_target / (1 + V_guess * (K2 - 1)))
    return [eq1, eq2, eq3]


sol_f = fsolve(flash_equations, [0.5, z1_target, z2_target])
V_flash = sol_f[0]
x1_f, x2_f = sol_f[1], sol_f[2]
x3_f = 1.0 - x1_f - x2_f

g1_f, g2_f, g3_f = calc_nrtl_ternary(x1_f, x2_f, x3_f, T_flash + 273.15)
y1_f = ((g1_f * P1_sat_f) / P_sys) * x1_f
y2_f = ((g2_f * P2_sat_f) / P_sys) * x2_f
y3_f = ((g3_f * P3_sat_f) / P_sys) * x3_f

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Ternary Isobaric VLE Results (NRTL)")

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
# 6. RESULTS TABLE, RESIDUALS & DOWNLOADS
# ==========================================
st.subheader("Convergence Residuals & State Summary")

res_dew = dew_equations(sol_d)
res_flash = flash_equations(sol_f)
st.write("**Dew Point Residuals (P_constraint, x1_bal, x2_bal):**", [f"{r:.2e}" for r in res_dew])
st.write("**Flash Residuals (RR_sum, x1_bal, x2_bal):**", [f"{r:.2e}" for r in res_flash])

# Create combined dataframe for export
summary_data = [
    {"Phase State": "Bubble (a)", "T (°C)": T_bubl, "x1": x1_target, "x2": x2_target, "x3": x3_bubl, "y1": y1_bubl,
     "y2": y2_bubl, "y3": y3_bubl, "V_frac": 0.0},
    {"Phase State": "Dew (b)", "T (°C)": T_dew, "x1": x1_d, "x2": x2_d, "x3": x3_d, "y1": y1_target, "y2": y2_target,
     "y3": y3_dew, "V_frac": 1.0},
    {"Phase State": "Flash (c)", "T (°C)": T_flash, "x1": x1_f, "x2": x2_f, "x3": x3_f, "y1": y1_f, "y2": y2_f,
     "y3": y3_f, "V_frac": V_flash},
]
df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary.style.format({col: "{:.4f}" for col in df_summary.columns if col != "Phase State"}),
             use_container_width=True)

csv_data = df_summary.to_csv(index=False).encode('utf-8')
st.download_button(label="Download VLE Summary (CSV)", data=csv_data, file_name='Q12_23_Summary.csv', mime='text/csv')