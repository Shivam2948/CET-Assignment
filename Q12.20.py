import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.20 Ternary VLE Calculator", layout="wide")

st.title("Q12.20 — Ternary Wilson VLE (BUBL, DEW, FLASH)")
st.markdown("**Acetone (1) / Methanol (2) / Water (3) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Main system inputs
T_sys = st.sidebar.number_input("System Temperature, T (K)", value=338.15, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

st.sidebar.subheader("Target Compositions")
x1_target = st.sidebar.number_input("Liquid x₁ (BUBL P)", value=0.30, min_value=0.0, max_value=1.0)
x2_target = st.sidebar.number_input("Liquid x₂ (BUBL P)", value=0.40, min_value=0.0, max_value=1.0)

y1_target = st.sidebar.number_input("Vapor y₁ (DEW P)", value=0.30, min_value=0.0, max_value=1.0)
y2_target = st.sidebar.number_input("Vapor y₂ (DEW P)", value=0.40, min_value=0.0, max_value=1.0)

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
def calc_psat(T_K, A, B, C):
    t_C = T_K - 273.15
    return np.exp(A - (B / (t_C + C)))


def calc_wilson_ternary(x1, x2, x3, T_K):
    """Calculates Ternary Wilson Activity Coefficients"""
    RT = R * T_K

    # Calculate binary Lambda interactions
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

    # Calculate Activity Coefficients
    ln_gamma1 = 1 - np.log(D1) - (x1 / D1 + (x2 * L21) / D2 + (x3 * L31) / D3)
    ln_gamma2 = 1 - np.log(D2) - ((x1 * L12) / D1 + x2 / D2 + (x3 * L32) / D3)
    ln_gamma3 = 1 - np.log(D3) - ((x1 * L13) / D1 + (x2 * L23) / D2 + x3 / D3)

    return np.exp(ln_gamma1), np.exp(ln_gamma2), np.exp(ln_gamma3)


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
P1_sat = calc_psat(T_sys, A1, B1, C1)
P2_sat = calc_psat(T_sys, A2, B2, C2)
P3_sat = calc_psat(T_sys, A3, B3, C3)

# --- (a) BUBL P Calculation ---
x3_bubl = 1.0 - x1_target - x2_target
g1_b, g2_b, g3_b = calc_wilson_ternary(x1_target, x2_target, x3_bubl, T_sys)

P1_bubl = x1_target * g1_b * P1_sat
P2_bubl = x2_target * g2_b * P2_sat
P3_bubl = x3_bubl * g3_b * P3_sat

P_bubl = P1_bubl + P2_bubl + P3_bubl
y1_bubl, y2_bubl, y3_bubl = P1_bubl / P_bubl, P2_bubl / P_bubl, P3_bubl / P_bubl

# --- (b) DEW P Calculation ---
y3_dew = 1.0 - y1_target - y2_target
dew_iterations = []

# Ideal liquid guess for the first iteration
x1_d, x2_d, x3_d = y1_target, y2_target, y3_dew

for i in range(1, 21):
    g1_d, g2_d, g3_d = calc_wilson_ternary(x1_d, x2_d, x3_d, T_sys)

    # Equilibrium pressure without normalization
    P_dew_calc = 1.0 / ((y1_target / (g1_d * P1_sat)) + (y2_target / (g2_d * P2_sat)) + (y3_dew / (g3_d * P3_sat)))

    # Extract unnormalized mole fractions explicitly
    x1_d_new = (y1_target * P_dew_calc) / (g1_d * P1_sat)
    x2_d_new = (y2_target * P_dew_calc) / (g2_d * P2_sat)
    x3_d_new = 1.0 - x1_d_new - x2_d_new

    error = abs(x1_d_new - x1_d) + abs(x2_d_new - x2_d)
    dew_iterations.append({
        "Iter": i, "x1": x1_d, "x2": x2_d, "x3": x3_d,
        "gamma1": g1_d, "gamma2": g2_d, "gamma3": g3_d,
        "P_calc": P_dew_calc, "Error": error
    })

    if error < 1e-6:
        P_dew = P_dew_calc
        break
    x1_d, x2_d, x3_d = x1_d_new, x2_d_new, x3_d_new

# --- (c) P-T FLASH Calculation ---
P_flash = 0.5 * (P_bubl + P_dew)
z3_target = 1.0 - z1_target - z2_target
flash_iterations = []

# Initialize liquid fractions (Bubble point logic guess)
x1_f, x2_f, x3_f = z1_target, z2_target, z3_target

for i in range(1, 25):
    g1_f, g2_f, g3_f = calc_wilson_ternary(x1_f, x2_f, x3_f, T_sys)

    K1 = (g1_f * P1_sat) / P_flash
    K2 = (g2_f * P2_sat) / P_flash
    K3 = (g3_f * P3_sat) / P_flash


    # Ternary Rachford-Rice Equation
    def rr_obj(V):
        term1 = (z1_target * (K1 - 1)) / (1 + V * (K1 - 1))
        term2 = (z2_target * (K2 - 1)) / (1 + V * (K2 - 1))
        term3 = (z3_target * (K3 - 1)) / (1 + V * (K3 - 1))
        return term1 + term2 + term3


    sol = root_scalar(rr_obj, bracket=[0.0, 1.0], method='brentq')
    V_calc = sol.root

    # Extract explicitly from flash definition (no forced normalization)
    x1_f_new = z1_target / (1 + V_calc * (K1 - 1))
    x2_f_new = z2_target / (1 + V_calc * (K2 - 1))
    x3_f_new = 1.0 - x1_f_new - x2_f_new

    error = abs(x1_f_new - x1_f) + abs(x2_f_new - x2_f)
    flash_iterations.append({
        "Iter": i, "V_frac": V_calc,
        "x1": x1_f, "x2": x2_f, "x3": x3_f,
        "gamma1": g1_f, "gamma2": g2_f, "gamma3": g3_f,
        "K1": K1, "K2": K2, "K3": K3, "Error": error
    })

    if error < 1e-6:
        x1_f, x2_f, x3_f = x1_f_new, x2_f_new, x3_f_new
        V_flash = V_calc
        break
    x1_f, x2_f, x3_f = x1_f_new, x2_f_new, x3_f_new

y1_f = K1 * x1_f
y2_f = K2 * x2_f
y3_f = K3 * x3_f

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Ternary VLE Results")

c1, c2, c3 = st.columns(3)
c1.metric("(a) BUBL Pressure", f"{P_bubl:.2f} kPa")
c2.metric("(b) DEW Pressure", f"{P_dew:.2f} kPa")
c3.metric("(c) FLASH Pressure", f"{P_flash:.2f} kPa")

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

# Rigorous Material Balance Proof
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

tab1, tab2 = st.tabs(["Dew Pressure Iterations", "Flash Iterations"])
df_dew = pd.DataFrame(dew_iterations)
df_flash = pd.DataFrame(flash_iterations)

with tab1:
    st.dataframe(df_dew.style.format({col: "{:.4f}" for col in df_dew.columns if col != "Iter"}),
                 use_container_width=True)
    csv_dew = df_dew.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Dew Data (CSV)", data=csv_dew, file_name='Q12_20_Dew.csv', mime='text/csv')

with tab2:
    st.dataframe(df_flash.style.format({col: "{:.4f}" for col in df_flash.columns if col != "Iter"}),
                 use_container_width=True)
    csv_flash = df_flash.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Flash Data (CSV)", data=csv_flash, file_name='Q12_20_Flash.csv', mime='text/csv')