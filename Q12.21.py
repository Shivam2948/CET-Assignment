import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, fsolve

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.21 Ternary NRTL VLE", layout="wide")

st.title("Q12.21 — Ternary NRTL VLE (BUBL, DEW, FLASH)")
st.markdown("**Acetone (1) / Methanol (2) / Water (3) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

T_sys = st.sidebar.number_input("System Temperature, T (K)", value=338.15, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

st.sidebar.subheader("Target Compositions")
x1_target = st.sidebar.number_input("Liquid x₁ (BUBL P)", value=0.30, min_value=0.0, max_value=1.0)
x2_target = st.sidebar.number_input("Liquid x₂ (BUBL P)", value=0.40, min_value=0.0, max_value=1.0)

y1_target = st.sidebar.number_input("Vapor y₁ (DEW P)", value=0.30, min_value=0.0, max_value=1.0)
y2_target = st.sidebar.number_input("Vapor y₂ (DEW P)", value=0.40, min_value=0.0, max_value=1.0)

z1_target = st.sidebar.number_input("Feed z₁ (FLASH)", value=0.30, min_value=0.0, max_value=1.0)
z2_target = st.sidebar.number_input("Feed z₂ (FLASH)", value=0.40, min_value=0.0, max_value=1.0)

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
def calc_psat(T_K, A, B, C):
    t_C = T_K - 273.15
    return np.exp(A - (B / (t_C + C)))


def calc_nrtl_ternary(x1, x2, x3, T_K):
    """Calculates Ternary NRTL Activity Coefficients using generic tensor logic"""
    RT = R * T_K

    # 1. Dimensionless interaction parameters (tau_ij)
    t11, t22, t33 = 0.0, 0.0, 0.0
    t12, t21 = b12 / RT, b21 / RT
    t13, t31 = b13 / RT, b31 / RT
    t23, t32 = b23 / RT, b32 / RT

    # 2. G_ij parameters
    G11, G22, G33 = 1.0, 1.0, 1.0
    G12 = np.exp(-a12 * t12);
    G21 = np.exp(-a12 * t21)
    G13 = np.exp(-a13 * t13);
    G31 = np.exp(-a13 * t31)
    G23 = np.exp(-a23 * t23);
    G32 = np.exp(-a23 * t32)

    # 3. Organize into arrays for compact matrix-like calculation
    x = np.array([x1, x2, x3])
    tau = np.array([[t11, t12, t13],
                    [t21, t22, t23],
                    [t31, t32, t33]])
    G = np.array([[G11, G12, G13],
                  [G21, G22, G23],
                  [G31, G32, G33]])

    # 4. Denominators for each component i: sum_k(x_k * G_ki)
    denom = np.zeros(3)
    for i in range(3):
        for k in range(3):
            denom[i] += x[k] * G[k, i]

    # 5. Activity Coefficients
    lng = np.zeros(3)
    for i in range(3):
        # Term 1
        term1_num = 0.0
        for j in range(3):
            term1_num += x[j] * tau[j, i] * G[j, i]
        term1 = term1_num / denom[i] if denom[i] != 0 else 0.0

        # Term 2
        term2 = 0.0
        for j in range(3):
            sub_num1 = x[j] * G[i, j]
            if denom[j] == 0: continue

            sub_sum2 = 0.0
            for k in range(3):
                sub_sum2 += x[k] * tau[k, j] * G[k, j]

            term2 += (sub_num1 / denom[j]) * (tau[i, j] - (sub_sum2 / denom[j]))

        lng[i] = term1 + term2

    return np.exp(lng[0]), np.exp(lng[1]), np.exp(lng[2])


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
P1_sat = calc_psat(T_sys, A1, B1, C1)
P2_sat = calc_psat(T_sys, A2, B2, C2)
P3_sat = calc_psat(T_sys, A3, B3, C3)

# --- (a) BUBL P Calculation ---
x3_bubl = 1.0 - x1_target - x2_target
g1_b, g2_b, g3_b = calc_nrtl_ternary(x1_target, x2_target, x3_bubl, T_sys)

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

for i in range(1, 31):
    g1_d, g2_d, g3_d = calc_nrtl_ternary(x1_d, x2_d, x3_d, T_sys)

    # Equilibrium pressure anchoring
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

# --- (c) P-T FLASH Calculation (Coupled Root Solver) ---
P_flash = 0.5 * (P_bubl + P_dew)
z3_target = 1.0 - z1_target - z2_target


# We solve for 3 unknowns: V, x1, x2. (x3 is strictly 1 - x1 - x2).
def flash_equations(vars):
    V_guess, x1_g, x2_g = vars
    x3_g = 1.0 - x1_g - x2_g

    g1, g2, g3 = calc_nrtl_ternary(x1_g, x2_g, x3_g, T_sys)
    K1 = (g1 * P1_sat) / P_flash
    K2 = (g2 * P2_sat) / P_flash
    K3 = (g3 * P3_sat) / P_flash

    # Eq 1: Rachford-Rice (Phase fraction balance)
    eq1 = (z1_target * (K1 - 1)) / (1 + V_guess * (K1 - 1)) + \
          (z2_target * (K2 - 1)) / (1 + V_guess * (K2 - 1)) + \
          (z3_target * (K3 - 1)) / (1 + V_guess * (K3 - 1))

    # Eq 2 & 3: Component mass balance definitions
    # (Forces x_i to structurally align with the K-values)
    eq2 = x1_g - (z1_target / (1 + V_guess * (K1 - 1)))
    eq3 = x2_g - (z2_target / (1 + V_guess * (K2 - 1)))

    return [eq1, eq2, eq3]


# Initial guesses
V_init = 0.5
x1_init, x2_init = z1_target, z2_target
sol_f = fsolve(flash_equations, [V_init, x1_init, x2_init])

V_flash = sol_f[0]
x1_f = sol_f[1]
x2_f = sol_f[2]
x3_f = 1.0 - x1_f - x2_f

g1_f, g2_f, g3_f = calc_nrtl_ternary(x1_f, x2_f, x3_f, T_sys)
K1_f = (g1_f * P1_sat) / P_flash
K2_f = (g2_f * P2_sat) / P_flash
K3_f = (g3_f * P3_sat) / P_flash

y1_f = K1_f * x1_f
y2_f = K2_f * x2_f
y3_f = K3_f * x3_f

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Ternary VLE Results (NRTL)")

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

tab1, tab2 = st.tabs(["Dew Pressure Iterations", "Flash Results"])
df_dew = pd.DataFrame(dew_iterations)

with tab1:
    st.dataframe(df_dew.style.format({col: "{:.4f}" for col in df_dew.columns if col != "Iter"}),
                 use_container_width=True)
    csv_dew = df_dew.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Dew Data (CSV)", data=csv_dew, file_name='Q12_21_Dew.csv', mime='text/csv')

with tab2:
    st.write(
        "The P-T Flash was solved simultaneously using SciPy's non-linear `fsolve` function rather than successive substitution due to the high non-ideality of the Ternary NRTL system.")
    res = flash_equations([V_flash, x1_f, x2_f])
    st.write(f"**Final Residuals [f(V)=0, x1_balance, x2_balance]:** `[{res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}]`")