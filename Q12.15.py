import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.15 T-x-y Calculator", layout="wide")

st.title("Q12.15 — NRTL T–x–y Calculator")
st.markdown("**Methanol (1) / Acetonitrile (2) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Main system inputs
P_sys = st.sidebar.number_input("System Pressure, P (kPa)", value=101.33, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")
x1_target = st.sidebar.number_input("Target Liquid Fraction (x₁) for Metrics", value=0.30, min_value=0.0, max_value=1.0,
                                    step=0.05)
num_points = st.sidebar.number_input("Number of Graph Points", value=101, min_value=10, max_value=500, step=10)

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

st.sidebar.button("Calculate T-x-y")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_tsat(P_kPa, A, B, C):
    """Calculates pure component boiling point (Celsius) using inverted Antoine eq."""
    return (B / (A - np.log(P_kPa))) - C


def calc_psat(T_C, A, B, C):
    """Calculates saturation pressure in kPa from temperature in Celsius."""
    return np.exp(A - (B / (T_C + C)))


def calc_nrtl(x1, x2, T_K, b12, b21, alpha, R):
    """Calculates NRTL activity coefficients and dimensionless parameters."""
    tau12 = b12 / (R * T_K)
    tau21 = b21 / (R * T_K)
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    # Avoid division by zero at endpoints
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


def bubble_objective(T_C, x1, x2, P_target, params):
    """
    Objective function for the root solver.
    Calculates the theoretical total pressure at a guessed Temperature (T_C).
    The solver finds the T_C where (P_calc - P_target) is exactly 0.
    """
    b12, b21, alpha, R, A1, B1, C1, A2, B2, C2 = params
    T_K = T_C + 273.15

    gamma1, gamma2, _, _, _, _ = calc_nrtl(x1, x2, T_K, b12, b21, alpha, R)
    P1_sat = calc_psat(T_C, A1, B1, C1)
    P2_sat = calc_psat(T_C, A2, B2, C2)

    P_calc = (x1 * gamma1 * P1_sat) + (x2 * gamma2 * P2_sat)

    return P_calc - P_target


# ==========================================
# 4. ITERATIVE CALCULATIONS
# ==========================================
# Calculate pure component boiling temperatures (Celsius)
T1_sat_C = calc_tsat(P_sys, A1, B1, C1)
T2_sat_C = calc_tsat(P_sys, A2, B2, C2)

results = []
params_tuple = (b12, b21, alpha, R, A1, B1, C1, A2, B2, C2)

x1_arr = np.linspace(0.0, 1.0, int(num_points))

# Define wide search bracket for Temperature solver (°C)
T_min_guess = min(T1_sat_C, T2_sat_C) - 30.0
T_max_guess = max(T1_sat_C, T2_sat_C) + 30.0

for x1 in x1_arr:
    x2 = 1.0 - x1  # Strict binary closure, no artificial normalization

    if x1 == 0.0:
        T_bubl_C = T2_sat_C
        converged = True
    elif x1 == 1.0:
        T_bubl_C = T1_sat_C
        converged = True
    else:
        # Iteratively solve for Bubble Temperature
        sol = root_scalar(
            bubble_objective,
            args=(x1, x2, P_sys, params_tuple),
            bracket=[T_min_guess, T_max_guess],
            method='brentq'
        )
        T_bubl_C = sol.root
        converged = sol.converged

    # Compute final thermodynamic state at the converged temperature
    T_bubl_K = T_bubl_C + 273.15
    gamma1, gamma2, tau12, tau21, G12, G21 = calc_nrtl(x1, x2, T_bubl_K, b12, b21, alpha, R)
    P1_sat = calc_psat(T_bubl_C, A1, B1, C1)
    P2_sat = calc_psat(T_bubl_C, A2, B2, C2)

    # Vapor compositions explicitly determined by Modified Raoult's Law
    y1 = (x1 * gamma1 * P1_sat) / P_sys
    y2 = (x2 * gamma2 * P2_sat) / P_sys
    y_sum = y1 + y2  # Should be identically 1.000 if converged

    results.append({
        "x1": x1, "x2": x2,
        "T_bubble (°C)": T_bubl_C,
        "gamma1": gamma1, "gamma2": gamma2,
        "y1": y1, "y2": y2, "Sum(y)": y_sum,
        "Converged": converged
    })

df = pd.DataFrame(results)

# ==========================================
# 5. METRICS FOR TARGET COMPOSITION
# ==========================================
x2_t = 1.0 - x1_target
sol_t = root_scalar(bubble_objective, args=(x1_target, x2_t, P_sys, params_tuple), bracket=[T_min_guess, T_max_guess],
                    method='brentq')
T_t_C = sol_t.root
g1_t, g2_t, tau12_t, tau21_t, _, _ = calc_nrtl(x1_target, x2_t, T_t_C + 273.15, b12, b21, alpha, R)
y1_t = (x1_target * g1_t * calc_psat(T_t_C, A1, B1, C1)) / P_sys

st.subheader("Calculated Results")
col1, col2, col3 = st.columns(3)
col1.metric(f"Methanol T_sat at {P_sys} kPa", f"{T1_sat_C:.2f} °C")
col2.metric(f"Acetonitrile T_sat at {P_sys} kPa", f"{T2_sat_C:.2f} °C")
col3.metric(f"Bubble Temp (at x₁={x1_target:.2f})", f"{T_t_C:.2f} °C")

col4, col5, col6 = st.columns(3)
col4.metric(f"NRTL τ₁₂ (at {T_t_C:.2f} °C)", f"{tau12_t:.5f}")
col5.metric(f"NRTL τ₂₁ (at {T_t_C:.2f} °C)", f"{tau21_t:.5f}")
col6.metric(f"Vapor y₁ (at x₁={x1_target:.2f})", f"{y1_t:.4f}")

st.markdown("---")

# ==========================================
# 6. T-X-Y DIAGRAM
# ==========================================
st.subheader("T–x–y Diagram")

fig, ax = plt.subplots(figsize=(10, 5))

# Plot Bubble point curve
ax.plot(df["x1"], df["T_bubble (°C)"], color="#1769aa", linewidth=2.5, label="Bubble curve (T vs x₁)")
# Plot Dew point curve
ax.plot(df["y1"], df["T_bubble (°C)"], color="#d94841", linewidth=2.5, linestyle="--", label="Dew curve (T vs y₁)")

ax.set_title(f"Methanol (1) + Acetonitrile (2) at P = {P_sys} kPa (NRTL)")
ax.set_xlabel("Mole fraction of Methanol (x₁, y₁)")
ax.set_ylabel("Temperature (°C)")
ax.set_xlim(0, 1.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

st.pyplot(fig)

# ==========================================
# 7. COMPOSITION DATA TABLE & DOWNLOAD
# ==========================================
st.subheader("Iterative Composition Data & Convergence")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='Q12_15_results.csv',
    mime='text/csv',
)

st.dataframe(df.style.format({
    "x1": "{:.4f}", "x2": "{:.4f}",
    "T_bubble (°C)": "{:.3f}",
    "gamma1": "{:.4f}", "gamma2": "{:.4f}",
    "y1": "{:.4f}", "y2": "{:.4f}", "Sum(y)": "{:.5f}"
}), use_container_width=True, height=350)

st.markdown(
    "*Note: 'Sum(y)' rigorously demonstrates thermodynamic closure. Vapor compositions $y_1$ and $y_2$ dynamically sum to exactly 1.000 because the iterative root solver explicitly zeroes the mathematical pressure gap, removing the need for manual normalization.*")