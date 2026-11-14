import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.13 T-x-y Calculator", layout="wide")

st.title("Q12.13 — Wilson T–x–y Calculator")
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

st.sidebar.subheader("Wilson Parameters")
V1 = st.sidebar.number_input("Molar volume V₁ (cm³/mol)", value=40.73, format="%.2f")
V2 = st.sidebar.number_input("Molar volume V₂ (cm³/mol)", value=66.30, format="%.2f")
a12 = st.sidebar.number_input("Wilson a₁₂ (J/mol)", value=2111.4, format="%.1f")
a21 = st.sidebar.number_input("Wilson a₂₁ (J/mol)", value=823.8, format="%.1f")

with st.sidebar.expander("Antoine Constants (Default)"):
    st.markdown("`ln(Psat in kPa) = A - B/(T_C + C)`")
    A1 = st.number_input("Methanol A₁", value=16.5785, format="%.4f")
    B1 = st.number_input("Methanol B₁", value=3638.27, format="%.2f")
    C1 = st.number_input("Methanol C₁", value=239.500, format="%.3f")

    A2 = st.number_input("Acetonitrile A₂", value=14.8950, format="%.4f")
    B2 = st.number_input("Acetonitrile B₂", value=3413.10, format="%.2f")
    C2 = st.number_input("Acetonitrile C₂", value=250.523, format="%.3f")

# Visual "Calculate" button (Streamlit updates reactively, but this provides a familiar UI)
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


def calc_wilson(x1, x2, T_K, V1, V2, a12, a21, R):
    """Calculates Wilson activity coefficients and Lambda parameters."""
    L12 = (V2 / V1) * np.exp(-a12 / (R * T_K))
    L21 = (V1 / V2) * np.exp(-a21 / (R * T_K))

    term1 = x1 + x2 * L12
    term2 = x2 + x1 * L21
    shared = (L12 / term1) - (L21 / term2)

    gamma1 = np.exp(-np.log(term1) + x2 * shared)
    gamma2 = np.exp(-np.log(term2) - x1 * shared)

    return gamma1, gamma2, L12, L21


def bubble_objective(T_C, x1, x2, P_target, params):
    """
    Objective function for the root solver.
    It calculates the theoretical total pressure at a guessed Temperature (T_C).
    The solver will find the T_C where this function returns exactly 0.
    """
    V1, V2, a12, a21, R, A1, B1, C1, A2, B2, C2 = params
    T_K = T_C + 273.15

    gamma1, gamma2, _, _ = calc_wilson(x1, x2, T_K, V1, V2, a12, a21, R)
    P1_sat = calc_psat(T_C, A1, B1, C1)
    P2_sat = calc_psat(T_C, A2, B2, C2)

    P_calc = (x1 * gamma1 * P1_sat) + (x2 * gamma2 * P2_sat)

    # We want P_calc - P_target to equal 0
    return P_calc - P_target


# ==========================================
# 4. ITERATIVE CALCULATIONS
# ==========================================
# Calculate pure component boiling temperatures (Celsius)
T1_sat_C = calc_tsat(P_sys, A1, B1, C1)
T2_sat_C = calc_tsat(P_sys, A2, B2, C2)

# Prepare lists to store results
results = []
params_tuple = (V1, V2, a12, a21, R, A1, B1, C1, A2, B2, C2)

# Generate composition grid
x1_arr = np.linspace(0.0, 1.0, int(num_points))

# Define wide search bracket for Temperature solver (°C)
T_min_guess = min(T1_sat_C, T2_sat_C) - 30.0
T_max_guess = max(T1_sat_C, T2_sat_C) + 30.0

for x1 in x1_arr:
    x2 = 1.0 - x1  # Strict binary closure, no arbitrary normalization

    # Handle pure components analytically
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

    # Calculate final thermodynamic state at the converged temperature
    T_bubl_K = T_bubl_C + 273.15
    gamma1, gamma2, L12, L21 = calc_wilson(x1, x2, T_bubl_K, V1, V2, a12, a21, R)
    P1_sat = calc_psat(T_bubl_C, A1, B1, C1)
    P2_sat = calc_psat(T_bubl_C, A2, B2, C2)

    # Calculate vapor compositions explicitly (Modified Raoult's Law)
    y1 = (x1 * gamma1 * P1_sat) / P_sys
    y2 = (x2 * gamma2 * P2_sat) / P_sys
    y_sum = y1 + y2  # Should be exactly 1.000 if solver converged properly

    results.append({
        "x1": x1, "x2": x2,
        "T_bubble (°C)": T_bubl_C,
        "gamma1": gamma1, "gamma2": gamma2,
        "y1": y1, "y2": y2, "Sum(y)": y_sum,
        "Converged": converged
    })

# Convert results into a Pandas DataFrame
df = pd.DataFrame(results)

# ==========================================
# 5. SPECIFIC METRICS FOR TARGET COMPOSITION
# ==========================================
# Calculate specifically for the user-requested target (default x1 = 0.3)
x2_t = 1.0 - x1_target
sol_t = root_scalar(bubble_objective, args=(x1_target, x2_t, P_sys, params_tuple), bracket=[T_min_guess, T_max_guess],
                    method='brentq')
T_t_C = sol_t.root
g1_t, g2_t, L12_t, L21_t = calc_wilson(x1_target, x2_t, T_t_C + 273.15, V1, V2, a12, a21, R)
y1_t = (x1_target * g1_t * calc_psat(T_t_C, A1, B1, C1)) / P_sys

st.subheader("Calculated Results")
col1, col2, col3 = st.columns(3)
col1.metric(f"Methanol T_sat at {P_sys} kPa", f"{T1_sat_C:.2f} °C")
col2.metric(f"Acetonitrile T_sat at {P_sys} kPa", f"{T2_sat_C:.2f} °C")
col3.metric(f"Bubble Temp (at x₁={x1_target:.2f})", f"{T_t_C:.2f} °C")

col4, col5, col6 = st.columns(3)
col4.metric(f"Wilson Λ₁₂ (at {T_t_C:.2f} °C)", f"{L12_t:.5f}")
col5.metric(f"Wilson Λ₂₁ (at {T_t_C:.2f} °C)", f"{L21_t:.5f}")
col6.metric(f"Vapor y₁ (at x₁={x1_target:.2f})", f"{y1_t:.4f}")

st.markdown("---")

# ==========================================
# 6. T-X-Y DIAGRAM
# ==========================================
st.subheader("T–x–y Diagram")

fig, ax = plt.subplots(figsize=(10, 5))

# Plot Bubble point curve (Temperature vs liquid fraction x1)
ax.plot(df["x1"], df["T_bubble (°C)"], color="#1769aa", linewidth=2.5, label="Bubble curve (T vs x₁)")
# Plot Dew point curve (Temperature vs vapor fraction y1)
ax.plot(df["y1"], df["T_bubble (°C)"], color="#d94841", linewidth=2.5, linestyle="--", label="Dew curve (T vs y₁)")

ax.set_title(f"Methanol (1) + Acetonitrile (2) at P = {P_sys} kPa")
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
    file_name='Q12_13_results.csv',
    mime='text/csv',
)

# Display interactive table with formatted floats
st.dataframe(df.style.format({
    "x1": "{:.4f}", "x2": "{:.4f}",
    "T_bubble (°C)": "{:.3f}",
    "gamma1": "{:.4f}", "gamma2": "{:.4f}",
    "y1": "{:.4f}", "y2": "{:.4f}", "Sum(y)": "{:.5f}"
}), use_container_width=True, height=350)

st.markdown(
    "*Note: The column 'Sum(y)' proves thermodynamic closure. Because the numerical root solver effectively zeroes the mass balance equation, y₁ + y₂ equals 1.000 without any artificial normalization logic.*")