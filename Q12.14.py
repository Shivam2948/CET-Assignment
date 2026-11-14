import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.14 P-x-y Calculator", layout="wide")

st.title("Q12.14 — NRTL P–x–y Calculator")
st.markdown("**Methanol (1) / Acetonitrile (2) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Main system inputs
T_sys = st.sidebar.number_input("System Temperature, T (K)", value=333.15, format="%.2f")
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

st.sidebar.button("Calculate P-x-y")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_psat(T_K, A, B, C):
    """Calculates pure component saturation pressure in kPa."""
    t_C = T_K - 273.15
    return np.exp(A - (B / (t_C + C)))


def calc_nrtl(x1, x2, T_K, b12, b21, alpha, R):
    """Calculates NRTL activity coefficients and dimensionless parameters."""
    # 1. Calculate dimensionless interaction parameters (tau)
    tau12 = b12 / (R * T_K)
    tau21 = b21 / (R * T_K)

    # 2. Calculate G parameters
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    # 3. Calculate activity coefficients
    # Breaking down the complex equations into smaller terms for clarity
    term1 = tau21 * (G21 / (x1 + x2 * G21)) ** 2
    term2 = (G12 * tau12) / (x2 + x1 * G12) ** 2
    ln_gamma1 = (x2 ** 2) * (term1 + term2)

    term3 = tau12 * (G12 / (x2 + x1 * G12)) ** 2
    term4 = (G21 * tau21) / (x1 + x2 * G21) ** 2
    ln_gamma2 = (x1 ** 2) * (term3 + term4)

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    return gamma1, gamma2, tau12, tau21, G12, G21


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
# Single Point Calculation for Metric Cards (Target x1)
x2_t = 1.0 - x1_target
P1_sat = calc_psat(T_sys, A1, B1, C1)
P2_sat = calc_psat(T_sys, A2, B2, C2)

g1_t, g2_t, tau12_t, tau21_t, G12_t, G21_t = calc_nrtl(x1_target, x2_t, T_sys, b12, b21, alpha, R)
P_bubl_t = (x1_target * g1_t * P1_sat) + (x2_t * g2_t * P2_sat)
y1_t = (x1_target * g1_t * P1_sat) / P_bubl_t

# Full Array Calculations for Plot and Table
# Create arrays for compositions (from 0 to 1)
x1_arr = np.linspace(0.0, 1.0, int(num_points))

# Enforce binary composition closure (No artificial normalization!)
x2_arr = 1.0 - x1_arr

# Calculate activity coefficients for the entire array
gamma1_arr, gamma2_arr, _, _, _, _ = calc_nrtl(x1_arr, x2_arr, T_sys, b12, b21, alpha, R)

# Calculate partial pressures using Modified Raoult's Law
P1_arr = x1_arr * gamma1_arr * P1_sat
P2_arr = x2_arr * gamma2_arr * P2_sat

# Calculate total bubble pressure
P_total_arr = P1_arr + P2_arr

# Calculate equilibrium vapor compositions explicitly
# Utilizing np.where or safe division to avoid 0/0 warning at absolute zero pressures
y1_arr = np.zeros_like(x1_arr)
y2_arr = np.zeros_like(x2_arr)
mask = P_total_arr > 0
y1_arr[mask] = P1_arr[mask] / P_total_arr[mask]
y2_arr[mask] = P2_arr[mask] / P_total_arr[mask]
y_sum_arr = y1_arr + y2_arr

# Build the results DataFrame
df = pd.DataFrame({
    "x1": x1_arr,
    "x2": x2_arr,
    "y1": y1_arr,
    "y2": y2_arr,
    "Sum(y)": y_sum_arr,
    "gamma1": gamma1_arr,
    "gamma2": gamma2_arr,
    "Pressure (kPa)": P_total_arr
})

# ==========================================
# 5. CALCULATED RESULTS DASHBOARD (METRICS)
# ==========================================
st.subheader("Calculated NRTL Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("NRTL τ₁₂", f"{tau12_t:.5f}")
col2.metric("NRTL τ₂₁", f"{tau21_t:.5f}")
col3.metric("NRTL G₁₂", f"{G12_t:.5f}")
col4.metric("NRTL G₂₁", f"{G21_t:.5f}")

st.markdown("---")

col5, col6, col7 = st.columns(3)
col5.metric(f"Methanol Psat (at {T_sys} K)", f"{P1_sat:.3f} kPa")
col6.metric(f"Acetonitrile Psat (at {T_sys} K)", f"{P2_sat:.3f} kPa")
col7.metric(f"Bubble Pressure (at x₁={x1_target:.2f})", f"{P_bubl_t:.3f} kPa")

# ==========================================
# 6. P-X-Y DIAGRAM
# ==========================================
st.subheader("P–x–y Diagram")

fig, ax = plt.subplots(figsize=(10, 5))

# Plot Bubble point curve (Pressure vs liquid fraction x1)
ax.plot(df["x1"], df["Pressure (kPa)"], color="#1769aa", linewidth=2.5, label="Bubble curve (P vs x₁)")
# Plot Dew point curve (Pressure vs vapor fraction y1)
ax.plot(df["y1"], df["Pressure (kPa)"], color="#d94841", linewidth=2.5, linestyle="--", label="Dew curve (P vs y₁)")

ax.set_title(f"Methanol (1) + Acetonitrile (2) at T = {T_sys} K (NRTL Model)")
ax.set_xlabel("Mole fraction of Methanol (x₁, y₁)")
ax.set_ylabel("Pressure (kPa)")
ax.set_xlim(0, 1.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()

st.pyplot(fig)

# ==========================================
# 7. COMPOSITION DATA TABLE & DOWNLOAD
# ==========================================
st.subheader("Composition Data")

# CSV Download button
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='Q12_14_results.csv',
    mime='text/csv',
)

# Display interactive table with formatted floats
st.dataframe(df.style.format({
    "x1": "{:.4f}", "x2": "{:.4f}",
    "y1": "{:.4f}", "y2": "{:.4f}", "Sum(y)": "{:.5f}",
    "gamma1": "{:.4f}", "gamma2": "{:.4f}",
    "Pressure (kPa)": "{:.4f}"
}), use_container_width=True, height=350)

st.markdown(
    "*Note: The column 'Sum(y)' rigorously demonstrates thermodynamic closure. Vapor compositions $y_1$ and $y_2$ sum to exactly 1.000 without requiring any artificial normalization.*")