import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.12 P-x-y Calculator", layout="wide")

st.title("Q12.12 — Wilson P–x–y Calculator")
st.markdown("**Methanol (1) / Acetonitrile (2) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

T = st.sidebar.number_input("Temperature, T (K)", value=333.15, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

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


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_psat(T_K, A, B, C):
    """Calculate saturation pressure in kPa."""
    t_C = T_K - 273.15
    return np.exp(A - (B / (t_C + C)))


def calc_wilson(x1, T_K, V1, V2, a12, a21, R):
    """Calculate Wilson activity coefficients for arrays."""
    x2 = 1.0 - x1

    L12 = (V2 / V1) * np.exp(-a12 / (R * T_K))
    L21 = (V1 / V2) * np.exp(-a21 / (R * T_K))

    term1 = x1 + x2 * L12
    term2 = x2 + x1 * L21
    shared_bracket = (L12 / term1) - (L21 / term2)

    gamma1 = np.exp(-np.log(term1) + x2 * shared_bracket)
    gamma2 = np.exp(-np.log(term2) - x1 * shared_bracket)

    return gamma1, gamma2, L12, L21


# ==========================================
# 4. CALCULATIONS
# ==========================================
# Single Point Calculations for Metrics (at x1 = 0.3)
P1_sat = calc_psat(T, A1, B1, C1)
P2_sat = calc_psat(T, A2, B2, C2)
g1_03, g2_03, L12, L21 = calc_wilson(np.array([0.3]), T, V1, V2, a12, a21, R)
P_bubl_03 = (0.3 * g1_03[0] * P1_sat) + (0.7 * g2_03[0] * P2_sat)
y1_03 = (0.3 * g1_03[0] * P1_sat) / P_bubl_03

# Full Array Calculations for Plot and Table
x1_arr = np.linspace(0, 1.0, 101)
x2_arr = 1.0 - x1_arr
gamma1_arr, gamma2_arr, _, _ = calc_wilson(x1_arr, T, V1, V2, a12, a21, R)

P1_arr = x1_arr * gamma1_arr * P1_sat
P2_arr = x2_arr * gamma2_arr * P2_sat
P_total_arr = P1_arr + P2_arr

# Handle y1 calculation securely to avoid 0/0 at endpoints if any numerical drift occurs
y1_arr = np.zeros_like(x1_arr)
mask = P_total_arr > 0
y1_arr[mask] = P1_arr[mask] / P_total_arr[mask]

# Create DataFrame
df = pd.DataFrame({
    "x1 (liquid)": x1_arr,
    "y1 (vapor)": y1_arr,
    "gamma1": gamma1_arr,
    "gamma2": gamma2_arr,
    "Pressure (kPa)": P_total_arr
})

# ==========================================
# 5. CALCULATED RESULTS DASHBOARD (METRICS)
# ==========================================
st.subheader("Calculated Results")
col1, col2, col3 = st.columns(3)
col1.metric("Methanol Psat", f"{P1_sat:.3f} kPa")
col2.metric("Acetonitrile Psat", f"{P2_sat:.3f} kPa")
col3.metric("Bubble Pressure (at x₁=0.3)", f"{P_bubl_03:.3f} kPa")

col4, col5, col6 = st.columns(3)
col4.metric("Wilson Λ₁₂", f"{L12:.5f}")
col5.metric("Wilson Λ₂₁", f"{L21:.5f}")
col6.metric("Vapor y₁ (at x₁=0.3)", f"{y1_03:.4f}")

st.markdown("---")

# ==========================================
# 6. P-X-Y DIAGRAM
# ==========================================
st.subheader("P–x–y Diagram")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x1_arr, P_total_arr, color="#1769aa", linewidth=2.5, label="Bubble curve (P vs x₁)")
ax.plot(y1_arr, P_total_arr, color="#d94841", linewidth=2.5, linestyle="--", label="Dew curve (P vs y₁)")

ax.set_title(f"Methanol (1) + Acetonitrile (2) at T = {T:.2f} K")
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
    file_name='Q12_12_results.csv',
    mime='text/csv',
)

# Render interactive table
st.dataframe(df.style.format("{:.4f}"), use_container_width=True, height=300)