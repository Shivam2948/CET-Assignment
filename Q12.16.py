import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
# ==========================================
st.set_page_config(page_title="Q12.16 VLE Calculator", layout="wide")

st.title("Q12.16 — Wilson VLE Calculations (BUBL, DEW, FLASH)")
st.markdown("**Methanol (1) / Acetonitrile (2) · Chemical Engineering Thermodynamics**")
st.markdown("---")

# ==========================================
# 2. SIDEBAR / INPUT PARAMETERS
# ==========================================
st.sidebar.header("Input Parameters")

# Main system inputs
T_sys = st.sidebar.number_input("System Temperature, T (K)", value=333.15, format="%.2f")
R = st.sidebar.number_input("Gas constant, R (J/mol·K)", value=8.314, format="%.3f")

st.sidebar.subheader("Target Compositions")
x1_target = st.sidebar.number_input("Liquid Fraction for BUBL P (x₁)", value=0.30, min_value=0.0, max_value=1.0)
y1_target = st.sidebar.number_input("Vapor Fraction for DEW P (y₁)", value=0.30, min_value=0.0, max_value=1.0)
z1_target = st.sidebar.number_input("Feed Fraction for FLASH (z₁)", value=0.30, min_value=0.0, max_value=1.0)

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

st.sidebar.button("Calculate VLE")


# ==========================================
# 3. THERMODYNAMIC FUNCTIONS
# ==========================================
def calc_psat(T_K, A, B, C):
    """Calculates saturation pressure in kPa."""
    t_C = T_K - 273.15
    return np.exp(A - (B / (t_C + C)))


def calc_wilson(x1, x2, T_K, V1, V2, a12, a21, R):
    """Calculates Wilson activity coefficients."""
    L12 = (V2 / V1) * np.exp(-a12 / (R * T_K))
    L21 = (V1 / V2) * np.exp(-a21 / (R * T_K))

    term1 = x1 + x2 * L12
    term2 = x2 + x1 * L21
    shared = (L12 / term1) - (L21 / term2)

    gamma1 = np.exp(-np.log(term1) + x2 * shared)
    gamma2 = np.exp(-np.log(term2) - x1 * shared)

    return gamma1, gamma2, L12, L21


# ==========================================
# 4. MAIN CALCULATIONS
# ==========================================
P1_sat = calc_psat(T_sys, A1, B1, C1)
P2_sat = calc_psat(T_sys, A2, B2, C2)

# --- (a) BUBL P Calculation ---
x2_bubl = 1.0 - x1_target
g1_bubl, g2_bubl, L12, L21 = calc_wilson(x1_target, x2_bubl, T_sys, V1, V2, a12, a21, R)
P_bubl = (x1_target * g1_bubl * P1_sat) + (x2_bubl * g2_bubl * P2_sat)
y1_bubl = (x1_target * g1_bubl * P1_sat) / P_bubl

# --- (b) DEW P Calculation (Iterative) ---
y2_dew = 1.0 - y1_target
x1_dew_guess = y1_target  # Initial ideal guess
dew_iterations = []

for i in range(1, 21):
    x2_dew_guess = 1.0 - x1_dew_guess
    g1_dew, g2_dew, _, _ = calc_wilson(x1_dew_guess, x2_dew_guess, T_sys, V1, V2, a12, a21, R)

    # Calculate P enforcing exact thermodynamic closure sum
    P_dew_calc = 1.0 / ((y1_target / (g1_dew * P1_sat)) + (y2_dew / (g2_dew * P2_sat)))

    # Extract unnormalized x1
    x1_dew_new = (y1_target * P_dew_calc) / (g1_dew * P1_sat)

    error = abs(x1_dew_new - x1_dew_guess)
    dew_iterations.append({
        "Iteration": i, "x1": x1_dew_guess, "x2": x2_dew_guess,
        "gamma1": g1_dew, "gamma2": g2_dew, "P_calc (kPa)": P_dew_calc, "Error": error
    })

    if error < 1e-6:
        x1_dew = x1_dew_new
        P_dew = P_dew_calc
        break
    x1_dew_guess = x1_dew_new

# --- (c) P-T FLASH Calculation (Iterative) ---
P_flash = 0.5 * (P_bubl + P_dew)
z2_target = 1.0 - z1_target
x1_flash_guess = z1_target
flash_iterations = []

for i in range(1, 21):
    x2_flash_guess = 1.0 - x1_flash_guess
    g1_f, g2_f, _, _ = calc_wilson(x1_flash_guess, x2_flash_guess, T_sys, V1, V2, a12, a21, R)

    K1 = (g1_f * P1_sat) / P_flash
    K2 = (g2_f * P2_sat) / P_flash


    # Rachford-Rice Objective Function
    def rr_obj(V):
        return (z1_target * (K1 - 1)) / (1 + V * (K1 - 1)) + (z2_target * (K2 - 1)) / (1 + V * (K2 - 1))


    # Root solver for V
    sol = root_scalar(rr_obj, bracket=[0.0, 1.0], method='brentq')
    V_calc = sol.root

    # Unnormalized x1 directly from flash definition
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
def azeo_obj(x1_guess):
    """At an azeotrope under modified Raoult's Law, gamma1 * P1sat = gamma2 * P2sat"""
    g1, g2, _, _ = calc_wilson(x1_guess, 1.0 - x1_guess, T_sys, V1, V2, a12, a21, R)
    return (g1 * P1_sat) - (g2 * P2_sat)


# Search for root between 0.001 and 0.999
azeo_sol = root_scalar(azeo_obj, bracket=[0.001, 0.999], method='brentq')
x1_az = azeo_sol.root
g1_az, g2_az, _, _ = calc_wilson(x1_az, 1.0 - x1_az, T_sys, V1, V2, a12, a21, R)
P_az = g1_az * P1_sat

# ==========================================
# 5. DASHBOARD METRICS
# ==========================================
st.subheader("Calculated Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("(a) BUBL Pressure", f"{P_bubl:.2f} kPa")
c2.metric("(b) DEW Pressure", f"{P_dew:.2f} kPa")
c3.metric("(c) FLASH Pressure", f"{P_flash:.2f} kPa")
c4.metric("(d) AZEO Pressure", f"{P_az:.2f} kPa")

c5, c6, c7, c8 = st.columns(4)
c5.metric(f"Vapor y₁ (at x₁={x1_target})", f"{y1_bubl:.4f}")
c6.metric(f"Liquid x₁ (at y₁={y1_target})", f"{x1_dew:.4f}")
c7.metric(f"Vapor Frac V (at z₁={z1_target})", f"{V_flash:.4f}")
c8.metric("Azeotrope Comp (x₁=y₁)", f"{x1_az:.4f}")

c9, c10, c11, c12 = st.columns(4)
c9.info(f"**P1_sat:** {P1_sat:.2f} kPa")
c10.info(f"**P2_sat:** {P2_sat:.2f} kPa")
c11.success(f"**FLASH x₁:** {x1_flash:.4f}")
c12.success(f"**FLASH y₁:** {y1_flash:.4f}")

st.markdown("---")

# ==========================================
# 6. P-X-Y DIAGRAM WITH OPERATION POINTS
# ==========================================
st.subheader("P–x–y Diagram & Operation Points")

# Generate plotting arrays
x_plot = np.linspace(0, 1, 101)
P_bub_plot = np.zeros_like(x_plot)
y_plot = np.zeros_like(x_plot)

for i, x_val in enumerate(x_plot):
    g1_p, g2_p, _, _ = calc_wilson(x_val, 1 - x_val, T_sys, V1, V2, a12, a21, R)
    P_bub_plot[i] = (x_val * g1_p * P1_sat) + ((1 - x_val) * g2_p * P2_sat)
    y_plot[i] = (x_val * g1_p * P1_sat) / P_bub_plot[i]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, P_bub_plot, color="#1769aa", linewidth=2.0, label="Bubble curve (P vs x₁)")
ax.plot(y_plot, P_bub_plot, color="#d94841", linewidth=2.0, linestyle="--", label="Dew curve (P vs y₁)")

# Scatter the 4 specific operational points calculated
ax.scatter([x1_target], [P_bubl], color="blue", s=80, zorder=5, label="(a) BUBL P")
ax.scatter([y1_target], [P_dew], color="red", s=80, zorder=5, label="(b) DEW P")
ax.scatter([z1_target], [P_flash], color="green", s=80, marker="s", zorder=5, label="(c) FLASH P (Feed z₁)")
ax.scatter([x1_flash, y1_flash], [P_flash, P_flash], color="green", s=40, zorder=5)
ax.plot([x1_flash, y1_flash], [P_flash, P_flash], color="green", linestyle=":", zorder=4)
ax.scatter([x1_az], [P_az], color="purple", s=100, marker="*", zorder=5, label="(d) Azeotrope")

ax.set_title(f"Methanol (1) + Acetonitrile (2) at T = {T_sys} K")
ax.set_xlabel("Mole fraction of Methanol (x₁, y₁, z₁)")
ax.set_ylabel("Pressure (kPa)")
ax.set_xlim(0, 1.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
st.pyplot(fig)

# ==========================================
# 7. ITERATION TABLES & DOWNLOADS
# ==========================================
st.subheader("Convergence Data Tracking")

tab1, tab2 = st.tabs(["Dew Pressure Iterations", "Flash Iterations"])
df_dew = pd.DataFrame(dew_iterations)
df_flash = pd.DataFrame(flash_iterations)

with tab1:
    st.dataframe(df_dew.style.format({col: "{:.4f}" for col in df_dew.columns if col != "Iteration"}),
                 use_container_width=True)
with tab2:
    st.dataframe(df_flash.style.format({col: "{:.4f}" for col in df_flash.columns if col != "Iteration"}),
                 use_container_width=True)

csv_dew = df_dew.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Dew Data (CSV)", data=csv_dew, file_name='Q12_16_Dew.csv', mime='text/csv')