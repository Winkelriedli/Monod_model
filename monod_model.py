import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Monod equation function
def monod_equation(S, Ks, mu_max):
    return mu_max * (S / (Ks + S))


# Differential equations for substrate and cell mass
def dynamics(y, t, Ks, mu_max, Y):
    S, X = y
    mu = monod_equation(S, Ks, mu_max)
    dSdt = -mu * X / Y  # Consumption of substrate with yield
    dXdt = mu * X  # Growth of cell mass
    return [dSdt, dXdt]


# Streamlit app title and description
st.title("Monod Model - Interactive Plot")
st.markdown(
    """
### Specific Growth Rate
μ = μ_max * (s / (Ks + s))

s >> Ks, then μ ≈ μ_max

s = Ks, then μ = μ_max / 2

s << Ks, then μ ≈ μ_max * S / Ks
"""
)
st.markdown(""" """)
st.markdown(
    """
### Exponetial growth rate of cells
x(t) = x0 * e^(μ * t)

x(t) = x0 * e^(μ_max * (s / (Ks + s)) * t)

##### If the substrate concentration is above KS: s >> Ks --> μ ≈ μ_max
x(t) ≈ x0 * e^(μ_max * t)


"""
)
st.markdown(
    """
            """
)

# Sidebar sliders for parameters
st.sidebar.header("Model Parameters")
Ks = st.sidebar.slider("Ks (g/l)", min_value=0.05, max_value=10.0, value=0.5, step=0.05)
mu_max = st.sidebar.slider("μ_max (1/h)", min_value=0.5, max_value=2.5, value=2.0, step=0.1)
S0 = st.sidebar.slider("S0 (g/l)", min_value=10.0, max_value=30.0, value=25.0, step=1.0)
X0 = st.sidebar.slider("X0 (g/l)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
Y = st.sidebar.slider("Yield Coefficient (Y)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)


# Function to plot the Monod model
def plot_monod_model(Ks, mu_max, S0, X0, Y):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2)

    # Substrate concentration range for Monod equation
    S = np.linspace(0, 60, 1000)

    # Time points for solving differential equations
    t = np.linspace(0, 11, 1000)

    # First plot: Monod equation (Specific Growth Rate vs Substrate Concentration)
    axs[0].axvline(x=Ks, color="black", linestyle="--", ymax=mu_max / 5.2, lw=1, label="Ks")
    # axs[0].axhline(y=mu_max / 2, color="black", linestyle="--", xmax=Ks / 55, lw=1)
    axs[0].plot(S, monod_equation(S, Ks, mu_max), lw=2, color="green")
    axs[0].set_ylim(0, 2.6)
    axs[0].set_xlim(0, 55)
    axs[0].set_xlabel("Substrate Concentration (g/l)")
    axs[0].set_ylabel("Specific Growth Rate μ (1/h)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Specific Growth Rate vs Substrate Concentration")

    # Second plot: Substrate and Cell Concentration over time
    y0 = [S0, X0]
    sol = odeint(dynamics, y0, t, args=(Ks, mu_max, Y))
    axs[1].axhline(y=Ks, color="black", linestyle="--", lw=1, label="Ks")
    axs[1].plot(t, sol[:, 0], label="Substrate Concentration (s)", lw=2, color="blue")
    axs[1].plot(t, sol[:, 1], label="Cell Concentration (x)", lw=2, color="red")
    axs[1].set_ylim(0, 35)
    axs[1].set_xlim(0, 10.5)
    axs[1].set_xlabel("Time (h)")
    axs[1].set_ylabel("Concentration (g/l)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Substrate and Cell Concentration over Time")

    st.pyplot(fig)


# Plot the Monod model with current slider values
plot_monod_model(Ks, mu_max, S0, X0, Y)

# Plot description

st.markdown(
    """
            The point where the blue line of the substrate concentration (s) crosses the dashed line of Ks the specific growth rate (μ) is half of the maximum growth rate (μ_max).
            
            s = Ks, then μ = μ_max / 2
             """
)

# Add copyright text at the end

st.markdown("---\n\n" "**Monod Model - Interactive Plot by Thomas Schwander**")
