import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Page Configuration
st.set_page_config(page_title="Visualizador de Distribución Weibull", layout="centered")

# Title and Description
st.title("📊 Análisis de Confiabilidad: Distribución de Weibull")
st.markdown("""
Esta aplicación permite visualizar la **Distribución de Weibull**, ampliamente utilizada en ingeniería de confiabilidad para modelar el tiempo hasta la falla.
""")

# Sidebar for Parameters
st.sidebar.header("Parámetros de la Distribución")

eta = st.sidebar.slider(
    "Escala ($\eta$ - Vida Característica)", 
    min_value=1.0, 
    max_value=1000.0, 
    value=100.0, 
    step=10.0,
    help="Representa el tiempo en el cual el 63.2% de los componentes habrán fallado."
)

beta = st.sidebar.slider(
    "Forma ($\\beta$ - Pendiente)", 
    min_value=0.1, 
    max_value=5.0, 
    value=1.5, 
    step=0.1,
    help="Determina el comportamiento de la tasa de falla: <1 (infantil), 1 (constante), >1 (desgaste)."
)

# Time Range
t_max = eta * 3
t = np.linspace(0.01, t_max, 500)

# Calculations
# PDF: f(t) = (beta/eta) * (t/eta)**(beta-1) * exp(-(t/eta)**beta)
pdf = (beta / eta) * (t / eta)**(beta - 1) * np.exp(-(t / eta)**beta)

# Reliability: R(t) = exp(-(t/eta)**beta)
reliability = np.exp(-(t / eta)**beta)

# Hazard Rate: lambda(t) = (beta/eta) * (t/eta)**(beta-1)
hazard_rate = (beta / eta) * (t / eta)**(beta - 1)

# Graphics - Stacked Vertically
st.subheader("Visualizaciones")

# 1. PDF
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(t, pdf, color='blue', lw=2, label=f'$\\eta={eta}, \\beta={beta}$')
ax1.fill_between(t, pdf, alpha=0.3, color='blue')
ax1.set_title("Función de Densidad de Probabilidad (PDF) $f(t)$")
ax1.set_xlabel("Tiempo (t)")
ax1.set_ylabel("Frecuencia de Falla")
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.legend()
st.pyplot(fig1)

# 2. Reliability
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(t, reliability, color='green', lw=2, label=f'$\\eta={eta}, \\beta={beta}$')
ax2.fill_between(t, reliability, alpha=0.3, color='green')
ax2.set_title("Función de Confiabilidad $R(t)$")
ax2.set_xlabel("Tiempo (t)")
ax2.set_ylabel("Probabilidad de Supervivencia")
ax2.set_ylim(0, 1.05)
ax2.grid(True, linestyle="--", alpha=0.7)
ax2.legend()
st.pyplot(fig2)

# 3. Hazard Rate
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(t, hazard_rate, color='red', lw=2, label=f'$\\eta={eta}, \\beta={beta}$')
ax3.fill_between(t, hazard_rate, alpha=0.3, color='red')
ax3.set_title("Tasa de Falla (Hazard Rate) $\\lambda(t)$")
ax3.set_xlabel("Tiempo (t)")
ax3.set_ylabel("Tasa de Falla")
# Limit y-axis for Hazard Rate if it explodes (beta < 1)
if beta < 1:
    ax3.set_ylim(0, hazard_rate[int(len(hazard_rate)*0.1)] * 2) 
ax3.grid(True, linestyle="--", alpha=0.7)
ax3.legend()
st.pyplot(fig3)

# Information Section
st.info(f"""
- **Tipo de falla:** {"Mortalidad Infantil (Tasa decreciente)" if beta < 1 else "Falla Aleatoria (Tasa constante)" if beta == 1 else "Desgaste (Tasa creciente)"}
- **Escala ($\\eta$):** {eta} unidades de tiempo.
- **Forma ($\\beta$):** {beta}
""")
