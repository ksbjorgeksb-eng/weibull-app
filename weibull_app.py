import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import weibull_min
import math

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

# Metrics Calculation
mttf = eta * math.gamma(1 + 1/beta)
b10 = eta * (-np.log(0.9))**(1/beta)

st.subheader("Métricas Clave")
col1, col2, col3 = st.columns(3)
col1.metric("Vida Característica (η)", f"{eta:.2f}")
col2.metric("MTTF (Tiempo Medio)", f"{mttf:.2f}")
col3.metric("Vida B10 (10% Falla)", f"{b10:.2f}")
st.divider()

# Graphics - Stacked Vertically
st.subheader("Visualizaciones")

# 1. PDF
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=pdf, mode='lines', name=f'η={eta}, β={beta}', fill='tozeroy', line=dict(color='blue')))
fig1.update_layout(title="Función de Densidad de Probabilidad (PDF) f(t)", xaxis_title="Tiempo (t)", yaxis_title="Frecuencia de Falla")
st.plotly_chart(fig1, use_container_width=True)

# 2. Reliability
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name=f'η={eta}, β={beta}', fill='tozeroy', line=dict(color='green')))
fig2.update_layout(title="Función de Confiabilidad R(t)", xaxis_title="Tiempo (t)", yaxis_title="Probabilidad de Supervivencia", yaxis=dict(range=[0, 1.05]))
st.plotly_chart(fig2, use_container_width=True)

# 3. Hazard Rate
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=hazard_rate, mode='lines', name=f'η={eta}, β={beta}', fill='tozeroy', line=dict(color='red')))
fig3.update_layout(title="Tasa de Falla (Hazard Rate) λ(t)", xaxis_title="Tiempo (t)", yaxis_title="Tasa de Falla")
if beta < 1:
    fig3.update_layout(yaxis=dict(range=[0, hazard_rate[int(len(hazard_rate)*0.1)] * 2]))
st.plotly_chart(fig3, use_container_width=True)

# Information Section
st.info(f"""
- **Tipo de falla:** {"Mortalidad Infantil (Tasa decreciente)" if beta < 1 else "Falla Aleatoria (Tasa constante)" if beta == 1 else "Desgaste (Tasa creciente)"}
- **Escala ($\\eta$):** {eta} unidades de tiempo.
- **Forma ($\\beta$):** {beta}
""")
