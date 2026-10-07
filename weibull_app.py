import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import weibull_min
import math
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Visualizador de Distribución Weibull", layout="centered")

# Title and Description
st.title("📊 Análisis de Confiabilidad: Distribución de Weibull")
st.markdown("""
Esta aplicación permite visualizar la **Distribución de Weibull**, ampliamente utilizada en ingeniería de confiabilidad para modelar el tiempo hasta la falla.
""")

# Default values
ETA_DEFAULT, BETA_DEFAULT = 100.0, 1.5
MIN_DATOS = 3

# Parameters panel (in the main body so it is visible on phones,
# where Streamlit hides the sidebar behind a menu)
with st.expander("⚙️ Configuración de Datos", expanded=True):
    input_mode = st.radio("Modo de Ingreso:", ["Manual", "Cargar Archivo CSV"], horizontal=True)

    if input_mode == "Manual":
        st.markdown("**Parámetros Teóricos**")
        eta = st.slider(
            "Escala (η - Vida Característica)",
            min_value=1.0,
            max_value=1000.0,
            value=ETA_DEFAULT,
            step=10.0,
            help="Representa el tiempo en el cual el 63.2% de los componentes habrán fallado."
        )

        beta = st.slider(
            "Forma (β - Pendiente)",
            min_value=0.1,
            max_value=5.0,
            value=BETA_DEFAULT,
            step=0.1,
            help="Determina el comportamiento de la tasa de falla: <1 (infantil), 1 (constante), >1 (desgaste)."
        )
    else:
        st.markdown("**Ajuste por Datos Empíricos**")
        uploaded_file = st.file_uploader(
            "Sube tu archivo (CSV)",
            type=["csv", "txt"],
            help="La primera columna debe contener los tiempos de falla."
        )
        eta, beta = ETA_DEFAULT, BETA_DEFAULT
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
                descartados = len(df) - len(data)
                no_positivos = int((data <= 0).sum())
                data = data[data > 0]

                if len(data) < MIN_DATOS:
                    st.error(
                        f"Se necesitan al menos {MIN_DATOS} tiempos de falla válidos (números mayores que 0). "
                        f"Se encontraron {len(data)}. Usando valores por defecto."
                    )
                else:
                    shape, loc, scale = weibull_min.fit(data, floc=0)
                    beta, eta = float(shape), float(scale)
                    st.success(
                        f"¡Ajuste Exitoso con {len(data)} datos!\n\n"
                        f"η estimado = {eta:.2f}  ·  β estimado = {beta:.2f}"
                    )
                    if descartados or no_positivos:
                        st.warning(
                            f"Se ignoraron {descartados + no_positivos} valores "
                            f"(vacíos, no numéricos o ≤ 0)."
                        )
            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")
                eta, beta = ETA_DEFAULT, BETA_DEFAULT
        else:
            st.info("Esperando archivo... Usando valores por defecto.")

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
etiqueta = f"η={eta:.2f}, β={beta:.2f}"

# 1. PDF
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=t, y=pdf, mode='lines', name=etiqueta, fill='tozeroy', line=dict(color='blue')))
fig1.update_layout(title="Función de Densidad de Probabilidad (PDF) f(t)", xaxis_title="Tiempo (t)", yaxis_title="Frecuencia de Falla")
st.plotly_chart(fig1, width='stretch')

# 2. Reliability
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name=etiqueta, fill='tozeroy', line=dict(color='green')))
fig2.update_layout(title="Función de Confiabilidad R(t)", xaxis_title="Tiempo (t)", yaxis_title="Probabilidad de Supervivencia", yaxis=dict(range=[0, 1.05]))
st.plotly_chart(fig2, width='stretch')

# 3. Hazard Rate
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=t, y=hazard_rate, mode='lines', name=etiqueta, fill='tozeroy', line=dict(color='red')))
fig3.update_layout(title="Tasa de Falla (Hazard Rate) λ(t)", xaxis_title="Tiempo (t)", yaxis_title="Tasa de Falla")
if beta < 1:
    fig3.update_layout(yaxis=dict(range=[0, hazard_rate[int(len(hazard_rate)*0.1)] * 2]))
st.plotly_chart(fig3, width='stretch')

# Failure type (with a tolerance band around β = 1, since a fitted β is never exactly 1)
TOLERANCIA_BETA = 0.05
if beta < 1 - TOLERANCIA_BETA:
    tipo_falla = "Mortalidad Infantil (Tasa decreciente)"
elif beta <= 1 + TOLERANCIA_BETA:
    tipo_falla = "Falla Aleatoria (Tasa aproximadamente constante)"
else:
    tipo_falla = "Desgaste (Tasa creciente)"

# Information Section
st.info(f"""
- **Tipo de falla:** {tipo_falla}
- **Escala ($\\eta$):** {eta:.2f} unidades de tiempo.
- **Forma ($\\beta$):** {beta:.2f}
""")
