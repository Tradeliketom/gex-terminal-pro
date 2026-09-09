import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="GEX & DEX Terminal Institutional", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 15px 20px; border-radius: 10px;
    }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton>button {
        width: 100%; background-color: #238636; color: white; border-radius: 6px; font-weight: 600; border: none; padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #2ea043; }
    </style>
""", unsafe_allow_html=True)

def calculate_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    return gamma, call_delta, put_delta

def fetch_option_chain_data(ticker_symbol):
    tk = yf.Ticker(ticker_symbol)
    todays_data = tk.history(period="1d")
    if todays_data.empty: return None, None, None
    return tk, todays_data['Close'].iloc[-1], tk.options

def process_multi_expiry_metrics(tk, expiration_dates, spot_price, target_future_price, multiplier_base, interest_rate=0.05):
    all_results = []
    
    for exp_date in expiration_dates:
        try:
            opt = tk.option_chain(exp_date)
            calls, puts = opt.calls.copy(), opt.puts.copy()
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            T = max((exp_dt - datetime.now()).days / 365.0, 1/365.0)
            
            calculated_base_fut = spot_price * multiplier_base
            calibration_offset = target_future_price - calculated_base_fut
            
            for _, row in calls.iterrows():
                K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, call_delta, _ = calculate_greeks(spot_price, K_etf, T, interest_rate, sigma)
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': gamma * oi * 100 * (spot_price ** 2) * 0.01,
                    'dex': call_delta * oi * 100 * spot_price
                })
                
            for _, row in puts.iterrows():
                K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, _, put_delta = calculate_greeks(spot_price, K_etf, T, interest_rate, sigma)
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': -1 * (gamma * oi * 100 * (spot_price ** 2) * 0.01),
                    'dex': put_delta * oi * 100 * spot_price
                })
        except Exception as e:
            continue
            
    if not all_results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price
    
    df = pd.DataFrame(all_results)
    df_grouped = df.groupby('strike')[['gex', 'dex']].sum().reset_index()
    
    call_wall = df_grouped.loc[df_grouped['gex'].idxmax()]['strike']
    put_wall = df_grouped.loc[df_grouped['gex'].idxmin()]['strike']
    
    df_grouped['cumsum_gex'] = df_grouped['gex'].cumsum()
    idx_flip = (df_grouped['gex'] * df_grouped['gex'].shift(-1) < 0).idxmax()
    gamma_flip = df_grouped.loc[idx_flip, 'strike'] if not df_grouped.empty else target_future_price
    
    return df_grouped, call_wall, put_wall, gamma_flip, df['gex'].sum(), target_future_price

st.title("⚡ GEX & DEX Institutional Terminal Pro")
st.markdown("Terminal cuantitativa multi-expiración para trading táctico de futuros (**MNQ / MES**).")

with st.expander("📖 GUÍA TÁCTICA: Muros Institucionales y Dinámica de Flujo", expanded=False):
    st.markdown("""
    * **Put Wall (Soporte Principal 🟢):** Zona masiva de cobertura bajista. Ideal para buscar rebotes en largo o vigilar aceleraciones en caso de ruptura con volumen.
    * **Call Wall (Resistencia Techo 🔴):** Concentración de gamma positivo. Actúa como imán o freno de precio; excelente para tomas de beneficio.
    * **Gamma Flip (Pivote 🟣):** Frontera donde el mercado pasa de rango estabilizado (gamma positivo) a tendencia direccional volátil (gamma negativo).
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuración de Activo")
    asset_choice = st.selectbox("Seleccionar Futuro", ["MNQ (Nasdaq vía QQQ)", "MES (S&P 500 vía SPY)"])
    
    if "MNQ" in asset_choice:
        ticker_input = "QQQ"
        default_mult = 40.0
        default_price = 29367.0
    else:
        ticker_input = "SPY"
        default_mult = 10.0
        default_price = 5900.0
        
    multiplier_base = st.number_input("Multiplicador Base de Conversión", value=default_mult, step=0.1)
    target_future_price = st.number_input("Precio Real del Futuro en Bróker", value=default_price, step=1.0, format="%.2f")
    
    interest_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 5.0) / 100.0
    metric_view = st.selectbox("Métrica Principal", ["Gamma Exposure (GEX)", "Delta Exposure (DEX)"])
    st.markdown("---")
    
    # Pre-cargamos para obtener expiraciones
    tk, spot_price, expirations = fetch_option_chain_data(ticker_input)
    
    selected_expirations = []
    if expirations is not None and len(expirations) > 0:
        st.subheader("Fechas de Expiración")
        # Por defecto seleccionamos las primeras 3 (0DTE y semanales cercanas)
        default_selection = list(expirations[:3])
        selected_expirations = st.multiselect("Seleccionar Vencimientos (Agregado)", expirations, default=default_selection)
        
    calcular_btn = st.button("🚀 Calcular Perfil Institucional")

if calcular_btn:
    st.session_state['loaded'] = True

if st.session_state.get('loaded', False) and selected_expirations:
    with st.spinner(f"Agregando cadenas de opciones y calibrando niveles para {asset_choice}..."):
        df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut = process_multi_expiry_metrics(
            tk, selected_expirations, spot_price, target_future_price, multiplier_base, interest_rate
        )
        
        if df_metrics.empty:
            st.warning("No hay suficiente información para las fechas seleccionadas.")
        else:
            st.markdown(f"### 📊 Dashboard Agregado [{asset_choice}] &nbsp;&nbsp;|&nbsp;&nbsp; *Actualizado: {datetime.now().strftime('%H:%M:%S')}*")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precio Spot Futuro", f"{spot_fut:,.2f}")
            c2.metric("Gamma Flip (Pivot)", f"{gamma_flip:,.2f}")
            c3.metric("Call Wall (Techo)", f"{call_wall:,.2f}", delta="Resistencia", delta_color="inverse")
            c4.metric("Put Wall (Suelo)", f"{put_wall:,.2f}", delta="Soporte")
            c5.metric("GEX Neto Global", f"${total_gex:,.0f}")
            
            st.markdown("---")
            
            col_target = 'gex' if "Gamma" in metric_view else 'dex'
            df_metrics['Color'] = np.where(df_metrics[col_target] >= 0, 'Positivo', 'Negativo')
            
            # Gráfico avanzado con Plotly Graph Objects para combinar barras y línea de acumulación
            fig = go.Figure()
            
            # Barras de exposición
            fig.add_trace(go.Bar(
                x=df_metrics[col_target],
                y=df_metrics['strike'],
                orientation='h',
                name=metric_view,
                marker=dict(color=np.where(df_metrics[col_target] >= 0, '#00b4d8', '#ef476f'))
            ))
            
            # Líneas de referencia clave
            fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", annotation_text=f"Spot Fut: {spot_fut:.2f}", annotation_position="top right", annotation_font_color="white")
            fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#a855f7", annotation_text=f"Gamma Flip: {gamma_flip:.2f}", annotation_position="bottom right", annotation_font_color="#a855f7")
            fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", annotation_text=f"Call Wall: {call_wall:.2f}", annotation_position="top left", annotation_font_color="#22c55e")
            fig.add_hline(y=put_wall, line_dash="solid", line_color="#ef4444", annotation_text=f"Put Wall: {put_wall:.2f}", annotation_position="bottom left", annotation_font_color="#ef4444")
            
            fig.update_layout(
                title=f"Perfil Institucional Agregado ({len(selected_expirations)} Vencimientos) — {asset_choice}",
                xaxis_title=f'Exposición Neta (${metric_view})',
                yaxis_title='Nivel de Strike Calibrado',
                height=850, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff", size=12), title_font=dict(size=20, color="#ffffff"),
                xaxis=dict(showgrid=True, gridcolor='#30363d'), yaxis=dict(showgrid=True, gridcolor='#30363d', autorange="reversed"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
            
            with st.expander("🔍 Ver desglose tabular completo por Strike Agregado"):
                st.dataframe(df_metrics.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
else:
    st.info("👈 Selecciona los vencimientos deseados en la barra lateral, introduce el precio real de tu futuro y haz clic en **Calcular Perfil Institucional**.")