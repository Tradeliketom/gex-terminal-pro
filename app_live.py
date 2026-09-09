import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="GEX & DEX Terminal Pro (Guía Táctica)", 
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
    if T <= 0 or sigma <= 0: return 0, 0
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

def process_trader_metrics(tk, expiration_date, spot_price, target_future_price, multiplier_base, interest_rate=0.05):
    opt = tk.option_chain(expiration_date)
    calls, puts = opt.calls.copy(), opt.puts.copy()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - datetime.now()).days / 365.0, 1/365.0)
    
    calculated_base_fut = spot_price * multiplier_base
    calibration_offset = target_future_price - calculated_base_fut
    
    results = []
    for _, row in calls.iterrows():
        K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
        if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
        
        K_fut = (K_etf * multiplier_base) + calibration_offset
        gamma, call_delta, _ = calculate_greeks(spot_price, K_etf, T, interest_rate, sigma)
        
        results.append({
            'strike': K_fut, 
            'gex': gamma * oi * 100 * (spot_price ** 2) * 0.01,
            'dex': call_delta * oi * 100 * spot_price
        })
        
    for _, row in puts.iterrows():
        K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
        if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
        
        K_fut = (K_etf * multiplier_base) + calibration_offset
        gamma, _, put_delta = calculate_greeks(spot_price, K_etf, T, interest_rate, sigma)
        
        results.append({
            'strike': K_fut, 
            'gex': -1 * (gamma * oi * 100 * (spot_price ** 2) * 0.01),
            'dex': put_delta * oi * 100 * spot_price
        })
        
    if not results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price
    
    df = pd.DataFrame(results)
    df_grouped = df.groupby('strike')[['gex', 'dex']].sum().reset_index()
    
    call_wall = df_grouped.loc[df_grouped['gex'].idxmax()]['strike']
    put_wall = df_grouped.loc[df_grouped['gex'].idxmin()]['strike']
    
    df_grouped['cumsum_gex'] = df_grouped['gex'].cumsum()
    idx_flip = (df_grouped['gex'] * df_grouped['gex'].shift(-1) < 0).idxmax()
    gamma_flip = df_grouped.loc[idx_flip, 'strike'] if not df_grouped.empty else target_future_price
    
    return df_grouped, call_wall, put_wall, gamma_flip, df['gex'].sum(), target_future_price

st.title("⚡ GEX & DEX Institutional Terminal Pro")
st.markdown("Terminal cuantitativa de opciones aplicada al trading de futuros (**MNQ / MES**).")

# SECCIÓN DE GUÍA TÁCTICA PARA TRADERS
with st.expander("📖 GUÍA TÁCTICA: Conceptos Institucionales y Cómo Operarlos", expanded=False):
    st.markdown("""
    ### 1. Put Wall (Muro de Puts) 🟢 — *El Suelo Institucional*
    * **Qué es:** Es el strike con la mayor concentración de gamma negativo / soporte masivo en opciones de venta. Representa el nivel donde los creadores de mercado (*market makers*) tienen la mayor cobertura de riesgo bajista.
    * **Cómo operarlo:** 
      * **Rebote:** Cuando el precio se acerca al Put Wall, es una zona de alta probabilidad de parada y rebote en largo (compras). Los institucionales suelen defender este nivel.
      * **Ruptura:** Si el precio rompe con volumen y cierra por debajo del Put Wall, se activa una venta en cascada (aceleración bajista), ya que los creadores de mercado se ven obligados a vender futuros agresivamente para cubrir su delta.

    ### 2. Call Wall (Muro de Calls) 🔴 — *El Techo Magnético*
    * **Qué es:** Es el strike con mayor gamma positivo acumulado. Actúa como una gran resistencia magnética de corto plazo.
    * **Cómo operarlo:**
      * **Toma de Beneficios / Cortos en Contratendencia:** Es el objetivo ideal (*take profit*) para posiciones alcistas. Al tocarlo, el mercado tiende a frenarse y hacer rangos o pequeños retrocesos.
      * **Ruptura Alcista (Squeeze):** Si el precio perfora el Call Wall hacia arriba, provoca un efecto imán muy fuerte (*Gamma Squeeze*), obligando a los creadores de mercado a comprar masivamente el subyacente para equilibrar sus carteras.

    ### 3. Gamma Flip (Punto Pivote) 🟣 — *El Cambio de Régimen*
    * **Qué es:** Es el precio exacto donde la exposición neta cambia de signo (de positivo a negativo o viceversa).
    * **Cómo operarlo:**
      * **Por encima del Gamma Flip (Gamma Positivo):** El mercado es de rangos y rebotes. Los market makers compran las caídas y venden las subidas (comportamiento estabilizador).
      * **Por debajo del Gamma Flip (Gamma Negativo):** El mercado se vuelve altamente direccional, volátil y acelerado. La volatilidad implícita y real explotan.

    ### 4. DEX (Delta Exposure) 🔵 — *El Sesgo Direccional*
    * **Qué es:** Mide el desequilibrio de delta neto acumulado por los creadores de mercado. Indica si el trasfondo institucional empuja estructuralmente al mercado al alza (positivo) o a la baja (negativo).
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
    target_future_price = st.number_input("Precio Real del Futuro en tu Bróker", value=default_price, step=1.0, format="%.2f")
    
    interest_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 5.0) / 100.0
    metric_view = st.selectbox("Métrica Principal", ["Gamma Exposure (GEX)", "Delta Exposure (DEX)"])
    st.markdown("---")
    calcular_btn = st.button("🚀 Calcular y Sincronizar Niveles")

if calcular_btn:
    st.session_state['loaded'] = True

if st.session_state.get('loaded', False):
    with st.spinner(f"Calibrando niveles para {asset_choice} con base en {target_future_price:,.2f}..."):
        tk, spot_price, expirations = fetch_option_chain_data(ticker_input)
        
        if tk is None or expirations is None or len(expirations) == 0:
            st.error("No se pudieron obtener datos para el ticker proporcionado.")
        else:
            selected_expiry = st.sidebar.selectbox("Fecha de Expiración", expirations, index=0)
            df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut = process_trader_metrics(
                tk, selected_expiry, spot_price, target_future_price, multiplier_base, interest_rate
            )
            
            if df_metrics.empty:
                st.warning("No hay suficiente información de opciones para esta fecha.")
            else:
                st.markdown(f"### 📊 Dashboard Calibrado [{asset_choice}] &nbsp;&nbsp;|&nbsp;&nbsp; *Actualizado: {datetime.now().strftime('%H:%M:%S')}*")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Precio Futuro Spot", f"{spot_fut:,.2f}")
                c2.metric("Gamma Flip (Pivot)", f"{gamma_flip:,.2f}")
                c3.metric("Call Wall (Techo)", f"{call_wall:,.2f}", delta="Resistencia", delta_color="inverse")
                c4.metric("Put Wall (Suelo)", f"{put_wall:,.2f}", delta="Soporte")
                c5.metric("GEX Neto", f"${total_gex:,.0f}")
                
                st.markdown("---")
                
                col_target = 'gex' if "Gamma" in metric_view else 'dex'
                df_metrics['Color'] = np.where(df_metrics[col_target] >= 0, 'Positivo', 'Negativo')
                
                fig = px.bar(
                    df_metrics, x=col_target, y='strike', orientation='h',
                    title=f"Perfil de {metric_view} Sincronizado — {asset_choice} [Exp: {selected_expiry}]",
                    labels={col_target: f'Exposición Neta (${metric_view})', 'strike': 'Nivel de Strike Calibrado'},
                    color='Color', color_discrete_map={'Positivo': '#00b4d8', 'Negativo': '#ef476f'}
                )
                
                fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", annotation_text=f"Spot Fut: {spot_fut:.2f}", annotation_position="top right", annotation_font_color="white")
                fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#a855f7", annotation_text=f"Gamma Flip: {gamma_flip:.2f}", annotation_position="bottom right", annotation_font_color="#a855f7")
                fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", annotation_text=f"Call Wall: {call_wall:.2f}", annotation_position="top left", annotation_font_color="#22c55e")
                fig.add_hline(y=put_wall, line_dash="solid", line_color="#ef4444", annotation_text=f"Put Wall: {put_wall:.2f}", annotation_position="bottom left", annotation_font_color="#ef4444")
                
                fig.update_layout(
                    height=850, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#ffffff", size=12), title_font=dict(size=20, color="#ffffff"),
                    xaxis=dict(showgrid=True, gridcolor='#30363d'), yaxis=dict(showgrid=True, gridcolor='#30363d', autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                
                with st.expander("🔍 Ver desglose tabular completo por Strike"):
                    st.dataframe(df_metrics.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
else:
    st.info("👈 Selecciona tu activo, introduce el precio real en la barra lateral y haz clic en **Calcular y Sincronizar Niveles**.")