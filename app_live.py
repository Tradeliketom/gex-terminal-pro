import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="GEX & DEX Institutional Terminal Pro", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO (Arreglo menú móvil y contraste) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 15px 20px; border-radius: 10px;
    }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    /* FORZAR VISIBILIDAD DE LAS 3 BARRAS DEL MENÚ MÓVIL Y HEADER */
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    [data-testid="stToolbar"] { right: 2rem; }
    svg[data-baseweb="icon"] { stroke: #ffffff !important; fill: #ffffff !important; }
    button[kind="header"] { color: #ffffff !important; }
    
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

def fetch_market_data(etf_symbol, future_symbol):
    tk_etf = yf.Ticker(etf_symbol)
    tk_fut = yf.Ticker(future_symbol)
    
    etf_hist = tk_etf.history(period="1d")
    fut_hist = tk_fut.history(period="1d")
    
    spot_etf = etf_hist['Close'].iloc[-1] if not etf_hist.empty else 0.0
    spot_fut = fut_hist['Close'].iloc[-1] if not fut_hist.empty else (spot_etf * 10)
    
    return tk_etf, spot_etf, spot_fut, tk_etf.options

def process_multi_expiry_metrics(tk_etf, expiration_dates, spot_etf, target_future_price, multiplier_base, interest_rate=0.05):
    all_results = []
    call_ivs = []
    put_ivs = []
    
    for exp_date in expiration_dates:
        try:
            opt = tk_etf.option_chain(exp_date)
            calls, puts = opt.calls.copy(), opt.puts.copy()
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            T = max((exp_dt - datetime.now()).days / 365.0, 1/365.0)
            
            calculated_base_fut = spot_etf * multiplier_base
            calibration_offset = target_future_price - calculated_base_fut
            
            for _, row in calls.iterrows():
                K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
                
                if abs(K_etf - spot_etf) / spot_etf < 0.05:
                    call_ivs.append(sigma)
                    
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, call_delta, _ = calculate_greeks(spot_etf, K_etf, T, interest_rate, sigma)
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': gamma * oi * 100 * (spot_etf ** 2) * 0.01,
                    'dex': call_delta * oi * 100 * spot_etf
                })
                
            for _, row in puts.iterrows():
                K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi): continue
                
                if abs(K_etf - spot_etf) / spot_etf < 0.05:
                    put_ivs.append(sigma)
                    
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, _, put_delta = calculate_greeks(spot_etf, K_etf, T, interest_rate, sigma)
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': -1 * (gamma * oi * 100 * (spot_etf ** 2) * 0.01),
                    'dex': put_delta * oi * 100 * spot_etf
                })
        except Exception:
            continue
            
    if not all_results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0
    
    df = pd.DataFrame(all_results)
    df_grouped = df.groupby('strike')[['gex', 'dex']].sum().reset_index()
    
    call_wall = df_grouped.loc[df_grouped['gex'].idxmax()]['strike']
    put_wall = df_grouped.loc[df_grouped['gex'].idxmin()]['strike']
    
    df_grouped['cumsum_gex'] = df_grouped['gex'].cumsum()
    idx_flip = (df_grouped['gex'] * df_grouped['gex'].shift(-1) < 0).idxmax()
    gamma_flip = df_grouped.loc[idx_flip, 'strike'] if not df_grouped.empty else target_future_price
    
    avg_call_iv = np.mean(call_ivs) if call_ivs else 0.0
    avg_put_iv = np.mean(put_ivs) if put_ivs else 0.0
    iv_skew = (avg_put_iv - avg_call_iv) * 100
    
    return df_grouped, call_wall, put_wall, gamma_flip, df['gex'].sum(), target_future_price, iv_skew

st.title("⚡ GEX & DEX Institutional Terminal Pro")
st.markdown("Terminal cuantitativa multi-expiración adaptada para **móvil, índices y small caps**.")

# --- GUÍA RÁPIDA INTEGRADA ---
with st.expander("📖 GUÍA RÁPIDA: Cómo operar la terminal y configurar parámetros", expanded=False):
    st.markdown("""
    ### 🛠️ Pasos de Configuración Inicial
    1. **Seleccionar el Modo:** Elige en la barra lateral si operas **Futuros / Índices** (con multiplicador) o **Acciones / Small Caps** (multiplicador 1:1).
    2. **Configurar el Ticker:** Ingresa el ETF o acción con opciones líquidas (ej. `QQQ`, `IWM` para small caps, o `AAPL`).
    3. **Seleccionar Vencimientos:** Marca los **3 o 4 vencimientos más cercanos** para capturar con precisión el flujo institucional de corto plazo (*Open Interest*).
    4. **Ajustar el Zoom para Móvil:** Usa el deslizador de rango en la barra lateral (un valor de **±3%** es ideal para evitar amontonamientos).

    ### 🎯 Claves de Interpretación
    * **Put Wall (Soporte Principal 🟢):** Zona masiva de cobertura bajista. Alta probabilidad de rebote institucional.
    * **Call Wall (Techo de Resistencia 🔴):** Resistencia magnética de corto plazo; ideal para tomas de beneficios.
    * **Gamma Flip (Pivote 🟣):** Frontera de régimen. Por encima = mercado en rango (estabilizador). Por debajo = mercado direccional y volátil.
    """, unsafe_allow_html=True)
# ------------------------------

with st.sidebar:
    st.header("⚙️ Configuración del Activo")
    
    # NUEVO: Selector de Modo (Futuros vs Acciones/Small Caps)
    modo_operativa = st.selectbox("Modo de Operativa", ["📈 Futuros / Índices", "📉 Acciones / Small Caps (Directas)"])
    
    if "Futuros" in modo_operativa:
        etf_ticker = st.text_input("Ticker de Opciones (Ej: QQQ, SPY)", value="QQQ").upper()
        fut_ticker = st.text_input("Ticker del Futuro / Activo", value="MNQ=F").upper()
        multiplier_base = st.number_input("Multiplicador de Conversión", value=40.0, step=0.1)
    else:
        etf_ticker = st.text_input("Ticker de la Small Cap / Acciones", value="IWM").upper()
        fut_ticker = etf_ticker  # Se auto-iguala al mismo ticker
        multiplier_base = 1.0    # Se fija automáticamente a 1.0 para lectura directa
        st.info("💡 **Modo Small Cap Activo:** Multiplicador ajustado a 1.0 de forma automática.")

    try:
        _, _, spot_fut_live, expirations = fetch_market_data(etf_ticker, fut_ticker)
    except:
        spot_fut_live = 100.0
        expirations = []

    target_future_price = st.number_input("Precio Live del Activo", value=float(spot_fut_live), step=0.05, format="%.2f")
    
    interest_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 5.0) / 100.0
    metric_view = st.selectbox("Métrica Principal", ["Gamma Exposure (GEX)", "Delta Exposure (DEX)"])
    
    st.markdown("---")
    st.subheader("📱 Zoom Óptimo para Móvil")
    range_pct = st.slider("Rango de Strikes (±%)", 0.5, 10.0, 3.0, step=0.5) / 100.0
    
    selected_expirations = []
    if expirations is not None and len(expirations) > 0:
        st.subheader("Fechas de Expiración")
        default_selection = list(expirations[:min(3, len(expirations))])
        selected_expirations = st.multiselect("Vencimientos (Agregado)", expirations, default=default_selection)
        
    calcular_btn = st.button("🚀 Actualizar Terminal")

if calcular_btn:
    st.session_state['loaded'] = True

if st.session_state.get('loaded', False) and selected_expirations:
    with st.spinner(f"Procesando flujos para {etf_ticker}..."):
        tk_etf, spot_etf, _, _ = fetch_market_data(etf_ticker, fut_ticker)
        
        df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut, iv_skew = process_multi_expiry_metrics(
            tk_etf, selected_expirations, spot_etf, target_future_price, multiplier_base, interest_rate
        )
        
        if df_metrics.empty:
            st.warning("No hay suficiente información disponible para este ticker.")
        else:
            min_strike = spot_fut * (1 - range_pct)
            max_strike = spot_fut * (1 + range_pct)
            df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
            
            if df_filtered.empty:
                df_filtered = df_metrics
                
            regimen = "🟢 GAMMA POSITIVO (Rango / Rebotes)" if spot_fut >= gamma_flip else "🔴 GAMMA NEGATIVO (Alta Volatilidad)"
            
            st.markdown(f"### 📊 Dashboard [{fut_ticker}] &nbsp;&nbsp;|&nbsp;&nbsp; *{datetime.now().strftime('%H:%M:%S')}*")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Spot", f"{spot_fut:,.2f}")
            c2.metric("Gamma Flip", f"{gamma_flip:,.2f}")
            c3.metric("Call Wall", f"{call_wall:,.2f}", delta="Resistencia", delta_color="inverse")
            c4.metric("Put Wall", f"{put_wall:,.2f}", delta="Soporte")
            c5.metric("IV Skew", f"{iv_skew:+.2f}%")
            
            st.info(f"**Régimen:** {regimen}")
            
            st.markdown("---")
            st.markdown("### 🎯 Plan de Acción Táctico")
            
            dist_call_wall = (call_wall - spot_fut) / spot_fut * 100
            dist_put_wall = (spot_fut - put_wall) / spot_fut * 100
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### 🧭 Sesgo Estructural")
                if spot_fut >= gamma_flip:
                    st.success("**Estrategia (Rango):** Buscar compras en soportes/Put Wall y tomas de beneficios en resistencias.")
                else:
                    st.error("**Estrategia (Direccional):** Entorno inestable. Seguir la tendencia de corto plazo.")
                if iv_skew > 0:
                    st.warning(f"**Alerta IV Skew (+{iv_skew:.1f}%):** Mayor demanda de cobertura bajista.")
            
            with col_t2:
                st.markdown("#### 📍 Proximidad a Muros")
                near_call = abs(spot_fut - call_wall) / spot_fut <= 0.004
                near_put = abs(spot_fut - put_wall) / spot_fut <= 0.004
                
                if near_call:
                    st.warning(f"⚠️ **¡Alerta Call Wall!** A **+{dist_call_wall:.2f}%** del techo ({call_wall:,.2f}).")
                elif near_put:
                    st.warning(f"⚠️ **¡Alerta Put Wall!** A **-{dist_put_wall:.2f}%** del suelo ({put_wall:,.2f}).")
                else:
                    st.info(f"ℹ️ Distancia Techo: **+{dist_call_wall:.2f}%** | Suelo: **-{dist_put_wall:.2f}%**")

            st.markdown("---")
            
            col_target = 'gex' if "Gamma" in metric_view else 'dex'
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_filtered[col_target],
                y=df_filtered['strike'],
                orientation='h',
                name=metric_view,
                marker=dict(color=np.where(df_filtered[col_target] >= 0, '#00b4d8', '#ef476f'))
            ))
            
            fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", annotation_text=f"Spot: {spot_fut:.2f}", annotation_position="top right", annotation_font_color="white")
            fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#a855f7", annotation_text=f"Flip: {gamma_flip:.2f}", annotation_position="bottom right", annotation_font_color="#a855f7")
            fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", annotation_text=f"Call Wall: {call_wall:.2f}", annotation_position="top left", annotation_font_color="#22c55e")
            fig.add_hline(y=put_wall, line_dash="solid", line_color="#ef4444", annotation_text=f"Put Wall: {put_wall:.2f}", annotation_position="bottom left", annotation_font_color="#ef4444")
            
            fig.update_layout(
                title=f"Perfil de {metric_view} (Zoom Móvil ±{int(range_pct*100)}%)",
                xaxis_title=f'Exposición Neta',
                yaxis_title='Strike',
                height=650, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff", size=11), title_font=dict(size=16, color="#ffffff"),
                xaxis=dict(showgrid=True, gridcolor='#30363d'), yaxis=dict(showgrid=True, gridcolor='#30363d', autorange="reversed"),
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
            
            with st.expander("🔍 Ver desglose tabular completo"):
                st.dataframe(df_filtered.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
else:
    st.info("👈 Configura los parámetros en la barra lateral y pulsa **Actualizar Terminal**.")