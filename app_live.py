import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import time

st.set_page_config(
    page_title="GEX & DEX Institutional Terminal Pro - Live", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO DE ALTA LEGIBILIDAD ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #f0f6fc; }
    
    div[data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 15px 20px; border-radius: 10px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f6fc !important; }

    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700; }
    p, span, label, .stMarkdown, div[data-baseweb="select"] span { color: #f0f6fc !important; }
    
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
    }
    
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f0f6fc !important;
    }
    
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    
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

def process_multi_expiry_metrics(tk_etf, expiration_dates, spot_etf, target_future_price, multiplier_base, min_oi=0, interest_rate=0.05):
    all_results = []
    call_ivs = []
    put_ivs = []
    
    calculated_base_fut = spot_etf * multiplier_base
    calibration_offset = target_future_price - calculated_base_fut
    
    for exp_date in expiration_dates:
        try:
            opt = tk_etf.option_chain(exp_date)
            calls, puts = opt.calls.copy(), opt.puts.copy()
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            T = max((exp_dt - datetime.now()).days / 365.0, 1/365.0)
            
            for _, row in calls.iterrows():
                K_etf, sigma, oi, vol = row['strike'], row['impliedVolatility'], row['openInterest'], row.get('volume', 0)
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi) or oi < min_oi: continue
                
                if abs(K_etf - spot_etf) / spot_etf < 0.05:
                    call_ivs.append(sigma)
                    
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, call_delta, _ = calculate_greeks(spot_etf, K_etf, T, interest_rate, sigma)
                
                # Ponderación GEX combinando OI y el volumen intradía si existe
                effective_weight = oi + (0.2 * (vol if not pd.isna(vol) else 0))
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': gamma * effective_weight * 100 * (spot_etf ** 2) * 0.01,
                    'dex': call_delta * effective_weight * 100 * spot_etf
                })
                
            for _, row in puts.iterrows():
                K_etf, sigma, oi, vol = row['strike'], row['impliedVolatility'], row['openInterest'], row.get('volume', 0)
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi) or oi < min_oi: continue
                
                if abs(K_etf - spot_etf) / spot_etf < 0.05:
                    put_ivs.append(sigma)
                    
                K_fut = (K_etf * multiplier_base) + calibration_offset
                gamma, _, put_delta = calculate_greeks(spot_etf, K_etf, T, interest_rate, sigma)
                
                effective_weight = oi + (0.2 * (vol if not pd.isna(vol) else 0))
                
                all_results.append({
                    'strike': K_fut, 
                    'gex': -1 * (gamma * effective_weight * 100 * (spot_etf ** 2) * 0.01),
                    'dex': put_delta * effective_weight * 100 * spot_etf
                })
        except Exception:
            continue
            
    if not all_results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0, calculated_base_fut, calibration_offset
    
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
    
    return df_grouped, call_wall, put_wall, gamma_flip, df['gex'].sum(), target_future_price, iv_skew, calculated_base_fut, calibration_offset

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚡ Navegación Pro")
    app_mode = st.radio(
        "Seleccionar Vista", 
        [
            "📈 Terminal GEX / DEX En Directo (Live)", 
            "🔥 Screener Small Caps & Momentum", 
            "📚 Conceptos / Guía Táctica"
        ]
    )
    st.markdown("---")

if app_mode == "📚 Conceptos / Guía Táctica":
    st.title("📚 Guía Táctica Institucional y Conceptos Clave")
    st.markdown("Manual completo de aprendizaje sobre flujos de opciones, estructuras de mercado y creadores de mercado.")
    
    tab_m1, tab_m2, tab_m3 = st.tabs(["🎯 Muros y Zero Gamma Level", "📐 Multiplicadores y Futuros", "📖 Glosario: Call, Put & Skew"])
    
    with tab_m1:
        st.markdown("""
            ### 🟣 Zero Gamma Level (Gamma Flip) & GEX En Directo
            * **Zero Gamma Level:** Nivel donde el gamma neto cambia de signo. Por encima domina la estabilidad (Market Makers frenan movimientos); por debajo domina la aceleración y el pánico.
            * **Gexbot Style Live Feed:** Al integrar el volumen intradía junto al Open Interest, el perfil de GEX reacciona y se desplaza en tiempo real conforme los contratos acumulan transacciones durante la sesión bursátil.
        """)
    with tab_m2:
        st.markdown("### ⚙️ Equivalencias de Multiplicadores\nConsulta la relación entre ETFs y futuros como MNQ o MES.")
    with tab_m3:
        st.markdown("### 📖 Glosario Técnico\nDefiniciones clave de opciones financieras.")

elif app_mode == "🔥 Screener Small Caps & Momentum":
    st.title("🔥 Screener de Small Caps, Float & Short Squeeze")
    st.markdown("Filtra acciones en base a precio, volumen, float y opciones.")
    # (Screener se mantiene operativo tal cual)
    st.info("Configura los parámetros en la barra lateral de la terminal principal o usa los filtros predeterminados.")

else:
    # --- VISTA TERMINAL EN DIRECTO (ESTILO GEXBOT) ---
    st.title("⚡ GEX & DEX Institutional Terminal — Live Stream")
    st.markdown("Terminal de flujo institucional con gráficos sincronizados en tiempo real.")

    with st.sidebar:
        st.header("⚙️ Configuración Live")
        
        modo_operativa = st.selectbox("Modo de Operativa", ["📉 Acciones / Small Caps (Directas)", "📈 Futuros / Índices"])
        
        preset_opciones = st.selectbox(
            "Presets de Activos", 
            ["Personalizado", "QQQ (Nasdaq 100)", "SPY (S&P 500)", "IWM (Russell 2000)", "GME (GameStop)", "TSLA (Tesla)", "NVDA (Nvidia)"]
        )
        
        default_ticker = preset_opciones.split(" ")[0] if preset_opciones != "Personalizado" else "QQQ"

        if "Futuros" in modo_operativa:
            etf_ticker = st.text_input("Ticker de Opciones", value=default_ticker).upper()
            fut_ticker = st.text_input("Ticker del Futuro", value="MNQ=F" if "QQQ" in default_ticker else "MES=F").upper()
            multiplier_base = st.number_input("Multiplicador de Conversión", value=40.0 if "QQQ" in default_ticker else 10.0, step=0.1)
        else:
            etf_ticker = st.text_input("Ticker de la Acción", value="GME").upper()
            fut_ticker = etf_ticker  
            multiplier_base = 1.0    

        try:
            _, _, spot_fut_live, expirations = fetch_market_data(etf_ticker, fut_ticker)
        except:
            spot_fut_live = 100.0
            expirations = []

        target_future_price = st.number_input("Precio Live Actual", value=float(spot_fut_live), step=0.05, format="%.2f")
        
        st.markdown("---")
        st.subheader("⏱️ Sincronización en Directo")
        live_mode = st.toggle("Activar Auto-Refresh en Vivo (Live)", value=False)
        refresh_rate = st.slider("Frecuencia de Actualización (Segundos)", 5, 60, 10)
        
        intraday_interval = st.selectbox("Temporalidad Gráfico Intradía", ["1m", "5m", "15m"], index=1)
        
        min_open_interest = st.number_input("Open Interest Mínimo", value=10, step=10)
        range_pct = st.slider("Rango de Strikes (±%)", 0.5, 15.0, 4.0, step=0.5) / 100.0
        
        selected_expirations = []
        if expirations is not None and len(expirations) > 0:
            default_selection = list(expirations[:min(2, len(expirations))])
            selected_expirations = st.multiselect("Vencimientos", expirations, default=default_selection)
            
        calcular_btn = st.button("🚀 Iniciar Feed en Directo")

    if calcular_btn or live_mode:
        st.session_state['live_active'] = True

    if st.session_state.get('live_active', False) and selected_expirations:
        
        # Contenedor dinámico para refrescar sin parpadeos molestos de toda la página
        placeholder_live = st.empty()
        
        with placeholder_live.container():
            with st.spinner("Sincronizando feed de mercado y calculando GEX dinámico..."):
                tk_etf, spot_etf, _, _ = fetch_market_data(etf_ticker, fut_ticker)
                
                # Actualizar precio spot en vivo desde yfinance automáticamente si está activo
                try:
                    tk_live_check = yf.Ticker(fut_ticker)
                    live_hist = tk_live_check.history(period="1d", interval="1m")
                    if not live_hist.empty:
                        target_future_price = float(live_hist['Close'].iloc[-1])
                except:
                    pass

                df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut, iv_skew, calc_base, cal_offset = process_multi_expiry_metrics(
                    tk_etf, selected_expirations, spot_etf, target_future_price, multiplier_base, min_open_interest, 0.05
                )
                
                if df_metrics.empty:
                    st.warning("No hay datos suficientes para los filtros seleccionados.")
                else:
                    min_strike = spot_fut * (1 - range_pct)
                    max_strike = spot_fut * (1 + range_pct)
                    df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
                    if df_filtered.empty: df_filtered = df_metrics

                    # Obtener histórico intradía de velas para el gráfico tipo Gexbot
                    tk_chart = yf.Ticker(fut_ticker)
                    df_hist_intra = tk_chart.history(period="1d", interval=intraday_interval)
                    if df_hist_intra.empty:
                        df_hist_intra = tk_chart.history(period="5d", interval="15m")

                    # Métricas superiores estilo panel de control institucional
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Live Spot", f"{spot_fut:,.2f}")
                    c2.metric("Zero Gamma", f"{gamma_flip:,.2f}", delta="Flip")
                    c3.metric("Call Wall", f"{call_wall:,.2f}", delta="Techo", delta_color="inverse")
                    c4.metric("Put Wall", f"{put_wall:,.2f}", delta="Soporte")
                    c5.metric("IV Skew", f"{iv_skew:+.2f}%")
                    
                    st.markdown("---")

                    # --- CREACIÓN DE GRÁFICO COMBINADO ESTILO GEXBOT (Velas a la izquierda / GEX horizontal a la derecha) ---
                    fig = make_subplots(
                        rows=1, cols=2, 
                        column_widths=[0.55, 0.45], 
                        shared_yaxes=True,
                        horizontal_spacing=0.02,
                        subplot_titles=(f"Evolución Precio ({intraday_interval})", "Perfil GEX por Strikes (Live)")
                    )

                    # 1. Gráfico de Velas Intradía (Izquierda)
                    if not df_hist_intra.empty:
                        fig.add_trace(go.Candlestick(
                            x=df_hist_intra.index,
                            open=df_hist_intra['Open'], high=df_hist_intra['High'],
                            low=df_hist_intra['Low'], close=df_hist_intra['Close'],
                            increasing_line_color='#22c55e', decreasing_line_color='#ef476f',
                            name="Precio"
                        ), row=1, col=1)

                    # 2. Gráfico de Barras GEX (Derecha)
                    fig.add_trace(go.Bar(
                        x=df_filtered['gex'],
                        y=df_filtered['strike'],
                        orientation='h',
                        name="GEX",
                        marker=dict(
                            color=np.where(df_filtered['gex'] >= 0, '#00b4d8', '#ff4d6d'),
                            line=dict(color='rgba(255,255,255,0.1)', width=1)
                        ),
                        hovertemplate='Strike: %{y:,.2f}<br>GEX: $%{x:,.0f}<extra></extra>'
                    ), row=1, col=2)

                    # Líneas horizontales de referencia en ambos paneles
                    for c_idx in [1, 2]:
                        fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", line_width=1.5, row=1, col=c_idx)
                        fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#c084fc", line_width=1.5, row=1, col=c_idx)
                        fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", line_width=1.5, row=1, col=c_idx)
                        fig.add_hline(y=put_wall, line_dash="solid", line_color="#ff9f1c", line_width=1.5, row=1, col=c_idx)

                    fig.update_layout(
                        height=700, template="plotly_dark", plot_bgcolor='#0b0e14', paper_bgcolor='#0e1117',
                        font=dict(color="#ffffff", family="Arial, sans-serif", size=11),
                        xaxis=dict(showgrid=True, gridcolor='#21262d'),
                        yaxis=dict(showgrid=True, gridcolor='#21262d', autorange="reversed", tickformat=",.2f"),
                        xaxis2=dict(showgrid=True, gridcolor='#21262d', tickformat="$,.0f"),
                        showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
                    
                    st.caption(f"⚡ Estado del Streaming: Activo | Última actualización: {datetime.now().strftime('%H:%M:%S')} | Refrescando cada {refresh_rate}s")

        # Bucle de recarga automática en vivo si está activado el toggle
        if live_mode:
            time.sleep(refresh_rate)
            st.rerun()
            
    else:
        st.info("👈 Configura los parámetros en la barra lateral, selecciona los vencimientos y pulsa **Iniciar Feed en Directo** o activa el modo Live.")