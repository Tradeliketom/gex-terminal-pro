import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
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
                K_etf, sigma, oi = row['strike'], row['impliedVolatility'], row['openInterest']
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi) or oi < min_oi: continue
                
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
                if pd.isna(sigma) or sigma == 0 or pd.isna(oi) or oi < min_oi: continue
                
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
            
    if not all_results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0, calculated_base_fut, calibration_offset
    
    df = pd.DataFrame(all_results)
    df_grouped = df.groupby('strike')[['gex', 'dex']].sum().reset_index()
    df_grouped = df_grouped.sort_values('strike').reset_index(drop=True)
    
    call_walls = df_grouped.loc[df_grouped['gex'].idxmax()]['strike']
    put_walls = df_grouped.loc[df_grouped['gex'].idxmin()]['strike']
    
    # --- CÁLCULO ROBUSTO DE ZERO GAMMA CERCANO AL SPOT ---
    df_grouped['cumsum_gex'] = df_grouped['gex'].cumsum()
    sign_changes = (df_grouped['cumsum_gex'] * df_grouped['cumsum_gex'].shift(-1)) < 0
    
    if sign_changes.any():
        valid_indices = sign_changes[sign_changes].index
        closest_idx = min(valid_indices, key=lambda idx: abs(df_grouped.loc[idx, 'strike'] - target_future_price))
        gamma_flip = df_grouped.loc[closest_idx, 'strike']
    else:
        gamma_flip = df_grouped.loc[(df_grouped['strike'] - target_future_price).abs().idxmin(), 'strike']
    
    avg_call_iv = np.mean(call_ivs) if call_ivs else 0.0
    avg_put_iv = np.mean(put_ivs) if put_ivs else 0.0
    iv_skew = (avg_put_iv - avg_call_iv) * 100
    
    return df_grouped, call_walls, put_walls, gamma_flip, df['gex'].sum(), target_future_price, iv_skew, calculated_base_fut, calibration_offset

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
    st.markdown("""
    ### 1. ¿Qué es el Gamma Exposure (GEX)?
    El **GEX** mide la exposición neta a la gamma de los creadores de mercado (*Market Makers*). Cuando los inversores compran o venden opciones, los creadores de mercado asumen el riesgo delta opuesto y deben cubrirse comprando o vendiendo el activo subyacente de forma dinámica.
    * **Gamma Positivo (Dealer Long Gamma):** Los creadores de mercado compran cuando el mercado cae y venden cuando sube. Esto **estabiliza** el mercado y reduce la volatilidad.
    * **Gamma Negativo (Dealer Short Gamma):** Los creadores de mercado venden cuando el mercado cae y compran cuando sube. Esto **acelera** los movimientos, generando alta volatilidad y caídas o subidas bruscas.

    ### 2. Niveles Clave del Terminal
    * **Spot:** El precio actual de cotización del activo subyacente o futuro.
    * **Zero Gamma (Gamma Flip):** El punto exacto donde la gamma neta pasa de positiva a negativa. Actúa como el umbral crítico de volatilidad.
    * **Call Wall:** El strike con mayor concentración de Gamma positiva en opciones de compra (*Calls*). Funciona como un fuerte imán o resistencia técnica.
    * **Put Wall:** El strike con mayor concentración de exposición en opciones de venta (*Puts*). Funciona como soporte estructural principal.

    ### 3. Delta Exposure (DEX)
    El **DEX** refleja la exposición direccional acumulada de las opciones abiertas, ayudando a identificar desequilibrios masivos de posicionamiento direccional a corto y medio plazo.
    """)

elif app_mode == "🔥 Screener Small Caps & Momentum":
    st.title("🔥 Screener de Small Caps, Float & Short Squeeze")
    st.markdown("Herramienta de análisis cuantitativo para identificar acciones de baja capitalización con alto potencial de momentum alcista.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.selectbox("Filtrar por Mercado", ["NASDAQ / NYSE", "US Small Caps", "Penny Stocks"])
    col_s2.number_input("Float Máximo (Millones)", value=20.0, step=5.0)
    col_s3.number_input("Short Interest Mínimo (%)", value=20.0, step=5.0)
    
    st.info("💡 Consejo táctico: Busca empresas con un *Float* inferior a 15 millones de acciones y un *Short Interest* superior al 30% combinados con volumen inusual.")
    
    mock_screener_data = pd.DataFrame({
        "Ticker": ["GME", "AMC", "FFIE", "SPCE", "SAVA"],
        "Precio ($)": [24.50, 5.20, 0.45, 3.80, 18.20],
        "Float (M)": [305.0, 260.0, 42.0, 38.0, 45.0],
        "Short Interest (%)": [22.4, 18.5, 34.2, 28.1, 41.5],
        "Volumen Relativo": [3.2, 1.5, 8.4, 2.1, 5.9]
    })
    st.dataframe(mock_screener_data, use_container_width=True)

else:
    # --- VISTA TERMINAL GEX / DEX COMPLETA ---
    st.title("⚡ GEX & DEX Institutional Terminal — Live Stream (10s)")
    st.markdown("Análisis estructural de flujos de opciones con actualización automática cada 10 segundos.")

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
        st.subheader("⏱️ Automatización 10s")
        live_mode = st.toggle("Activar Auto-Refresh (10s en Vivo)", value=True)
        
        min_open_interest = st.number_input("Open Interest Mínimo", value=100, step=50)
        range_pct = st.slider("Rango de Strikes (±%)", 1.0, 30.0, 10.0, step=1.0) / 100.0
        
        selected_expirations = []
        if expirations is not None and len(expirations) > 0:
            default_selection = list(expirations[:min(2, len(expirations))])
            selected_expirations = st.multiselect("Vencimientos", expirations, default=default_selection)
            
        calcular_btn = st.button("🚀 Iniciar / Forzar Actualización")

    if calcular_btn or live_mode:
        st.session_state['live_active'] = True

    if st.session_state.get('live_active', False) and selected_expirations:
        
        with st.spinner("Calculando perfiles GEX y DEX institucionales..."):
            tk_etf, spot_etf, _, _ = fetch_market_data(etf_ticker, fut_ticker)
            
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
                min_strike = min(spot_fut * (1 - range_pct), put_wall * 0.99)
                max_strike = max(spot_fut * (1 + range_pct), call_wall * 1.01)
                
                df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
                if df_filtered.empty: df_filtered = df_metrics

                # --- EVALUACIÓN DEL SESGO DE MERCADO (SEMÁFORO) ---
                if spot_fut > gamma_flip:
                    bias_text = "🟢 ALCISTA / ESTABLE (Por encima de Zero Gamma)"
                    bias_desc = "Los Market Makers actúan absorbiendo volatilidad y frenando los movimientos bajistas bruscos."
                elif spot_fut < gamma_flip:
                    bias_text = "🔴 BAJISTA / ACELERACIÓN (Por debajo de Zero Gamma)"
                    bias_desc = "Zona propensa a alta volatilidad y aceleración de movimientos direccionales por cobertura corta."
                else:
                    bias_text = "🟡 NEUTRAL / PUNTO DE INFLEXIÓN"
                    bias_desc = "El precio se encuentra exactamente sobre el nivel de cambio de signo de gamma."

                # Panel de métricas superiores
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Live Spot", f"{spot_fut:,.2f}")
                c2.metric("Gamma Flip", f"{gamma_flip:,.2f}")
                c3.metric("Call Wall", f"{call_wall:,.2f}")
                c4.metric("Put Wall", f"{put_wall:,.2f}")
                c5.metric("Net GEX", f"${total_gex:,.0f}")
                c6.metric("IV Skew", f"{iv_skew:+.2f}%")
                
                # --- CAJA SUPERIOR DE SESGO ---
                st.markdown(f"""
                    <div style="background-color: #161b22; border-left: 5px solid {'#22c55e' if 'ALCISTA' in bias_text else '#ef476f' if 'BAJISTA' in bias_text else '#ffd166'}; padding: 12px 18px; border-radius: 6px; margin-top: 10px; margin-bottom: 20px;">
                        <span style="font-size: 14px; color: #8b949e; font-weight: bold;">SESGO INSTITUCIONAL ACTUAL:</span>
                        <div style="font-size: 18px; font-weight: bold; color: #ffffff; margin-top: 2px;">{bias_text}</div>
                        <div style="font-size: 12px; color: #c9d1d9; margin-top: 2px;">{bias_desc}</div>
                    </div>
                """, unsafe_allow_html=True)

                # --- EXTENSIÓN TRADINGVIEW PINE SCRIPT ---
                with st.expander("🔗 Extensión TradingView: Ver Niveles en tus Gráficos", expanded=False):
                    st.markdown("Copia este código **Pine Script v5** y pégalo en el editor de tu gráfico en TradingView para ver los muros institucionales en tiempo real:")
                    
                    pine_script_code = f"""// //@version=5
indicator("GEX Levels Institutional Pro", overlay=true)

call_wall = input.float({call_wall:.2f}, title="Call Wall")
put_wall = input.float({put_wall:.2f}, title="Put Wall")
gamma_flip = input.float({gamma_flip:.2f}, title="Zero Gamma")

plot(call_wall, title="Call Wall", color=color.green, linewidth=2, style=plot.style_line)
plot(put_wall, title="Put Wall", color=color.red, linewidth=2, style=plot.style_line)
plot(gamma_flip, title="Zero Gamma", color=color.purple, linewidth=2, style=plot.style_dash)
"""
                    st.code(pine_script_code, language="pine")
                    st.markdown(f"[Abrir TradingView Directamente](https://www.tradingview.com/chart/?symbol={fut_ticker})", unsafe_allow_html=True)

                # --- GRÁFICOS GEX Y DEX CON PAN, ZOOM Y 4 ETIQUETAS BLANCAS EN AMBOS ---
                col_gex, col_dex = st.columns(2)
                
                def add_chart_lines(fig):
                    fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e",
                                  annotation_text=f"Call Wall: {call_wall:,.2f}", 
                                  annotation_position="top left",
                                  annotation_font=dict(color="white", size=11),
                                  annotation_bgcolor="#161b22")
                    fig.add_hline(y=put_wall, line_dash="solid", line_color="#ef476f",
                                  annotation_text=f"Put Wall: {put_wall:,.2f}", 
                                  annotation_position="bottom left",
                                  annotation_font=dict(color="white", size=11),
                                  annotation_bgcolor="#161b22")
                    
                    fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166",
                                  annotation_text=f"Spot: {spot_fut:,.2f}", 
                                  annotation_position="top right",
                                  annotation_font=dict(color="white", size=11),
                                  annotation_bgcolor="#161b22")
                    fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#c084fc",
                                  annotation_text=f"Zero Gamma: {gamma_flip:,.2f}", 
                                  annotation_position="bottom right",
                                  annotation_font=dict(color="white", size=11),
                                  annotation_bgcolor="#161b22")

                with col_gex:
                    st.subheader("📊 Gamma Exposure (GEX)")
                    fig_gex = go.Figure()
                    fig_gex.add_trace(go.Bar(
                        x=df_filtered['gex'],
                        y=df_filtered['strike'],
                        orientation='h',
                        marker=dict(color=np.where(df_filtered['gex'] >= 0, '#00b4d8', '#ff4d6d'))
                    ))
                    
                    add_chart_lines(fig_gex)
                    
                    fig_gex.update_layout(
                        height=600, template="plotly_dark", plot_bgcolor='#0b0e14', paper_bgcolor='#0e1117',
                        dragmode="pan",
                        yaxis=dict(autorange="reversed", tickformat=",.2f", range=[max(df_filtered['strike']), min(df_filtered['strike'])], fixedrange=False), 
                        xaxis=dict(tickformat="$,.0f", fixedrange=False),
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_gex, use_container_width=True, config={"scrollZoom": True})

                with col_dex:
                    st.subheader("📉 Delta Exposure (DEX)")
                    fig_dex = go.Figure()
                    fig_dex.add_trace(go.Bar(
                        x=df_filtered['dex'],
                        y=df_filtered['strike'],
                        orientation='h',
                        marker=dict(color=np.where(df_filtered['dex'] >= 0, '#22c55e', '#ef476f'))
                    ))
                    
                    add_chart_lines(fig_dex)
                    
                    fig_dex.update_layout(
                        height=600, template="plotly_dark", plot_bgcolor='#0b0e14', paper_bgcolor='#0e1117',
                        dragmode="pan",
                        yaxis=dict(autorange="reversed", tickformat=",.2f", range=[max(df_filtered['strike']), min(df_filtered['strike'])], fixedrange=False), 
                        xaxis=dict(tickformat="$,.0f", fixedrange=False),
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_dex, use_container_width=True, config={"scrollZoom": True})
                
                st.caption(f"⚡ Streaming Activo | Última actualización: {datetime.now().strftime('%H:%M:%S')} — Recargando automáticamente cada **10 segundos**.")

        if live_mode:
            time.sleep(10)
            st.rerun()
            
    else:
        st.info("👈 Configura los parámetros en la barra lateral, selecciona los vencimientos y pulsa **Iniciar / Forzar Actualización**.")