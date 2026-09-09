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
            "📈 Terminal GEX / DEX Individual", 
            "🔥 Screener Small Caps & Momentum", 
            "📚 Conceptos / Guía Táctica"
        ]
    )
    st.markdown("---")

if app_mode == "📚 Conceptos / Guía Táctica":
    st.title("📚 Guía Táctica Institucional y Conceptos Clave")
    st.markdown("Manual completo de aprendizaje sobre flujos de opciones, estructuras de mercado, cálculo de multiplicadores y perfil de creadores de mercado.")
    
    tab_m1, tab_m2, tab_m3 = st.tabs(["🎯 Muros y Dinámica de GEX", "📐 Multiplicadores y Futuros", "📖 Glosario: Call, Put & Skew"])
    
    with tab_m1:
        st.markdown("""
            ### 🧱 Put Wall y Call Wall (Los Muros Institucionales)
            * **Put Wall (🟠 Suelo Institucional):** Es el nivel de Strike que acumula la mayor concentración neta de contratos Put abiertos. Representa un soporte crítico del mercado. ¿Por qué? Porque los creadores de mercado (Market Makers) venden masivamente opciones Put a los inversores y, para cubrir su riesgo direccional (Delta/Gamma), compran acciones o futuros subyacentes masivamente a medida que el precio se acerca a este nivel, frenando la caída.
            * **Call Wall (🟢 Techo Institucional):** Strike con la mayor concentración de contratos Call. Actúa como una resistencia o imán superior. Los creadores de mercado se ven obligados a vender subyacentes o desentenderse de coberturas al alcanzar este nivel, creando un techo rígido.
            * **Gamma Flip (🟣):** La línea divisoria donde la exposición neta de gamma pasa de positiva a negativa.
              * **Gamma Positivo (+):** Estabilidad. Los Market Makers actúan como amortiguadores (compran caídas, venden subidas).
              * **Gamma Negativo (-):** Aceleración. Los Market Makers amplifican los movimientos (venden en caídas, compran en pánicos).
        """)
        
    with tab_m2:
        st.markdown("""
            ### ⚙️ Equivalencias de Multiplicadores (ETF vs Futuros)
            Cuando operamos futuros apalancados sobre índices (como el Nasdaq `MNQ=F` o el S&P `MES=F`), las opciones líquidas de referencia se negocian en un ETF subyacente (`QQQ` o `SPY`). 
            
            Como cotizan en escalas numéricas completamente distintas, utilizamos un **Multiplicador de Conversión** y un **Offset de Calibración**:
        """)
        
        st.markdown("""
| Activo / Futuro | ETF de Referencia | Multiplicador Base Típico | Notas de Calibración |
| :--- | :--- | :--- | :--- |
| **MNQ / NQ** (Nasdaq) | QQQ | ~40.0x | Relación matemática aproximada entre el precio de QQQ y el Nasdaq 100. |
| **MES / ES** (S&P 500) | SPY | ~10.0x | Relación proporcional entre el SPY y el contrato del S&P 500. |
| **Acciones / Small Caps** | Misma acción (ej. GME, TSLA) | 1.0x | Sin conversión necesaria; el precio del activo coincide directamente con el subyacente. |
        """)
        
    with tab_m3:
        st.markdown("""
            ### 📖 Glosario Técnico de Derivados
            * **Call (Opción de Compra):** Contrato que otorga el derecho a comprar un activo a un precio fijado (Strike) en una fecha de expiración concreta.
            * **Put (Opción de Venta):** Contrato que otorga el derecho a vender un activo a un precio fijado. Vital para coberturas de cartera.
            * **IV Skew (Sesgo de Volatilidad Implícita):** Mide la diferencia de precio (volatilidad) que pagan las Puts frente a las Calls. Un **IV Skew positivo elevado** indica que los inversores están pagando primas muy altas por protegerse ante caídas (miedo institucional).
            * **Gamma Exposure (GEX):** Volumen de acciones o futuros que los creadores de mercado deben comprar/vender obligatoriamente ante un movimiento de 1% en el subyacente.
            * **Delta Exposure (DEX):** Exposición direccional neta de los contratos abiertos en función de la delta de cada opción.
        """)

elif app_mode == "🔥 Screener Small Caps & Momentum":
    st.title("🔥 Screener de Small Caps, Float & Short Squeeze")
    st.markdown("Filtra acciones en base a precio, volumen, cambio porcentual, **Float (acciones libres)** y **Short Float (%)**.")
    
    with st.expander("⚙️ Filtros Avanzados del Scanner", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_price = st.number_input("Precio Mínimo ($)", value=0.5, step=0.5)
            max_price = st.number_input("Precio Máximo ($)", value=100.0, step=5.0)
            min_vol = st.number_input("Volumen Mínimo Diario", value=50000, step=50000)
        with col_f2:
            min_change_pct = st.number_input("Variación Mínima Día (%)", value=0.0, step=0.5)
            min_open_interest_scr = st.number_input("Open Interest Opciones Mín.", value=10, step=10)
        with col_f3:
            max_float_shares = st.number_input("Float Máximo (Millones)", value=500.0, step=50.0)
            min_short_float = st.number_input("Short Float Mínimo (%)", value=0.0, step=1.0)
            
    st.markdown("### 📋 Universo Inicial de Small Caps / Watchlist")
    default_small_caps = ["GME", "AMC", "IWM", "RIOT", "MARA", "PLTR", "SOFI", "NIO", "COIN", "HOOD", "BITF", "HUT", "HLGN", "SENS", "SPY", "QQQ", "TSLA", "NVDA"]
    watchlist_input = st.text_area("Lista de Tickers (separados por comas)", value=", ".join(default_small_caps))
    watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    
    if st.button("🚀 Escanear Mercado y Estructuras"):
        screener_data = []
        progress_bar = st.progress(0)
        total_items = len(watchlist)
        
        for i, ticker in enumerate(watchlist):
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="5d")
                if hist.empty or len(hist) < 2:
                    continue
                
                spot = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                day_vol = hist['Volume'].iloc[-1]
                change_pct = ((spot - prev_close) / prev_close) * 100
                
                info = tk.info
                float_shares = info.get('floatShares', 0)
                float_millions = (float_shares / 1e6) if float_shares else 0.0
                short_ratio_pct = (info.get('shortPercentOfFloat', 0.0) or 0.0) * 100
                
                if not (min_price <= spot <= max_price): continue
                if day_vol < min_vol: continue
                if change_pct < min_change_pct: continue
                if max_float_shares > 0 and float_millions > 0 and float_millions > max_float_shares: continue
                if min_short_float > 0 and short_ratio_pct > 0 and short_ratio_pct < min_short_float: continue
                
                exps = tk.options
                sub_exps = list(exps[:2]) if exps else []
                
                c_wall, p_wall, g_flip, skew = 0.0, 0.0, spot, 0.0
                if sub_exps:
                    df_res, c_wall, p_wall, g_flip, _, _, skew, _, _ = process_multi_expiry_metrics(
                        tk, sub_exps, spot, spot, 1.0, min_oi=min_open_interest_scr, interest_rate=0.05
                    )
                
                if p_wall > 0 and spot <= p_wall * 1.005:
                    signal = "🟢 LONG SOPORTE"
                elif c_wall > 0 and spot >= c_wall * 0.995:
                    signal = "🔴 SHORT / TECHO"
                else:
                    signal = "🟡 MOMENTUM ACTIVO"
                        
                regime = "Gamma Positivo (+)" if spot >= g_flip else "Gamma Negativo (-)"
                
                screener_data.append({
                    "Ticker": ticker,
                    "Spot ($)": round(spot, 2),
                    "Cambio (%)": round(change_pct, 2),
                    "Volumen": f"{day_vol:,}",
                    "Float (M)": round(float_millions, 2) if float_millions > 0 else "N/D",
                    "Short Float (%)": round(short_ratio_pct, 1) if short_ratio_pct > 0 else "N/D",
                    "Señal Táctica": signal,
                    "Régimen": regime,
                    "Put Wall": round(p_wall, 2) if p_wall > 0 else "N/D",
                    "Call Wall": round(c_wall, 2) if c_wall > 0 else "N/D"
                })
            except Exception:
                continue
            progress_bar.progress((i + 1) / total_items)
            
        progress_bar.empty()
        
        if screener_data:
            df_screener = pd.DataFrame(screener_data)
            st.success(f"¡Escaneo completado! Se detectaron **{len(df_screener)} Activos** cumpliendo los criterios.")
            
            st.dataframe(
                df_screener.style.map(
                    lambda v: 'color: #22c55e; font-weight: bold;' if 'LONG' in str(v) else ('color: #ef476f; font-weight: bold;' if 'SHORT' in str(v) else ''),
                    subset=['Señal Táctica']
                ),
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("🕯️ Gráfico Profesional de Velas Japonesas")
            tickers_found = df_screener["Ticker"].tolist()
            
            col_t1, col_t2 = st.columns([2, 2])
            with col_t1:
                selected_ticker_chart = st.selectbox("Selecciona un Ticker de la lista:", tickers_found)
            with col_t2:
                interval_map = {
                    "1 Minuto (1m)": "1m",
                    "5 Minutos (5m)": "5m",
                    "15 Minutos (15m)": "15m",
                    "1 Hora (1h)": "1h",
                    "1 Día (1d)": "1d",
                    "1 Semana (1wk)": "1wk",
                    "1 Mes (1mo)": "1mo"
                }
                selected_interval_label = st.selectbox("Selecciona la Temporalidad:", list(interval_map.keys()), index=4)
                selected_interval = interval_map[selected_interval_label]
            
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max", "1wk": "max", "1mo": "max"}
            fetch_period = period_map.get(selected_interval, "1y")
            
            if selected_ticker_chart:
                tk_chart = yf.Ticker(selected_ticker_chart)
                df_history = tk_chart.history(period=fetch_period, interval=selected_interval)
                
                if not df_history.empty:
                    fig_candle = go.Figure(data=[go.Candlestick(
                        x=df_history.index,
                        open=df_history['Open'],
                        high=df_history['High'],
                        low=df_history['Low'],
                        close=df_history['Close'],
                        increasing_line_color='#22c55e', decreasing_line_color='#ef476f',
                        increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef476f',
                        name="Velas"
                    )])
                    
                    fig_candle.update_layout(
                        title=dict(text=f"<b>Gráfico de Velas ({selected_interval_label}) — {selected_ticker_chart}</b>", font=dict(size=16, color="#ffffff")),
                        xaxis_title="Fecha / Hora",
                        yaxis_title="Precio ($)",
                        height=500, template="plotly_dark", plot_bgcolor='#0b0e14', paper_bgcolor='#0e1117',
                        xaxis=dict(showgrid=True, gridcolor='#21262d', rangeslider=dict(visible=False)),
                        yaxis=dict(showgrid=True, gridcolor='#21262d', tickformat="$,.2f"),
                        margin=dict(l=30, r=30, t=50, b=30)
                    )
                    st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.warning("Ningún ticker cumple con los filtros. Prueba a ampliar el precio máximo o relajar el volumen.")
    else:
        st.info("👈 Configura los filtros y presiona **Escanear Mercado y Estructuras**.")

else:
    # --- VISTA TERMINAL INDIVIDUAL ORIGINAL ---
    st.title("⚡ GEX & DEX Institutional Terminal Pro")
    st.markdown("Terminal cuantitativa multi-expiración adaptada para **móvil, índices y small caps**.")

    with st.sidebar:
        st.header("⚙️ Configuración del Activo")
        
        modo_operativa = st.selectbox("Modo de Operativa", ["📉 Acciones / Small Caps (Directas)", "📈 Futuros / Índices"])
        
        preset_opciones = st.selectbox(
            "Presets de Activos Populares", 
            ["Personalizado", "GME (GameStop)", "IWM (Russell 2000 - Small Caps)", "QQQ (Nasdaq 100)", "SPY (S&P 500)", "TSLA (Tesla)", "NVDA (Nvidia)"]
        )
        
        if preset_opciones != "Personalizado":
            default_ticker = preset_opciones.split(" ")[0]
        else:
            default_ticker = "GME" if "Small Caps" in modo_operativa else "QQQ"

        if "Futuros" in modo_operativa:
            etf_ticker = st.text_input("Ticker de Opciones", value=default_ticker).upper()
            fut_ticker = st.text_input("Ticker del Futuro / Activo", value="MNQ=F").upper()
            multiplier_base = st.number_input("Multiplicador de Conversión", value=40.0, step=0.1, help="Equivalencia de puntos de opciones a puntos del contrato de futuros.")
        else:
            etf_ticker = st.text_input("Ticker de la Small Cap / Acción", value=default_ticker).upper()
            fut_ticker = etf_ticker  
            multiplier_base = 1.0    

        try:
            _, _, spot_fut_live, expirations = fetch_market_data(etf_ticker, fut_ticker)
        except:
            spot_fut_live = 100.0
            expirations = []

        target_future_price = st.number_input("Precio Live del Activo", value=float(spot_fut_live), step=0.05, format="%.2f")
        interest_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 5.0) / 100.0
        metric_view = st.selectbox("Métrica Principal", ["Gamma Exposure (GEX)", "Delta Exposure (DEX)"])
        
        st.markdown("---")
        st.subheader("🧹 Filtros y Zoom Móvil")
        min_open_interest = st.number_input("Open Interest Mínimo por Strike", value=10, step=10)
        range_pct = st.slider("Rango de Strikes (±%)", 0.5, 15.0, 4.0, step=0.5) / 100.0
        
        selected_expirations = []
        if expirations is not None and len(expirations) > 0:
            st.subheader("Fechas de Expiración")
            default_selection = list(expirations[:min(3, len(expirations))])
            selected_expirations = st.multiselect("Vencimientos (Agregado)", expirations, default=default_selection)
            
        calcular_btn = st.button("🚀 Actualizar Terminal")

    if calcular_btn:
        st.session_state['loaded'] = True

    if st.session_state.get('loaded', False) and selected_expirations:
        with st.spinner(f"Procesando flujos y optimizando gráfico para {etf_ticker}..."):
            tk_etf, spot_etf, _, _ = fetch_market_data(etf_ticker, fut_ticker)
            
            df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut, iv_skew, calc_base, cal_offset = process_multi_expiry_metrics(
                tk_etf, selected_expirations, spot_etf, target_future_price, multiplier_base, min_open_interest, interest_rate
            )
            
            if df_metrics.empty:
                st.warning("No hay suficiente información disponible o el filtro de Open Interest es muy alto para este ticker.")
            else:
                min_strike = spot_fut * (1 - range_pct)
                max_strike = spot_fut * (1 + range_pct)
                df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
                
                if df_filtered.empty:
                    df_filtered = df_metrics
                    
                dist_to_put = (spot_fut - put_wall) / spot_fut * 100
                
                if spot_fut <= put_wall * 1.003 or (spot_fut >= gamma_flip and dist_to_put < 1.5):
                    semaforo_emoji = "🟢"
                    semaforo_texto = "POSICIÓN ALCISTA (BUSCAR COMPRAS / LONG)"
                    semaforo_color = "#238636"
                    consejo_dir = "El precio está apoyado sobre el suelo institucional (Put Wall)."
                elif spot_fut >= call_wall * 0.997 or spot_fut < gamma_flip:
                    semaforo_emoji = "🔴"
                    semaforo_texto = "PRECAUCIÓN / SESGO BAJISTA (SHORT O COBERTURA)"
                    semaforo_color = "#da3633"
                    consejo_dir = "Zona de techo (Call Wall) o régimen de alta volatilidad."
                else:
                    semaforo_emoji = "🟡"
                    semaforo_texto = "MERCADO EN RANGO (ESPERAR EXTREMOS)"
                    semaforo_color = "#9e6a03"
                    consejo_dir = "El precio está en tierra de nadie."

                st.markdown(f"### 📊 Dashboard [{fut_ticker}] &nbsp;&nbsp;|&nbsp;&nbsp; *{datetime.now().strftime('%H:%M:%S')}*")
                
                st.markdown(f"""
                    <div style="background-color: #161b22; border-left: 6px solid {semaforo_color}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h3 style="margin: 0; color: #ffffff !important;">{semaforo_emoji} Dirección Sugerida: {semaforo_texto}</h3>
                        <p style="margin: 8px 0 0 0; color: #8b949e !important; font-size: 14px;">{consejo_dir}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Spot", f"{spot_fut:,.2f}")
                c2.metric("Gamma Flip", f"{gamma_flip:,.2f}")
                c3.metric("Call Wall", f"{call_wall:,.2f}", delta="Resistencia", delta_color="inverse")
                c4.metric("Put Wall", f"{put_wall:,.2f}", delta="Soporte")
                c5.metric("IV Skew", f"{iv_skew:+.2f}%")
                
                st.markdown("---")
                
                col_target = 'gex' if "Gamma" in metric_view else 'dex'
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_filtered[col_target],
                    y=df_filtered['strike'],
                    orientation='h',
                    name=metric_view,
                    marker=dict(
                        color=np.where(df_filtered[col_target] >= 0, '#00b4d8', '#ff4d6d'),
                        line=dict(color='rgba(255,255,255,0.15)', width=1)
                    ),
                    hovertemplate='Strike: %{y:,.2f}<br>Exposición: $%{x:,.0f}<extra></extra>'
                ))
                
                fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", line_width=2, annotation_text=f" ⚡ SPOT: {spot_fut:,.2f} ", annotation_position="top right", annotation_font_color="#ffd166", annotation_font_size=11, annotation_bgcolor="#161b22")
                fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#c084fc", line_width=1.5, annotation_text=f" 🟣 FLIP: {gamma_flip:,.2f} ", annotation_position="bottom right", annotation_font_color="#c084fc", annotation_font_size=11, annotation_bgcolor="#161b22")
                fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", line_width=2, annotation_text=f" 🟢 CALL WALL: {call_wall:,.2f} ", annotation_position="top left", annotation_font_color="#22c55e", annotation_font_size=11, annotation_bgcolor="#161b22")
                fig.add_hline(y=put_wall, line_dash="solid", line_color="#ff9f1c", line_width=2, annotation_text=f" 🟠 PUT WALL: {put_wall:,.2f} ", annotation_position="bottom left", annotation_font_color="#ff9f1c", annotation_font_size=11, annotation_bgcolor="#161b22")
                
                fig.update_layout(
                    title=dict(text=f"<b>Perfil Dinámico de {metric_view}</b> &nbsp;|&nbsp; <span style='font-size:12px; color:#8b949e;'>±{int(range_pct*100)}% Rango</span>", font=dict(size=16, color="#ffffff")),
                    xaxis_title='<b>Exposición Neta Acumulada ($)</b>',
                    yaxis_title='<b>Niveles de Strike</b>',
                    height=720, template="plotly_dark", plot_bgcolor='#0b0e14', paper_bgcolor='#0e1117',
                    font=dict(color="#ffffff", family="Arial, sans-serif", size=12), dragmode='pan',
                    xaxis=dict(showgrid=True, gridcolor='#21262d', zeroline=True, zerolinecolor='#484f58', tickformat="$,.0f"), 
                    yaxis=dict(showgrid=True, gridcolor='#21262d', autorange="reversed", tickformat=",.2f"),
                    showlegend=False, margin=dict(l=30, r=30, t=60, b=30), hoverlabel=dict(bgcolor="#161b22", font_size=13, font_family="Arial")
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d']})
                
    else:
        st.info("👈 Selecciona un activo preconfigurado o ajusta los parámetros en la barra lateral y pulsa **Actualizar Terminal**.")