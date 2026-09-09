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

# --- CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="metric-container"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 15px 20px; border-radius: 10px;
    }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
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
    app_mode = st.radio("Seleccionar Vista", ["📈 Terminal GEX / DEX Individual", "🔥 Screener Small Caps & Momentum"])
    st.markdown("---")

if app_mode == "🔥 Screener Small Caps & Momentum":
    st.title("🔥 Screener de Small Caps y Momentum (Estilo Trade Ideas)")
    st.markdown("Filtra automáticamente aquellas acciones con opciones líquidas que cumplen con criterios estrictos de **volumen, precio y cambio porcentual** antes de analizar sus muros.")
    
    with st.expander("⚙️ Filtros del Scanner de Mercado", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_price = st.number_input("Precio Mínimo ($)", value=1.0, step=0.5)
            max_price = st.number_input("Precio Máximo ($)", value=50.0, step=1.0)
            min_vol = st.number_input("Volumen Mínimo Diario", value=500000, step=100000)
        with col_f2:
            min_change_pct = st.number_input("Variación Mínima del Día (%)", value=2.0, step=0.5)
            min_open_interest_scr = st.number_input("Open Interest Opciones Mín.", value=50, step=10)
            
    st.markdown("### 📋 Universo Inicial de Small Caps / Activos Calientes")
    default_small_caps = ["IWM", "GME", "AMC", "RIOT", "MARA", "PLTR", "SOFI", "NIO", "COIN", "HOOD", "BITF", "HUT"]
    watchlist_input = st.text_area("Lista de Tickers (separados por comas)", value=", ".join(default_small_caps))
    watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    
    if st.button("🚀 Filtrar y Escanear Small Caps en Vivo"):
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
                
                # --- APLICAR FILTROS DE SCREENER ---
                if not (min_price <= spot <= max_price): continue
                if day_vol < min_vol: continue
                if change_pct < min_change_pct: continue
                
                exps = tk.options
                if not exps:
                    continue
                
                sub_exps = list(exps[:2]) # Tomar los 2 vencimientos más cercanos
                df_res, c_wall, p_wall, g_flip, _, _, skew, _, _ = process_multi_expiry_metrics(
                    tk, sub_exps, spot, spot, 1.0, min_oi=min_open_interest_scr, interest_rate=0.05
                )
                
                if not df_res.empty:
                    dist_put_pct = (spot - p_wall) / spot * 100
                    dist_call_pct = (c_wall - spot) / spot * 100
                    
                    if spot <= p_wall * 1.005 or (spot >= g_flip and dist_put_pct < 1.5):
                        signal = "🟢 LONG SOPORTE (Rebote)"
                    elif spot >= c_wall * 0.995 or spot < g_flip:
                        signal = "🔴 SHORT / TECHO (Resistencia)"
                    else:
                        signal = "🟡 RANGO / MOMENTUM"
                        
                    regime = "Gamma Positivo (+)" if spot >= g_flip else "Gamma Negativo (-)"
                    
                    screener_data.append({
                        "Ticker": ticker,
                        "Spot ($)": round(spot, 2),
                        "Cambio Día (%)": round(change_pct, 2),
                        "Volumen": f"{day_vol:,}",
                        "Señal Táctica": signal,
                        "Régimen Gamma": regime,
                        "Put Wall ($)": round(p_wall, 2),
                        "Dist. Put (%)": round(dist_put_pct, 2),
                        "Call Wall ($)": round(c_wall, 2),
                        "IV Skew (%)": round(skew, 2)
                    })
            except Exception:
                continue
            progress_bar.progress((i + 1) / total_items)
            
        progress_bar.empty()
        
        if screener_data:
            df_screener = pd.DataFrame(screener_data)
            st.success(f"¡Filtros aplicados con éxito! Se encontraron **{len(df_screener)} Small Caps** activas cumpliendo los criterios.")
            
            st.dataframe(
                df_screener.style.applymap(
                    lambda v: 'color: #22c55e; font-weight: bold;' if 'LONG' in str(v) else ('color: #ef476f; font-weight: bold;' if 'SHORT' in str(v) else ''),
                    subset=['Señal Táctica']
                ),
                use_container_width=True
            )
        else:
            st.warning("Ningún ticker de la lista cumple simultáneamente con los filtros de precio, volumen, cambio porcentual y liquidez de opciones exigidos hoy.")
    else:
        st.info("👈 Ajusta los filtros de premarket/volumen y pulsa **Filtrar y Escanear Small Caps en Vivo**.")

else:
    # --- VISTA TERMINAL INDIVIDUAL ORIGINAL ---
    st.title("⚡ GEX & DEX Institutional Terminal Pro")
    st.markdown("Terminal cuantitativa multi-expiración adaptada para **móvil, índices y small caps**.")

    with st.expander("📖 GUÍA TÁCTICA Y MANUAL DE CONCEPTOS INSTITUCIONALES", expanded=False):
        st.markdown("""
        ### 🚦 1. Semáforo de Dirección Táctica
        * **🟢 POSICIÓN ALCISTA (LONG):** Ideal para buscar compras si el precio está apoyado sobre el suelo institucional (**Put Wall**).
        * **🔴 POSICIÓN BAJISTA (SHORT):** Precaución o cortos si el precio perfora soportes o se acerca al techo con alta volatilidad.
        * **🟡 RANGO / ESPERAR:** El precio está en tierra de nadie entre muros.
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuración del Activo")
        
        modo_operativa = st.selectbox("Modo de Operativa", ["📉 Acciones / Small Caps (Directas)", "📈 Futuros / Índices"])
        
        preset_opciones = st.selectbox(
            "Presets de Activos Populares", 
            ["Personalizado", "IWM (Russell 2000 - Small Caps)", "QQQ (Nasdaq 100)", "SPY (S&P 500)", "TSLA (Tesla)", "NVDA (Nvidia)", "GME (GameStop)"]
        )
        
        if preset_opciones != "Personalizado":
            default_ticker = preset_opciones.split(" ")[0]
        else:
            default_ticker = "IWM" if "Small Caps" in modo_operativa else "QQQ"

        if "Futuros" in modo_operativa:
            etf_ticker = st.text_input("Ticker de Opciones", value=default_ticker).upper()
            fut_ticker = st.text_input("Ticker del Futuro / Activo", value="MNQ=F").upper()
            multiplier_base = st.number_input("Multiplicador de Conversión", value=40.0, step=0.1)
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
                dist_to_call = (call_wall - spot_fut) / spot_fut * 100
                
                if spot_fut <= put_wall * 1.003 or (spot_fut >= gamma_flip and dist_to_put < 1.5):
                    semaforo_emoji = "🟢"
                    semaforo_texto = "POSICIÓN ALCISTA (BUSCAR COMPRAS / LONG)"
                    semaforo_color = "#238636"
                    consejo_dir = "El precio está apoyado sobre el suelo institucional (Put Wall) o zona de rebote."
                elif spot_fut >= call_wall * 0.997 or spot_fut < gamma_flip:
                    semaforo_emoji = "🔴"
                    semaforo_texto = "PRECAUCIÓN / SESGO BAJISTA (SHORT O COBERTURA)"
                    semaforo_color = "#da3633"
                    consejo_dir = "Zona de techo (Call Wall) o régimen de alta volatilidad."
                else:
                    semaforo_emoji = "🟡"
                    semaforo_texto = "MERCADO EN RANGO (ESPERAR EXTREMOS)"
                    semaforo_color = "#9e6a03"
                    consejo_dir = "El precio está en tierra de nadie entre el Put Wall y el Call Wall."

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
                
                with st.expander("⚙️ Ver detalles de Conversión y Multiplicador Activo", expanded=False):
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Spot ETF Subyacente", f"{spot_etf:,.2f}")
                    col_m2.metric("Multiplicador Base", f"{multiplier_base:,.2f}")
                    col_m3.metric("Base Calculada", f"{calc_base:,.2f}")
                    col_m4.metric("Offset de Calibración", f"{cal_offset:+,.2f}")
                
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
                
                with st.expander("🔍 Ver desglose tabular completo"):
                    st.dataframe(df_filtered.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
    else:
        st.info("👈 Selecciona un activo preconfigurado o ajusta los parámetros en la barra lateral y pulsa **Actualizar Terminal**.")