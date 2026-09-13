import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from ib_insync import IB, Stock, Option

st.set_page_config(
    page_title="GEX & DEX Institutional Terminal Pro - IBKR Live", 
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

# --- CONEXIÓN Y GESTIÓN DE IBKR ---
@st.cache_resource
def get_ib_connection(port=7497):
    ib = IB()
    try:
        ib.connect('127.0.0.1', port, clientId=np.random.randint(100, 9999), timeout=5)
    except Exception as e:
        st.error(f"Error conectando a IBKR (Asegúrate de tener TWS o IB Gateway abierto): {e}")
    return ib

def fetch_ibkr_market_data(etf_symbol, port=7497):
    ib = get_ib_connection(port)
    if not ib.isConnected():
        try:
            ib.connect('127.0.0.1', port, clientId=np.random.randint(100, 9999), timeout=5)
        except:
            return None, 0.0, 0.0, []
            
    ib.reqMarketDataType(3) # 3 = Datos con delay (15 min gratis sin OPRA)
    
    underlying = Stock(etf_symbol, 'SMART', 'USD')
    ib.qualifyContracts(underlying)
    
    [underlying_ticker] = ib.reqTickers(underlying)
    spot_etf = underlying_ticker.marketPrice()
    if pd.isna(spot_etf) or spot_etf == 0:
        spot_etf = underlying_ticker.close
    if pd.isna(spot_etf) or spot_etf == 0:
        spot_etf = 100.0

    spot_fut = spot_etf * 10 
    
    try:
        chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
        chain = next((c for c in chains if c.exchange == 'SMART'), None)
        expirations = sorted(list(chain.expirations)) if chain else []
    except:
        expirations = []
        
    return underlying, spot_etf, spot_fut, expirations

def process_multi_expiry_metrics_ibkr(etf_symbol, expiration_dates, spot_etf, target_future_price, multiplier_base, min_oi=0, port=7497):
    ib = get_ib_connection(port)
    if not ib.isConnected():
        return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0, 0, 0
        
    ib.reqMarketDataType(3) 
    
    underlying = Stock(etf_symbol, 'SMART', 'USD')
    ib.qualifyContracts(underlying)
    
    all_results = []
    call_ivs = []
    put_ivs = []
    
    calculated_base_fut = spot_etf * multiplier_base
    calibration_offset = target_future_price - calculated_base_fut
    
    contracts_to_fetch = []
    for exp_date in expiration_dates:
        try:
            chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
            chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not chain: continue
            
            valid_strikes = [s for s in chain.strikes if spot_etf * 0.9 <= s <= spot_etf * 1.1]
            
            for strike in valid_strikes:
                for right in ['C', 'P']:
                    contracts_to_fetch.append(Option(etf_symbol, exp_date, strike, right, 'SMART'))
        except:
            continue
            
    if not contracts_to_fetch:
        return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0, calculated_base_fut, calibration_offset
        
    qualified_contracts = ib.qualifyContracts(*contracts_to_fetch)
    tickers = ib.reqTickers(*qualified_contracts)
    
    for t in tickers:
        contract = t.contract
        greeks = t.modelGreeks
        oi = t.openInterest if not pd.isna(t.openInterest) else 0
        
        if pd.isna(oi) or oi < min_oi: continue
        if greeks and not pd.isna(greeks.gamma):
            gamma = greeks.gamma
            delta = greeks.delta
            sigma = greeks.impliedVolatility if not pd.isna(greeks.impliedVolatility) else 0.2
            
            K_etf = contract.strike
            K_fut = (K_etf * multiplier_base) + calibration_offset
            
            if abs(K_etf - spot_etf) / spot_etf < 0.05:
                if contract.right == 'C': call_ivs.append(sigma)
                else: put_ivs.append(sigma)
                
            if contract.right == 'C':
                gex_val = gamma * oi * 100 * (spot_etf ** 2) * 0.01
                dex_val = delta * oi * 100 * spot_etf
            else:
                gex_val = -1 * (gamma * oi * 100 * (spot_etf ** 2) * 0.01)
                dex_val = delta * oi * 100 * spot_etf
                
            all_results.append({
                'strike': K_fut, 
                'gex': gex_val,
                'dex': dex_val
            })
            
    if not all_results: return pd.DataFrame(), 0, 0, 0, 0, target_future_price, 0.0, calculated_base_fut, calibration_offset
    
    df = pd.DataFrame(all_results)
    df_grouped = df.groupby('strike')[['gex', 'dex']].sum().reset_index()
    df_grouped = df_grouped.sort_values('strike').reset_index(drop=True)
    
    call_walls = df_grouped.loc[df_grouped['gex'].idxmax()]['strike']
    put_walls = df_grouped.loc[df_grouped['gex'].idxmin()]['strike']
    
    # --- CÁLCULO DE ZERO GAMMA CON INTERPOLACIÓN LINEAL ---
    df_grouped['cumsum_gex'] = df_grouped['gex'].cumsum()
    
    sub_df = df_grouped[(df_grouped['strike'] >= target_future_price * 0.8) & (df_grouped['strike'] <= target_future_price * 1.2)].copy()
    if sub_df.empty:
        sub_df = df_grouped.copy()
    sub_df = sub_df.reset_index(drop=True)
        
    sub_df['sign'] = np.sign(sub_df['cumsum_gex'])
    sub_df['sign_change'] = sub_df['sign'].diff() != 0
    
    crossings = sub_df[sub_df['sign_change'] & (sub_df.index > 0)]
    
    if not crossings.empty:
        crossings['distance'] = (crossings['strike'] - target_future_price).abs()
        best_crossing_idx = crossings['distance'].idxmin()
        
        idx_loc = sub_df.index.get_loc(best_crossing_idx)
        if idx_loc > 0:
            x1 = sub_df.loc[idx_loc - 1, 'strike']
            y1 = sub_df.loc[idx_loc - 1, 'cumsum_gex']
            x2 = sub_df.loc[idx_loc, 'strike']
            y2 = sub_df.loc[idx_loc, 'cumsum_gex']
            
            if y2 != y1:
                gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1)
            else:
                gamma_flip = x2
        else:
            gamma_flip = sub_df.loc[best_crossing_idx, 'strike']
    else:
        gamma_flip = df_grouped.loc[(df_grouped['strike'] - target_future_price).abs().idxmin(), 'strike']
    
    avg_call_iv = np.mean(call_ivs) if call_ivs else 0.0
    avg_put_iv = np.mean(put_ivs) if put_ivs else 0.0
    iv_skew = (avg_put_iv - avg_call_iv) * 100
    
    return df_grouped, call_walls, put_walls, gamma_flip, df['gex'].sum(), target_future_price, iv_skew, calculated_base_fut, calibration_offset

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚡ Navegación Pro (IBKR)")
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
    ### 1. ¿Qué es el Gamma Exposure (GEX) y 0DTE?
    * **GEX Estructural:** Agrega el posicionamiento global de toda la cadena de opciones (semanales, mensuales).
    * **GEX 0DTE (Zero Days to Expiration):** Aísla los contratos que expiran hoy. Su gamma se dispara de forma exponencial cerca del precio actual, gobernando la acción del precio intradía.
    """)

elif app_mode == "🔥 Screener Small Caps & Momentum":
    st.title("🔥 Screener de Small Caps, Float & Short Squeeze")
    mock_screener_data = pd.DataFrame({
        "Ticker": ["GME", "AMC", "FFIE", "SPCE", "SAVA"],
        "Precio ($)": [24.50, 5.20, 0.45, 3.80, 18.20],
        "Float (M)": [305.0, 260.0, 42.0, 38.0, 45.0],
        "Short Interest (%)": [22.4, 18.5, 34.2, 28.1, 41.5]
    })
    st.dataframe(mock_screener_data, use_container_width=True)

else:
    # --- VISTA TERMINAL GEX / DEX COMPLETA ---
    st.title("⚡ GEX & DEX Institutional Terminal — IBKR Live Stream (10s)")
    st.markdown("Análisis estructural y de flujos 0DTE conectado a la API de Interactive Brokers.")

    with st.sidebar:
        st.header("⚙️ Configuración IBKR")
        
        ib_port = st.number_input("Puerto TWS / Gateway", value=7497, step=1)
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
            _, spot_etf_live, spot_fut_live, expirations = fetch_ibkr_market_data(etf_ticker, ib_port)
        except:
            spot_etf_live, spot_fut_live, expirations = 100.0, 100.0, []

        target_future_price = st.number_input("Precio Live Actual", value=float(spot_fut_live), step=0.05, format="%.2f")
        
        st.markdown("---")
        
        # --- SELECTOR DE TIPO DE ANÁLISIS (GLOBAL vs 0DTE) ---
        tipo_analisis = st.radio(
            "Modo de Análisis GEX", 
            ["📊 Global (Multi-vencimiento)", "⚡ Exclusivo 0DTE (Solo Vencimiento de Hoy)"]
        )
        
        selected_expirations = []
        if expirations is not None and len(expirations) > 0:
            if "0DTE" in tipo_analisis:
                # Filtrar fecha de hoy (formato IBKR suele ser YYYYMMDD o YYYY-MM-DD)
                today_str1 = datetime.now().strftime("%Y%m%d")
                today_str2 = datetime.now().strftime("%Y-%m-%d")
                selected_expirations = [exp for exp in expirations if exp == today_str1 or exp == today_str2]
                if not selected_expirations:
                    st.warning("⚠️ No hay opciones 0DTE exactas para hoy en este ticker. Seleccionando la más cercana disponible.")
                    selected_expirations = [expirations[0]]
            else:
                default_selection = list(expirations[:min(2, len(expirations))])
                selected_expirations = st.multiselect("Vencimientos", expirations, default=default_selection)

        st.markdown("---")
        st.subheader("⏱️ Automatización 10s")
        live_mode = st.toggle("Activar Auto-Refresh (10s en Vivo)", value=True)
        
        min_open_interest = st.number_input("Open Interest Mínimo", value=100, step=50)
        range_pct = st.slider("Rango de Strikes (±%)", 1.0, 30.0, 10.0, step=1.0) / 100.0
            
        calcular_btn = st.button("🚀 Iniciar / Forzar Actualización")

    if calcular_btn or live_mode:
        st.session_state['live_active'] = True

    if st.session_state.get('live_active', False) and selected_expirations:
        
        with st.spinner(f"Calculando perfiles {'0DTE' if '0DTE' in tipo_analisis else 'Globales'} desde IBKR..."):
            _, spot_etf, _, _ = fetch_ibkr_market_data(etf_ticker, ib_port)

            df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut, iv_skew, calc_base, cal_offset = process_multi_expiry_metrics_ibkr(
                etf_ticker, selected_expirations, spot_etf, target_future_price, multiplier_base, min_open_interest, ib_port
            )
            
            if df_metrics.empty:
                st.warning("No hay datos suficientes para los filtros seleccionados o TWS/Gateway no está respondiendo.")
            else:
                min_strike = min(spot_fut * (1 - range_pct), put_wall * 0.99)
                max_strike = max(spot_fut * (1 + range_pct), call_wall * 1.01)
                
                df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
                if df_filtered.empty: df_filtered = df_metrics

                if spot_fut > gamma_flip:
                    bias_text = "🟢 ALCISTA / ESTABLE (Por encima de Zero Gamma)"
                    bias_desc = "Los Market Makers actúan absorbiendo volatilidad y frenando los movimientos bajistas bruscos."
                elif spot_fut < gamma_flip:
                    bias_text = "🔴 BAJISTA / ACELERACIÓN (Por debajo de Zero Gamma)"
                    bias_desc = "Zona propensa a alta volatilidad y aceleración de movimientos direccionales por cobertura corta."
                else:
                    bias_text = "🟡 NEUTRAL / PUNTO DE INFLEXIÓN"
                    bias_desc = "El precio se encuentra exactamente sobre el nivel de cambio de signo de gamma."

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Live Spot", f"{spot_fut:,.2f}")
                c2.metric("Gamma Flip", f"{gamma_flip:,.2f}")
                c3.metric("Call Wall", f"{call_wall:,.2f}")
                c4.metric("Put Wall", f"{put_wall:,.2f}")
                c5.metric("Net GEX", f"${total_gex:,.0f}")
                c6.metric("IV Skew", f"{iv_skew:+.2f}%")
                
                st.markdown(f"""
                    <div style="background-color: #161b22; border-left: 5px solid {'#22c55e' if 'ALCISTA' in bias_text else '#ef476f' if 'BAJISTA' in bias_text else '#ffd166'}; padding: 12px 18px; border-radius: 6px; margin-top: 10px; margin-bottom: 20px;">
                        <span style="font-size: 14px; color: #8b949e; font-weight: bold;">SESGO INSTITUCIONAL ({tipo_analisis.upper()}):</span>
                        <div style="font-size: 18px; font-weight: bold; color: #ffffff; margin-top: 2px;">{bias_text}</div>
                        <div style="font-size: 12px; color: #c9d1d9; margin-top: 2px;">{bias_desc}</div>
                    </div>
                """, unsafe_allow_html=True)

                with st.expander("🔗 Extensión TradingView: Ver Niveles en tus Gráficos", expanded=False):
                    st.markdown("Copia este código **Pine Script v5** para ver los muros en tiempo real:")
                    pine_script_code = f"""// //@version=5
indicator("GEX Levels 0DTE/Pro", overlay=true)

call_wall = input.float({call_wall:.2f}, title="Call Wall")
put_wall = input.float({put_wall:.2f}, title="Put Wall")
gamma_flip = input.float({gamma_flip:.2f}, title="Zero Gamma")

plot(call_wall, title="Call Wall", color=color.green, linewidth=2, style=plot.style_line)
plot(put_wall, title="Put Wall", color=color.red, linewidth=2, style=plot.style_line)
plot(gamma_flip, title="Zero Gamma", color=color.purple, linewidth=2, style=plot.style_line)
"""
                    st.code(pine_script_code, language="pine")

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
                    st.subheader(f"📊 Gamma Exposure ({tipo_analisis})")
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
                    st.subheader(f"📉 Delta Exposure ({tipo_analisis})")
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
                
                st.caption(f"⚡ Streaming IBKR Activo | Modo: {tipo_analisis} | Última actualización: {datetime.now().strftime('%H:%M:%S')} (10s)")

        if live_mode:
            time.sleep(10)
            st.rerun()
            
    else:
        st.info("👈 Selecciona si deseas ver el análisis Global o el GEX 0DTE exclusivo de hoy y pulsa **Iniciar / Forzar Actualización**.")