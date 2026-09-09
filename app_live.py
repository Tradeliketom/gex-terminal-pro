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

st.title("⚡ GEX & DEX Institutional Terminal Pro")
st.markdown("Terminal cuantitativa multi-expiración adaptada para **móvil, índices y small caps**.")

# --- GUÍA TÁCTICA Y DE CONCEPTOS AMPLIADA ---
with st.expander("📖 GUÍA TÁCTICA Y MANUAL DE CONCEPTOS INSTITUCIONALES", expanded=False):
    st.markdown("""
    ### 🚦 1. Semáforo de Dirección Táctica
    * **🟢 POSICIÓN ALCISTA (LONG):** Ideal para buscar compras si el precio está apoyado sobre el suelo institucional (**Put Wall**) o el régimen de mercado es estable.
    * **🔴 POSICIÓN BAJISTA (SHORT):** Precaución o cortos si el precio perfora soportes o se acerca peligrosamente al techo con alta volatilidad.
    * **🟡 RANGO / ESPERAR:** El precio está en tierra de nadie entre muros. No persigas el precio; opera solo si llega a los extremos.

    ### 🧠 2. Sesgo Estructural y Régimen de Mercado
    * **Gamma Positivo (Precio > Gamma Flip):** Entorno de mercado normal/rango. Los creadores de mercado (Market Makers) actúan comprando caídas y vendiendo subidas, amortiguando la volatilidad.
    * **Gamma Negativo (Precio < Gamma Flip):** Entorno inestable de alta volatilidad. Los movimientos se aceleran porque los creadores de mercado se ven obligados a vender cuando el precio cae.
    * **IV Skew (Sesgo de Volatilidad):** Mide la diferencia de volatilidad implícita entre Puts y Calls. Un skew muy positivo indica una demanda inusual de seguros bajistas (miedo en el mercado).

    ### 🧱 3. Conceptos Clave: Call Wall, Put Wall y Muros
    * **Call Wall (Techo Verde):** Strike con la mayor concentración de GEX positivo (Calls). Actúa como resistencia magnética importante.
    * **Put Wall (Suelo Naranja):** Strike con la mayor concentración de protección. Funciona como soporte duro donde los institucionales suelen defender el precio.

    ### ⚙️ 4. Multiplicadores y Conversión (Futuros vs ETFs)
    * Permite escalar el strike de los ETFs subyacentes (como QQQ) al precio de sus futuros equivalentes (como MNQ) aplicando un factor de conversión y un desplazamiento de calibración (`calibration_offset`) para alinear el precio exacto con el mercado en vivo.
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
                
            # --- SEMÁFORO DE DIRECCIÓN AUTOMÁTICO ---
            dist_to_put = (spot_fut - put_wall) / spot_fut * 100
            dist_to_call = (call_wall - spot_fut) / spot_fut * 100
            
            if spot_fut <= put_wall * 1.003 or (spot_fut >= gamma_flip and dist_to_put < 1.5):
                semaforo_emoji = "🟢"
                semaforo_texto = "POSICIÓN ALCISTA (BUSCAR COMPRAS / LONG)"
                semaforo_color = "#238636"
                consejo_dir = "El precio está apoyado sobre el suelo institucional (Put Wall) o zona de rebote. Excelente zona para buscar largos con stop ceñido."
            elif spot_fut >= call_wall * 0.997 or spot_fut < gamma_flip:
                semaforo_emoji = "🔴"
                semaforo_texto = "PRECAUCIÓN / SESGO BAJISTA (SHORT O COBERTURA)"
                semaforo_color = "#da3633"
                consejo_dir = "Zona de techo (Call Wall) o régimen de alta volatilidad. Evita comprar en altos; prioriza tomas de beneficios o cortos tácticos."
            else:
                semaforo_emoji = "🟡"
                semaforo_texto = "MERCADO EN RANGO (ESPERAR EXTREMOS)"
                semaforo_color = "#9e6a03"
                consejo_dir = "El precio está en tierra de nadie entre el Put Wall y el Call Wall. No persigas el precio; opera solo si llega a los extremos."

            st.markdown(f"### 📊 Dashboard [{fut_ticker}] &nbsp;&nbsp;|&nbsp;&nbsp; *{datetime.now().strftime('%H:%M:%S')}*")
            
            # --- BLOQUE VISUAL DEL SEMÁFORO ---
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
            
            # --- NUEVO: DESGLOSE DE EQUIVALENCIAS Y MULTIPLICADOR ---
            with st.expander("⚙️ Ver detalles de Conversión y Multiplicador Activo", expanded=False):
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Spot ETF Subyacente", f"{spot_etf:,.2f}")
                col_m2.metric("Multiplicador Base", f"{multiplier_base:,.2f}")
                col_m3.metric("Base Calculada", f"{calc_base:,.2f}")
                col_m4.metric("Offset de Calibración", f"{cal_offset:+,.2f}")
            
            st.markdown("---")
            
            col_target = 'gex' if "Gamma" in metric_view else 'dex'
            
            # --- MOTOR DE GRÁFICO (DRAGMODE='PAN' CORREGIDO EN LAYOUT) ---
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_filtered[col_target],
                y=df_filtered['strike'],
                orientation='h',
                name=metric_view,
                marker=dict(
                    color=np.where(df_filtered[col_target] >= 0, '#00b4d8', '#ef476f'),
                    line=dict(color='rgba(255,255,255,0.1)', width=0.5)
                )
            ))
            
            fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", annotation_text=f" Spot: {spot_fut:.2f} ", annotation_position="top right", annotation_font_color="white")
            fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#a855f7", annotation_text=f" Flip: {gamma_flip:.2f} ", annotation_position="bottom right", annotation_font_color="#a855f7")
            fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", annotation_text=f" Call Wall: {call_wall:.2f} ", annotation_position="top left", annotation_font_color="#22c55e")
            fig.add_hline(y=put_wall, line_dash="solid", line_color="#ff9f1c", annotation_text=f" Put Wall: {put_wall:.2f} ", annotation_position="bottom left", annotation_font_color="#ff9f1c")
            
            fig.update_layout(
                title=f"Perfil de {metric_view} (Modo Pan Activo ±{int(range_pct*100)}%)",
                xaxis_title='Exposición Neta ($)',
                yaxis_title='Niveles de Strike',
                height=700, 
                template="plotly_dark", 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff", size=12), 
                title_font=dict(size=16, color="#ffffff"),
                dragmode='pan',
                xaxis=dict(showgrid=True, gridcolor='#30363d', zeroline=True, zerolinecolor='#ffffff'), 
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='#30363d', 
                    autorange="reversed",
                    tickformat=".2f"
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={
                    'scrollZoom': True, 
                    'displayModeBar': True, 
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            )
            
            with st.expander("🔍 Ver desglose tabular completo"):
                st.dataframe(df_filtered.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
else:
    st.info("👈 Selecciona un activo preconfigurado o ajusta los parámetros en la barra lateral y pulsa **Actualizar Terminal**.")