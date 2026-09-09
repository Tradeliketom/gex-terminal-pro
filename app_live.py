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
st.markdown("Terminal cuantitativa multi-expiración avanzada para trading de futuros (**MNQ / MES**).")

with st.expander("📖 GUÍA TÁCTICA: Muros Institucionales y Régimen de Gamma", expanded=False):
    st.markdown("""
    * **Put Wall (Soporte Principal 🟢):** Zona masiva de cobertura bajista. Alta probabilidad de rebote institucional.
    * **Call Wall (Techo de Resistencia 🔴):** Resistencia magnética de corto plazo; ideal para tomas de beneficios.
    * **Gamma Flip (Pivote 🟣):** Frontera de régimen. Por encima = mercado en rango (estabilizador). Por debajo = mercado direccional y volátil.
    * **IV Skew (Sesgo):** Si las Puts encarecen su volatilidad respecto a las Calls, alerta de cobertura bajista institucional.
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuración de Activo")
    asset_choice = st.selectbox("Seleccionar Futuro", ["MNQ (Nasdaq)", "MES (S&P 500)"])
    
    if "MNQ" in asset_choice:
        etf_ticker = "QQQ"
        fut_ticker = "MNQ=F"
        default_mult = 40.0
    else:
        etf_ticker = "SPY"
        fut_ticker = "MES=F"
        default_mult = 10.0
        
    multiplier_base = st.number_input("Multiplicador Base", value=default_mult, step=0.1)
    
    try:
        _, _, spot_fut_live, expirations = fetch_market_data(etf_ticker, fut_ticker)
    except:
        spot_fut_live = 29000.0 if "MNQ" in asset_choice else 5900.0
        expirations = []

    target_future_price = st.number_input("Precio Live del Futuro", value=float(spot_fut_live), step=1.0, format="%.2f")
    
    interest_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 5.0) / 100.0
    metric_view = st.selectbox("Métrica Principal", ["Gamma Exposure (GEX)", "Delta Exposure (DEX)"])
    
    st.markdown("---")
    st.subheader("Filtro de Rango (Zoom)")
    range_pct = st.slider("Rango de Strikes (±%)", 1.0, 15.0, 5.0) / 100.0
    
    selected_expirations = []
    if expirations is not None and len(expirations) > 0:
        st.subheader("Fechas de Expiración")
        default_selection = list(expirations[:3])
        selected_expirations = st.multiselect("Vencimientos (Agregado)", expirations, default=default_selection)
        
    calcular_btn = st.button("🚀 Actualizar Terminal Institucional")

if calcular_btn:
    st.session_state['loaded'] = True

if st.session_state.get('loaded', False) and selected_expirations:
    with st.spinner(f"Procesando flujos institucionales para {asset_choice}..."):
        tk_etf, spot_etf, _, _ = fetch_market_data(etf_ticker, fut_ticker)
        
        df_metrics, call_wall, put_wall, gamma_flip, total_gex, spot_fut, iv_skew = process_multi_expiry_metrics(
            tk_etf, selected_expirations, spot_etf, target_future_price, multiplier_base, interest_rate
        )
        
        if df_metrics.empty:
            st.warning("No hay suficiente información para las fechas seleccionadas.")
        else:
            min_strike = spot_fut * (1 - range_pct)
            max_strike = spot_fut * (1 + range_pct)
            df_filtered = df_metrics[(df_metrics['strike'] >= min_strike) & (df_metrics['strike'] <= max_strike)].copy()
            
            if df_filtered.empty:
                df_filtered = df_metrics
                
            regimen = "🟢 GAMMA POSITIVO (Rango / Rebotes Estabilizadores)" if spot_fut >= gamma_flip else "🔴 GAMMA NEGATIVO (Direccional / Alta Volatilidad)"
            
            st.markdown(f"### 📊 Dashboard Institucional [{asset_choice}] &nbsp;&nbsp;|&nbsp;&nbsp; *Actualizado: {datetime.now().strftime('%H:%M:%S')}*")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precio Spot Futuro", f"{spot_fut:,.2f}")
            c2.metric("Gamma Flip (Pivot)", f"{gamma_flip:,.2f}")
            c3.metric("Call Wall (Techo)", f"{call_wall:,.2f}", delta="Resistencia", delta_color="inverse")
            c4.metric("Put Wall (Suelo)", f"{put_wall:,.2f}", delta="Soporte")
            c5.metric("IV Skew (Puts-Calls)", f"{iv_skew:+.2f}%")
            
            st.info(f"**Régimen de Mercado Actual:** {regimen}")
            
            # --- NUEVO MÓDULO: PLAN DE ACCIÓN TÁCTICO EN VIVO ---
            st.markdown("---")
            st.markdown("### 🎯 Plan de Acción Táctico & Recomendación Operativa")
            
            dist_call_wall = (call_wall - spot_fut) / spot_fut * 100
            dist_put_wall = (spot_fut - put_wall) / spot_fut * 100
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### 🧭 Sesgo Estructural & Régimen")
                if spot_fut >= gamma_flip:
                    st.success("**Estrategia Recomendada (Rango):**\n* El precio está sobre el Gamma Flip. Los Market Makers actúan como estabilizadores (compran caídas/venden subidas).\n* **Acción:** Buscar compras en acercamientos a soportes o Put Wall, y tomas de beneficios o cortos rápidos en resistencias.")
                else:
                    st.error("**Estrategia Recomendada (Direccional):**\n* El precio está bajo el Gamma Flip. Entorno inestable con aceleración de precios.\n* **Acción:** Evitar contratendencias largas. Seguir la tendencia de corto plazo o operar rupturas con stops ajustados.")
                
                if iv_skew > 0:
                    st.warning(f"**Alerta IV Skew (+{iv_skew:.1f}%):** Las opciones de venta están más caras. Mayor demanda institucional de cobertura bajista.")
                else:
                    st.info(f"**IV Skew Neutral ({iv_skew:.1f}%):** Sin tensiones extremas en el sesgo de puts.")

            with col_t2:
                st.markdown("#### 📍 Proximidad a Muros & Disparadores")
                
                # Evaluamos proximidad a menos del 0.4%
                near_call = abs(spot_fut - call_wall) / spot_fut <= 0.004
                near_put = abs(spot_fut - put_wall) / spot_fut <= 0.004
                
                if near_call:
                    st.warning(f"⚠️ **¡ZONA DE ALERTA EN CALL WALL!**\n* Precio a **+{dist_call_wall:.2f}%** del techo ({call_wall:,.2f}).\n* **Acción:** Alta probabilidad de freno o rechazo bajista. Si rompe con volumen, buscar continuidad alcista (*Gamma Squeeze*).")
                elif near_put:
                    st.warning(f"⚠️ **¡ZONA DE ALERTA EN PUT WALL!**\n* Precio a **-{dist_put_wall:.2f}%** del suelo ({put_wall:,.2f}).\n* **Acción:** Zona clave de defensa institucional. Vigilar formaciones de giro en largo para rebote.")
                else:
                    st.info(f"ℹ️ **Distancias Actuales a Límites:**\n* Distancia al Techo (Call Wall): **+{dist_call_wall:.2f}%**\n* Distancia al Suelo (Put Wall): **-{dist_put_wall:.2f}%**\n* *El precio opera en zona intermedia sin tests inmediatos a los muros principales.*")
            # -----------------------------------------------------

            st.markdown("---")
            
            col_target = 'gex' if "Gamma" in metric_view else 'dex'
            df_filtered['Color'] = np.where(df_filtered[col_target] >= 0, 'Positivo', 'Negativo')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_filtered[col_target],
                y=df_filtered['strike'],
                orientation='h',
                name=metric_view,
                marker=dict(color=np.where(df_filtered[col_target] >= 0, '#00b4d8', '#ef476f'))
            ))
            
            fig.add_hline(y=spot_fut, line_dash="dash", line_color="#ffd166", annotation_text=f"Spot Fut: {spot_fut:.2f}", annotation_position="top right", annotation_font_color="white")
            fig.add_hline(y=gamma_flip, line_dash="dot", line_color="#a855f7", annotation_text=f"Gamma Flip: {gamma_flip:.2f}", annotation_position="bottom right", annotation_font_color="#a855f7")
            fig.add_hline(y=call_wall, line_dash="solid", line_color="#22c55e", annotation_text=f"Call Wall: {call_wall:.2f}", annotation_position="top left", annotation_font_color="#22c55e")
            fig.add_hline(y=put_wall, line_dash="solid", line_color="#ef4444", annotation_text=f"Put Wall: {put_wall:.2f}", annotation_position="bottom left", annotation_font_color="#ef4444")
            
            fig.update_layout(
                title=f"Perfil de {metric_view} (Zoom ±{int(range_pct*100)}%) — {asset_choice}",
                xaxis_title=f'Exposición Neta (${metric_view})',
                yaxis_title='Nivel de Strike Calibrado',
                height=850, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#ffffff", size=12), title_font=dict(size=20, color="#ffffff"),
                xaxis=dict(showgrid=True, gridcolor='#30363d'), yaxis=dict(showgrid=True, gridcolor='#30363d', autorange="reversed"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
            
            with st.expander("🔍 Ver desglose tabular completo por Strike"):
                st.dataframe(df_filtered.style.format({'strike': '{:,.2f}', 'gex': '${:,.2f}', 'dex': '${:,.2f}'}), use_container_width=True)
else:
    st.info("👈 Selecciona los vencimientos, ajusta el rango de zoom en la barra lateral y haz clic en **Actualizar Terminal Institucional**.")