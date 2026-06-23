import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Forecasting Engines
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="AlphaQuant Pro | Hybrid Engine", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; text-align: center; }
    .metric-value { font-size: 22px; font-weight: 700; color: #58a6ff; font-family: 'Courier New', monospace; }
    .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .verdict-box { padding: 15px; border-radius: 6px; margin-top: 15px; text-align: center; font-size: 18px; font-weight: bold; border: 1px solid #30363d; }
    .v-bull { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; border-color: rgba(46, 160, 67, 0.4); }
    .v-bear { background-color: rgba(248, 81, 73, 0.15); color: #f85149; border-color: rgba(248, 81, 73, 0.4); }
    .v-neutral { background-color: rgba(210, 153, 34, 0.15); color: #d29922; border-color: rgba(210, 153, 34, 0.4); }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ AlphaQuant Pro: Hybrid Forecasting Terminal")
st.markdown("Advanced technical momentum, Fibonacci modeling, and Hybrid ARIMA (Auto & Manual) predictive engine.")
st.markdown("---")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    # FIX: Extract data into a standard Python dictionary so Streamlit can serialize it
    info = {}
    try:
        fast = yf.Ticker(ticker).fast_info
        # Convert custom yfinance objects to standard Python types (like float)
        info['marketCap'] = float(fast.market_cap) if fast.market_cap else "N/A"
    except Exception:
        pass
        
    return df, info

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Core Parameters")
    ticker_symbol = st.text_input("Target Asset Ticker (e.g., RELIANCE.NS):", value="RELIANCE.NS").upper()
    
    st.markdown("### 🔮 Forecasting Engine")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    
    arima_mode = st.radio("ARIMA Configuration:", ["Auto-ARIMA (AI Optimized)", "Manual Override (Custom p,d,q)"])
    
    if arima_mode == "Manual Override (Custom p,d,q)":
        st.markdown("**Set ARIMA Parameters:**")
        col1, col2, col3 = st.columns(3)
        with col1: p_val = st.number_input("p (Lag)", 0, 10, 5)
        with col2: d_val = st.number_input("d (Diff)", 0, 3, 1)
        with col3: q_val = st.number_input("q (MA)", 0, 10, 0)
    
    run_terminal = st.button("🚀 Execute Terminal", type="primary", use_container_width=True)

# --- Terminal Execution ---
if run_terminal:
    if ticker_symbol:
        with st.spinner("Fetching data and computing technical matrices..."):
            try:
                df, info = fetch_market_data(ticker_symbol)
                
                if df.empty:
                    st.error("Asset data unavailable. Verify ticker.")
                    st.stop()
                    
                # Clean Data
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                    df_open = df['Open'][ticker_symbol].dropna()
                    df_high = df['High'][ticker_symbol].dropna()
                    df_low = df['Low'][ticker_symbol].dropna()
                    df_vol = df['Volume'][ticker_symbol].dropna()
                else:
                    df_close = df['Close'].dropna()
                    df_open = df['Open'].dropna()
                    df_high = df['High'].dropna()
                    df_low = df['Low'].dropna()
                    df_vol = df['Volume'].dropna()

                current_price = df_close.iloc[-1]
                
                # --- TECHNICAL CALCULATIONS ---
                sma_50 = df_close.rolling(50).mean()
                sma_200 = df_close.rolling(200).mean()
                
                delta = df_close.diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rsi = 100 - (100 / (1 + (gain / loss)))

                # Fibonacci Retracement Levels (Based on 52-Week High/Low)
                high_52 = df_high.tail(252).max()
                low_52 = df_low.tail(252).min()
                diff = high_52 - low_52
                fib_levels = {
                    "Level 0.0% (High)": high_52,
                    "Level 23.6%": high_52 - 0.236 * diff,
                    "Level 38.2%": high_52 - 0.382 * diff,
                    "Level 50.0%": high_52 - 0.5 * diff,
                    "Level 61.8%": high_52 - 0.618 * diff,
                    "Level 100% (Low)": low_52
                }

                # --- CURRENT TECHNICAL VERDICT ---
                score = 0
                if rsi.iloc[-1] < 35: score += 1
                elif rsi.iloc[-1] > 70: score -= 1
                if current_price > sma_200.iloc[-1]: score += 1
                else: score -= 1

                if score > 0: tech_class, tech_text = "v-bull", "CURRENT MOMENTUM: BULLISH"
                elif score < 0: tech_class, tech_text = "v-bear", "CURRENT MOMENTUM: BEARISH"
                else: tech_class, tech_text = "v-neutral", "CURRENT MOMENTUM: NEUTRAL"

                tab_tech, tab_forecast = st.tabs(["📊 Technical & Volume Desk", "🔮 ARIMA Projection Engine"])

                # ==========================================
                # TAB 1: TECHNICALS & FIBONACCI
                # ==========================================
                with tab_tech:
                    st.markdown(f'<div class="verdict-box {tech_class}">{tech_text}</div>', unsafe_allow_html=True)
                    
                    # Key Metrics row
                    st.markdown("<br>", unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">₹{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="metric-card"><div class="metric-label">50-Day SMA</div><div class="metric-value">₹{sma_50.iloc[-1]:,.2f}</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="metric-card"><div class="metric-label">200-Day SMA</div><div class="metric-value">₹{sma_200.iloc[-1]:,.2f}</div></div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="metric-card"><div class="metric-label">Current RSI</div><div class="metric-value">{rsi.iloc[-1]:.1f}</div></div>', unsafe_allow_html=True)

                    # Institutional Chart with Volume and RSI
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
                    
                    # 1. Price
                    fig_tech.add_trace(go.Candlestick(x=df_close.index, open=df_open, high=df_high, low=df_low, close=df_close, name="Price"), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#d29922", width=1.5, dash='dot')), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#58a6ff", width=2)), row=1, col=1)
                    
                    # Add Fibonacci lines
                    for name, level in fib_levels.items():
                        fig_tech.add_hline(y=level, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text=name, row=1, col=1)

                    # 2. Volume
                    colors = np.where(df_close > df_open, '#3fb950', '#f85149')
                    fig_tech.add_trace(go.Bar(x=df_vol.index, y=df_vol.values, name="Volume", marker_color=colors), row=2, col=1)

                    # 3. RSI
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a371f7")), row=3, col=1)
                    fig_tech.add_hline(y=70, line_dash="dot", line_color="#f85149", row=3, col=1)
                    fig_tech.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=3, col=1)

                    fig_tech.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

                # ==========================================
                # TAB 2: ARIMA FORECASTING ENGINE
                # ==========================================
                with tab_forecast:
                    trading_days = int(horizon_years * 252)
                    
                    with st.spinner("🧠 Calculating ARIMA Forward Projection..."):
                        # Apply Log Transformation to prevent negative numbers
                        log_close = np.log(df_close)
                        
                        try:
                            if arima_mode == "Auto-ARIMA (AI Optimized)":
                                model = pm.auto_arima(log_close, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, stepwise=True, suppress_warnings=True)
                                st.info(f"✅ AI Selected Optimal Parameters: **ARIMA {model.order}**")
                                forecast_log, conf_int_log = model.predict(n_periods=trading_days, return_conf_int=True)
                                
                            else: # Manual Override
                                model = ARIMA(log_close, order=(p_val, d_val, q_val))
                                fitted_model = model.fit()
                                st.info(f"✅ Executing Manual Parameters: **ARIMA ({p_val}, {d_val}, {q_val})**")
                                
                                # Fetch forecast and confidence intervals manually for statsmodels
                                forecast_res = fitted_model.get_forecast(steps=trading_days)
                                forecast_log = forecast_res.predicted_mean
                                conf_int_log = forecast_res.conf_int().values

                            # Reverse the log transformation (np.exp guarantees > 0)
                            forecast_actual = np.exp(forecast_log)
                            lower_bound = np.exp(conf_int_log[:, 0])
                            upper_bound = np.exp(conf_int_log[:, 1])
                            
                            # Future Dates
                            future_dates = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                            target_price = forecast_actual.iloc[-1] if hasattr(forecast_actual, 'iloc') else forecast_actual[-1]
                            
                            # --- FORECAST VERDICT ENGINE ---
                            expected_return = ((target_price - current_price) / current_price) * 100
                            
                            if expected_return > 10:
                                proj_class = "v-bull"
                                proj_text = f"🚀 PREDICTIVE VERDICT: STRONGLY BULLISH (+{expected_return:.2f}% Expected Target)"
                            elif 2 < expected_return <= 10:
                                proj_class = "v-bull"
                                proj_text = f"📈 PREDICTIVE VERDICT: BULLISH (+{expected_return:.2f}% Expected Target)"
                            elif -2 <= expected_return <= 2:
                                proj_class = "v-neutral"
                                proj_text = f"⚖️ PREDICTIVE VERDICT: RANGEBOUND ({expected_return:.2f}% Expected Target)"
                            elif -10 <= expected_return < -2:
                                proj_class = "v-bear"
                                proj_text = f"📉 PREDICTIVE VERDICT: BEARISH ({expected_return:.2f}% Expected Target)"
                            else:
                                proj_class = "v-bear"
                                proj_text = f"⚠️ PREDICTIVE VERDICT: STRONGLY BEARISH ({expected_return:.2f}% Expected Target)"

                            st.markdown(f'<div class="verdict-box {proj_class}">{proj_text}</div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Plotly Forecast Chart
                            fig_arima = go.Figure()
                            
                            # Historical
                            fig_arima.add_trace(go.Scatter(x=df_close.index[-500:], y=df_close.values[-500:], name='Historical Close', line=dict(color='#8b949e')))
                            
                            # Forecast line
                            fig_arima.add_trace(go.Scatter(x=future_dates, y=forecast_actual, name='ARIMA Projection', line=dict(color='#58a6ff', width=3, dash='dash')))
                            
                            # Confidence Interval
                            fig_arima.add_trace(go.Scatter(
                                x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                                y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                                fill='toself', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                                name='Statistical Confidence Boundary'
                            ))
                            
                            fig_arima.update_layout(template="plotly_dark", height=600, hovermode="x unified")
                            st.plotly_chart(fig_arima, use_container_width=True)

                        except Exception as e:
                            st.error(f"ARIMA Calculation Error. Try adjusting your manual (p, d, q) parameters. Details: {e}")

            except Exception as e:
                st.error(f"Terminal Execution Error: {e}")
