import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import pmdarima as pm # We are using PMDARIMA for the ultimate Auto-ARIMA search

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="AlphaQuant | ARIMA Edition", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    .metric-value { font-size: 22px; font-weight: 800; color: #2c3e50; }
    .metric-label { font-size: 13px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    .verdict-box { padding: 20px; border-radius: 10px; margin-top: 20px; color: white; text-align: center; font-size: 20px; font-weight: bold; }
    .verdict-buy { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .verdict-sell { background: linear-gradient(135deg, #cb2d3e, #ef473f); }
    .verdict-hold { background: linear-gradient(135deg, #f1c40f, #f39c12); }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ AlphaQuant: Advanced ARIMA Forecasting Terminal")
st.markdown("Utilizing Log-Transformed Auto-ARIMA optimization to predict future market trajectories based on historical volatility.")
st.markdown("---")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    info = {}
    ticker_obj = yf.Ticker(ticker)
    try:
        info = ticker_obj.info
    except Exception:
        pass
    if not info or 'marketCap' not in info:
        try:
            fast = ticker_obj.fast_info
            info['marketCap'] = fast.market_cap
            info['longName'] = ticker
            info['sector'] = "Sector Data Limited"
        except Exception:
            pass
    return df, info

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ticker_symbol = st.text_input("Stock Ticker (e.g., RELIANCE.NS, INFY.NS)", value="RELIANCE.NS").upper()
    
    st.markdown("### 🔮 Forecast Horizon")
    horizon_years = st.slider("Years into the future:", min_value=1, max_value=5, value=2)
    
    run_analysis = st.button("🚀 Execute Pipeline", type="primary", use_container_width=True)

# --- Main Logic ---
if run_analysis:
    if ticker_symbol:
        with st.spinner(f"📡 Downloading data and computing mathematical matrices for {ticker_symbol}..."):
            try:
                df, info = fetch_stock_data(ticker_symbol)
                
                if df.empty:
                    st.error("No data found. Please verify the ticker symbol.")
                    st.stop()
                
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                else:
                    df_close = df['Close'].dropna()
                
                current_price = df_close.iloc[-1]
                
                # --- TECHNICAL CALCULATIONS ---
                cagr = (((current_price / df_close.iloc[0]) ** (1/5)) - 1) * 100
                sma_50 = df_close.rolling(window=50).mean()
                sma_200 = df_close.rolling(window=200).mean()
                
                delta = df_close.diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                ema_12 = df_close.ewm(span=12, adjust=False).mean()
                ema_26 = df_close.ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                signal_line = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - signal_line

                # --- VERDICT SCORING ---
                score = 0
                if rsi.iloc[-1] < 35: score += 1
                elif rsi.iloc[-1] > 70: score -= 1
                
                if macd.iloc[-1] > signal_line.iloc[-1]: score += 1
                else: score -= 1
                
                if current_price > sma_200.iloc[-1]: score += 1
                else: score -= 1

                if score >= 2:
                    verdict_class = "verdict-buy"
                    verdict_text = "🎯 SYSTEM VERDICT: BUY / ACCUMULATE"
                elif score <= -2:
                    verdict_class = "verdict-sell"
                    verdict_text = "⚠️ SYSTEM VERDICT: SELL / REDUCE"
                else:
                    verdict_class = "verdict-hold"
                    verdict_text = "⚖️ SYSTEM VERDICT: HOLD"

                # UI Layout
                tab_tech, tab_forecast, tab_export = st.tabs(["📊 Technicals", "🔮 Deep ARIMA Forecast", "💾 Data Export"])

                # ==========================================
                # TAB 1: TECHNICALS
                # ==========================================
                with tab_tech:
                    st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Close Price", line=dict(color="#2c3e50")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#e67e22", dash="dot")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#c0392b", dash="dash")), row=1, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#8e44ad")), row=2, col=1)
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#2980b9")), row=3, col=1)
                    fig_tech.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", line=dict(color="#e74c3c")), row=3, col=1)
                    fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Histogram", marker_color=np.where(macd_hist>0, '#27ae60', '#c0392b')), row=3, col=1)

                    fig_tech.update_layout(height=800, hovermode="x unified", template="plotly_white", margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

                # ==========================================
                # TAB 2: DEEP ARIMA FORECAST
                # ==========================================
                with tab_forecast:
                    st.header(f"Log-Transformed Auto-ARIMA ({horizon_years} Year Horizon)")
                    st.markdown("The algorithm is running a highly intensive search across multiple (p, d, q) parameters on log-transformed data to find the optimal statistical fit. **This may take 15-45 seconds.**")
                    
                    with st.spinner("🧠 Calculating hundreds of ARIMA permutations. Please wait..."):
                        
                        # 1. APPLY LOG TRANSFORMATION (Secret to preventing negative numbers and smoothing exponential growth)
                        log_close = np.log(df_close)
                        
                        # 2. RUN DEEP AUTO-ARIMA SEARCH
                        model = pm.auto_arima(
                            log_close,
                            start_p=0, start_q=0,
                            max_p=5, max_q=5, # Allow it to search deeply
                            seasonal=False,   # Daily stock data is technically non-seasonal for ARIMA
                            d=None,           # Let the algorithm find the best differencing order
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True
                        )
                        
                        st.success(f"✅ Optimization Complete! Selected Parameters: **ARIMA {model.order}**")
                        
                        # 3. FORECAST FUTURE LOG VALUES
                        # Roughly 252 trading days in a year
                        trading_days_future = int(horizon_years * 252)
                        
                        forecast_log, conf_int_log = model.predict(n_periods=trading_days_future, return_conf_int=True)
                        
                        # 4. REVERSE THE LOG TRANSFORMATION (np.exp)
                        # This mathematically guarantees prices can never drop below zero
                        forecast_actual = np.exp(forecast_log)
                        lower_bound = np.exp(conf_int_log[:, 0])
                        upper_bound = np.exp(conf_int_log[:, 1])
                        
                        # 5. CREATE FUTURE DATES (Excluding weekends)
                        last_date = df_close.index[-1]
                        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=trading_days_future)
                        
                        # Plotly Chart
                        fig_arima = go.Figure()
                        
                        # Historical
                        fig_arima.add_trace(go.Scatter(x=df_close.index, y=df_close.values, name='Historical Close', line=dict(color='#2c3e50')))
                        
                        # Forecast line
                        fig_arima.add_trace(go.Scatter(x=future_dates, y=forecast_actual, name='Optimal ARIMA Trend', line=dict(color='#e74c3c', width=3, dash='dash')))
                        
                        # Confidence Interval
                        fig_arima.add_trace(go.Scatter(
                            x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                            y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                            fill='toself', fillcolor='rgba(231, 76, 60, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                            name='Mathematical Confidence Boundary'
                        ))
                        
                        fig_arima.update_layout(height=600, hovermode="x unified", template="plotly_white")
                        st.plotly_chart(fig_arima, use_container_width=True)
                        
                        # Save dataframe to session state for the export tab
                        export_df = pd.DataFrame({
                            'Date': future_dates.date,
                            'ARIMA Forecast': np.round(forecast_actual.values, 2),
                            'Pessimistic Bound': np.round(lower_bound, 2),
                            'Optimistic Bound': np.round(upper_bound, 2)
                        }).set_index('Date')
                        st.session_state['arima_export'] = export_df

                # ==========================================
                # TAB 3: EXPORT
                # ==========================================
                with tab_export:
                    st.header("💾 Download ARIMA Trajectory")
                    if 'arima_export' in st.session_state:
                        st.dataframe(st.session_state['arima_export'], use_container_width=True)
                        csv = st.session_state['arima_export'].to_csv().encode('utf-8')
                        st.download_button("📥 Download Forecast as CSV", data=csv, file_name=f'{ticker_symbol}_ARIMA_{horizon_years}Y.csv', mime='text/csv')
                    else:
                        st.info("Run the forecast in the previous tab to generate export data.")

            except Exception as e:
                st.error(f"Analysis failed. {e}")
