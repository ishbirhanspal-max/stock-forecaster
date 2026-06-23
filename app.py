import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# --- PAGE SETUP & CSS ---
st.set_page_config(page_title="AlphaQuant Pro | Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #58a6ff; font-family: 'Courier New', monospace; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .analysis-box { background-color: #1c2128; border-left: 4px solid #58a6ff; padding: 15px; margin-top: 10px; margin-bottom: 20px; border-radius: 4px; font-size: 14px; }
    .verdict-box { padding: 15px; border-radius: 6px; margin-top: 15px; text-align: center; font-size: 20px; font-weight: bold; border: 1px solid #30363d; }
    .v-buy { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; border-color: rgba(46, 160, 67, 0.4); }
    .v-sell { background-color: rgba(248, 81, 73, 0.15); color: #f85149; border-color: rgba(248, 81, 73, 0.4); }
    .v-hold { background-color: rgba(210, 153, 34, 0.15); color: #d29922; border-color: rgba(210, 153, 34, 0.4); }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ AlphaQuant: Institutional Analysis & Forecasting")
st.markdown("Dynamic momentum tracking, automated technical interpretation, and hybrid AI/ARIMA forecasting.")
st.markdown("---")

# --- DATA FETCHING (Memory Caching) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    info = {}
    try:
        fast = yf.Ticker(ticker).fast_info
        info['marketCap'] = float(fast.market_cap) if fast.market_cap else "N/A"
        info['previousClose'] = float(fast.previous_close) if fast.previous_close else "N/A"
    except: pass
    return df, info

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Terminal Configuration")
    ticker_symbol = st.text_input("Asset Ticker (e.g., RELIANCE.NS):", value="RELIANCE.NS").upper()
    
    st.markdown("### 🔮 Forecasting Engine")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    
    engine_choice = st.radio("Select Prediction Algorithm:", [
        "Meta Prophet (Dynamic AI Trend)", 
        "Manual ARIMA (Statistical Math)"
    ])
    
    if engine_choice == "Manual ARIMA (Statistical Math)":
        st.markdown("**Manual ARIMA Parameters:**")
        st.caption("Note: ARIMA naturally calculates flat/linear paths to represent the safest statistical mean.")
        c1, c2, c3 = st.columns(3)
        with c1: p_val = st.number_input("p (Lag)", 0, 10, 5)
        with c2: d_val = st.number_input("d (Diff)", 0, 3, 1)
        with c3: q_val = st.number_input("q (MA)", 0, 10, 0)
        
    run_analysis = st.button("🚀 Execute Terminal", type="primary", use_container_width=True)

# --- EXECUTION LOGIC ---
if run_analysis and ticker_symbol:
    with st.spinner(f"📡 Fetching data and generating algorithmic analysis for {ticker_symbol}..."):
        try:
            df, info = fetch_data(ticker_symbol)
            if df.empty:
                st.error("No data found. Verify ticker.")
                st.stop()
            
            # Clean data array
            if isinstance(df.columns, pd.MultiIndex):
                df_close = df['Close'][ticker_symbol].dropna()
                df_vol = df['Volume'][ticker_symbol].dropna()
            else:
                df_close = df['Close'].dropna()
                df_vol = df['Volume'].dropna()

            current_price = df_close.iloc[-1]
            
            # --- CALCULATE TECHNICALS ---
            sma_50 = df_close.rolling(50).mean()
            sma_200 = df_close.rolling(200).mean()
            
            # Bollinger Bands
            sma_20 = df_close.rolling(20).mean()
            std_20 = df_close.rolling(20).std()
            upper_bb = sma_20 + (std_20 * 2)
            lower_bb = sma_20 - (std_20 * 2)
            
            # RSI
            delta = df_close.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))

            # MACD
            ema_12 = df_close.ewm(span=12, adjust=False).mean()
            ema_26 = df_close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            sig_line = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - sig_line

            # --- MASTER VERDICT LOGIC ---
            score = 0
            if rsi.iloc[-1] < 35: score += 1
            elif rsi.iloc[-1] > 70: score -= 1
            if macd.iloc[-1] > sig_line.iloc[-1]: score += 1
            else: score -= 1
            if current_price > sma_200.iloc[-1]: score += 1
            else: score -= 1

            if score >= 2: master_class, master_text = "v-buy", "MASTER VERDICT: STRONGLY BULLISH (ACCUMULATE)"
            elif score <= -2: master_class, master_text = "v-sell", "MASTER VERDICT: STRONGLY BEARISH (REDUCE)"
            else: master_class, master_text = "v-hold", "MASTER VERDICT: NEUTRAL (HOLD)"

            tab_tech, tab_fore, tab_data = st.tabs(["📊 Technical Dashboard & Analysis", "🔮 Predictive Forecasting", "💾 Data Summary"])

            # ==========================================
            # TAB 1: TECHNICAL DESK & WRITTEN ANALYSIS
            # ==========================================
            with tab_tech:
                st.markdown(f'<div class="verdict-box {master_class}">{master_text}</div>', unsafe_allow_html=True)
                
                # --- CHART 1: PRICE, SMA, BOLLINGER ---
                st.markdown("### 1. Macro Trend & Volatility Analysis (Price, SMA, Bollinger Bands)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#c9d1d9")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#d29922", dash="dot")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#58a6ff")))
                fig1.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, line=dict(color='rgba(41, 128, 185, 0.2)'), name='Upper Band', showlegend=False))
                fig1.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, line=dict(color='rgba(41, 128, 185, 0.2)'), fill='tonexty', fillcolor='rgba(41, 128, 185, 0.1)', name='Bollinger Bands'))
                fig1.update_layout(template="plotly_dark", height=450, margin=dict(t=20, b=20))
                st.plotly_chart(fig1, use_container_width=True)
                
                # Dynamic Written Analysis 1
                trend_verdict = "BULLISH 🟢" if current_price > sma_200.iloc[-1] else "BEARISH 🔴"
                bb_status = "near the Upper Band (Potentially Overvalued)" if current_price > upper_bb.iloc[-1] * 0.98 else "near the Lower Band (Potentially Undervalued)" if current_price < lower_bb.iloc[-1] * 1.02 else "in the middle of the bands (Stable Volatility)"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> This chart tracks the raw price against moving averages (long-term trends) and Bollinger Bands (volatility limits).<br><br>
                <b>🤖 AI Chart Analysis & Verdict:</b> The asset is currently trading at ₹{current_price:,.2f}. The macro trend is <b>{trend_verdict}</b> because the price is {"above" if current_price > sma_200.iloc[-1] else "below"} the 200-Day SMA (Blue Line). Additionally, the price is currently trading {bb_status}.
                </div>
                """, unsafe_allow_html=True)

                # --- CHART 2: RSI ---
                st.markdown("### 2. Relative Strength Index (Momentum)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a371f7")))
                fig2.add_hline(y=70, line_dash="dot", line_color="#f85149")
                fig2.add_hline(y=30, line_dash="dot", line_color="#3fb950")
                fig2.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)
                
                # Dynamic Written Analysis 2
                current_rsi = rsi.iloc[-1]
                if current_rsi > 70: rsi_verdict, rsi_exp = "BEARISH 🔴", "OVERBOUGHT (Buyers are exhausted, high risk of a price drop)."
                elif current_rsi < 30: rsi_verdict, rsi_exp = "BULLISH 🟢", "OVERSOLD (Panic selling has occurred, high probability of a bounce)."
                else: rsi_verdict, rsi_exp = "NEUTRAL ⚪", "in a healthy, stable momentum range."
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> RSI is a momentum oscillator measuring the speed and change of price movements on a scale of 0 to 100.<br><br>
                <b>🤖 AI Chart Analysis & Verdict:</b> The RSI is currently at <b>{current_rsi:.1f}</b>. This verdict is <b>{rsi_verdict}</b> because the stock is {rsi_exp}
                </div>
                """, unsafe_allow_html=True)

                # --- CHART 3: MACD ---
                st.markdown("### 3. MACD (Trend Direction & Strength)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#2980b9")))
                fig3.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#e74c3c")))
                fig3.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Histogram", marker_color=np.where(macd_hist>0, '#3fb950', '#f85149')))
                fig3.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig3, use_container_width=True)
                
                # Dynamic Written Analysis 3
                macd_v = "BULLISH 🟢" if macd.iloc[-1] > sig_line.iloc[-1] else "BEARISH 🔴"
                macd_dir = "accelerating upwards" if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2] else "losing momentum"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> MACD shows the relationship between two moving averages. When the Blue line crosses above the Red line, it triggers a buy signal.<br><br>
                <b>🤖 AI Chart Analysis & Verdict:</b> The MACD verdict is <b>{macd_v}</b>. The MACD line is {"above" if macd.iloc[-1] > sig_line.iloc[-1] else "below"} the Signal line, and the histogram indicates the trend is {macd_dir}.
                </div>
                """, unsafe_allow_html=True)

            # ==========================================
            # TAB 2: FORECASTING ENGINE
            # ==========================================
            with tab_fore:
                st.header(f"Algorithmic Price Projection ({engine_choice})")
                trading_days = int(horizon_years * 252)
                
                with st.spinner("🧠 Calculating future trajectory..."):
                    # ----------------------------------
                    # ENGINE 1: PROPHET (Dynamic AI Curve)
                    # ----------------------------------
                    if engine_choice == "Meta Prophet (Dynamic AI Trend)":
                        prophet_df = df_close.reset_index()
                        prophet_df.columns = ['ds', 'y']
                        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                        
                        model = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, weekly_seasonality=True)
                        model.fit(prophet_df)
                        
                        target_date = datetime.today() + timedelta(days=horizon_years * 365)
                        days_ahead = (target_date - prophet_df['ds'].max()).days
                        future = model.make_future_dataframe(periods=days_ahead)
                        future = future[future['ds'].dt.weekday < 5] 
                        
                        forecast = model.predict(future)
                        # Clip negative values
                        forecast['yhat'] = forecast['yhat'].clip(lower=0)
                        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                        
                        future_dates = forecast['ds']
                        forecast_actual = forecast['yhat']
                        lower_bound = forecast['yhat_lower']
                        upper_bound = forecast['yhat_upper']
                        
                        engine_desc = "Meta's Prophet AI algorithm. It breaks down historical data to find hidden weekly and yearly seasonal patterns, producing a realistic, curved momentum forecast."

                    # ----------------------------------
                    # ENGINE 2: MANUAL ARIMA 
                    # ----------------------------------
                    else:
                        log_close = np.log(df_close)
                        model = ARIMA(log_close, order=(p_val, d_val, q_val))
                        fitted_model = model.fit()
                        
                        forecast_res = fitted_model.get_forecast(steps=trading_days)
                        # Reverse log to prevent negative prices
                        forecast_actual = np.exp(forecast_res.predicted_mean)
                        conf_int_log = forecast_res.conf_int().values
                        lower_bound = np.exp(conf_int_log[:, 0])
                        upper_bound = np.exp(conf_int_log[:, 1])
                        
                        future_dates = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                        
                        engine_desc = f"A strict, user-defined ARIMA ({p_val}, {d_val}, {q_val}) statistical model. Note: ARIMA models inherently calculate the safest statistical mean, resulting in a flat or strictly linear trajectory."

                    # --- FORECAST VERDICT ---
                    target_price = forecast_actual.iloc[-1] if hasattr(forecast_actual, 'iloc') else forecast_actual[-1]
                    roi = ((target_price - current_price) / current_price) * 100
                    
                    if roi > 5: fore_v, fore_c = f"BULLISH (+{roi:.2f}%)", "v-buy"
                    elif roi < -5: fore_v, fore_c = f"BEARISH ({roi:.2f}%)", "v-sell"
                    else: fore_v, fore_c = f"NEUTRAL/FLAT ({roi:.2f}%)", "v-hold"

                    st.markdown(f'<div class="verdict-box {fore_c}">FORECAST VERDICT: {fore_v}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="analysis-box">
                    <b>📌 Engine Definition:</b> You are currently using {engine_desc}<br><br>
                    <b>🤖 Predictive Analysis:</b> The current price is ₹{current_price:,.2f}. Based on this mathematical model, the projected price in {horizon_years} years is <b>₹{target_price:,.2f}</b>. Because this represents a change of {roi:.2f}%, the algorithmic outlook is <b>{fore_v.split(' ')[0]}</b>.
                    </div>
                    """, unsafe_allow_html=True)

                    # --- PLOT FORECAST ---
                    fig_fore = go.Figure()
                    fig_fore.add_trace(go.Scatter(x=df_close.index[-500:], y=df_close.values[-500:], name='Past Actuals', line=dict(color='#8b949e')))
                    fig_fore.add_trace(go.Scatter(x=future_dates, y=forecast_actual, name='Predicted Path', line=dict(color='#58a6ff', width=3, dash='dash')))
                    fig_fore.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                        fill='toself', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='Mathematical Boundary'
                    ))
                    fig_fore.update_layout(template="plotly_dark", height=500, hovermode="x unified", margin=dict(t=20, b=20))
                    st.plotly_chart(fig_fore, use_container_width=True)

            # ==========================================
            # TAB 3: DATA SUMMARY & FUNDAMENTALS
            # ==========================================
            with tab_data:
                st.header("📋 High-Level Data Summary")
                st.markdown("A quick overview of the current asset's statistical footprint.")
                
                c1, c2, c3, c4 = st.columns(4)
                
                mcap = info.get('marketCap', 'N/A')
                mcap_str = f"₹{mcap / 1e9:,.2f} Billion" if isinstance(mcap, (int, float)) else "N/A"
                
                daily_change = ((current_price / df_close.iloc[-2]) - 1) * 100
                
                c1.markdown(f'<div class="metric-card"><div class="metric-label">Latest Close</div><div class="metric-value">₹{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-label">24H Change</div><div class="metric-value" style="color:{"#3fb950" if daily_change > 0 else "#f85149"}">{daily_change:+.2f}%</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-label">52-Week High</div><div class="metric-value">₹{df_close.tail(252).max():,.2f}</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{mcap_str}</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Execution Error: {e}")
