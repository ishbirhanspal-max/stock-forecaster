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

# --- PAGE SETUP & ADVANCED CSS THEME ---
st.set_page_config(page_title="AlphaQuant | Executive Terminal", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Global Theme Background */
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Glassmorphism Metric Cards */
    .metric-card { 
        background: rgba(30, 41, 59, 0.7); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-radius: 12px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-5px); border: 1px solid rgba(56, 189, 248, 0.3); }
    .metric-value { font-size: 26px; font-weight: 800; color: #38bdf8; font-family: 'Courier New', monospace; margin-top: 5px; }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    
    /* Analysis & Summary Boxes */
    .summary-box { background-color: #0f172a; border-left: 4px solid #38bdf8; padding: 20px; border-radius: 6px; font-size: 15px; line-height: 1.6; color: #cbd5e1; }
    
    /* Gradient Verdict Boxes */
    .verdict-box { padding: 20px; border-radius: 12px; margin-top: 10px; margin-bottom: 20px; text-align: center; font-size: 22px; font-weight: 900; letter-spacing: 1px; color: #ffffff; text-transform: uppercase; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
    .v-buy { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .v-sell { background: linear-gradient(135deg, #be123c 0%, #e11d48 100%); }
    .v-hold { background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%); }
    
    /* Section Headers */
    h1, h2, h3 { color: #f8fafc; font-weight: 700; }
    hr { border-color: #334155; }
    </style>
""", unsafe_allow_html=True)

# --- INTELLIGENT CURRENCY ROUTER ---
def get_currency_config(ticker):
    """Dynamically assigns currency symbol based on global ticker suffix."""
    ticker_up = ticker.upper()
    if ticker_up.endswith('.NS') or ticker_up.endswith('.BO'): return '₹', 'INR'
    elif ticker_up.endswith('.L'): return '£', 'GBP'
    elif ticker_up.endswith('.TO'): return 'C$', 'CAD'
    elif ticker_up.endswith('.AX'): return 'A$', 'AUD'
    elif ticker_up.endswith('.DE') or ticker_up.endswith('.PA') or ticker_up.endswith('.AS'): return '€', 'EUR'
    else: return '$', 'USD' # Default to US Markets

# --- DATA FETCHING (Robust Caching) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    info = {}
    try:
        # Try full info scrape for business summary and deep fundamentals
        t = yf.Ticker(ticker)
        info = t.info
    except:
        pass
    
    # Fallback to fast_info if full scrape fails due to rate limits
    if not info or 'longBusinessSummary' not in info:
        try:
            fast = yf.Ticker(ticker).fast_info
            info['marketCap'] = fast.market_cap
            info['previousClose'] = fast.previous_close
            info['longName'] = ticker
            info['sector'] = "Data Unavailable"
            info['longBusinessSummary'] = "Company description is currently unavailable due to market data rate limits. Quantitative algorithms remain fully functional."
        except: pass
        
    return df, info

# --- TERMINAL HEADER ---
st.title("🌐 AlphaQuant: Global Executive Terminal")
st.markdown("Institutional grade technical, fundamental, and algorithmic intelligence. Auto-calibrated for global exchanges.")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Terminal Setup")
    ticker_symbol = st.text_input("Global Asset Ticker (e.g., RELIANCE.NS, AAPL, TSLA):", value="RELIANCE.NS").upper()
    
    st.markdown("### 🔮 Predictive Engine")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    
    engine_choice = st.radio("Select Algorithm:", [
        "Meta Prophet (AI Momentum Curve)", 
        "Manual ARIMA (Statistical Math)"
    ])
    
    if engine_choice == "Manual ARIMA (Statistical Math)":
        st.markdown("**ARIMA Parameters:**")
        c1, c2, c3 = st.columns(3)
        with c1: p_val = st.number_input("p (Lag)", 0, 10, 5)
        with c2: d_val = st.number_input("d (Diff)", 0, 3, 1)
        with c3: q_val = st.number_input("q (MA)", 0, 10, 0)
        
    run_analysis = st.button("🚀 Execute Briefing", type="primary", use_container_width=True)

# --- EXECUTION PIPELINE ---
if run_analysis and ticker_symbol:
    with st.spinner(f"📡 Establishing secure uplink for {ticker_symbol}... Fetching global data matrices..."):
        try:
            df, info = fetch_data(ticker_symbol)
            if df.empty:
                st.error(f"Critical Failure: No market data found for '{ticker_symbol}'. Verify the ticker and exchange suffix.")
                st.stop()
            
            # Identify Currency
            sym, curr_code = get_currency_config(ticker_symbol)
            
            # Clean data array
            if isinstance(df.columns, pd.MultiIndex):
                df_close = df['Close'][ticker_symbol].dropna()
                df_vol = df['Volume'][ticker_symbol].dropna()
            else:
                df_close = df['Close'].dropna()
                df_vol = df['Volume'].dropna()

            current_price = df_close.iloc[-1]
            
            # --- QUANTITATIVE CALCULATIONS ---
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

            # --- MASTER VERDICT ENGINE ---
            score = 0
            if rsi.iloc[-1] < 35: score += 1
            elif rsi.iloc[-1] > 70: score -= 1
            if macd.iloc[-1] > sig_line.iloc[-1]: score += 1
            else: score -= 1
            if current_price > sma_200.iloc[-1]: score += 1
            else: score -= 1

            if score >= 2: master_class, master_text, signal = "v-buy", "MASTER VERDICT: STRONGLY BULLISH", "Buy/Accumulate"
            elif score <= -2: master_class, master_text, signal = "v-sell", "MASTER VERDICT: STRONGLY BEARISH", "Sell/Reduce"
            else: master_class, master_text, signal = "v-hold", "MASTER VERDICT: NEUTRAL", "Hold Position"

            # --- UI TABS ---
            tab_summary, tab_tech, tab_fore, tab_data = st.tabs([
                "📑 Executive Summary", 
                "📊 Technical Dashboard", 
                "🔮 Predictive AI", 
                "💾 Raw Data"
            ])

            # ==========================================
            # TAB 1: EXECUTIVE SUMMARY (NEW)
            # ==========================================
            with tab_summary:
                st.markdown(f'<div class="verdict-box {master_class}">{master_text}</div>', unsafe_allow_html=True)
                
                # Top Level Metrics
                st.markdown("### 🏢 Fundamental Snapshot")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                mcap = info.get('marketCap', 'N/A')
                mcap_str = f"{sym}{mcap / 1e9:,.2f}B" if isinstance(mcap, (int, float)) else "N/A"
                pe = info.get('trailingPE', 'N/A')
                pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
                div = info.get('dividendYield', 'N/A')
                div_str = f"{div*100:.2f}%" if isinstance(div, (int, float)) else "0.00%"
                
                col_s1.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">{sym}{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                col_s2.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{mcap_str}</div></div>', unsafe_allow_html=True)
                col_s3.markdown(f'<div class="metric-card"><div class="metric-label">P/E Ratio</div><div class="metric-value">{pe_str}</div></div>', unsafe_allow_html=True)
                col_s4.markdown(f'<div class="metric-card"><div class="metric-label">Dividend Yield</div><div class="metric-value">{div_str}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Technical Snapshot
                st.markdown("### 📈 Technical Health Status")
                col_t1, col_t2, col_t3 = st.columns(3)
                
                trend_status = "Uptrend 🟢" if current_price > sma_200.iloc[-1] else "Downtrend 🔴"
                rsi_status = "Oversold 🟢" if rsi.iloc[-1] < 30 else "Overbought 🔴" if rsi.iloc[-1] > 70 else "Neutral ⚪"
                macd_status = "Bullish Cross 🟢" if macd.iloc[-1] > sig_line.iloc[-1] else "Bearish Cross 🔴"
                
                col_t1.markdown(f'<div class="summary-box"><b>Macro Trend (200 SMA):</b><br>{trend_status}</div>', unsafe_allow_html=True)
                col_t2.markdown(f'<div class="summary-box"><b>Momentum (RSI):</b><br>{rsi_status} ({rsi.iloc[-1]:.1f})</div>', unsafe_allow_html=True)
                col_t3.markdown(f'<div class="summary-box"><b>Trend Direction (MACD):</b><br>{macd_status}</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Company Description
                st.markdown(f"### 📖 About {info.get('longName', ticker_symbol)}")
                st.markdown(f"<div style='font-size: 16px; color: #94a3b8; line-height: 1.6;'>{info.get('longBusinessSummary', 'Business summary not available.')}</div>", unsafe_allow_html=True)

            # ==========================================
            # TAB 2: TECHNICAL DASHBOARD
            # ==========================================
            with tab_tech:
                st.header("Deep Technical Analysis")
                
                fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Baselines", "RSI Momentum", "MACD Strength"))
                
                # Price Chart
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#f1f5f9")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#f59e0b", dash="dot")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#0ea5e9")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, line=dict(color='rgba(14, 165, 233, 0.2)'), name='Upper BB', showlegend=False), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, line=dict(color='rgba(14, 165, 233, 0.2)'), fill='tonexty', fillcolor='rgba(14, 165, 233, 0.05)', name='Bollinger Bands'), row=1, col=1)
                
                # RSI Chart
                fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a855f7")), row=2, col=1)
                fig_tech.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=2, col=1)
                
                # MACD Chart
                fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#3b82f6")), row=3, col=1)
                fig_tech.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#f43f5e")), row=3, col=1)
                fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Hist", marker_color=np.where(macd_hist>0, '#22c55e', '#ef4444')), row=3, col=1)

                fig_tech.update_layout(template="plotly_dark", height=850, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=20))
                st.plotly_chart(fig_tech, use_container_width=True)

            # ==========================================
            # TAB 3: PREDICTIVE FORECASTING
            # ==========================================
            with tab_fore:
                st.header(f"Algorithmic Projection ({engine_choice})")
                trading_days = int(horizon_years * 252)
                
                with st.spinner("🧠 Compiling Future Matrices..."):
                    if engine_choice == "Meta Prophet (AI Momentum Curve)":
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
                        forecast['yhat'] = forecast['yhat'].clip(lower=0)
                        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                        
                        future_dates = forecast['ds']
                        forecast_actual = forecast['yhat']
                        lower_bound = forecast['yhat_lower']
                        upper_bound = forecast['yhat_upper']
                        
                    else: # Manual ARIMA
                        log_close = np.log(df_close)
                        model = ARIMA(log_close, order=(p_val, d_val, q_val))
                        fitted_model = model.fit()
                        
                        forecast_res = fitted_model.get_forecast(steps=trading_days)
                        forecast_actual = np.exp(forecast_res.predicted_mean)
                        conf_int_log = forecast_res.conf_int().values
                        lower_bound = np.exp(conf_int_log[:, 0])
                        upper_bound = np.exp(conf_int_log[:, 1])
                        future_dates = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)

                    # Forecast Verdict
                    target_price = forecast_actual.iloc[-1] if hasattr(forecast_actual, 'iloc') else forecast_actual[-1]
                    roi = ((target_price - current_price) / current_price) * 100
                    
                    if roi > 5: fore_v, fore_c = f"BULLISH (+{roi:.2f}%)", "v-buy"
                    elif roi < -5: fore_v, fore_c = f"BEARISH ({roi:.2f}%)", "v-sell"
                    else: fore_v, fore_c = f"NEUTRAL ({roi:.2f}%)", "v-hold"

                    st.markdown(f'<div class="verdict-box {fore_c}">FORECAST EXPECTATION: {fore_v}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="summary-box">
                    <b>🤖 Algorithmic Conclusion:</b> From the current price of {sym}{current_price:,.2f}, the math projects a future value of <b>{sym}{target_price:,.2f}</b> in {horizon_years} years.
                    </div>
                    """, unsafe_allow_html=True)

                    # Plot Forecast
                    fig_fore = go.Figure()
                    fig_fore.add_trace(go.Scatter(x=df_close.index[-500:], y=df_close.values[-500:], name='Historical Data', line=dict(color='#94a3b8')))
                    fig_fore.add_trace(go.Scatter(x=future_dates, y=forecast_actual, name='Predicted Trend', line=dict(color='#38bdf8', width=3, dash='dash')))
                    fig_fore.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                        fill='toself', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        name='Volatility Tolerance'
                    ))
                    fig_fore.update_layout(template="plotly_dark", height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                    st.plotly_chart(fig_fore, use_container_width=True)

            # ==========================================
            # TAB 4: RAW DATA EXPORT
            # ==========================================
            with tab_data:
                st.header("💾 Algorithmic Data Matrix")
                st.markdown(f"Export the generated {horizon_years}-year timeline for integration into external Excel/financial models.")
                
                # Align formats for both engines
                if engine_choice == "Meta Prophet (Dynamic AI Momentum Curve)":
                    last_date = prophet_df['ds'].max()
                    future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                else:
                    future_only = pd.DataFrame({'ds': future_dates, 'yhat': forecast_actual.values, 'yhat_lower': lower_bound, 'yhat_upper': upper_bound})
                
                future_only.columns = ['Date', f'Target ({sym})', f'Worst Case ({sym})', f'Best Case ({sym})']
                future_only['Date'] = future_only['Date'].dt.date
                future_only.set_index('Date', inplace=True)
                future_rounded = future_only.round(2)
                
                st.dataframe(future_rounded, use_container_width=True)
                csv = future_rounded.to_csv().encode('utf-8')
                st.download_button("📥 Download Projection as CSV", data=csv, file_name=f'{ticker_symbol}_AI_Forecast.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Execution Error: {e}")
