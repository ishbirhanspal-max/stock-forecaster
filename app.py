import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from prophet import Prophet

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="AlphaQuant | Institutional Forecaster", page_icon="🏛️", layout="wide")

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

st.title("🏛️ AlphaQuant: Institutional Market Analysis & AI Engine")
st.markdown("A unified quantitative terminal combining Fundamental health, Technical momentum, and Algorithmic AI forecasting.")
st.markdown("---")

# --- DATA FETCHING WITH ADVANCED CACHING & FALLBACKS ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    
    # 1. Fetch historical price data (Rarely rate-limited)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # 2. Try to fetch fundamental info
    info = {}
    ticker_obj = yf.Ticker(ticker)
    
    try:
        # First, attempt the heavy web scrape for full data
        info = ticker_obj.info
    except Exception:
        pass # Caught the rate limit!
        
    # 3. If the heavy scrape failed, use the lightweight 'fast_info' fallback
    if not info or 'marketCap' not in info:
        try:
            fast = ticker_obj.fast_info
            info['marketCap'] = fast.market_cap
            info['longName'] = ticker
            info['sector'] = "Sector Data Limited"
            info['trailingPE'] = "N/A"
            info['dividendYield'] = "N/A"
            info['longBusinessSummary'] = "⚠️ **Note:** Yahoo Finance rate limits prevented downloading the full text company profile. However, Market Cap, Technicals, and AI Forecasting models have been successfully compiled and are fully operational."
        except Exception:
            pass
            
    return df, info

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ticker_symbol = st.text_input("Stock Ticker (e.g., RELIANCE.NS, INFY.NS)", value="RELIANCE.NS").upper()
    
    st.markdown("### 🔮 Forecast Horizon")
    horizon_years = st.slider("Years into the future:", min_value=1, max_value=5, value=2)
    
    run_analysis = st.button("🚀 Execute Terminal Pipeline", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Anti-Rate Limit Active:** Data is cached for 1 hour to prevent Yahoo Finance blocks.")

# --- Main Logic ---
if run_analysis:
    if ticker_symbol:
        with st.spinner(f"📡 Establishing uplink to market data for {ticker_symbol}..."):
            try:
                # Use the cached function!
                df, info = fetch_stock_data(ticker_symbol)
                
                if df.empty:
                    st.error("No data found. Please verify the ticker symbol or wait a few minutes if Yahoo is blocking you.")
                    st.stop()
                
                # Format Data safely
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                else:
                    df_close = df['Close'].dropna()
                
                current_price = df_close.iloc[-1]
                
                # --- CALCULATION ENGINE ---
                daily_returns = df_close.pct_change().dropna()
                cagr = (((current_price / df_close.iloc[0]) ** (1/5)) - 1) * 100
                
                # Technicals
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
                
                sma_20 = df_close.rolling(window=20).mean()
                std_20 = df_close.rolling(window=20).std()
                upper_bb = sma_20 + (std_20 * 2)
                lower_bb = sma_20 - (std_20 * 2)

                # --- VERDICT SCORING SYSTEM ---
                score = 0
                reasons = []
                
                curr_rsi = rsi.iloc[-1]
                if curr_rsi < 35:
                    score += 1
                    reasons.append(f"🟢 **Bullish:** RSI is at {curr_rsi:.1f} (Oversold).")
                elif curr_rsi > 70:
                    score -= 1
                    reasons.append(f"🔴 **Bearish:** RSI is at {curr_rsi:.1f} (Overbought).")
                else:
                    reasons.append(f"⚪ **Neutral:** RSI is at {curr_rsi:.1f}.")

                if macd.iloc[-1] > signal_line.iloc[-1]:
                    score += 1
                    reasons.append("🟢 **Bullish:** MACD line is above the Signal line.")
                else:
                    score -= 1
                    reasons.append("🔴 **Bearish:** MACD line is below the Signal line.")

                if current_price > sma_200.iloc[-1]:
                    score += 1
                    reasons.append("🟢 **Bullish:** Price is above the 200-Day SMA (Long-term uptrend).")
                else:
                    score -= 1
                    reasons.append("🔴 **Bearish:** Price is below the 200-Day SMA.")

                if score >= 2:
                    verdict_class = "verdict-buy"
                    verdict_text = "🎯 SYSTEM VERDICT: STRONG BUY / ACCUMULATE"
                elif score <= -2:
                    verdict_class = "verdict-sell"
                    verdict_text = "⚠️ SYSTEM VERDICT: STRONG SELL / REDUCE"
                else:
                    verdict_class = "verdict-hold"
                    verdict_text = "⚖️ SYSTEM VERDICT: HOLD / WAIT FOR CONFIRMATION"

                # UI Layout: TABS
                tab_fund, tab_tech, tab_forecast, tab_season, tab_export = st.tabs([
                    "🏢 Fundamentals", "📊 Technicals", "🔮 Forecast", "🧠 Seasonality", "💾 Export"
                ])

                # ==========================================
                # TAB 1: FUNDAMENTALS
                # ==========================================
                with tab_fund:
                    st.header(f"Company Overview: {info.get('longName', ticker_symbol)}")
                    
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    mcap = info.get('marketCap', 'N/A')
                    if mcap != 'N/A': mcap = f"₹{mcap / 1e9:,.2f} Billion"
                    
                    pe = info.get('trailingPE', 'N/A')
                    if pe != 'N/A': pe = f"{pe:.2f}"
                    
                    div_yield = info.get('dividendYield', 'N/A')
                    if div_yield != 'N/A' and isinstance(div_yield, (int, float)): 
                        div_yield = f"{div_yield * 100:.2f}%"

                    col_f1.markdown(f'<div class="metric-card"><div class="metric-label">Sector</div><div class="metric-value">{info.get("sector", "N/A")}</div></div>', unsafe_allow_html=True)
                    col_f2.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{mcap}</div></div>', unsafe_allow_html=True)
                    col_f3.markdown(f'<div class="metric-card"><div class="metric-label">P/E Ratio</div><div class="metric-value">{pe}</div></div>', unsafe_allow_html=True)
                    col_f4.markdown(f'<div class="metric-card"><div class="metric-label">Dividend Yield</div><div class="metric-value">{div_yield}</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write(info.get('longBusinessSummary', 'Fundamental summary is temporarily unavailable due to Yahoo Finance rate limits. Technical and forecasting tools are fully operational.'))

                # ==========================================
                # TAB 2: TECHNICALS & VERDICT
                # ==========================================
                with tab_tech:
                    st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
                    with st.expander("🔍 View Algorithmic Reasoning"):
                        for r in reasons: st.markdown(r)

                    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Close Price", line=dict(color="#2c3e50")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#e67e22", dash="dot")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#c0392b", dash="dash")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, line=dict(color='rgba(41, 128, 185, 0.2)'), name='Upper Band', showlegend=False), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, line=dict(color='rgba(41, 128, 185, 0.2)'), fill='tonexty', fillcolor='rgba(41, 128, 185, 0.1)', name='Bollinger Bands'), row=1, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#8e44ad")), row=2, col=1)
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#2980b9")), row=3, col=1)
                    fig_tech.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", line=dict(color="#e74c3c")), row=3, col=1)
                    fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Histogram", marker_color=np.where(macd_hist>0, '#27ae60', '#c0392b')), row=3, col=1)

                    fig_tech.update_layout(height=800, hovermode="x unified", template="plotly_white", margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

                # ==========================================
                # TAB 3: AI FORECAST
                # ==========================================
                with tab_forecast:
                    st.header(f"Algorithmic Price Projection ({horizon_years} Year Horizon)")
                    prophet_df = df_close.reset_index()
                    prophet_df.columns = ['ds', 'y']
                    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                    
                    model = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, yearly_seasonality=True, weekly_seasonality=True)
                    model.add_country_holidays(country_name='IN')
                    model.fit(prophet_df)
                    
                    target_date = datetime.today() + timedelta(days=horizon_years * 365)
                    days_ahead = (target_date - prophet_df['ds'].max()).days
                    future = model.make_future_dataframe(periods=days_ahead)
                    future = future[future['ds'].dt.weekday < 5] 
                    
                    # 1. Generate Raw Forecast
                    forecast = model.predict(future)
                    
                    # 2. Prevent Negative Prices (Clipping)
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                    
                    # Forecast Chart
                    fig_fore = go.Figure()
                    fig_fore.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Historical Close', line=dict(color='#2c3e50')))
                    fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Target', line=dict(color='#27ae60', width=3)))
                    fig_fore.add_trace(go.Scatter(x=pd.concat([forecast['ds'], forecast['ds'][::-1]]), y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]), fill='toself', fillcolor='rgba(39, 174, 96, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Confidence'))
                    
                    fig_fore.update_layout(height=600, hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig_fore, use_container_width=True)

                # ==========================================
                # TAB 4: SEASONALITY
                # ==========================================
                with tab_season:
                    col_y, col_w = st.columns(2)
                    with col_y:
                        st.subheader("🗓️ Yearly Institutional Cycle")
                        fig_yearly = go.Figure(go.Scatter(x=forecast['ds'].dt.dayofyear, y=forecast['yearly'], mode='lines', line=dict(color='#8e44ad', width=3)))
                        fig_yearly.update_layout(template="plotly_white", height=350)
                        st.plotly_chart(fig_yearly, use_container_width=True)
                    with col_w:
                        st.subheader("📆 Weekly Trading Cycle")
                        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
                        weekly_data = forecast[['ds', 'weekly']].copy()
                        weekly_data['weekday'] = weekly_data['ds'].dt.weekday
                        weekly_avg = weekly_data[weekly_data['weekday'] < 5].groupby('weekday')['weekly'].mean()
                        fig_weekly = go.Figure(go.Bar(x=days, y=weekly_avg.values, marker_color='#2980b9'))
                        fig_weekly.update_layout(template="plotly_white", height=350)
                        st.plotly_chart(fig_weekly, use_container_width=True)

                # ==========================================
                # TAB 5: EXPORT
                # ==========================================
                with tab_export:
                    last_date = prophet_df['ds'].max()
                    future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    future_only.columns = ['Date', 'Projected Target', 'Bearish Target', 'Bullish Target']
                    future_only['Date'] = future_only['Date'].dt.date
                    future_only.set_index('Date', inplace=True)
                    future_rounded = future_only.round(2)
                    
                    st.dataframe(future_rounded, use_container_width=True)
                    csv = future_rounded.to_csv().encode('utf-8')
                    st.download_button("📥 Download Forecast as CSV", data=csv, file_name=f'{ticker_symbol}_forecast.csv', mime='text/csv')

            except Exception as e:
                st.error(f"Analysis failed. {e}")
