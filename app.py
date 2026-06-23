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
st.set_page_config(page_title="Quantitative Stock Analyst", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Quantitative Market Analysis & AI Forecasting")
st.markdown("A deep-dive financial intelligence platform combining traditional technical indicators with advanced algorithmic forecasting.")
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ticker_symbol = st.text_input("Indian Stock Ticker (e.g., RELIANCE.NS, INFY.NS)", value="RELIANCE.NS").upper()
    run_analysis = st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Disclaimer:** This tool relies purely on mathematical historical data. It does not account for fundamental news, earnings reports, or macroeconomic geopolitical events.")

# --- Main Logic ---
if run_analysis:
    if ticker_symbol:
        with st.spinner(f"📡 Fetching market data, calculating technicals, and training AI for {ticker_symbol}..."):
            try:
                # 1. Fetch 5 Years of Data
                end_date = datetime.today()
                start_date = end_date - timedelta(days=5*365)
                df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error("No data found. Please verify the ticker symbol.")
                    st.stop()
                
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                    df_high = df['High'][ticker_symbol].dropna()
                    df_low = df['Low'][ticker_symbol].dropna()
                else:
                    df_close = df['Close'].dropna()
                    df_high = df['High'].dropna()
                    df_low = df['Low'].dropna()
                
                current_price = df_close.iloc[-1]
                
                # --- CALCULATION ENGINE ---
                # Core Metrics
                daily_returns = df_close.pct_change().dropna()
                annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
                cagr = (((current_price / df_close.iloc[0]) ** (1/5)) - 1) * 100
                high_52 = df_close.tail(252).max()
                low_52 = df_close.tail(252).min()

                # Technicals: SMAs
                sma_50 = df_close.rolling(window=50).mean()
                sma_200 = df_close.rolling(window=200).mean()
                
                # Technicals: RSI (Relative Strength Index)
                delta = df_close.diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # Technicals: MACD (Moving Average Convergence Divergence)
                ema_12 = df_close.ewm(span=12, adjust=False).mean()
                ema_26 = df_close.ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                signal_line = macd.ewm(span=9, adjust=False).mean()
                macd_histogram = macd - signal_line

                # UI Layout: TABS
                tab_tech, tab_forecast, tab_components, tab_data = st.tabs([
                    "📊 Technical Analysis", 
                    "🔮 AI Price Forecast", 
                    "🧠 Seasonal Components", 
                    "📋 Raw Data Matrix"
                ])

                # ==========================================
                # TAB 1: TECHNICAL ANALYSIS
                # ==========================================
                with tab_tech:
                    st.header("Historical Performance & Technical Indicators")
                    
                    # Top Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">₹{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                    col2.markdown(f'<div class="metric-card"><div class="metric-label">5-Year CAGR</div><div class="metric-value">{cagr:,.2f}%</div></div>', unsafe_allow_html=True)
                    col3.markdown(f'<div class="metric-card"><div class="metric-label">Annual Volatility</div><div class="metric-value">{annualized_volatility:,.2f}%</div></div>', unsafe_allow_html=True)
                    col4.markdown(f'<div class="metric-card"><div class="metric-label">52-Week Range</div><div class="metric-value">₹{low_52:,.0f} - ₹{high_52:,.0f}</div></div>', unsafe_allow_html=True)
                    st.write("")

                    with st.expander("📚 Definition: What do these metrics mean?"):
                        st.markdown("""
                        * **CAGR (Compound Annual Growth Rate):** The smoothed mathematical rate at which the investment grew each year to reach its current price.
                        * **Annual Volatility:** A measure of risk. It calculates how wildly the stock's price swings. A high percentage means the stock is highly unpredictable; a low percentage means it is stable.
                        * **52-Week Range:** The lowest and highest price the stock has hit over the last year, useful for spotting historical support and resistance levels.
                        """)

                    # Master Technical Chart (Price, RSI, MACD)
                    fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                             vertical_spacing=0.05, 
                                             row_heights=[0.6, 0.2, 0.2],
                                             subplot_titles=("Price & Moving Averages", "RSI (Momentum)", "MACD (Trend Direction)"))
                    
                    # Row 1: Price and SMAs
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Close", line=dict(color="#2c3e50")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#e67e22", dash="dot")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#c0392b", dash="dash")), row=1, col=1)
                    
                    # Row 2: RSI
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#8e44ad")), row=2, col=1)
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # Row 3: MACD
                    fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#2980b9")), row=3, col=1)
                    fig_tech.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal", line=dict(color="#e74c3c")), row=3, col=1)
                    fig_tech.add_trace(go.Bar(x=macd_histogram.index, y=macd_histogram, name="Histogram", marker_color=np.where(macd_histogram>0, '#27ae60', '#c0392b')), row=3, col=1)

                    fig_tech.update_layout(height=800, hovermode="x unified", template="plotly_white", margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

                    with st.expander("📚 Definition: How to read this chart"):
                        st.markdown("""
                        * **Moving Averages (SMA):** The 50-day SMA tracks short-term momentum. The 200-day tracks long-term trend. If the 50 crosses *above* the 200, it is a bullish "Golden Cross". If it crosses *below*, it is a bearish "Death Cross".
                        * **RSI (Relative Strength Index):** A score from 0 to 100. If RSI is above 70 (Red Line), the stock is considered **Overbought** (potentially due for a drop). If below 30 (Green Line), it is **Oversold** (potentially a buying opportunity).
                        * **MACD:** Measures trend strength. When the blue MACD line crosses above the red Signal line, it indicates upward momentum. 
                        """)

                # ==========================================
                # TAB 2: AI PRICE FORECAST
                # ==========================================
                with tab_forecast:
                    st.header("Algorithmic Price Projection (June 2027)")
                    
                    # AI Processing
                    prophet_df = df_close.reset_index()
                    prophet_df.columns = ['ds', 'y']
                    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                    
                    model = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                    model.add_country_holidays(country_name='IN')
                    model.fit(prophet_df)
                    
                    target_date = datetime(2027, 6, 30)
                    days_ahead = (target_date - prophet_df['ds'].max()).days
                    future = model.make_future_dataframe(periods=days_ahead)
                    future = future[future['ds'].dt.weekday < 5] 
                    forecast = model.predict(future)
                    
                    # Forecast Chart
                    fig_fore = go.Figure()
                    fig_fore.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Historical Close', line=dict(color='#2c3e50')))
                    fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Target Trend', line=dict(color='#27ae60', width=3)))
                    fig_fore.add_trace(go.Scatter(
                        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(39, 174, 96, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval (Expected Volatility)'
                    ))
                    
                    fig_fore.update_layout(height=500, hovermode="x unified", template="plotly_white")
                    st.plotly_chart(fig_fore, use_container_width=True)

                    with st.expander("📚 Definition: How does the AI generate this?"):
                        st.markdown("""
                        * **The Engine:** This uses Meta's Prophet algorithm, originally designed to forecast highly volatile user data. It breaks the stock down into a base "Trend", adds mathematical "Seasonality" waves on top, and subtracts major "Holidays" where the market is closed.
                        * **The Confidence Interval (Green Shaded Area):** The AI calculates an 80% uncertainty boundary based on how much the stock historically deviated from its trend. The wider the shadow, the more unpredictable the stock is.
                        """)

                # ==========================================
                # TAB 3: SEASONAL COMPONENTS
                # ==========================================
                with tab_components:
                    st.header("Inside the Algorithm: Market Seasonality")
                    st.markdown("These charts explain exactly *why* the AI's forecast curves the way it does.")
                    
                    col_y, col_w = st.columns(2)
                    
                    with col_y:
                        st.subheader("🗓️ Yearly Cycle Impact")
                        fig_yearly = go.Figure(go.Scatter(x=forecast['ds'].dt.dayofyear, y=forecast['yearly'], mode='lines', line=dict(color='#8e44ad', width=3)))
                        fig_yearly.update_layout(xaxis_title="Day of Year", yaxis_title="Impact on Stock Price (INR)", template="plotly_white", height=350)
                        st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        with st.expander("📚 Definition: Yearly Cycle"):
                            st.write("This isolates the 'time of year' effect. If the line spikes in October, it means that mathematically, over the last 5 years, this stock almost always experiences a surge in October regardless of the overall market trend.")

                    with col_w:
                        st.subheader("📆 Weekly Trading Cycle")
                        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
                        weekly_data = forecast[['ds', 'weekly']].copy()
                        weekly_data['weekday'] = weekly_data['ds'].dt.weekday
                        weekly_avg = weekly_data[weekly_data['weekday'] < 5].groupby('weekday')['weekly'].mean()
                        
                        fig_weekly = go.Figure(go.Bar(x=days, y=weekly_avg.values, marker_color='#2980b9'))
                        fig_weekly.update_layout(yaxis_title="Impact on Stock Price (INR)", template="plotly_white", height=350)
                        st.plotly_chart(fig_weekly, use_container_width=True)
                        
                        with st.expander("📚 Definition: Weekly Cycle"):
                            st.write("This shows how the stock typically behaves based on the day of the week. Many stocks show a dip on Fridays as traders sell off before the weekend, or a spike on Mondays.")

                # ==========================================
                # TAB 4: RAW DATA MATRIX
                # ==========================================
                with tab_data:
                    st.header("Future Price Data Matrix")
                    st.markdown("Tabular breakdown of the forecasted trajectory.")
                    
                    last_date = prophet_df['ds'].max()
                    future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    future_only.columns = ['Date', 'Projected Target (INR)', 'Bearish Target (Lower)', 'Bullish Target (Upper)']
                    future_only['Date'] = future_only['Date'].dt.date
                    future_only.set_index('Date', inplace=True)
                    
                    st.dataframe(future_only.round(2), use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed. Ensure the ticker symbol is valid (e.g., INFY.NS). Error Details: {e}")
