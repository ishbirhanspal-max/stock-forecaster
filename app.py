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
st.set_page_config(page_title="Pro Stock Analyst & Forecaster", page_icon="📊", layout="wide")

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Algorithmic Market Analysis & Forecasting Suite")
st.markdown("A comprehensive quantitative analysis tool combining historical technical indicators with Meta's advanced Prophet AI for forward-looking projections.")
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ticker_symbol = st.text_input("Indian Stock Ticker (e.g., RELIANCE.NS, TCS.NS)", value="RELIANCE.NS").upper()
    run_analysis = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🧠 Model Parameters")
    st.markdown("- **Engine:** Facebook Prophet\n- **Seasonality:** Weekly & Yearly\n- **Holidays:** Indian Market Holidays Included\n- **Horizon:** June 2027")

# --- Main Logic ---
if run_analysis:
    if ticker_symbol:
        with st.spinner(f"📡 Fetching market data and computing analytics for {ticker_symbol}..."):
            try:
                # 1. Fetch 5 Years of Data
                end_date = datetime.today()
                start_date = end_date - timedelta(days=5*365)
                df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error("No data found. Please verify the ticker symbol.")
                    st.stop()
                
                # Format Data
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                    df_volume = df['Volume'][ticker_symbol].dropna()
                else:
                    df_close = df['Close'].dropna()
                    df_volume = df['Volume'].dropna()
                
                current_price = df_close.iloc[-1]
                
                # ==========================================
                # SECTION 1: DESCRIPTIVE DATA ANALYSIS
                # ==========================================
                st.header("1️⃣ Historical Performance & Volatility Analysis")
                st.markdown("Understanding how the asset has behaved over the last 5 years before looking into the future.")
                
                # Calculate Metrics
                daily_returns = df_close.pct_change().dropna()
                annualized_volatility = daily_returns.std() * np.sqrt(252) * 100 # 252 trading days
                total_return = ((current_price / df_close.iloc[0]) - 1) * 100
                cagr = (((current_price / df_close.iloc[0]) ** (1/5)) - 1) * 100
                high_52 = df_close.tail(252).max()
                low_52 = df_close.tail(252).min()
                
                # Display Metrics in Custom Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f'<div class="metric-card"><div class="metric-label">Current Price</div><div class="metric-value">₹{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><div class="metric-label">5-Year CAGR</div><div class="metric-value">{cagr:,.2f}%</div></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><div class="metric-label">Annual Volatility (Risk)</div><div class="metric-value">{annualized_volatility:,.2f}%</div></div>', unsafe_allow_html=True)
                col4.markdown(f'<div class="metric-card"><div class="metric-label">52-Week Range</div><div class="metric-value">₹{low_52:,.0f} - ₹{high_52:,.0f}</div></div>', unsafe_allow_html=True)
                st.write("")

                # Technical Indicators (Moving Averages)
                sma_50 = df_close.rolling(window=50).mean()
                sma_200 = df_close.rolling(window=200).mean()
                
                fig_tech = go.Figure()
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close.values, name="Close Price", line=dict(color="#2c3e50")))
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50-Day SMA", line=dict(color="#e67e22", dash="dot")))
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200-Day SMA", line=dict(color="#c0392b", dash="dash")))
                
                fig_tech.update_layout(title="Historical Price with Moving Averages (Trend Identification)", hovermode="x unified", template="plotly_white", height=450)
                st.plotly_chart(fig_tech, use_container_width=True)

                # ==========================================
                # SECTION 2: PROPHET FORECASTING
                # ==========================================
                st.header("2️⃣ Algorithmic Price Projection (Prophet AI)")
                st.markdown("The AI decomposes historical data to isolate the core trend from market noise, then projects it forward factoring in historical volatility boundaries.")
                
                # Prepare data for Prophet
                prophet_df = df_close.reset_index()
                prophet_df.columns = ['ds', 'y']
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                
                # Model Fine-Tuning
                model = Prophet(
                    changepoint_prior_scale=0.05, 
                    seasonality_prior_scale=10.0,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                # Add Indian Holidays to improve accuracy
                model.add_country_holidays(country_name='IN')
                model.fit(prophet_df)
                
                # Forecast till June 30, 2027
                target_date = datetime(2027, 6, 30)
                last_date = prophet_df['ds'].max()
                days_ahead = (target_date - last_date).days
                
                future = model.make_future_dataframe(periods=days_ahead)
                future = future[future['ds'].dt.weekday < 5] # Remove weekends
                forecast = model.predict(future)
                
                # Plotly Forecast Chart
                fig_fore = go.Figure()
                fig_fore.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Historical Actuals', line=dict(color='#2c3e50', width=2)))
                fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Trend Projection', line=dict(color='#27ae60', width=3)))
                fig_fore.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(39, 174, 96, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval (Expected Volatility)'
                ))
                
                fig_fore.update_layout(title=f"{ticker_symbol} Path to June 2027", hovermode="x unified", template="plotly_white", height=550)
                st.plotly_chart(fig_fore, use_container_width=True)

                # ==========================================
                # SECTION 3: COMPONENT ANALYSIS
                # ==========================================
                st.header("3️⃣ Inside the Algorithm: Market Seasonality")
                st.markdown("Why did the AI draw that specific curve? By breaking down the components, we can see the hidden seasonal cycles driving the stock's price.")
                
                col_y, col_w = st.columns(2)
                
                with col_y:
                    st.subheader("🗓️ Yearly Seasonality")
                    st.write("Shows which months historically perform best or worst for this specific stock.")
                    fig_yearly = go.Figure(go.Scatter(x=forecast['ds'].dt.dayofyear, y=forecast['yearly'], mode='lines', line=dict(color='#8e44ad')))
                    fig_yearly.update_layout(xaxis_title="Day of Year", yaxis_title="Price Impact (INR)", template="plotly_white", height=300, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_yearly, use_container_width=True)
                
                with col_w:
                    st.subheader("📆 Weekly Seasonality")
                    st.write("Shows the average price momentum based on the day of the trading week.")
                    # Map 0-4 to Mon-Fri
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    weekly_data = forecast[['ds', 'weekly']].copy()
                    weekly_data['weekday'] = weekly_data['ds'].dt.weekday
                    weekly_avg = weekly_data[weekly_data['weekday'] < 5].groupby('weekday')['weekly'].mean()
                    
                    fig_weekly = go.Figure(go.Bar(x=days, y=weekly_avg.values, marker_color='#2980b9'))
                    fig_weekly.update_layout(yaxis_title="Price Impact (INR)", template="plotly_white", height=300, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_weekly, use_container_width=True)

                # ==========================================
                # SECTION 4: DATA EXPORT
                # ==========================================
                st.header("4️⃣ Future Data Matrix")
                st.markdown("Raw forecasted numbers for integration into external financial models.")
                
                future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                future_only.columns = ['Date', 'Projected Target (INR)', 'Bearish Case (Lower Bound)', 'Bullish Case (Upper Bound)']
                future_only['Date'] = future_only['Date'].dt.date
                future_only.set_index('Date', inplace=True)
                
                st.dataframe(future_only.round(2), use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed. Ensure the ticker symbol is valid (e.g., INFY.NS). Error Details: {e}")
