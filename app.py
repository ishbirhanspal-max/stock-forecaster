import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# Import the advanced Prophet forecasting engine
from prophet import Prophet

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Stock Forecaster", page_icon="📈", layout="wide")
st.title("📈 Advanced Algorithmic Stock Forecaster")
st.markdown("This engine uses **Facebook Prophet** to analyze 5 years of historical data, extracting yearly trends, weekly patterns, and market momentum to project prices until **June 2027**.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter an Indian stock ticker (e.g., **RELIANCE.NS**, **TCS.NS**, **INFY.NS**).")
    
    ticker_symbol = st.text_input("Stock Ticker Symbol", value="RELIANCE.NS").upper()
    
    run_forecast = st.button("🚀 Run Advanced Forecast", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Why Prophet?** Unlike basic models that draw flat lines, this engine decomposes the stock's history to find hidden seasonal patterns and long-term momentum, generating a highly dynamic predictive curve.")

# --- Main Logic ---
if run_forecast:
    if ticker_symbol:
        # Step 1: Fetch Data
        with st.spinner(f"📡 Fetching live market data for {ticker_symbol}..."):
            try:
                end_date = datetime.today()
                start_date = end_date - timedelta(days=5*365)
                
                df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error(f"No data found for {ticker_symbol}. Please check the symbol.")
                else:
                    # Clean the data for Prophet (Prophet requires columns named 'ds' for dates and 'y' for values)
                    if isinstance(df.columns, pd.MultiIndex):
                        close_data = df['Close'][ticker_symbol].dropna().reset_index()
                    else:
                        close_data = df['Close'].dropna().reset_index()
                    
                    close_data.columns = ['ds', 'y'] # Rename for Prophet compatibility
                    
                    # Ensure timezone-naive dates for Prophet
                    close_data['ds'] = close_data['ds'].dt.tz_localize(None)

                    st.success("✅ Data successfully downloaded. Initializing algorithmic modeling...")
                    
                    # Step 2: Prophet Modeling & Fine-tuning
                    with st.spinner("🧠 Analyzing trends and seasonal market patterns..."):
                        
                        # Initialize Prophet with fine-tuned parameters for financial data
                        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=True,
                            changepoint_prior_scale=0.05 # Allows the trend line to be flexible and catch recent momentum
                        )
                        
                        # Fit the model
                        model.fit(close_data)
                        
                        # Step 3: Forecast until June 30, 2027
                        target_date = datetime(2027, 6, 30)
                        last_date = close_data['ds'].max()
                        days_into_future = (target_date - last_date).days
                        
                        # Generate future dates
                        future = model.make_future_dataframe(periods=days_into_future)
                        
                        # Filter out weekends (Saturdays and Sundays) for realistic stock market days
                        future = future[future['ds'].dt.weekday < 5]
                        
                        # Predict the future!
                        forecast = model.predict(future)
                        
                        # Extract just the future portion for the table
                        future_forecast = forecast[forecast['ds'] > last_date]
                        
                        # Step 4: Interactive Plotly Graph
                        st.subheader(f"Projected Price Trajectory: {ticker_symbol}")
                        
                        fig = go.Figure()
                        
                        # Historical Data (Actuals)
                        fig.add_trace(go.Scatter(
                            x=close_data['ds'], y=close_data['y'],
                            mode='lines', name='Historical Close Price',
                            line=dict(color='#2c3e50', width=2)
                        ))
                        
                        # Forecast Data Line (Trend)
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['yhat'],
                            mode='lines', name='AI Projected Trend',
                            line=dict(color='#e74c3c', width=3)
                        ))
                        
                        # Confidence Interval (Shaded Area)
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(231, 76, 60, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval (Volatility)',
                            showlegend=True
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker_symbol} Price Forecast till June 2027",
                            xaxis_title="Timeline",
                            yaxis_title="Price (INR)",
                            hovermode="x unified",
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Step 5: Tabular Data display
                        st.subheader("📅 Future Valuation Metrics (Tabular)")
                        
                        # Clean up the output table
                        clean_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        clean_table.columns = ['Date', 'Predicted Price', 'Worst Case', 'Best Case']
                        
                        # Format dates and round numbers
                        clean_table['Date'] = clean_table['Date'].dt.date
                        clean_table['Predicted Price'] = clean_table['Predicted Price'].round(2)
                        clean_table['Worst Case'] = clean_table['Worst Case'].round(2)
                        clean_table['Best Case'] = clean_table['Best Case'].round(2)
                        
                        st.dataframe(clean_table, use_container_width=True)
                        
            except Exception as e:
                st.error(f"An execution error occurred: {e}")
    else:
        st.warning("Please enter a valid stock ticker.")
