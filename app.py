import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings

# Suppress standard statsmodels warnings for a cleaner Streamlit terminal
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Indian Stocks ARIMA Forecaster", layout="wide")
st.title("📈 Indian Stock Price ARIMA Forecaster")
st.markdown("Forecast stock prices until **June 2027** based on the last 5 years of historical Yahoo Finance data.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter an Indian stock ticker. Add **.NS** for NSE or **.BO** for BSE.")
    
    # Ticker Input
    ticker_symbol = st.text_input("Ticker Symbol", value="RELIANCE.NS").upper()
    
    # ARIMA Parameters (p, d, q)
    st.markdown("### ARIMA Parameters")
    st.markdown("*(p: lag order, d: degree of differencing, q: order of moving average)*")
    p = st.number_input("p (Autoregressive)", min_value=0, max_value=10, value=5)
    d = st.number_input("d (Differencing)", min_value=0, max_value=5, value=1)
    q = st.number_input("q (Moving Average)", min_value=0, max_value=10, value=0)
    
    run_forecast = st.button("🚀 Run Forecast", type="primary")

# --- Main Logic ---
if run_forecast:
    if ticker_symbol:
        with st.spinner(f"Fetching 5 years of data for {ticker_symbol}..."):
            try:
                # 1. Calculate Date Ranges (Last 5 Years)
                end_date = datetime.today()
                start_date = end_date - timedelta(days=5*365)
                
                # Fetch Data
                df = yf.download(ticker_symbol, start=start_date, end=end_date)
                
                if df.empty:
                    st.error("No data found. Please check the ticker symbol and try again.")
                else:
                    # Isolate the 'Close' prices
                    # Handling yfinance multi-index format if it occurs
                    if isinstance(df.columns, pd.MultiIndex):
                        close_data = df['Close'][ticker_symbol].dropna()
                    else:
                        close_data = df['Close'].dropna()

                    st.success(f"Data successfully fetched for {ticker_symbol}!")
                    
                    # 2. Run the ARIMA Model
                    with st.spinner("Calculating ARIMA Forecast (this may take a moment)..."):
                        # Fit the model
                        model = ARIMA(close_data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        # 3. Forecast until June 30, 2027
                        target_date = datetime(2027, 6, 30)
                        
                        # Calculate future business days (ignoring weekends)
                        last_historical_date = close_data.index[-1]
                        future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), 
                                                     end=target_date, 
                                                     freq='B')
                        forecast_steps = len(future_dates)
                        
                        # Generate Forecast
                        forecast_values = fitted_model.forecast(steps=forecast_steps)
                        
                        # 4. Interactive Plotly Graph
                        st.subheader(f"Price Forecast for {ticker_symbol}")
                        
                        fig = go.Figure()
                        
                        # Historical Data Line
                        fig.add_trace(go.Scatter(
                            x=close_data.index, 
                            y=close_data.values,
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='#1f77b4')
                        ))
                        
                        # Forecast Data Line
                        fig.add_trace(go.Scatter(
                            x=future_dates, 
                            y=forecast_values,
                            mode='lines',
                            name='ARIMA Forecast',
                            line=dict(color='#ff7f0e', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker_symbol} Forecast till June 2027",
                            xaxis_title="Date",
                            yaxis_title="Price (INR)",
                            hovermode="x unified",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 5. Display the Numbers (Data Table)
                        st.subheader("Forecasted Numbers (Tabular Data)")
                        
                        # Create a clean dataframe for the forecast
                        forecast_df = pd.DataFrame({
                            "Date": future_dates.date,
                            "Predicted Price": forecast_values.values
                        })
                        
                        # Format the numbers to 2 decimal places
                        forecast_df['Predicted Price'] = forecast_df['Predicted Price'].round(2)
                        
                        st.dataframe(forecast_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ticker symbol.")
