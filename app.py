import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# Import the Auto-ARIMA library
from pmdarima.arima import auto_arima

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Auto-ARIMA Stock Forecaster", page_icon="📈", layout="wide")
st.title("📈 Fully Automated Stock Price Forecaster")
st.markdown("This app fetches the last 5 years of live data from Yahoo Finance, **automatically discovers the best ARIMA parameters**, and projects the trend until **June 2027**.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Enter an Indian stock ticker (e.g., **RELIANCE.NS**, **TCS.NS**, **INFY.NS**).")
    
    # Ticker Input (No manual parameters needed anymore!)
    ticker_symbol = st.text_input("Stock Ticker Symbol", value="RELIANCE.NS").upper()
    
    run_forecast = st.button("🚀 Run Auto-Forecast", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **How it works:** The engine uses `Auto-ARIMA` to test dozens of mathematical combinations in the background to find the perfect statistical fit for your specific stock before predicting the future.")

# --- Main Logic ---
if run_forecast:
    if ticker_symbol:
        # Step 1: Fetch Data
        with st.spinner(f"📡 Fetching 5 years of live data for {ticker_symbol} from Yahoo Finance..."):
            try:
                end_date = datetime.today()
                start_date = end_date - timedelta(days=5*365)
                
                df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error(f"No data found for {ticker_symbol}. Please check the symbol and try again.")
                else:
                    # Robustly extract 'Close' prices (handles yfinance formatting updates)
                    if isinstance(df.columns, pd.MultiIndex):
                        close_data = df['Close'][ticker_symbol].dropna()
                    else:
                        close_data = df['Close'].dropna()

                    st.success(f"✅ Data successfully downloaded ({len(close_data)} trading days).")
                    
                    # Step 2: Auto-ARIMA Optimization
                    with st.spinner("🧠 Finding the optimal ARIMA parameters automatically (This takes 10-30 seconds)..."):
                        
                        # The auto_arima function does all the heavy lifting
                        model = auto_arima(
                            close_data, 
                            start_p=0, start_q=0,
                            max_p=5, max_q=5, # Search bounds
                            seasonal=False,   # Daily stock data is generally treated as non-seasonal for ARIMA
                            stepwise=True,    # Uses a smart search algorithm to save time
                            suppress_warnings=True,
                            error_action="ignore"
                        )
                        
                        # Show the user the parameters the AI chose
                        best_order = model.order
                        st.info(f"🎯 **Optimization Complete:** The engine automatically selected ARIMA parameters **(p={best_order[0]}, d={best_order[1]}, q={best_order[2]})** as the most accurate fit for this specific stock.")
                        
                        # Step 3: Forecast until June 30, 2027
                        target_date = datetime(2027, 6, 30)
                        last_historical_date = close_data.index[-1]
                        
                        # Calculate future business days
                        future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), 
                                                     end=target_date, 
                                                     freq='B')
                        forecast_steps = len(future_dates)
                        
                        # Generate Forecast
                        forecast_values = model.predict(n_periods=forecast_steps)
                        
                        # Step 4: Interactive Plotly Graph
                        st.subheader(f"Projected Price Trajectory: {ticker_symbol}")
                        
                        fig = go.Figure()
                        
                        # Historical Data Line
                        fig.add_trace(go.Scatter(
                            x=close_data.index, y=close_data.values,
                            mode='lines', name='Historical Close Price',
                            line=dict(color='#2c3e50', width=2)
                        ))
                        
                        # Forecast Data Line
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=forecast_values,
                            mode='lines', name='Auto-ARIMA Forecast',
                            line=dict(color='#e74c3c', dash='dash', width=3)
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
                        
                        forecast_df = pd.DataFrame({
                            "Future Date": future_dates.date,
                            "Predicted Price (INR)": forecast_values.values
                        })
                        
                        forecast_df['Predicted Price (INR)'] = forecast_df['Predicted Price (INR)'].round(2)
                        
                        # Display as a clean, scrollable dataframe
                        st.dataframe(forecast_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"An execution error occurred: {e}")
    else:
        st.warning("Please enter a valid stock ticker.")
