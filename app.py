import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="AlphaQuant | Monte Carlo Edition", page_icon="🏛️", layout="wide")

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

st.title("🏛️ AlphaQuant: Stochastic Monte Carlo Terminal")
st.markdown("Simulating thousands of alternate future trajectories using Geometric Brownian Motion (GBM) to model real-world market risk and volatility.")
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
    except:
        pass
    
    if not info or 'marketCap' not in info:
        try:
            fast = ticker_obj.fast_info
            info['marketCap'] = fast.market_cap
            info['sector'] = "Limited Data"
        except:
            pass
    return df, info

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    ticker_symbol = st.text_input("Stock Ticker (e.g., RELIANCE.NS, TSLA)", value="RELIANCE.NS").upper()
    
    st.markdown("### 🎲 Monte Carlo Parameters")
    horizon_years = st.slider("Forecast Horizon (Years):", min_value=1, max_value=5, value=2)
    num_simulations = st.slider("Number of Alternate Realities:", min_value=10, max_value=500, value=100)
    
    run_analysis = st.button("🚀 Execute Simulation", type="primary", use_container_width=True)

# --- Main Logic ---
if run_analysis:
    if ticker_symbol:
        with st.spinner(f"📡 Downloading data and running {num_simulations} mathematical simulations for {ticker_symbol}..."):
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
                
                # --- TECHNICALS & VERDICT ---
                daily_returns = df_close.pct_change().dropna()
                sma_50 = df_close.rolling(window=50).mean()
                sma_200 = df_close.rolling(window=200).mean()
                
                # RSI
                delta = df_close.diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # MACD
                ema_12 = df_close.ewm(span=12, adjust=False).mean()
                ema_26 = df_close.ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                signal_line = macd.ewm(span=9, adjust=False).mean()

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

                # UI TABS
                tab_tech, tab_mc, tab_export = st.tabs(["📊 Technical Dashboard", "🎲 Monte Carlo Simulation", "💾 Data Matrix"])

                # ==========================================
                # TAB 1: TECHNICALS
                # ==========================================
                with tab_tech:
                    st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Close Price", line=dict(color="#2c3e50")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#e67e22", dash="dot")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#c0392b", dash="dash")), row=1, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#8e44ad")), row=2, col=1)
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig_tech.update_layout(height=600, hovermode="x unified", template="plotly_white", margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

                # ==========================================
                # TAB 2: MONTE CARLO SIMULATION
                # ==========================================
                with tab_mc:
                    st.header(f"Geometric Brownian Motion ({horizon_years} Year Risk Profile)")
                    st.markdown(f"Running **{num_simulations}** unique simulations based on the stock's historical drift and volatility.")
                    
                    # --- GBM MATH ENGINE ---
                    trading_days = int(horizon_years * 252)
                    mu = daily_returns.mean()
                    sigma = daily_returns.std()
                    
                    # Create simulation matrix
                    simulations = np.zeros((trading_days, num_simulations))
                    simulations[0] = current_price
                    
                    # Generate random shocks and calculate prices
                    for t in range(1, trading_days):
                        # Random shock (Brownian Motion)
                        random_shock = np.random.normal(0, 1, num_simulations)
                        # Price calculation formula
                        simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * random_shock)
                    
                    # Generate Future Dates
                    last_date = df_close.index[-1]
                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=trading_days)
                    
                    # Calculate Probability Bands
                    sim_df = pd.DataFrame(simulations, index=future_dates)
                    median_path = sim_df.median(axis=1)
                    upper_95 = sim_df.quantile(0.95, axis=1) # Top 5% Best Case Scenario
                    lower_05 = sim_df.quantile(0.05, axis=1) # Bottom 5% Worst Case Scenario
                    
                    # Plotting
                    fig_mc = go.Figure()
                    
                    # Plot Historical
                    fig_mc.add_trace(go.Scatter(x=df_close.index[-500:], y=df_close.values[-500:], name="Historical Data", line=dict(color="black", width=2)))
                    
                    # Plot a few random "Squiggly" Paths to show volatility (max 5 lines to keep it clean)
                    for i in range(min(5, num_simulations)):
                        fig_mc.add_trace(go.Scatter(x=future_dates, y=simulations[:, i], mode='lines', line=dict(width=1, opacity=0.3), showlegend=False))
                    
                    # Plot Median (The Expected Average Trend)
                    fig_mc.add_trace(go.Scatter(x=future_dates, y=median_path, name="Median Trajectory", line=dict(color="#2980b9", width=3, dash='dash')))
                    
                    # Plot 95% Confidence Interval Shadows
                    fig_mc.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_95), pd.Series(lower_05[::-1])]),
                        fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                        name='90% Probability Boundary'
                    ))
                    
                    fig_mc.update_layout(title="Monte Carlo Probability Cone", hovermode="x unified", template="plotly_white", height=700)
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Save for export
                    export_df = pd.DataFrame({
                        'Date': future_dates.date,
                        'Expected Median Price': np.round(median_path.values, 2),
                        'Worst Case (5th Percentile)': np.round(lower_05.values, 2),
                        'Best Case (95th Percentile)': np.round(upper_95.values, 2)
                    }).set_index('Date')
                    st.session_state['mc_export'] = export_df

                # ==========================================
                # TAB 3: EXPORT
                # ==========================================
                with tab_export:
                    st.header("💾 Download Probability Matrix")
                    if 'mc_export' in st.session_state:
                        st.dataframe(st.session_state['mc_export'], use_container_width=True)
                        csv = st.session_state['mc_export'].to_csv().encode('utf-8')
                        st.download_button("📥 Download Simulation Matrix (CSV)", data=csv, file_name=f'{ticker_symbol}_MonteCarlo.csv', mime='text/csv')

            except Exception as e:
                st.error(f"Analysis failed. {e}")
