import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import gc

warnings.filterwarnings('ignore')

# --- PAGE SETUP & HIGH-CONTRAST CSS THEME ---
st.set_page_config(page_title="AlphaQuant | Institutional Terminal", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Ultra-Modern Dark Theme */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Neon Accent Metric Cards */
    .metric-card { 
        background: #111111; 
        border: 1px solid #333333; 
        border-radius: 8px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    .metric-value { font-size: 28px; font-weight: 900; color: #00f2fe; font-family: 'Courier New', monospace; margin-top: 5px; }
    .metric-label { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }
    
    /* Analysis & Summary Boxes */
    .analysis-box { background-color: #0a0a0a; border-left: 4px solid #4facfe; padding: 20px; border-radius: 4px; font-size: 15px; line-height: 1.6; color: #cccccc; border: 1px solid #222; }
    
    /* BIG Company Name */
    .company-title { font-size: 4em; font-weight: 900; text-align: center; text-transform: uppercase; background: -webkit-linear-gradient(#00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2;}
    .company-ticker { font-size: 1.5em; font-weight: 600; text-align: center; color: #666; letter-spacing: 3px; margin-top: 0px; padding-top: 0px; margin-bottom: 30px;}
    
    /* Neon Verdict Boxes */
    .verdict-box { padding: 20px; border-radius: 8px; margin-top: 10px; margin-bottom: 20px; text-align: center; font-size: 24px; font-weight: 900; letter-spacing: 2px; color: #ffffff; text-transform: uppercase; border: 2px solid transparent; }
    .v-buy { background: rgba(0, 255, 135, 0.1); border-color: #00ff87; color: #00ff87; box-shadow: 0 0 20px rgba(0, 255, 135, 0.2); }
    .v-sell { background: rgba(255, 15, 123, 0.1); border-color: #ff0f7b; color: #ff0f7b; box-shadow: 0 0 20px rgba(255, 15, 123, 0.2); }
    .v-hold { background: rgba(248, 216, 43, 0.1); border-color: #f8d82b; color: #f8d82b; box-shadow: 0 0 20px rgba(248, 216, 43, 0.2); }
    
    /* Custom Dividers */
    hr { border-color: #222222; }
    </style>
""", unsafe_allow_html=True)

# --- INTELLIGENT CURRENCY ROUTER ---
def get_currency_symbol(ticker):
    t = ticker.upper()
    if t.endswith('.NS') or t.endswith('.BO'): return '₹'
    elif t.endswith('.L'): return '£'
    elif t.endswith('.TO'): return 'C$'
    elif t.endswith('.AX'): return 'A$'
    elif t.endswith('.DE') or t.endswith('.PA'): return '€'
    else: return '$' 

# --- DATA FETCHING (Robust Caching) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    info = {}
    try:
        t = yf.Ticker(ticker)
        # Fetch basic fast info to avoid rate limits
        fast = t.fast_info
        info['marketCap'] = fast.market_cap
        info['previousClose'] = fast.previous_close
        
        # Try to get the proper long name
        full_info = t.info
        info['longName'] = full_info.get('longName', ticker)
        info['longBusinessSummary'] = full_info.get('longBusinessSummary', 'Business summary temporarily unavailable.')
        info['trailingPE'] = full_info.get('trailingPE', 'N/A')
        info['dividendYield'] = full_info.get('dividendYield', 'N/A')
    except:
        info['longName'] = ticker
        info['longBusinessSummary'] = "Data temporarily unavailable due to exchange rate limits."
        
    return df, info

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency-systems-regular/96/00f2fe/combo-chart.png", width=60)
    st.markdown("## ⚙️ Quant Setup")
    
    # EMPTIED DEFAULT INPUT AS REQUESTED
    ticker_symbol = st.text_input("Asset Ticker (e.g., RELIANCE.NS, AAPL):", value="", placeholder="Enter ticker symbol...").upper()
    
    st.markdown("### 🎲 Monte Carlo Parameters")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    simulations = st.slider("Simulated Realities:", 100, 1000, 500, 100)
    
    run_analysis = st.button("🚀 Execute Terminal", type="primary", use_container_width=True)

# ==========================================
# LANDING PAGE (GUIDELINES & DISCLAIMER)
# ==========================================
if not run_analysis or not ticker_symbol:
    st.markdown("<h1 style='text-align: center; font-size: 4em; color: #ffffff;'>ALPHA<span style='color: #00f2fe;'>QUANT</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2em; letter-spacing: 2px;'>INSTITUTIONAL STOCHASTIC FORECASTING TERMINAL</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 How to Use This Terminal
        This platform uses institutional-grade quantitative mathematics to analyze assets. 
        
        1. **Enter a Ticker:** In the left sidebar, enter a valid Yahoo Finance ticker. 
           * *Indian Stocks:* Must include the suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `ZOMATO.NS`).
           * *US Stocks:* Just the ticker (e.g., `AAPL`, `TSLA`, `NVDA`).
        2. **Set Parameters:** Choose how far into the future you want to predict, and how many alternate realities the engine should simulate.
        3. **Execute:** Click 'Execute Terminal' to generate the reports.
        
        ### 🧠 The Mathematical Philosophy (Why No Flat Lines?)
        Retail traders use basic algorithms like ARIMA, which inevitably draw a single "flat line" into the future because they mathematically revert to the mean. **Professional Quants know this is impossible.** This terminal uses **Geometric Brownian Motion (Monte Carlo)**. It simulates hundreds of randomized, hyper-realistic market paths based on the stock's historical volatility and drift. The result is a **Probability Cone**—telling you mathematically the maximum upside and maximum downside risk.
        """)
        
    with col2:
        st.markdown("""
        ### ⚠️ Legal Disclaimer
        <div style="background-color: rgba(255, 15, 123, 0.1); border: 1px solid #ff0f7b; padding: 15px; border-radius: 8px; font-size: 12px; color: #ccc;">
        <b>NO FINANCIAL ADVICE.</b><br><br>
        The data, analysis, and forecasts provided by this terminal are generated purely through mathematical algorithms and historical data mapping. <br><br>
        This tool does <b>not</b> account for macroeconomic events, sudden earnings crashes, geopolitical news, or Black Swan events. <br><br>
        The simulations represent statistical probabilities, not guarantees. Trading equities involves significant risk of loss. Use this data strictly for educational and technical analysis purposes.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if run_analysis and ticker_symbol:
    with st.spinner(f"📡 Establishing secure uplink for {ticker_symbol}... Compiling Stochastic Matrices..."):
        try:
            df, info = fetch_data(ticker_symbol)
            if df.empty:
                st.error(f"Critical Failure: No market data found for '{ticker_symbol}'. Ensure you are using the correct suffix (e.g., .NS for India).")
                st.stop()
            
            sym = get_currency_symbol(ticker_symbol)
            
            # Extract data
            if isinstance(df.columns, pd.MultiIndex):
                df_close = df['Close'][ticker_symbol].dropna()
                df_vol = df['Volume'][ticker_symbol].dropna()
            else:
                df_close = df['Close'].dropna()
                df_vol = df['Volume'].dropna()

            current_price = df_close.iloc[-1]
            
            # --- TECHNICAL CALCULATIONS ---
            sma_50 = df_close.rolling(50).mean()
            sma_200 = df_close.rolling(200).mean()
            
            delta = df_close.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))

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

            if score >= 2: master_class, master_text = "v-buy", "TERMINAL VERDICT: STRONGLY BULLISH"
            elif score <= -2: master_class, master_text = "v-sell", "TERMINAL VERDICT: STRONGLY BEARISH"
            else: master_class, master_text = "v-hold", "TERMINAL VERDICT: NEUTRAL / RANGEBOUND"

            # --- BIG COMPANY NAME DISPLAY ---
            st.markdown(f"<div class='company-title'>{info.get('longName', ticker_symbol)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='company-ticker'>| {ticker_symbol} |</div>", unsafe_allow_html=True)

            tab_summary, tab_tech, tab_fore, tab_data = st.tabs([
                "📑 Executive Summary", 
                "📊 Technical Diagnostics", 
                "🎲 Professional Monte Carlo", 
                "💾 Matrix Export"
            ])

            # ==========================================
            # TAB 1: EXECUTIVE SUMMARY
            # ==========================================
            with tab_summary:
                st.markdown(f'<div class="verdict-box {master_class}">{master_text}</div>', unsafe_allow_html=True)
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                mcap = info.get('marketCap', 'N/A')
                mcap_str = f"{sym}{mcap / 1e9:,.2f}B" if isinstance(mcap, (int, float)) else "N/A"
                pe = info.get('trailingPE', 'N/A')
                pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
                div = info.get('dividendYield', 'N/A')
                div_str = f"{div*100:.2f}%" if isinstance(div, (int, float)) else "0.00%"
                
                col_s1.markdown(f'<div class="metric-card"><div class="metric-label">Last Close</div><div class="metric-value">{sym}{current_price:,.2f}</div></div>', unsafe_allow_html=True)
                col_s2.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{mcap_str}</div></div>', unsafe_allow_html=True)
                col_s3.markdown(f'<div class="metric-card"><div class="metric-label">P/E Ratio</div><div class="metric-value">{pe_str}</div></div>', unsafe_allow_html=True)
                col_s4.markdown(f'<div class="metric-card"><div class="metric-label">Dividend Yield</div><div class="metric-value">{div_str}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### 📖 Corporate Profile")
                st.markdown(f"<div class='analysis-box'>{info.get('longBusinessSummary', 'Business profile not available.')}</div>", unsafe_allow_html=True)

            # ==========================================
            # TAB 2: TECHNICAL DASHBOARD
            # ==========================================
            with tab_tech:
                st.header("Algorithmic Technical Tracking")
                
                fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Baselines", "RSI Momentum", "MACD Strength"))
                
                # Price Chart
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#ffffff")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#f8d82b", dash="dot")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#00f2fe")), row=1, col=1)
                
                # RSI Chart
                fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#b066ff")), row=2, col=1)
                fig_tech.add_hline(y=70, line_dash="dot", line_color="#ff0f7b", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dot", line_color="#00ff87", row=2, col=1)
                
                # MACD Chart
                fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#4facfe")), row=3, col=1)
                fig_tech.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#ff0f7b")), row=3, col=1)
                fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Hist", marker_color=np.where(macd_hist>0, '#00ff87', '#ff0f7b')), row=3, col=1)

                fig_tech.update_layout(template="plotly_dark", height=850, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tech, use_container_width=True)

                st.markdown(f"""
                <div class="analysis-box">
                <b>🤖 AI Technical Analysis:</b><br><br>
                <b>1. Trend (Price vs 200 SMA):</b> The asset is trading <b>{"above" if current_price > sma_200.iloc[-1] else "below"}</b> the 200-Day Macro Baseline. This indicates a long-term {"Bullish Uptrend" if current_price > sma_200.iloc[-1] else "Bearish Downtrend"}.<br>
                <b>2. Momentum (RSI):</b> Current RSI is <b>{rsi.iloc[-1]:.1f}</b>. {"It is approaching Overbought territory, signaling potential exhaustion." if rsi.iloc[-1] > 65 else "It is in Oversold territory, signaling panic selling and a potential bounce." if rsi.iloc[-1] < 35 else "It sits in a stable, neutral momentum zone."}<br>
                <b>3. Direction (MACD):</b> The MACD line is {"above" if macd.iloc[-1] > sig_line.iloc[-1] else "below"} the Signal line, confirming {"positive upward" if macd.iloc[-1] > sig_line.iloc[-1] else "negative downward"} momentum.
                </div>
                """, unsafe_allow_html=True)

            # ==========================================
            # TAB 3: PROFESSIONAL MONTE CARLO FORECAST
            # ==========================================
            with tab_fore:
                st.header(f"Stochastic Volatility Engine ({horizon_years} Year Horizon)")
                
                with st.spinner("🧠 Simulating Alternate Realities..."):
                    # Math Engine
                    trading_days = int(horizon_years * 252)
                    daily_returns = df_close.pct_change().dropna()
                    mu = daily_returns.mean()
                    sigma = daily_returns.std()
                    
                    # Memory Optimized Matrix
                    sim_array = np.zeros((trading_days, simulations), dtype=np.float32)
                    sim_array[0] = current_price
                    
                    # Generate Random Walk
                    for t in range(1, trading_days):
                        shock = np.random.normal(0, 1, simulations)
                        sim_array[t] = sim_array[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * shock)
                    
                    future_dates = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                    
                    # Calculate Percentiles (Probabilities)
                    median_path = np.median(sim_array, axis=1)
                    upper_95 = np.percentile(sim_array, 95, axis=1)
                    lower_05 = np.percentile(sim_array, 5, axis=1)
                    
                    # Clear RAM
                    del sim_array
                    gc.collect()
                    
                    # Target Metrics
                    final_median = median_path[-1]
                    roi = ((final_median - current_price) / current_price) * 100
                    
                    if roi > 5: fore_v, fore_c = f"STATISTICAL PROJECTION: BULLISH (+{roi:.2f}%)", "v-buy"
                    elif roi < -5: fore_v, fore_c = f"STATISTICAL PROJECTION: BEARISH ({roi:.2f}%)", "v-sell"
                    else: fore_v, fore_c = f"STATISTICAL PROJECTION: NEUTRAL ({roi:.2f}%)", "v-hold"

                    st.markdown(f'<div class="verdict-box {fore_c}">{fore_v}</div>', unsafe_allow_html=True)
                    
                    # Plotly Chart
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(x=df_close.index[-252:], y=df_close.values[-252:], name='Historical Data', line=dict(color='#cccccc', width=2)))
                    fig_mc.add_trace(go.Scatter(x=future_dates, y=median_path, name='Statistical Median', line=dict(color='#00f2fe', width=3, dash='dash')))
                    fig_mc.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_95), pd.Series(lower_05[::-1])]),
                        fill='toself', fillcolor='rgba(0, 242, 254, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        name='90% Probability Cone (Volatility)'
                    ))
                    
                    fig_mc.update_layout(template="plotly_dark", height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                    st.plotly_chart(fig_mc, use_container_width=True)

                    st.markdown(f"""
                    <div class="analysis-box">
                    <b>📌 Engine Definition:</b> This uses Geometric Brownian Motion (GBM). Instead of drawing a single flat line, it calculates {simulations} randomized future paths based on how volatile the stock has been in the past. It then calculates the exact statistical median and maximum risk boundaries.<br><br>
                    <b>🤖 Predictive Risk Analysis:</b> The math states that in {horizon_years} years, the median expected price is <b>{sym}{final_median:,.2f}</b>. Furthermore, there is a 90% statistical probability that the price will remain between the Worst-Case Scenario of <b>{sym}{lower_05[-1]:,.2f}</b> and the Best-Case Scenario of <b>{sym}{upper_95[-1]:,.2f}</b>.
                    </div>
                    """, unsafe_allow_html=True)

            # ==========================================
            # TAB 4: RAW DATA EXPORT
            # ==========================================
            with tab_data:
                st.header("💾 Probability Matrix Export")
                st.markdown(f"Export the raw Monte Carlo probability bounds for integration into external financial modeling.")
                
                export_df = pd.DataFrame({
                    'Date': future_dates.date,
                    f'Median Target ({sym})': np.round(median_path, 2),
                    f'Lower Risk Bound ({sym})': np.round(lower_05, 2),
                    f'Upper Potential Bound ({sym})': np.round(upper_95, 2)
                }).set_index('Date')
                
                st.dataframe(export_df, use_container_width=True)
                csv = export_df.to_csv().encode('utf-8')
                st.download_button("📥 Download Projection Matrix as CSV", data=csv, file_name=f'{ticker_symbol}_Monte_Carlo_Matrix.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Execution Error: {e}")
