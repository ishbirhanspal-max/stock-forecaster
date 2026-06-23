import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import gc
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# --- PAGE SETUP & CYBER THEME ---
st.set_page_config(page_title="AlphaQuant | Master Terminal", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Ultra-Modern Dark Theme */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Neon Accent Metric Cards */
    .metric-card { background: #111111; border: 1px solid #333333; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); }
    .metric-value { font-size: 26px; font-weight: 900; color: #00f2fe; font-family: 'Courier New', monospace; margin-top: 5px; }
    .metric-label { font-size: 11px; color: #888888; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }
    
    /* Analysis & Summary Boxes */
    .analysis-box { background-color: #0a0a0a; border-left: 4px solid #4facfe; padding: 20px; border-radius: 4px; font-size: 15px; line-height: 1.6; color: #cccccc; border: 1px solid #222; margin-top: 15px; margin-bottom: 25px;}
    
    /* BIG Company Name */
    .company-title { font-size: 3.5em; font-weight: 900; text-align: center; text-transform: uppercase; background: -webkit-linear-gradient(#00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2;}
    .company-ticker { font-size: 1.2em; font-weight: 600; text-align: center; color: #666; letter-spacing: 3px; margin-top: 0px; padding-top: 0px; margin-bottom: 30px;}
    
    /* Neon Verdict Boxes */
    .verdict-box { padding: 20px; border-radius: 8px; margin-top: 10px; margin-bottom: 20px; text-align: center; font-size: 20px; font-weight: 900; letter-spacing: 2px; color: #ffffff; text-transform: uppercase; border: 2px solid transparent; }
    .v-buy { background: rgba(0, 255, 135, 0.1); border-color: #00ff87; color: #00ff87; box-shadow: 0 0 20px rgba(0, 255, 135, 0.2); }
    .v-sell { background: rgba(255, 15, 123, 0.1); border-color: #ff0f7b; color: #ff0f7b; box-shadow: 0 0 20px rgba(255, 15, 123, 0.2); }
    .v-hold { background: rgba(248, 216, 43, 0.1); border-color: #f8d82b; color: #f8d82b; box-shadow: 0 0 20px rgba(248, 216, 43, 0.2); }
    
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
        fast = t.fast_info
        info['marketCap'] = fast.market_cap
        
        full_info = t.info
        info['longName'] = full_info.get('longName', ticker)
        info['longBusinessSummary'] = full_info.get('longBusinessSummary', 'Business profile currently unavailable.')
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
    
    ticker_symbol = st.text_input("Asset Ticker (e.g., RELIANCE.NS, AAPL):", value="", placeholder="Enter ticker symbol...").upper()
    
    st.markdown("### ⏱️ Global Timeline")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    
    st.markdown("### 📐 Manual ARIMA Override")
    st.caption("Standard statistical modeling configuration. Auto-Simulations run independently.")
    c1, c2, c3 = st.columns(3)
    with c1: p_val = st.number_input("p (Lag)", 0, 10, 5)
    with c2: d_val = st.number_input("d (Diff)", 0, 3, 1)
    with c3: q_val = st.number_input("q (MA)", 0, 10, 0)
    
    run_analysis = st.button("🚀 Execute Terminal", type="primary", use_container_width=True)

# ==========================================
# LANDING PAGE (GUIDELINES & DISCLAIMER)
# ==========================================
if not run_analysis or not ticker_symbol:
    st.markdown("<h1 style='text-align: center; font-size: 5em; color: #ffffff;'>ALPHA<span style='color: #00f2fe;'>QUANT</span> <span style='font-size:0.5em; color:#555;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2em; letter-spacing: 4px;'>INSTITUTIONAL ANALYSIS & FORECASTING DESK</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Terminal Operation Guidelines
        This platform runs three concurrent analytical engines to evaluate market conditions and future probabilities.
        
        1. **Target Selection:** Enter a ticker in the sidebar. **Use exchange suffixes for international stocks** (e.g., `.NS` for India's NSE, `.L` for London). US Stocks require no suffix.
        2. **Algorithmic Automation:** The system will automatically select the mathematically optimal number of Stochastic Simulations based on your chosen time horizon to protect server memory while maximizing accuracy.
        3. **Manual Override:** You may manually configure the `p, d, q` parameters for the classic ARIMA forecasting engine. 
        
        ### 🔬 The Three Engines
        * **1. Technical Confluence:** Evaluates present-day Momentum, Volatility, and Macro Trends.
        * **2. Stochastic Monte Carlo:** Simulates hundreds of alternate realities (Geometric Brownian Motion) to map risk probability cones.
        * **3. Deterministic ARIMA:** A user-defined, strict statistical mean-reversion algorithm.
        """)
        
    with col2:
        st.markdown("""
        ### ⚠️ Institutional Disclaimer
        <div style="background-color: rgba(255, 15, 123, 0.1); border: 1px solid #ff0f7b; padding: 15px; border-radius: 8px; font-size: 13px; color: #ccc;">
        <b>NOT FINANCIAL ADVICE.</b><br><br>
        Models do not account for macroeconomic variables, earnings shocks, or geopolitical Black Swan events. <br><br>
        ARIMA inherently calculates flat statistical means. Monte Carlo calculates risk probability distributions.<br><br>
        Capital is at risk. Use exclusively for quantitative research and educational analysis.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
if run_analysis and ticker_symbol:
    with st.spinner(f"📡 Establishing secure uplink for {ticker_symbol}... Compiling Stochastic Matrices..."):
        try:
            df, info = fetch_data(ticker_symbol)
            if df.empty:
                st.error(f"Critical Failure: No market data found. Ensure you are using the correct suffix (e.g., .NS).")
                st.stop()
            
            sym = get_currency_symbol(ticker_symbol)
            
            # Extract data
            if isinstance(df.columns, pd.MultiIndex):
                df_close = df['Close'][ticker_symbol].dropna()
                df_vol = df['Volume'][ticker_symbol].dropna()
                df_high = df['High'][ticker_symbol].dropna()
                df_low = df['Low'][ticker_symbol].dropna()
            else:
                df_close = df['Close'].dropna()
                df_vol = df['Volume'].dropna()
                df_high = df['High'].dropna()
                df_low = df['Low'].dropna()

            current_price = df_close.iloc[-1]
            
            # --- AUTOMATED SIMULATION OPTIMIZER ---
            # Automatically adjusts simulation depth based on time horizon to prevent memory crashes
            # Shorter horizon = deeper simulation allowed
            auto_sims = int(500 / horizon_years)
            auto_sims = min(max(auto_sims, 100), 1000) 

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

            tab_summary, tab_tech, tab_mc, tab_arima, tab_data = st.tabs([
                "📑 Executive Summary", 
                "📊 Technical Diagnostics", 
                "🎲 Professional Monte Carlo", 
                "📐 Manual ARIMA",
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
                
                st.markdown(f"<div class='analysis-box'><b>📖 Corporate Profile:</b><br>{info.get('longBusinessSummary', 'Business profile not available.')}</div>", unsafe_allow_html=True)

            # ==========================================
            # TAB 2: TECHNICAL DASHBOARD
            # ==========================================
            with tab_tech:
                fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Baselines", "RSI Momentum", "MACD Strength"))
                
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#ffffff")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#f8d82b", dash="dot")), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#00f2fe")), row=1, col=1)
                
                fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#b066ff")), row=2, col=1)
                fig_tech.add_hline(y=70, line_dash="dot", line_color="#ff0f7b", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dot", line_color="#00ff87", row=2, col=1)
                
                fig_tech.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#4facfe")), row=3, col=1)
                fig_tech.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#ff0f7b")), row=3, col=1)
                fig_tech.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Hist", marker_color=np.where(macd_hist>0, '#00ff87', '#ff0f7b')), row=3, col=1)

                fig_tech.update_layout(template="plotly_dark", height=850, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tech, use_container_width=True)

                st.markdown(f"""
                <div class="analysis-box">
                <b>🤖 Technical Breakdown & Verdict:</b><br><br>
                <b>1. Macro Trend:</b> The asset is trading <b>{"above" if current_price > sma_200.iloc[-1] else "below"}</b> the 200-Day Macro Baseline. Verdict: <b>{"Bullish Uptrend 🟢" if current_price > sma_200.iloc[-1] else "Bearish Downtrend 🔴"}</b>.<br>
                <b>2. Momentum (RSI):</b> Current RSI is <b>{rsi.iloc[-1]:.1f}</b>. Verdict: <b>{"Approaching Overbought (Exhaustion Risk) 🔴" if rsi.iloc[-1] > 65 else "Approaching Oversold (Bounce Potential) 🟢" if rsi.iloc[-1] < 35 else "Stable / Neutral ⚪"}</b>.<br>
                <b>3. Direction (MACD):</b> The MACD line is {"above" if macd.iloc[-1] > sig_line.iloc[-1] else "below"} the Signal line. Verdict: <b>{"Bullish Upward Momentum 🟢" if macd.iloc[-1] > sig_line.iloc[-1] else "Bearish Downward Momentum 🔴"}</b>.
                </div>
                """, unsafe_allow_html=True)

            # ==========================================
            # TAB 3: STOCHASTIC MONTE CARLO
            # ==========================================
            with tab_mc:
                with st.spinner("🧠 Booting Stochastic Engine..."):
                    trading_days = int(horizon_years * 252)
                    daily_returns = df_close.pct_change().dropna()
                    mu = daily_returns.mean()
                    sigma = daily_returns.std()
                    
                    sim_array = np.zeros((trading_days, auto_sims), dtype=np.float32)
                    sim_array[0] = current_price
                    
                    for t in range(1, trading_days):
                        shock = np.random.normal(0, 1, auto_sims)
                        sim_array[t] = sim_array[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * shock)
                    
                    future_dates_mc = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                    median_path = np.median(sim_array, axis=1)
                    upper_95 = np.percentile(sim_array, 95, axis=1)
                    lower_05 = np.percentile(sim_array, 5, axis=1)
                    
                    del sim_array
                    gc.collect()
                    
                    final_median = median_path[-1]
                    roi_mc = ((final_median - current_price) / current_price) * 100
                    
                    if roi_mc > 5: mc_v, mc_c = f"STOCHASTIC EXPECTATION: BULLISH (+{roi_mc:.2f}%)", "v-buy"
                    elif roi_mc < -5: mc_v, mc_c = f"STOCHASTIC EXPECTATION: BEARISH ({roi_mc:.2f}%)", "v-sell"
                    else: mc_v, mc_c = f"STOCHASTIC EXPECTATION: NEUTRAL ({roi_mc:.2f}%)", "v-hold"

                    st.markdown(f'<div class="verdict-box {mc_c}">{mc_v}</div>', unsafe_allow_html=True)
                    
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(x=df_close.index[-252:], y=df_close.values[-252:], name='Historical Data', line=dict(color='#cccccc', width=2)))
                    fig_mc.add_trace(go.Scatter(x=future_dates_mc, y=median_path, name='Statistical Median', line=dict(color='#00f2fe', width=3, dash='dash')))
                    fig_mc.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates_mc), pd.Series(future_dates_mc[::-1])]),
                        y=pd.concat([pd.Series(upper_95), pd.Series(lower_05[::-1])]),
                        fill='toself', fillcolor='rgba(0, 242, 254, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        name='90% Probability Cone'
                    ))
                    fig_mc.update_layout(template="plotly_dark", height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                    st.plotly_chart(fig_mc, use_container_width=True)

                    st.markdown(f"""
                    <div class="analysis-box">
                    <b>📌 Engine Definition:</b> Geometric Brownian Motion (GBM). The algorithm analyzed the stock's historic volatility and <b>automatically executed {auto_sims} alternate simulated realities</b> into the future.<br><br>
                    <b>🤖 Predictive Risk Analysis:</b> Over a {horizon_years}-year timeline, the expected median outcome is <b>{sym}{final_median:,.2f}</b>. There is a 90% statistical probability that extreme market volatility will trap the price between a floor of <b>{sym}{lower_05[-1]:,.2f}</b> and a ceiling of <b>{sym}{upper_95[-1]:,.2f}</b>.
                    </div>
                    """, unsafe_allow_html=True)

            # ==========================================
            # TAB 4: MANUAL ARIMA OVERRIDE
            # ==========================================
            with tab_arima:
                with st.spinner("🧠 Executing strict statistical ARIMA..."):
                    try:
                        log_close = np.log(df_close)
                        model = ARIMA(log_close, order=(p_val, d_val, q_val))
                        fitted_model = model.fit()
                        
                        forecast_res = fitted_model.get_forecast(steps=trading_days)
                        forecast_actual = np.exp(forecast_res.predicted_mean)
                        conf_int_log = forecast_res.conf_int().values
                        lower_arima = np.exp(conf_int_log[:, 0])
                        upper_arima = np.exp(conf_int_log[:, 1])
                        future_dates_arima = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)

                        target_price_ar = forecast_actual.iloc[-1] if hasattr(forecast_actual, 'iloc') else forecast_actual[-1]
                        roi_ar = ((target_price_ar - current_price) / current_price) * 100
                        
                        if roi_ar > 5: ar_v, ar_c = f"DETERMINISTIC TARGET: BULLISH (+{roi_ar:.2f}%)", "v-buy"
                        elif roi_ar < -5: ar_v, ar_c = f"DETERMINISTIC TARGET: BEARISH ({roi_ar:.2f}%)", "v-sell"
                        else: ar_v, ar_c = f"DETERMINISTIC TARGET: NEUTRAL ({roi_ar:.2f}%)", "v-hold"

                        st.markdown(f'<div class="verdict-box {ar_c}">{ar_v}</div>', unsafe_allow_html=True)

                        fig_ar = go.Figure()
                        fig_ar.add_trace(go.Scatter(x=df_close.index[-252:], y=df_close.values[-252:], name='Historical Data', line=dict(color='#cccccc', width=2)))
                        fig_ar.add_trace(go.Scatter(x=future_dates_arima, y=forecast_actual, name='ARIMA Projection', line=dict(color='#ff0f7b', width=3, dash='dash')))
                        fig_ar.add_trace(go.Scatter(
                            x=pd.concat([pd.Series(future_dates_arima), pd.Series(future_dates_arima[::-1])]),
                            y=pd.concat([pd.Series(upper_arima), pd.Series(lower_arima[::-1])]),
                            fill='toself', fillcolor='rgba(255, 15, 123, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                            name='Statistical Bounds'
                        ))
                        fig_ar.update_layout(template="plotly_dark", height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                        st.plotly_chart(fig_ar, use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="analysis-box">
                        <b>📌 Engine Definition:</b> You are currently executing a strict <b>ARIMA ({p_val}, {d_val}, {q_val})</b> regression model. This algorithm mathematically reverts to the safest statistical mean (resulting in a linear/flat trajectory).<br><br>
                        <b>🤖 Predictive Analysis:</b> The manual statistical override projects a final deterministic value of <b>{sym}{target_price_ar:,.2f}</b>, generating a return expectation of {roi_ar:.2f}%.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"ARIMA Error: {e}. Please adjust your manual p, d, q parameters.")

            # ==========================================
            # TAB 5: RAW DATA EXPORT
            # ==========================================
            with tab_data:
                st.header("💾 Probability Matrix Export")
                st.markdown("Export the generated timeline matrices for integration into external financial models.")
                
                export_df = pd.DataFrame({
                    'Date': future_dates_mc.date,
                    f'Monte Carlo Expected Median ({sym})': np.round(median_path, 2),
                    f'MC Worst Case Bound ({sym})': np.round(lower_05, 2),
                    f'MC Best Case Bound ({sym})': np.round(upper_95, 2),
                }).set_index('Date')
                
                st.dataframe(export_df, use_container_width=True)
                csv = export_df.to_csv().encode('utf-8')
                st.download_button("📥 Download Master Matrix (CSV)", data=csv, file_name=f'{ticker_symbol}_Master_Quant_Matrix.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Execution Error: {e}")
