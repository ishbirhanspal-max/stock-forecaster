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

# --- PAGE SETUP & CYBER THEME ---
st.set_page_config(page_title="AlphaQuant | Executive Terminal", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    .metric-card { 
        background: rgba(30, 41, 59, 0.7); 
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-radius: 12px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 26px; font-weight: 800; color: #38bdf8; font-family: 'Courier New', monospace; margin-top: 5px; }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    
    .analysis-box { background-color: #0f172a; border-left: 4px solid #38bdf8; padding: 20px; border-radius: 6px; font-size: 15px; line-height: 1.6; color: #cbd5e1; margin-top: 15px; margin-bottom: 25px; }
    
    .verdict-box { padding: 20px; border-radius: 12px; margin-top: 10px; margin-bottom: 20px; text-align: center; font-size: 22px; font-weight: 900; letter-spacing: 1px; color: #ffffff; text-transform: uppercase; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
    .v-buy { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .v-sell { background: linear-gradient(135deg, #be123c 0%, #e11d48 100%); }
    .v-hold { background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%); }
    </style>
""", unsafe_allow_html=True)

# --- INTELLIGENT CURRENCY ROUTER ---
def get_currency_config(ticker):
    ticker_up = ticker.upper()
    if ticker_up.endswith('.NS') or ticker_up.endswith('.BO'): return '₹', 'INR'
    elif ticker_up.endswith('.L'): return '£', 'GBP'
    elif ticker_up.endswith('.TO'): return 'C$', 'CAD'
    elif ticker_up.endswith('.AX'): return 'A$', 'AUD'
    elif ticker_up.endswith('.DE') or ticker_up.endswith('.PA') or ticker_up.endswith('.AS'): return '€', 'EUR'
    else: return '$', 'USD'

# --- DATA FETCHING (Robust Caching) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365) # Strictly forces minimum 5 years of data
    
    # Using auto_adjust=True prevents the -100% crash bug caused by stock splits
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    info = {}
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        info['marketCap'] = float(fast.market_cap) if fast.market_cap else "N/A"
        info['previousClose'] = float(fast.previous_close) if fast.previous_close else "N/A"
        info['longName'] = ticker
        info['longBusinessSummary'] = "Business summary is temporarily unavailable due to exchange rate limits. Quantitative algorithms remain fully functional."
        
        # Attempt full scrape safely
        full_info = t.info
        if 'longBusinessSummary' in full_info:
            info['longBusinessSummary'] = full_info['longBusinessSummary']
        if 'longName' in full_info:
            info['longName'] = full_info['longName']
        info['trailingPE'] = full_info.get('trailingPE', 'N/A')
        info['dividendYield'] = full_info.get('dividendYield', 'N/A')
    except:
        pass
        
    return df, info

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Terminal Setup")
    # Empty default ticker as requested
    ticker_symbol = st.text_input("Global Asset Ticker (e.g., RELIANCE.NS, AAPL):", value="", placeholder="Enter ticker symbol...").upper()
    
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

# ==========================================
# LANDING PAGE (GUIDELINES & DISCLAIMER)
# ==========================================
if not run_analysis or not ticker_symbol:
    st.markdown("<h1 style='text-align: center; font-size: 4em; color: #ffffff;'>ALPHA<span style='color: #38bdf8;'>QUANT</span> <span style='font-size:0.4em; color:#94a3b8;'>EXECUTIVE</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2em; letter-spacing: 2px;'>GLOBAL INSTITUTIONAL FORECASTING TERMINAL</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Terminal Operation Guidelines
        This platform runs concurrent analytical engines to evaluate market conditions based on a strict 5-year historical data footprint.
        
        1. **Target Selection:** Enter a valid ticker in the sidebar. **Use exchange suffixes for international stocks** (e.g., `.NS` for India's NSE, `.L` for London). US Stocks require no suffix.
        2. **Algorithmic Selection:** * **Meta Prophet:** Select this for AI-driven, curved momentum forecasting based on weekly and yearly seasonal patterns.
           * **Manual ARIMA:** Select this to manually input `p, d, q` parameters for strict, deterministic linear modeling.
        3. **Execution:** Click 'Execute Briefing' to generate the executive reports and automated chart analyses.
        """)
        
    with col2:
        st.markdown("""
        ### ⚠️ Institutional Disclaimer
        <div style="background-color: rgba(190, 18, 60, 0.1); border: 1px solid #be123c; padding: 15px; border-radius: 8px; font-size: 13px; color: #cbd5e1;">
        <b>NOT FINANCIAL ADVICE.</b><br><br>
        Models do not account for macroeconomic variables, fundamental earnings shocks, or geopolitical Black Swan events. <br><br>
        Capital is at risk. Use exclusively for quantitative research and educational technical analysis.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
elif run_analysis and ticker_symbol:
    with st.spinner(f"📡 Establishing secure uplink for {ticker_symbol}... Fetching global data matrices..."):
        try:
            df, info = fetch_data(ticker_symbol)
            if df.empty:
                st.error(f"Critical Failure: No market data found for '{ticker_symbol}'. Verify the ticker and exchange suffix.")
                st.stop()
            
            sym, curr_code = get_currency_config(ticker_symbol)
            
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
            
            sma_20 = df_close.rolling(20).mean()
            std_20 = df_close.rolling(20).std()
            upper_bb = sma_20 + (std_20 * 2)
            lower_bb = sma_20 - (std_20 * 2)
            
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

            if score >= 2: master_class, master_text = "v-buy", "MASTER VERDICT: STRONGLY BULLISH"
            elif score <= -2: master_class, master_text = "v-sell", "MASTER VERDICT: STRONGLY BEARISH"
            else: master_class, master_text = "v-hold", "MASTER VERDICT: NEUTRAL"

            # --- UI TABS ---
            tab_summary, tab_tech, tab_fore, tab_data = st.tabs([
                "📑 Executive Summary", 
                "📊 Technical Dashboard", 
                "🔮 Predictive AI", 
                "💾 Raw Data"
            ])

            # ==========================================
            # TAB 1: EXECUTIVE SUMMARY
            # ==========================================
            with tab_summary:
                st.markdown(f'<div class="verdict-box {master_class}">{master_text}</div>', unsafe_allow_html=True)
                
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
                
                st.markdown(f"### 📖 About {info.get('longName', ticker_symbol)}")
                st.markdown(f"<div style='font-size: 16px; color: #cbd5e1; line-height: 1.6;'>{info.get('longBusinessSummary', 'Business summary not available.')}</div>", unsafe_allow_html=True)

            # ==========================================
            # TAB 2: TECHNICAL DASHBOARD & TEXTUAL ANALYSIS
            # ==========================================
            with tab_tech:
                st.header("Deep Technical Analysis")
                
                # --- CHART 1: PRICE & BASELINES ---
                st.markdown("### 1. Macro Trend & Volatility Bounds")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#f1f5f9")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#f59e0b", dash="dot")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#0ea5e9")))
                fig1.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, line=dict(color='rgba(14, 165, 233, 0.2)'), name='Upper BB', showlegend=False))
                fig1.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, line=dict(color='rgba(14, 165, 233, 0.2)'), fill='tonexty', fillcolor='rgba(14, 165, 233, 0.05)', name='Bollinger Bands'))
                fig1.update_layout(template="plotly_dark", height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig1, use_container_width=True)
                
                trend_status = "Bullish Uptrend 🟢" if current_price > sma_200.iloc[-1] else "Bearish Downtrend 🔴"
                bb_status = "near the Upper Band (indicating overextension or high expense)" if current_price >= upper_bb.iloc[-1] * 0.98 else "near the Lower Band (indicating a deep discount or oversold conditions)" if current_price <= lower_bb.iloc[-1] * 1.02 else "within the central band (indicating stable volatility)"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> Tracks raw price against moving averages (long-term trends) and Bollinger Bands (volatility limits).<br><br>
                <b>🤖 Automated Analysis:</b> The macro trend is currently in a <b>{trend_status}</b> because the price ({sym}{current_price:,.2f}) is {"above" if current_price > sma_200.iloc[-1] else "below"} the 200-Day Macro Baseline. Additionally, the asset is trading <b>{bb_status}</b>.
                </div>
                """, unsafe_allow_html=True)

                # --- CHART 2: RSI ---
                st.markdown("### 2. Relative Strength Index (Momentum)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a855f7")))
                fig2.add_hline(y=70, line_dash="dot", line_color="#ef4444")
                fig2.add_hline(y=30, line_dash="dot", line_color="#22c55e")
                fig2.update_layout(template="plotly_dark", height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)
                
                rsi_status = "Oversold 🟢 (High bounce probability)" if rsi.iloc[-1] < 30 else "Overbought 🔴 (High exhaustion risk)" if rsi.iloc[-1] > 70 else "Neutral ⚪ (Stable momentum)"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> A momentum oscillator that measures the speed and change of price movements on a scale of 0 to 100.<br><br>
                <b>🤖 Automated Analysis:</b> The RSI is currently reading <b>{rsi.iloc[-1]:.1f}</b>. This indicates the asset is <b>{rsi_status}</b>.
                </div>
                """, unsafe_allow_html=True)

                # --- CHART 3: MACD ---
                st.markdown("### 3. Moving Average Convergence Divergence (Direction)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#3b82f6")))
                fig3.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#f43f5e")))
                fig3.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Hist", marker_color=np.where(macd_hist>0, '#22c55e', '#ef4444')))
                fig3.update_layout(template="plotly_dark", height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig3, use_container_width=True)
                
                macd_status = "Bullish Cross 🟢" if macd.iloc[-1] > sig_line.iloc[-1] else "Bearish Cross 🔴"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> Displays trend direction and strength. A crossover of the lines generates buy/sell signals.<br><br>
                <b>🤖 Automated Analysis:</b> The MACD indicates a <b>{macd_status}</b> because the MACD line is {"above" if macd.iloc[-1] > sig_line.iloc[-1] else "below"} the Signal line.
                </div>
                """, unsafe_allow_html=True)

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
                        
                        engine_desc = "Meta's Prophet AI algorithm. It breaks down historical data to find hidden weekly and yearly seasonal patterns, producing a realistic, curved momentum forecast."
                        
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
                        
                        engine_desc = f"A strict, user-defined ARIMA ({p_val}, {d_val}, {q_val}) statistical regression. ARIMA models inherently calculate the safest statistical mean, resulting in a linear/flat mathematical trajectory."

                    # Forecast Verdict
                    target_price = forecast_actual.iloc[-1] if hasattr(forecast_actual, 'iloc') else forecast_actual[-1]
                    roi = ((target_price - current_price) / current_price) * 100
                    
                    if roi > 5: fore_v, fore_c = f"BULLISH (+{roi:.2f}%)", "v-buy"
                    elif roi < -5: fore_v, fore_c = f"BEARISH ({roi:.2f}%)", "v-sell"
                    else: fore_v, fore_c = f"NEUTRAL ({roi:.2f}%)", "v-hold"

                    st.markdown(f'<div class="verdict-box {fore_c}">FORECAST EXPECTATION: {fore_v}</div>', unsafe_allow_html=True)

                    # Plot Forecast
                    fig_fore = go.Figure()
                    fig_fore.add_trace(go.Scatter(x=df_close.index[-500:], y=df_close.values[-500:], name='Historical Data', line=dict(color='#94a3b8')))
                    fig_fore.add_trace(go.Scatter(x=future_dates, y=forecast_actual, name='Predicted Trend', line=dict(color='#38bdf8', width=3, dash='dash')))
                    fig_fore.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_bound), pd.Series(lower_bound[::-1])]),
                        fill='toself', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Boundary'
                    ))
                    fig_fore.update_layout(template="plotly_dark", height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                    st.plotly_chart(fig_fore, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="analysis-box">
                    <b>📌 Engine Definition:</b> You are analyzing this asset using {engine_desc}<br><br>
                    <b>🤖 Automated Analysis:</b> From the current price of {sym}{current_price:,.2f}, the math projects a median future value of <b>{sym}{target_price:,.2f}</b> in {horizon_years} years. The shaded blue area represents the mathematical volatility boundary.
                    </div>
                    """, unsafe_allow_html=True)
# ==========================================
            # TAB 4: STOCHASTIC MONTE CARLO
            # ==========================================
            with tab_mc:
                with st.spinner("🧠 Booting Stochastic Engine..."):
                    trading_days = int(horizon_years * 252)
                    daily_returns = df_close.pct_change().dropna()
                    mu = daily_returns.mean()
                    sigma = daily_returns.std()
                    
                    # --- THE FIX: CALCULATE DYNAMIC SUPPORT FLOOR ---
                    # Find the lowest price in the last 5 years
                    historical_low = df_close.min()
                    # Set a hard floor (e.g., 20% below the 5-year low) to prevent zero-crashes
                    support_floor = historical_low * 0.80 
                    
                    sim_array = np.zeros((trading_days, auto_sims), dtype=np.float32)
                    sim_array[0] = current_price
                    
                    for t in range(1, trading_days):
                        shock = np.random.normal(0, 1, auto_sims)
                        
                        # Calculate the next step
                        next_step = sim_array[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * shock)
                        
                        # THE FIX: Apply the Support Floor boundary
                        # If the math tries to push the price below the floor, force it to bounce
                        sim_array[t] = np.maximum(next_step, support_floor)
                        
                    future_dates_mc = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                    median_path = np.median(sim_array, axis=1)
                    upper_95 = np.percentile(sim_array, 95, axis=1)
                    lower_05 = np.percentile(sim_array, 5, axis=1)
            # ==========================================
            # TAB 5: RAW DATA EXPORT
            # ==========================================
            with tab_data:
                st.header("💾 Algorithmic Data Matrix")
                st.markdown(f"Export the generated {horizon_years}-year timeline for integration into external Excel/financial models.")
                
                # Align formats for both engines
                if engine_choice == "Meta Prophet (AI Momentum Curve)":
                    last_date = prophet_df['ds'].max()
                    future_only = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                else:
                    future_only = pd.DataFrame({'ds': future_dates, 'yhat': forecast_actual.values, 'yhat_lower': lower_bound, 'yhat_upper': upper_bound})
                
                future_only.columns = ['Date', f'Target ({sym})', f'Worst Case ({sym})', f'Best Case ({sym})']
                future_only['Date'] = pd.to_datetime(future_only['Date']).dt.date
                future_only.set_index('Date', inplace=True)
                future_rounded = future_only.round(2)
                
                st.dataframe(future_rounded, use_container_width=True)
                csv = future_rounded.to_csv().encode('utf-8')
                st.download_button("📥 Download Projection as CSV", data=csv, file_name=f'{ticker_symbol}_AI_Forecast.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Execution Error: {e}")
