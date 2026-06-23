import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import requests
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# --- PAGE SETUP & BLACK/BLUE THEME ---
st.set_page_config(page_title="AlphaQuant | Executive Terminal", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Deep Black & Neon Blue Theme */
    .stApp { background-color: #000000; color: #d0e0ff; font-family: 'Inter', sans-serif; }
    
    .metric-card { 
        background: #030814; 
        border: 1px solid #003380; 
        border-radius: 12px; 
        padding: 15px 10px; 
        text-align: center; 
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.15);
        margin-bottom: 15px;
    }
    .metric-value { font-size: 22px; font-weight: 800; color: #00aaff; font-family: 'Courier New', monospace; margin-top: 5px; }
    .metric-label { font-size: 11px; color: #7090c0; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    
    .analysis-box { background-color: #020610; border-left: 4px solid #00aaff; padding: 20px; border-radius: 6px; font-size: 15px; line-height: 1.6; color: #a0c0ff; border-top: 1px solid #001a4d; border-right: 1px solid #001a4d; border-bottom: 1px solid #001a4d; margin-top: 15px; margin-bottom: 25px; }
    
    .company-title { font-size: 3.8em; font-weight: 900; text-align: center; text-transform: uppercase; background: -webkit-linear-gradient(#ffffff, #00aaff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2; letter-spacing: -1px;}
    .company-ticker { font-size: 1.2em; font-weight: 600; text-align: center; color: #0066ff; letter-spacing: 4px; margin-top: 0px; padding-top: 5px; margin-bottom: 30px;}
    
    /* VIBRANT NEON VERDICT BOXES */
    .verdict-box { padding: 20px; border-radius: 12px; margin-top: 10px; margin-bottom: 25px; text-align: center; font-size: 22px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.8); }
    .v-buy { background-color: rgba(0, 255, 136, 0.05); border: 2px solid #00ff88; color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
    .v-sell { background-color: rgba(255, 51, 102, 0.05); border: 2px solid #ff3366; color: #ff3366; text-shadow: 0 0 10px rgba(255, 51, 102, 0.5); }
    .v-hold { background-color: rgba(0, 170, 255, 0.05); border: 2px solid #00aaff; color: #00aaff; text-shadow: 0 0 10px rgba(0, 170, 255, 0.5); }
    
    hr { border-color: #001a4d; }
    h1, h2, h3 { color: #ffffff; }
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

# --- DATA FETCHING (With Stealth Backend Bypass - NO WIKIPEDIA) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    
    session = requests.Session()
    # Stealth User-Agent to bypass standard bot blocks
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    
    df = yf.download(ticker, start=start, end=end, progress=False, session=session, auto_adjust=True)
    
    # Initialize clean default dictionary
    info = {
        'longName': ticker,
        'longBusinessSummary': "Corporate briefing currently unavailable due to exchange rate limits. Quantitative matrices remain operational.",
        'marketCap': 'N/A', 'trailingPE': 'N/A', 'dividendYield': 'N/A',
        'beta': 'N/A', 'fiftyTwoWeekHigh': 'N/A', 'fiftyTwoWeekLow': 'N/A', 'averageVolume': 'N/A'
    }
    
    try:
        t = yf.Ticker(ticker, session=session)
        fast = t.fast_info
        info['marketCap'] = float(fast.market_cap) if fast.market_cap else "N/A"
        info['fiftyTwoWeekHigh'] = float(fast.year_high) if fast.year_high else "N/A"
        info['fiftyTwoWeekLow'] = float(fast.year_low) if fast.year_low else "N/A"
    except:
        pass

    # Attempt 1: Standard yfinance extraction
    try:
        full_info = t.info
        info['longName'] = full_info.get('longName', info['longName'])
        if 'longBusinessSummary' in full_info:
            info['longBusinessSummary'] = full_info['longBusinessSummary']
        info['trailingPE'] = full_info.get('trailingPE', info['trailingPE'])
        info['dividendYield'] = full_info.get('dividendYield', info['dividendYield'])
        info['beta'] = full_info.get('beta', info['beta'])
        info['averageVolume'] = full_info.get('averageVolume', info['averageVolume'])
    except Exception:
        pass
        
    # THE FIX: Direct Yahoo Query2 Backend API Bypass (100% Accurate)
    # If the standard method got blocked, we slip past the firewall and ask for ONLY the text profile
    if "unavailable" in info['longBusinessSummary'] or len(info['longBusinessSummary']) < 50:
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile,price"
            res = session.get(url, timeout=5).json()
            
            result = res.get('quoteSummary', {}).get('result', [])
            if result and result[0]:
                profile = result[0].get('assetProfile', {})
                price_data = result[0].get('price', {})
                
                if profile.get('longBusinessSummary'):
                    info['longBusinessSummary'] = profile['longBusinessSummary']
                if price_data.get('longName'):
                    info['longName'] = price_data['longName']
        except Exception:
            pass

    return df, info

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Terminal Setup")
    ticker_symbol = st.text_input("Global Asset Ticker (e.g., RELIANCE.NS, AAPL):", value="", placeholder="Enter ticker symbol...").upper()
    
    st.markdown("### 🔮 Predictive Engine")
    horizon_years = st.slider("Forecast Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    
    engine_choice = st.radio("Select Main Algorithm:", [
        "Meta Prophet (AI Momentum)", 
        "Manual ARIMA (Statistical)"
    ])
    
    if engine_choice == "Manual ARIMA (Statistical)":
        st.markdown("**ARIMA Parameters:**")
        c1, c2, c3 = st.columns(3)
        with c1: p_val = st.number_input("p (Lag)", 0, 10, 5)
        with c2: d_val = st.number_input("d (Diff)", 0, 3, 1)
        with c3: q_val = st.number_input("q (MA)", 0, 10, 0)
        
    run_analysis = st.button("🚀 Execute Briefing", type="primary", use_container_width=True)

# ==========================================
# LANDING PAGE
# ==========================================
if not run_analysis or not ticker_symbol:
    st.markdown("<h1 style='text-align: center; font-size: 5em; color: #ffffff;'>ALPHA<span style='color: #00aaff;'>QUANT</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7090c0; font-size: 1.2em; letter-spacing: 2px;'>INSTITUTIONAL ANALYSIS & FORECASTING DESK</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📖 Terminal Operation Guidelines
        This platform runs concurrent analytical engines to evaluate market conditions based on a strict 5-year historical footprint.
        
        1. **Target Selection:** Enter a valid ticker in the sidebar. **Use exchange suffixes for international stocks** (e.g., `.NS` for India's NSE). US Stocks require no suffix.
        2. **Algorithmic Integration:** The Forecasting tab unifies your chosen Deterministic path (Prophet or ARIMA) alongside the Probabilistic Monte Carlo simulation in a single view.
        3. **Execution:** Click 'Execute Briefing' to generate the executive reports.
        """)
        
    with col2:
        st.markdown("""
        ### ⚠️ Institutional Disclaimer
        <div style="background-color: rgba(0, 85, 255, 0.1); border: 1px solid #0055ff; padding: 15px; border-radius: 8px; font-size: 13px; color: #a0c0ff;">
        <b>NOT FINANCIAL ADVICE.</b><br><br>
        Models mathematically establish support floors to prevent false-zero crashes, but do not account for external Black Swan events. <br><br>
        Capital is at risk. Use exclusively for quantitative research and technical analysis.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
elif run_analysis and ticker_symbol:
    with st.spinner(f"📡 Establishing secure uplink for {ticker_symbol}... Compiling 5-Year Data Matrices..."):
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
            
            support_floor = df_close.min() * 0.70
            
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
            else: master_class, master_text = "v-hold", "MASTER VERDICT: NEUTRAL / RANGEBOUND"

            # --- BIG COMPANY NAME DISPLAY ---
            st.markdown(f"<div class='company-title'>{info.get('longName', ticker_symbol)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='company-ticker'>| {ticker_symbol} |</div>", unsafe_allow_html=True)

            # --- UNIFIED TABS ---
            tab_summary, tab_tech, tab_fore, tab_data = st.tabs([
                "📑 Executive Summary", 
                "📊 Technical Dashboard", 
                "🔮 Predictive Forecasting", 
                "💾 Raw Data"
            ])

            # ==========================================
            # TAB 1: EXECUTIVE SUMMARY
            # ==========================================
            with tab_summary:
                st.markdown(f'<div class="verdict-box {master_class}">{master_text}</div>', unsafe_allow_html=True)
                st.markdown("### 🏢 In-Depth Fundamental Snapshot")
                
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
                
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                beta = info.get('beta', 'N/A')
                beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
                high52 = info.get('fiftyTwoWeekHigh', 'N/A')
                high52_str = f"{sym}{high52:,.2f}" if isinstance(high52, (int, float)) else "N/A"
                low52 = info.get('fiftyTwoWeekLow', 'N/A')
                low52_str = f"{sym}{low52:,.2f}" if isinstance(low52, (int, float)) else "N/A"
                vol = info.get('averageVolume', 'N/A')
                vol_str = f"{vol / 1e6:,.2f}M" if isinstance(vol, (int, float)) else "N/A"

                col_b1.markdown(f'<div class="metric-card"><div class="metric-label">Market Beta</div><div class="metric-value">{beta_str}</div></div>', unsafe_allow_html=True)
                col_b2.markdown(f'<div class="metric-card"><div class="metric-label">52-Week High</div><div class="metric-value">{high52_str}</div></div>', unsafe_allow_html=True)
                col_b3.markdown(f'<div class="metric-card"><div class="metric-label">52-Week Low</div><div class="metric-value">{low52_str}</div></div>', unsafe_allow_html=True)
                col_b4.markdown(f'<div class="metric-card"><div class="metric-label">Average Volume</div><div class="metric-value">{vol_str}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### 📖 Corporate Briefing")
                st.markdown(f"<div class='analysis-box'>{info.get('longBusinessSummary')}</div>", unsafe_allow_html=True)

            # ==========================================
            # TAB 2: TECHNICAL DASHBOARD
            # ==========================================
            with tab_tech:
                st.header("Deep Technical Analysis")
                
                st.markdown("### 1. Macro Trend & Volatility Bounds")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df_close.index, y=df_close, name="Price", line=dict(color="#ffffff")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50 SMA", line=dict(color="#0066ff", dash="dot")))
                fig1.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200 SMA", line=dict(color="#00aaff")))
                fig1.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, line=dict(color='rgba(0, 170, 255, 0.2)'), name='Upper BB', showlegend=False))
                fig1.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, line=dict(color='rgba(0, 170, 255, 0.2)'), fill='tonexty', fillcolor='rgba(0, 170, 255, 0.05)', name='Bollinger Bands'))
                fig1.update_layout(template="plotly_dark", height=450, plot_bgcolor='#000000', paper_bgcolor='#000000')
                st.plotly_chart(fig1, use_container_width=True)
                
                trend_status = "Bullish Uptrend 🟢" if current_price > sma_200.iloc[-1] else "Bearish Downtrend 🔴"
                bb_status = "near the Upper Band (indicating overextension or high expense)" if current_price >= upper_bb.iloc[-1] * 0.98 else "near the Lower Band (indicating a deep discount or oversold conditions)" if current_price <= lower_bb.iloc[-1] * 1.02 else "within the central band (indicating stable volatility)"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> Tracks raw price against moving averages (long-term trends) and Bollinger Bands (volatility limits).<br><br>
                <b>🤖 Automated Analysis:</b> The macro trend is currently in a <b>{trend_status}</b> because the price ({sym}{current_price:,.2f}) is {"above" if current_price > sma_200.iloc[-1] else "below"} the 200-Day Macro Baseline. Additionally, the asset is trading <b>{bb_status}</b>.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 2. Relative Strength Index (Momentum)")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#00aaff")))
                fig2.add_hline(y=70, line_dash="dot", line_color="#ff3366")
                fig2.add_hline(y=30, line_dash="dot", line_color="#00ff88")
                fig2.update_layout(template="plotly_dark", height=300, plot_bgcolor='#000000', paper_bgcolor='#000000')
                st.plotly_chart(fig2, use_container_width=True)
                
                rsi_status = "Oversold 🟢 (High bounce probability)" if rsi.iloc[-1] < 30 else "Overbought 🔴 (High exhaustion risk)" if rsi.iloc[-1] > 70 else "Neutral ⚪ (Stable momentum)"
                
                st.markdown(f"""
                <div class="analysis-box">
                <b>📌 Chart Definition:</b> A momentum oscillator measuring the speed and change of price movements.<br><br>
                <b>🤖 Automated Analysis:</b> The RSI is currently reading <b>{rsi.iloc[-1]:.1f}</b>. This indicates the asset is <b>{rsi_status}</b>.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 3. Moving Average Convergence Divergence (Direction)")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#00aaff")))
                fig3.add_trace(go.Scatter(x=sig_line.index, y=sig_line, name="Signal", line=dict(color="#0066ff")))
                fig3.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name="Hist", marker_color=np.where(macd_hist>0, '#00ff88', '#ff3366')))
                fig3.update_layout(template="plotly_dark", height=300, plot_bgcolor='#000000', paper_bgcolor='#000000')
                st.plotly_chart(fig3, use_container_width=True)
                
                macd_status = "Bullish Cross 🟢" if macd.iloc[-1] > sig_line.iloc[-1] else "Bearish Cross 🔴"
