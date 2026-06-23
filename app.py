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
        <b>NOT FINANCIAL
