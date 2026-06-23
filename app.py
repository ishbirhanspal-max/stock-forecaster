import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
import gc # Garbage Collector to clear RAM

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="AlphaQuant Pro | Memory Optimized", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: 700; color: #58a6ff; font-family: 'Courier New', monospace; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .risk-high { color: #f85149; } .risk-low { color: #3fb950; }
    .verdict-box { padding: 15px; border-radius: 6px; margin-top: 15px; text-align: center; font-size: 18px; font-weight: bold; border: 1px solid #30363d; }
    .v-buy { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; border-color: rgba(46, 160, 67, 0.4); }
    .v-sell { background-color: rgba(248, 81, 73, 0.15); color: #f85149; border-color: rgba(248, 81, 73, 0.4); }
    .v-hold { background-color: rgba(210, 153, 34, 0.15); color: #d29922; border-color: rgba(210, 153, 34, 0.4); }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AlphaQuant Pro: Institutional Quant Terminal")
st.markdown("Optimized for Streamlit Cloud deployment. Utilizing memory-safe stochastic modeling.")
st.markdown("---")

# --- DATA FETCHING (Memory Caching) ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker, benchmark="^NSEI"):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    bench_df = yf.download(benchmark, start=start, end=end, progress=False)
    
    return df, bench_df

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Quant Parameters")
    ticker_symbol = st.text_input("Target Asset Ticker (e.g., RELIANCE.NS):", value="RELIANCE.NS").upper()
    benchmark_symbol = st.text_input("Market Benchmark (e.g., ^NSEI):", value="^NSEI").upper()
    
    st.markdown("### 🎲 Stochastic Parameters")
    # Reduced max limits to prevent Streamlit 503 Out-of-Memory Errors
    horizon_years = st.slider("Projection Horizon (Years):", 0.5, 3.0, 1.0, 0.5)
    num_simulations = st.slider("Paths (Realities):", 50, 1000, 250, 50) 
    
    run_terminal = st.button("🚀 Initialize Quant Engine", type="primary", use_container_width=True)

# --- Terminal Execution ---
if run_terminal:
    if ticker_symbol:
        with st.spinner("Initializing quantitative risk engine..."):
            try:
                df, bench_df = fetch_market_data(ticker_symbol, benchmark_symbol)
                
                if df.empty:
                    st.error("Asset data unavailable. Verify ticker.")
                    st.stop()
                    
                # Clean MultiIndex and extract arrays cleanly to save RAM
                if isinstance(df.columns, pd.MultiIndex):
                    df_close = df['Close'][ticker_symbol].dropna()
                    bench_close = bench_df['Close'][benchmark_symbol].dropna() if not bench_df.empty else pd.Series()
                else:
                    df_close = df['Close'].dropna()
                    bench_close = bench_df['Close'].dropna() if not bench_df.empty else pd.Series()

                current_price = df_close.iloc[-1]
                returns = df_close.pct_change().dropna()
                
                # --- RISK METRICS CALCULATION ---
                ann_volatility = returns.std() * np.sqrt(252)
                ann_return = returns.mean() * 252
                var_95 = np.percentile(returns, 5)
                cvar_95 = returns[returns <= var_95].mean()
                
                rf = 0.05
                sharpe = (ann_return - rf) / ann_volatility
                
                beta = "N/A"
                if not bench_close.empty:
                    bench_returns = bench_close.pct_change().dropna()
                    aligned_data = pd.concat([returns, bench_returns], axis=1).dropna()
                    aligned_data.columns = ['Asset', 'Market']
                    cov_matrix = np.cov(aligned_data['Asset'], aligned_data['Market'])
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

                # --- TECHNICAL VERDICT ---
                sma_50 = df_close.rolling(50).mean()
                sma_200 = df_close.rolling(200).mean()
                
                delta = df_close.diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rsi = 100 - (100 / (1 + (gain / loss)))

                score = 0
                if rsi.iloc[-1] < 35: score += 1
                elif rsi.iloc[-1] > 70: score -= 1
                if current_price > sma_200.iloc[-1]: score += 1
                else: score -= 1
                if current_price > sma_50.iloc[-1]: score += 1
                else: score -= 1

                if score >= 2: v_class, v_text = "v-buy", "ACCUMULATE (BULLISH CONFLUENCE)"
                elif score <= -2: v_class, v_text = "v-sell", "REDUCE (BEARISH CONFLUENCE)"
                else: v_class, v_text = "v-hold", "HOLD (NEUTRAL MOMENTUM)"

                tab_risk, tab_jump, tab_tech = st.tabs(["🛡️ Risk Desk", "🎲 Jump-Diffusion", "📊 Technicals"])

                # ==========================================
                # TAB 1: RISK DESK
                # ==========================================
                with tab_risk:
                    st.markdown("### Institutional Risk Profile")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    c1.markdown(f'<div class="metric-card"><div class="metric-label">Value at Risk (95%)</div><div class="metric-value risk-high">{var_95*100:.2f}%</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="metric-card"><div class="metric-label">Expected Shortfall</div><div class="metric-value risk-high">{cvar_95*100:.2f}%</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="metric-card"><div class="metric-label">Market Beta</div><div class="metric-value">{beta if isinstance(beta, str) else round(beta, 2)}</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=returns, nbinsx=100, name="Daily Returns", marker_color='#3fb950', opacity=0.7))
                    fig_dist.add_vline(x=var_95, line_dash="dot", line_color="#f85149", annotation_text="95% VaR")
                    fig_dist.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)

                # ==========================================
                # TAB 2: MERTON JUMP DIFFUSION
                # ==========================================
                with tab_jump:
                    st.markdown("### Memory-Safe Stochastic Modeling")
                    
                    # --- MEMORY-OPTIMIZED MATH ENGINE ---
                    trading_days = int(horizon_years * 252)
                    dt = 1/252
                    mu = returns.mean() * 252
                    sigma = returns.std() * np.sqrt(252)
                    
                    jumps = returns[abs(returns) > (3 * returns.std())]
                    lambda_jump = len(jumps) / 5 
                    mu_j = jumps.mean() if len(jumps) > 0 else 0
                    sigma_j = jumps.std() if len(jumps) > 1 else 0.01
                    
                    # Pre-allocate numpy array for speed and memory efficiency
                    simulations = np.zeros((trading_days, num_simulations), dtype=np.float32) # Using float32 halves the memory requirement
                    simulations[0] = current_price
                    
                    for t in range(1, trading_days):
                        z1 = np.random.normal(0, 1, num_simulations)
                        poisson_rv = np.random.poisson(lambda_jump * dt, num_simulations)
                        jump_size = np.random.normal(mu_j, sigma_j, num_simulations) * poisson_rv
                        
                        drift = (mu - 0.5 * sigma**2) * dt
                        shock = sigma * np.sqrt(dt) * z1
                        simulations[t] = simulations[t-1] * np.exp(drift + shock + jump_size)
                    
                    future_dates = pd.bdate_range(start=df_close.index[-1] + pd.Timedelta(days=1), periods=trading_days)
                    
                    # Calculate percentiles DIRECTLY via NumPy to avoid heavy Pandas DataFrame creation
                    median_path = np.median(simulations, axis=1)
                    upper_95 = np.percentile(simulations, 95, axis=1)
                    lower_05 = np.percentile(simulations, 5, axis=1)
                    
                    # Force garbage collection to free up Streamlit RAM before plotting
                    gc.collect() 
                    
                    fig_mc = go.Figure()
                    
                    # Historical Data
                    fig_mc.add_trace(go.Scatter(x=df_close.index[-252:], y=df_close.values[-252:], name="Past 1Y Actuals", line=dict(color="#8b949e", width=2)))
                    
                    # Only plot 3 sample paths to save browser memory
                    for i in range(min(3, num_simulations)):
                        fig_mc.add_trace(go.Scatter(x=future_dates, y=simulations[:, i], mode='lines', line=dict(width=1, opacity=0.15), showlegend=False))
                    
                    fig_mc.add_trace(go.Scatter(x=future_dates, y=median_path, name="Statistical Median", line=dict(color="#58a6ff", width=2, dash='dot')))
                    
                    fig_mc.add_trace(go.Scatter(
                        x=pd.concat([pd.Series(future_dates), pd.Series(future_dates[::-1])]),
                        y=pd.concat([pd.Series(upper_95), pd.Series(lower_05[::-1])]),
                        fill='toself', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                        name='90% Probability Matrix'
                    ))
                    
                    fig_mc.update_layout(template="plotly_dark", height=600, hovermode="x unified")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Clear simulation array from RAM entirely
                    del simulations 
                    gc.collect()

                # ==========================================
                # TAB 3: TECHNICALS
                # ==========================================
                with tab_tech:
                    st.markdown(f'<div class="verdict-box {v_class}">{v_text}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=("Price Action", "RSI Momentum"))
                    
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=df_close.values, name="Close", line=dict(color="#c9d1d9")), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_50, name="50-Day Baseline", line=dict(color="#d29922", width=1.5, dash='dot')), row=1, col=1)
                    fig_tech.add_trace(go.Scatter(x=df_close.index, y=sma_200, name="200-Day Macro", line=dict(color="#58a6ff", width=2)), row=1, col=1)
                    
                    fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#a371f7")), row=2, col=1)
                    fig_tech.add_hline(y=70, line_dash="dot", line_color="#f85149", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=2, col=1)

                    fig_tech.update_layout(template="plotly_dark", height=800, margin=dict(t=40, b=40))
                    st.plotly_chart(fig_tech, use_container_width=True)

            except Exception as e:
                st.error(f"Quant Engine Error: {e}")
