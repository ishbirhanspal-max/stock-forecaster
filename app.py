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
    .v-hold { background-color: rgba(0, 170, 255, 0.05); border: 2px solid #00aaff; color: #00aaff; text-shadow: 0 0 1
