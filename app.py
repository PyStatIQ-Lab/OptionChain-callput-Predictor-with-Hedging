import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Portfolio Hedge Calculator",
    page_icon="🛡️",
    layout="wide"
)

# Constants
NIFTY_LOT_SIZE = 75

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Beta values for stocks (example values - should be fetched from API in production)
BETA_VALUES = {
    'STAR.NS': 1.2,
    'ORCHPHARMA.NS': 0.8,
    'APARINDS.NS': 1.1,
    'NEWGEN.NS': 1.3,
    'GENESYS.NS': 1.0,
    'PIXTRANS.NS': 0.9,
    'SHARDACROP.NS': 1.1,
    'OFSS.NS': 1.4,
    'GANECOS.NS': 0.7,
    'SALZERELEC.NS': 1.0,
    'ADFFOODS.NS': 0.6,
    'PGIL.NS': 0.9,
    'DSSL.NS': 1.2,
    'SANSERA.NS': 1.1,
    'INDOTECH.NS': 0.8,
    'AZAD.NS': 1.0,
    'UNOMINDA.NS': 0.9,
    'POLICYBZR.NS': 1.3,
    'DEEPINDS.NS': 0.7,
    'MAXHEALTH.NS': 0.8,
    'ONESOURCE.NS': 1.5
}

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .hedge-recommendation {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #e3f2fd;
        border-left: 5px solid #1565c0;
        color: #000;
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .strike-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #40404f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fetch data from API
@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Fetch live Nifty price
@st.cache_data(ttl=60)
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        return data['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code} - {response.text}")
        return None

# Process options data
def process_options_data(raw_data, spot_price):
    if not raw_data or 'data' not in raw_data:
        return None
    
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed_data = []
    
    for strike, data in strike_map.items():
        call_data = data.get('callOptionData', {})
        put_data = data.get('putOptionData', {})
        
        call_market = call_data.get('marketData', {})
        put_market = put_data.get('marketData', {})
        
        call_analytics = call_data.get('analytics', {})
        put_analytics = put_data.get('analytics', {})
        
        strike_float = float(strike)
        
        processed_data.append({
            'strike': strike_float,
            'pcr': data.get('pcr', 0),
            'call_moneyness': 'ITM' if strike_float < spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            'put_moneyness': 'ITM' if strike_float > spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            'call_ltp': call_market.get('ltp', 0),
            'call_bid': call_market.get('bidPrice', 0),
            'call_ask': call_market.get('askPrice', 0),
            'call_volume': call_market.get('volume', 0),
            'call_oi': call_market.get('oi', 0),
            'call_oi_change': call_market.get('oi', 0) - call_market.get('prevOi', 0),
            'call_iv': call_analytics.get('iv', 0),
            'call_delta': call_analytics.get('delta', 0),
            'put_ltp': put_market.get('ltp', 0),
            'put_bid': put_market.get('bidPrice', 0),
            'put_ask': put_market.get('askPrice', 0),
            'put_volume': put_market.get('volume', 0),
            'put_oi': put_market.get('oi', 0),
            'put_oi_change': put_market.get('oi', 0) - put_market.get('prevOi', 0),
            'put_iv': put_analytics.get('iv', 0),
            'put_delta': put_analytics.get('delta', 0),
        })
    
    return pd.DataFrame(processed_data)

def calculate_portfolio_metrics(portfolio_df):
    """Calculate portfolio metrics and hedge requirements"""
    portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
    total_value = portfolio_df['Current Value'].sum()
    
    portfolio_df['Beta'] = portfolio_df['Symbol'].map(BETA_VALUES).fillna(1.0)
    portfolio_df['Beta Exposure'] = portfolio_df['Current Value'] * portfolio_df['Beta']
    total_beta_exposure = portfolio_df['Beta Exposure'].sum()
    
    return portfolio_df, total_value, total_beta_exposure

def calculate_hedge(total_beta_exposure, nifty_price):
    """Calculate hedge requirements"""
    nifty_lot_value = nifty_price * NIFTY_LOT_SIZE
    hedge_lots = -total_beta_exposure / nifty_lot_value
    return round(hedge_lots), nifty_lot_value

def get_recommended_put(options_df, nifty_price):
    """Recommend optimal put option for hedging"""
    # Filter OTM puts (strike < current price)
    puts = options_df[options_df['strike'] < nifty_price].copy()
    
    if len(puts) == 0:
        return None
    
    # Calculate distance from current price (%)
    puts['distance_pct'] = (nifty_price - puts['strike']) / nifty_price * 100
    
    # Filter puts with 3-5% OTM
    target_puts = puts[(puts['distance_pct'] >= 3) & (puts['distance_pct'] <= 5)]
    
    if len(target_puts) > 0:
        # Select put with highest open interest
        recommended_put = target_puts.nlargest(1, 'put_oi').iloc[0]
    else:
        # Fallback to nearest OTM put
        recommended_put = puts.nlargest(1, 'strike').iloc[0]
    
    return recommended_put

def main():
    st.markdown("<div class='header'><h1>🛡️ Portfolio Hedge Calculator</h1></div>", unsafe_allow_html=True)
    
    # Sample portfolio data
    portfolio_data = {
        'Symbol': ['STAR.NS', 'ORCHPHARMA.NS', 'APARINDS.NS', 'NEWGEN.NS', 'GENESYS.NS', 
                  'PIXTRANS.NS', 'SHARDACROP.NS', 'OFSS.NS', 'GANECOS.NS', 'SALZERELEC.NS',
                  'ADFFOODS.NS', 'PGIL.NS', 'DSSL.NS', 'SANSERA.NS', 'INDOTECH.NS',
                  'AZAD.NS', 'UNOMINDA.NS', 'POLICYBZR.NS', 'DEEPINDS.NS', 'MAXHEALTH.NS',
                  'ONESOURCE.NS'],
        'Quantity': [30, 30, 3, 21, 35, 14, 41, 4, 21, 24, 120, 22, 57, 35, 14, 28, 33, 26, 90, 29, 15],
        'Avg. Cost Price': [1397.1, 1680.92, 11145, 1663.65, 991.75, 2454.89, 829.5, 12551.95, 2345.14, 1557.34,
                          338.95, 1571.69, 1488.39, 1588.43, 3102.76, 1775.26, 1069.88, 1925.29, 555.84, 1208.4, 333.05],
        'Current Price': [575.8, 720.35, 4974.35, 842.4, 550.05, 1402.45, 480.7, 7294.15, 1469.6, 980.55,
                        215.22, 1001.4, 961.15, 1051.8, 2083.2, 1218.25, 802.85, 1445.65, 428.3, 1075.5, 1376.4]
    }
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # Fetch market data
    with st.spinner("Fetching market data..."):
        nifty_price = fetch_nifty_price()
        if nifty_price is None:
            st.error("Failed to fetch NIFTY price. Using default value.")
            nifty_price = 22000  # Default fallback
        
        options_data = fetch_options_data()
        if options_data:
            options_df = process_options_data(options_data, nifty_price)
        else:
            st.error("Failed to fetch options data")
            return
    
    # Calculate portfolio metrics
    portfolio_df, total_value, total_beta_exposure = calculate_portfolio_metrics(portfolio_df)
    
    # Calculate hedge requirements
    hedge_lots, nifty_lot_value = calculate_hedge(total_beta_exposure, nifty_price)
    
    # Get recommended put option
    recommended_put = get_recommended_put(options_df, nifty_price)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Portfolio Value**")
        st.markdown(f"<h2>₹{total_value:,.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Beta Exposure**")
        st.markdown(f"<h2>₹{total_beta_exposure:,.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**NIFTY Current Price**")
        st.markdown(f"<h2>₹{nifty_price:,.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**NIFTY Lot Value**")
        st.markdown(f"<h2>₹{nifty_lot_value:,.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Hedge recommendation
    if recommended_put is not None:
        st.markdown(f"""
            <div class='hedge-recommendation'>
                <h3>Hedge Recommendation</h3>
                <p><b>Action:</b> Buy {abs(hedge_lots)} NIFTY Put Options</p>
                <p><b>Recommended Strike:</b> ₹{recommended_put['strike']:,.2f} ({(nifty_price - recommended_put['strike']) / nifty_price * 100:.1f}% OTM)</p>
                <p><b>Current Premium:</b> ₹{recommended_put['put_ltp']:,.2f}</p>
                <p><b>Open Interest:</b> {recommended_put['put_oi']:,}</p>
                <p><b>Delta:</b> {recommended_put['put_delta']:.2f}</p>
                <p><b>Total Hedge Cost:</b> ₹{abs(hedge_lots) * recommended_put['put_ltp'] * NIFTY_LOT_SIZE:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Could not find suitable put options for hedging")
    
    # Portfolio details
    with st.expander("View Portfolio Details"):
        st.dataframe(
            portfolio_df.style.format({
                'Avg. Cost Price': '{:,.2f}',
                'Current Price': '{:,.2f}',
                'Current Value': '{:,.2f}',
                'Beta': '{:.2f}',
                'Beta Exposure': '{:,.2f}'
            }),
            use_container_width=True,
            height=600
        )
    
    # Options chain visualization
    st.markdown("### NIFTY Options Chain")
    
    # Filter nearby strikes
    all_strikes = sorted(options_df['strike'].unique())
    current_idx = all_strikes.index(min(all_strikes, key=lambda x: abs(x - nifty_price)))
    nearby_strikes = all_strikes[max(0, current_idx-5):min(len(all_strikes), current_idx+6)]
    nearby_df = options_df[options_df['strike'].isin(nearby_strikes)]
    
    # Plot OI and volume
    fig = px.bar(
        nearby_df,
        x='strike',
        y=['put_oi', 'call_oi'],
        barmode='group',
        title='Open Interest',
        labels={'value': 'OI', 'strike': 'Strike Price'},
        color_discrete_map={'put_oi': '#e74c3c', 'call_oi': '#3498db'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(
        nearby_df,
        x='strike',
        y=['put_volume', 'call_volume'],
        barmode='group',
        title='Volume',
        labels={'value': 'Volume', 'strike': 'Strike Price'},
        color_discrete_map={'put_volume': '#e74c3c', 'call_volume': '#3498db'}
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
