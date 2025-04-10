import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
NIFTY_SYMBOL = "^NSEI"

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
</style>
""", unsafe_allow_html=True)

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

def get_nifty_data():
    """Fetch current NIFTY price and options data"""
    nifty = yf.Ticker(NIFTY_SYMBOL)
    
    # Get current price
    hist = nifty.history(period='1d')
    current_price = hist['Close'].iloc[-1]
    
    # Get options chain (nearest expiry)
    options = nifty.option_chain()
    puts = options.puts
    calls = options.calls
    
    return current_price, puts, calls

def calculate_portfolio_metrics(portfolio_df):
    """Calculate portfolio metrics and hedge requirements"""
    # Calculate current values
    portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
    total_value = portfolio_df['Current Value'].sum()
    
    # Assign beta values
    portfolio_df['Beta'] = portfolio_df['Symbol'].map(BETA_VALUES).fillna(1.0)
    
    # Calculate beta-weighted exposure
    portfolio_df['Beta Exposure'] = portfolio_df['Current Value'] * portfolio_df['Beta']
    total_beta_exposure = portfolio_df['Beta Exposure'].sum()
    
    return portfolio_df, total_value, total_beta_exposure

def calculate_hedge(total_beta_exposure, nifty_price):
    """Calculate hedge requirements"""
    nifty_lot_value = nifty_price * NIFTY_LOT_SIZE
    hedge_lots = -total_beta_exposure / nifty_lot_value
    return round(hedge_lots), nifty_lot_value

def get_recommended_strike(puts, nifty_price, hedge_lots):
    """Recommend optimal strike price for hedging"""
    # Filter OTM puts (strike < current price)
    otm_puts = puts[puts['strike'] < nifty_price].copy()
    
    # Calculate distance from current price (%)
    otm_puts['distance_pct'] = (nifty_price - otm_puts['strike']) / nifty_price * 100
    
    # Filter puts with 3-5% OTM
    target_puts = otm_puts[(otm_puts['distance_pct'] >= 3) & (otm_puts['distance_pct'] <= 5)]
    
    if len(target_puts) > 0:
        # Select put with highest open interest
        recommended_put = target_puts.nlargest(1, 'openInterest').iloc[0]
    else:
        # Fallback to nearest OTM put
        recommended_put = otm_puts.nlargest(1, 'strike').iloc[0]
    
    return recommended_put

def main():
    st.markdown("<div class='header'><h1>🛡️ Portfolio Hedge Calculator</h1></div>", unsafe_allow_html=True)
    
    # Sample portfolio data (would normally come from uploaded file)
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
    
    # Get NIFTY data
    with st.spinner("Fetching NIFTY data..."):
        nifty_price, puts, calls = get_nifty_data()
    
    # Calculate portfolio metrics
    portfolio_df, total_value, total_beta_exposure = calculate_portfolio_metrics(portfolio_df)
    
    # Calculate hedge requirements
    hedge_lots, nifty_lot_value = calculate_hedge(total_beta_exposure, nifty_price)
    
    # Get recommended strike
    recommended_put = get_recommended_strike(puts, nifty_price, hedge_lots)
    
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
    st.markdown(f"""
        <div class='hedge-recommendation'>
            <h3>Hedge Recommendation</h3>
            <p><b>Action:</b> Buy {abs(hedge_lots)} NIFTY Put Options</p>
            <p><b>Recommended Strike:</b> ₹{recommended_put['strike']:,.2f} ({(nifty_price - recommended_put['strike']) / nifty_price * 100:.1f}% OTM)</p>
            <p><b>Current Premium:</b> ₹{recommended_put['lastPrice']:,.2f}</p>
            <p><b>Open Interest:</b> {recommended_put['openInterest']:,}</p>
            <p><b>Total Hedge Cost:</b> ₹{abs(hedge_lots) * recommended_put['lastPrice'] * NIFTY_LOT_SIZE:,.2f}</p>
        </div>
    """, unsafe_allow_html=True)
    
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
    
    # NIFTY options chain
    with st.expander("View NIFTY Options Chain"):
        tab1, tab2 = st.tabs(["Put Options", "Call Options"])
        
        with tab1:
            st.dataframe(
                puts.style.format({
                    'strike': '{:,.2f}',
                    'lastPrice': '{:,.2f}',
                    'bid': '{:,.2f}',
                    'ask': '{:,.2f}',
                    'change': '{:,.2f}',
                    'percentChange': '{:.2f}%',
                    'openInterest': '{:,}',
                    'impliedVolatility': '{:.2f}%'
                }),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            st.dataframe(
                calls.style.format({
                    'strike': '{:,.2f}',
                    'lastPrice': '{:,.2f}',
                    'bid': '{:,.2f}',
                    'ask': '{:,.2f}',
                    'change': '{:,.2f}',
                    'percentChange': '{:.2f}%',
                    'openInterest': '{:,}',
                    'impliedVolatility': '{:.2f}%'
                }),
                use_container_width=True,
                height=400
            )

if __name__ == "__main__":
    main()
