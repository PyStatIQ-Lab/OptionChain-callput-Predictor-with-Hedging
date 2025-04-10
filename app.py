import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Constants
NIFTY_LOT_SIZE = 75

def initialize_portfolio():
    # Portfolio Data
    portfolio_data = {
        'Symbol': ['STAR.NS', 'ORCHPHARMA.NS', 'APARINDS.NS', 'NEWGEN.NS', 'GENESYS.NS', 
                   'PIXTRANS.NS', 'SHARDACROP.NS', 'OFSS.NS', 'GANECOS.NS', 'SALZERELEC.NS',
                   'ADFFOODS.NS', 'PGIL.NS', 'DSSL.NS', 'SANSERA.NS', 'INDOTECH.NS',
                   'AZAD.NS', 'UNOMINDA.NS', 'POLICYBZR.NS', 'DEEPINDS.NS', 'MAXHEALTH.NS',
                   'ONESOURCE.NS'],
        'Quantity': [30, 30, 3, 21, 35, 14, 41, 4, 21, 24, 120, 22, 57, 35, 14, 28, 33, 26, 90, 29, 15],
        'Avg. Cost Price': [1397.1, 1680.92, 11145, 1663.65, 991.75, 2454.89, 829.5, 12551.95, 
                            2345.14, 1557.34, 338.95, 1571.69, 1488.39, 1588.43, 3102.76, 
                            1775.26, 1069.88, 1925.29, 555.84, 1208.4, 333.05],
        'Current Price': [575.8, 720.35, 4974.35, 842.4, 550.05, 1402.45, 480.7, 7294.15, 
                          1469.6, 980.55, 215.22, 1001.4, 961.15, 1051.8, 2083.2, 
                          1218.25, 802.85, 1445.65, 428.3, 1075.5, 1376.4]
    }
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_data)
    portfolio_df['Investment'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
    portfolio_df['P&L'] = (portfolio_df['Current Price'] - portfolio_df['Avg. Cost Price']) * portfolio_df['Quantity']
    portfolio_df['P&L%'] = (portfolio_df['Current Price'] / portfolio_df['Avg. Cost Price'] - 1) * 100
    
    return portfolio_df

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

# Process raw API data
def process_options_data(raw_data, spot_price):
    if not raw_data or 'data' not in raw_data:
        return None
    
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed_data = []
    
    for strike, data in strike_map.items():
        call_data = data.get('callOptionData', {})
        put_data = data.get('putOptionData', {})
        
        # Market data
        call_market = call_data.get('marketData', {})
        put_market = put_data.get('marketData', {})
        
        # Analytics data
        call_analytics = call_data.get('analytics', {})
        put_analytics = put_data.get('analytics', {})
        
        strike_float = float(strike)
        
        processed_data.append({
            'strike': strike_float,
            'pcr': data.get('pcr', 0),
            
            # Moneyness
            'call_moneyness': 'ITM' if strike_float < spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            'put_moneyness': 'ITM' if strike_float > spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            
            # Call data
            'call_ltp': call_market.get('ltp', 0),
            'call_bid': call_market.get('bidPrice', 0),
            'call_ask': call_market.get('askPrice', 0),
            'call_volume': call_market.get('volume', 0),
            'call_oi': call_market.get('oi', 0),
            'call_prev_oi': call_market.get('prevOi', 0),
            'call_oi_change': call_market.get('oi', 0) - call_market.get('prevOi', 0),
            'call_iv': call_analytics.get('iv', 0),
            'call_delta': call_analytics.get('delta', 0),
            'call_gamma': call_analytics.get('gamma', 0),
            'call_theta': call_analytics.get('theta', 0),
            'call_vega': call_analytics.get('vega', 0),
            
            # Put data
            'put_ltp': put_market.get('ltp', 0),
            'put_bid': put_market.get('bidPrice', 0),
            'put_ask': put_market.get('askPrice', 0),
            'put_volume': put_market.get('volume', 0),
            'put_oi': put_market.get('oi', 0),
            'put_prev_oi': put_market.get('prevOi', 0),
            'put_oi_change': put_market.get('oi', 0) - put_market.get('prevOi', 0),
            'put_iv': put_analytics.get('iv', 0),
            'put_delta': put_analytics.get('delta', 0),
            'put_gamma': put_analytics.get('gamma', 0),
            'put_theta': put_analytics.get('theta', 0),
            'put_vega': put_analytics.get('vega', 0),
        })
    
    return pd.DataFrame(processed_data)

# Calculate portfolio beta and hedging parameters
def calculate_portfolio_beta(portfolio_df):
    # Get current date for beta calculation
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    
    # Download Nifty data
    nifty = yf.download('^NSEI', start=start_date, end=end_date)['Adj Close']
    nifty_returns = nifty.pct_change().dropna()
    
    betas = []
    correlations = []
    volatilities = []
    
    for symbol in portfolio_df['Symbol']:
        try:
            # Download stock data
            stock = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
            stock_returns = stock.pct_change().dropna()
            
            # Align dates
            common_dates = nifty_returns.index.intersection(stock_returns.index)
            nifty_r = nifty_returns[common_dates]
            stock_r = stock_returns[common_dates]
            
            # Calculate beta
            cov_matrix = np.cov(stock_r, nifty_r)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            # Calculate correlation
            correlation = np.corrcoef(stock_r, nifty_r)[0, 1]
            
            # Calculate volatility (annualized)
            volatility = stock_r.std() * np.sqrt(252)
            
            betas.append(beta)
            correlations.append(correlation)
            volatilities.append(volatility)
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            betas.append(np.nan)
            correlations.append(np.nan)
            volatilities.append(np.nan)
    
    portfolio_df['Beta'] = betas
    portfolio_df['Correlation_with_Nifty'] = correlations
    portfolio_df['Volatility'] = volatilities
    
    # Calculate weighted average beta for the portfolio
    total_investment = portfolio_df['Investment'].sum()
    portfolio_df['Weight'] = portfolio_df['Investment'] / total_investment
    portfolio_beta = (portfolio_df['Beta'] * portfolio_df['Weight']).sum()
    
    return portfolio_df, portfolio_beta

# Calculate hedging requirements
def calculate_hedging(portfolio_df, portfolio_beta, nifty_spot):
    total_investment = portfolio_df['Investment'].sum()
    
    # Calculate portfolio delta (sensitivity to Nifty)
    portfolio_delta = portfolio_beta * total_investment / nifty_spot
    
    # Calculate number of lots needed for hedging (using put delta of -0.5 for ATM puts)
    put_delta = -0.5  # Typical ATM put delta
    lots_needed = abs(portfolio_delta / (put_delta * NIFTY_LOT_SIZE))
    
    return total_investment, portfolio_delta, lots_needed

# Get recommended strikes
def get_recommended_strikes(options_df, nifty_spot, lots_needed):
    # Filter puts only
    puts_df = options_df.copy()
    
    # Calculate distance from spot
    puts_df['distance_from_spot'] = abs(puts_df['strike'] - nifty_spot)
    
    # Sort by distance from spot (closest first)
    puts_df = puts_df.sort_values('distance_from_spot')
    
    # Select top 5 closest strikes
    recommended_strikes = puts_df.head(5).copy()
    
    # Calculate cost per lot
    recommended_strikes['premium_per_lot'] = recommended_strikes['put_ltp'] * NIFTY_LOT_SIZE
    recommended_strikes['total_hedging_cost'] = recommended_strikes['premium_per_lot'] * lots_needed
    
    # Add moneyness description
    recommended_strikes['moneyness'] = recommended_strikes.apply(
        lambda x: f"{'ATM' if x['distance_from_spot'] < 50 else 'Near ATM'} ({x['strike']})", axis=1)
    
    return recommended_strikes[['strike', 'moneyness', 'put_ltp', 'put_delta', 'put_iv', 
                               'premium_per_lot', 'total_hedging_cost']]

# Main function
def main():
    st.title("Portfolio Hedging Calculator")
    
    # Initialize portfolio
    portfolio_df = initialize_portfolio()
    
    # Calculate portfolio metrics
    portfolio_df, portfolio_beta = calculate_portfolio_beta(portfolio_df)
    
    # Fetch Nifty spot price
    nifty_spot = fetch_nifty_price()
    if nifty_spot is None:
        st.error("Unable to fetch Nifty spot price. Using default value of 22000.")
        nifty_spot = 22000  # Default fallback
    
    # Calculate hedging requirements
    total_investment, portfolio_delta, lots_needed = calculate_hedging(
        portfolio_df, portfolio_beta, nifty_spot)
    
    # Fetch options data
    expiry_date = "24-04-2025"  # You can make this dynamic
    options_data = fetch_options_data(expiry=expiry_date)
    options_df = process_options_data(options_data, nifty_spot)
    
    # Display portfolio summary
    st.subheader("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Investment", f"₹{total_investment:,.2f}")
    col2.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
    col3.metric("Nifty Spot", f"₹{nifty_spot:,.2f}")
    
    st.subheader("Portfolio Details")
    st.dataframe(portfolio_df.style.format({
        'Avg. Cost Price': '{:,.2f}',
        'Current Price': '{:,.2f}',
        'Investment': '{:,.2f}',
        'P&L': '{:,.2f}',
        'P&L%': '{:.2f}%',
        'Beta': '{:.2f}',
        'Correlation_with_Nifty': '{:.2f}',
        'Volatility': '{:.2f}',
        'Weight': '{:.2%}'
    }))
    
    # Display hedging requirements
    st.subheader("Hedging Requirements")
    st.write(f"Portfolio Delta (Nifty points equivalent): {portfolio_delta:.2f}")
    st.write(f"Number of Nifty lots needed for hedging: {lots_needed:.2f}")
    
    if options_df is not None:
        # Get recommended strikes
        recommended_puts = get_recommended_strikes(options_df, nifty_spot, lots_needed)
        
        st.subheader("Recommended Put Options for Hedging")
        st.dataframe(recommended_puts.style.format({
            'strike': '{:,.0f}',
            'put_ltp': '{:.2f}',
            'put_delta': '{:.2f}',
            'put_iv': '{:.2%}',
            'premium_per_lot': '₹{:,.2f}',
            'total_hedging_cost': '₹{:,.2f}'
        }))
        
        # Show options chain for analysis
        st.subheader("Nifty Options Chain (Puts)")
        st.dataframe(options_df[['strike', 'put_moneyness', 'put_ltp', 'put_delta', 
                                'put_gamma', 'put_iv', 'put_theta', 'put_oi']].sort_values('strike').style.format({
            'strike': '{:.0f}',
            'put_ltp': '{:.2f}',
            'put_delta': '{:.2f}',
            'put_gamma': '{:.4f}',
            'put_iv': '{:.2%}',
            'put_theta': '{:.2f}',
            'put_oi': '{:,.0f}'
        }))
    else:
        st.error("Failed to fetch options data. Please try again later.")

if __name__ == "__main__":
    main()
