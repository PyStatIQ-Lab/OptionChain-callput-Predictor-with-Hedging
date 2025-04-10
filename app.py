import yfinance as yf
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Constants
NIFTY_LOT_SIZE = 75
NIFTY_SYMBOL = "^NSEI"
RISK_FREE_RATE = 0.05  # 5% risk-free rate (approximate)
LOOKBACK_PERIOD = 252  # 1 year of trading days

# Portfolio data
portfolio = {
    "STAR.NS": {"quantity": 30, "avg_cost": 1397.1},
    "ORCHPHARMA.NS": {"quantity": 30, "avg_cost": 1680.92},
    "APARINDS.NS": {"quantity": 3, "avg_cost": 11145},
    "NEWGEN.NS": {"quantity": 21, "avg_cost": 1663.65},
    "GENESYS.NS": {"quantity": 35, "avg_cost": 991.75}
}

current_prices = {
    "STAR.NS": 575.8,
    "ORCHPHARMA.NS": 720.35,
    "APARINDS.NS": 4974.35,
    "NEWGEN.NS": 842.4,
    "GENESYS.NS": 550.05
}

def calculate_portfolio_metrics():
    # Get historical data for all stocks and Nifty
    tickers = list(portfolio.keys()) + [NIFTY_SYMBOL]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data returned from yfinance")
            
        # Check if multi-index DataFrame (multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data['Close']
        else:
            # Single ticker case (unlikely in this context)
            adj_close = pd.DataFrame(data['Close'])
            adj_close.columns = tickers
            
        adj_close = adj_close.dropna()
        
        # Calculate daily returns
        returns = adj_close.pct_change().dropna()
        
        # Calculate individual stock betas against Nifty
        cov_matrix = returns.cov()
        nifty_variance = returns[NIFTY_SYMBOL].var()
        betas = cov_matrix[NIFTY_SYMBOL] / nifty_variance
        betas = betas.drop(NIFTY_SYMBOL)
        
        # Calculate portfolio weights
        current_values = {stock: portfolio[stock]['quantity'] * current_prices[stock] for stock in portfolio}
        total_value = sum(current_values.values())
        weights = {stock: value/total_value for stock, value in current_values.items()}
        
        # Portfolio beta
        portfolio_beta = sum(weights[stock] * betas[stock] for stock in portfolio)
        
        # Portfolio volatility (annualized)
        portfolio_returns = pd.Series(0, index=returns.index)
        for stock in portfolio:
            portfolio_returns += weights[stock] * returns[stock]
        
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        return {
            "portfolio_beta": portfolio_beta,
            "portfolio_volatility": portfolio_volatility,
            "portfolio_value": total_value,
            "weights": weights,
            "betas": betas,
            "returns": returns
        }
        
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return {
            "portfolio_beta": 0,
            "portfolio_volatility": 0,
            "portfolio_value": 0,
            "weights": {},
            "betas": {},
            "returns": pd.DataFrame()
        }

def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="24-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        return data['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code} - {response.text}")
        return None

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

def recommend_hedge(portfolio_beta, portfolio_value, options_data, nifty_spot):
    # Calculate hedge notional (portfolio beta-adjusted exposure)
    hedge_notional = portfolio_value * portfolio_beta
    
    # Find ATM put (strike closest to spot)
    options_data['distance_to_spot'] = abs(options_data['strike'] - nifty_spot)
    atm_put = options_data[options_data['put_moneyness'] == 'ATM'].iloc[0]
    
    # Calculate how many lots we need
    put_delta = atm_put['put_delta']
    hedge_lots = (hedge_notional / (nifty_spot * NIFTY_LOT_SIZE)) / put_delta
    hedge_lots = round(hedge_lots)
    
    # Ensure at least 1 lot if some hedging is needed
    if hedge_lots == 0 and portfolio_beta > 0.1:
        hedge_lots = 1
    
    # Calculate cost of hedge
    hedge_cost = hedge_lots * NIFTY_LOT_SIZE * atm_put['put_ltp']
    
    # Also look at slightly OTM puts (5% below spot) as alternative
    otm_strike = nifty_spot * 0.95
    otm_puts = options_data[options_data['strike'] <= otm_strike].sort_values('strike', ascending=False)
    if not otm_puts.empty:
        otm_put = otm_puts.iloc[0]
        otm_hedge_lots = (hedge_notional / (nifty_spot * NIFTY_LOT_SIZE)) / otm_put['put_delta']
        otm_hedge_lots = round(otm_hedge_lots)
        otm_hedge_cost = otm_hedge_lots * NIFTY_LOT_SIZE * otm_put['put_ltp']
    else:
        otm_put = None
        otm_hedge_lots = 0
        otm_hedge_cost = 0
    
    return {
        'atm_put': atm_put,
        'hedge_lots': hedge_lots,
        'hedge_cost': hedge_cost,
        'otm_put': otm_put,
        'otm_hedge_lots': otm_hedge_lots,
        'otm_hedge_cost': otm_hedge_cost if otm_put else 0,
        'hedge_notional': hedge_notional
    }

def main():
    st.title("Portfolio Hedging Analysis")
    
    # Calculate portfolio metrics
    st.header("Portfolio Analysis")
    with st.spinner("Calculating portfolio beta and volatility..."):
        metrics = calculate_portfolio_metrics()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"₹{metrics['portfolio_value']:,.2f}")
    col2.metric("Portfolio Beta", f"{metrics['portfolio_beta']:.2f}")
    col3.metric("Annual Volatility", f"{metrics['portfolio_volatility']*100:.2f}%")
    
    st.subheader("Stock-wise Details")
    stock_data = []
    for stock in portfolio:
        stock_data.append({
            "Stock": stock,
            "Quantity": portfolio[stock]['quantity'],
            "Avg Cost": portfolio[stock]['avg_cost'],
            "Current Price": current_prices[stock],
            "Current Value": portfolio[stock]['quantity'] * current_prices[stock],
            "P&L": (current_prices[stock] - portfolio[stock]['avg_cost']) * portfolio[stock]['quantity'],
            "Weight": f"{metrics['weights'][stock]*100:.2f}%",
            "Beta": f"{metrics['betas'][stock]:.2f}"
        })
    
    st.dataframe(pd.DataFrame(stock_data))
    
    # Get options data for hedging
    st.header("Hedging Recommendation")
    nifty_spot = fetch_nifty_price()
    if nifty_spot:
        st.metric("Current Nifty Spot Price", f"₹{nifty_spot:,.2f}")
        
        with st.spinner("Fetching options data..."):
            raw_data = fetch_options_data()
            if raw_data:
                options_df = process_options_data(raw_data, nifty_spot)
                
                if options_df is not None:
                    hedge_rec = recommend_hedge(
                        metrics['portfolio_beta'],
                        metrics['portfolio_value'],
                        options_df,
                        nifty_spot
                    )
                    
                    st.subheader("ATM Put Hedge")
                    st.write(f"Strike: ₹{hedge_rec['atm_put']['strike']:,.2f}")
                    st.write(f"Premium: ₹{hedge_rec['atm_put']['put_ltp']:,.2f}")
                    st.write(f"Delta: {hedge_rec['atm_put']['put_delta']:,.2f}")
                    st.write(f"IV: {hedge_rec['atm_put']['put_iv']*100:,.2f}%")
                    st.write(f"Lots needed: {hedge_rec['hedge_lots']} (Quantity: {hedge_rec['hedge_lots'] * NIFTY_LOT_SIZE})")
                    st.write(f"Total hedge cost: ₹{hedge_rec['hedge_cost']:,.2f} ({hedge_rec['hedge_cost']/metrics['portfolio_value']*100:.2f}% of portfolio)")
                    
                    if hedge_rec['otm_put'] is not None:
                        st.subheader("OTM Put Alternative (5% below spot)")
                        st.write(f"Strike: ₹{hedge_rec['otm_put']['strike']:,.2f}")
                        st.write(f"Premium: ₹{hedge_rec['otm_put']['put_ltp']:,.2f}")
                        st.write(f"Delta: {hedge_rec['otm_put']['put_delta']:,.2f}")
                        st.write(f"IV: {hedge_rec['otm_put']['put_iv']*100:,.2f}%")
                        st.write(f"Lots needed: {hedge_rec['otm_hedge_lots']} (Quantity: {hedge_rec['otm_hedge_lots'] * NIFTY_LOT_SIZE})")
                        st.write(f"Total hedge cost: ₹{hedge_rec['otm_hedge_cost']:,.2f} ({hedge_rec['otm_hedge_cost']/metrics['portfolio_value']*100:.2f}% of portfolio)")
                    
                    st.info("Note: This hedge is designed to offset systematic risk (beta) in your portfolio. The quantity is calculated based on the delta-adjusted exposure to match your portfolio's beta.")
                else:
                    st.error("Failed to process options data")
            else:
                st.error("Failed to fetch options data")
    else:
        st.error("Failed to fetch Nifty spot price")

if __name__ == "__main__":
    main()
