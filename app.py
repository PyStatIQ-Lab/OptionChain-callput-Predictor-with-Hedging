import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
import json
import yfinance as yf

# Configure page
st.set_page_config(
    page_title="PyStatIQ Options Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

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
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .prediction-card {
        background-color: #f1f8fe;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .tabs {
        margin-bottom: 20px;
    }
    .trade-recommendation {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        color: #000;
    }
    .trade-recommendation.sell {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
    }
    .strike-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #40404f;
    }
    .probability-meter {
        height: 20px;
        background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
        border-radius: 10px;
        margin: 10px 0;
        position: relative;
    }
    .probability-indicator {
        position: absolute;
        height: 30px;
        width: 2px;
        background-color: black;
        top: -5px;
    }
    .risk-metric {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
    }
    .regime-indicator {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
    }
    .regime-low {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    .regime-normal {
        background-color: #e3f2fd;
        border-left: 5px solid #1565c0;
    }
    .regime-high {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
    }
    .arbitrage-opportunity {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #fff8e1;
        border-left: 5px solid #ff8f00;
    }
    .portfolio-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        background-color: #f0f7ff;
        border-left: 5px solid #1e88e5;
    }
    .hedge-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Portfolio Configuration (can be made editable by user)
PORTFOLIO = [
    {"symbol": "STAR.NS", "quantity": 30, "avg_cost": 1397.1},
    {"symbol": "ORCHPHARMA.NS", "quantity": 30, "avg_cost": 1680.92},
    {"symbol": "APARINDS.NS", "quantity": 3, "avg_cost": 11145},
    {"symbol": "NEWGEN.NS", "quantity": 21, "avg_cost": 1663.65}
]

# Fetch data from API
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="03-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Fetch live Nifty price
@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        return data['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code} - {response.text}")
        return None

# Fetch portfolio current prices
@st.cache_data(ttl=300)
def fetch_portfolio_prices(portfolio):
    try:
        symbols = [item['symbol'] for item in portfolio]
        data = yf.download(tickers=symbols, period='1d', progress=False)['Close']
        
        # Get the last available price
        current_prices = {}
        for symbol in symbols:
            current_prices[symbol] = data[symbol].iloc[-1] if not data[symbol].isnull().all() else None
        
        return current_prices
    except Exception as e:
        st.error(f"Failed to fetch portfolio prices: {str(e)}")
        return None

# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio, current_prices):
    if not current_prices:
        return None
    
    portfolio_value = 0
    portfolio_beta = 0
    total_investment = 0
    stocks = []
    
    for item in portfolio:
        symbol = item['symbol']
        quantity = item['quantity']
        avg_cost = item['avg_cost']
        current_price = current_prices.get(symbol)
        
        if current_price is None:
            continue
        
        # Get beta for the stock (simplified - in reality you'd need to calculate or get from API)
        # For demo, we'll use some reasonable estimates
        beta_estimates = {
            "STAR.NS": 1.2,
            "ORCHPHARMA.NS": 0.8,
            "APARINDS.NS": 1.1,
            "NEWGEN.NS": 1.3
        }
        beta = beta_estimates.get(symbol, 1.0)
        
        investment = quantity * avg_cost
        current_val = quantity * current_price
        pnl = current_val - investment
        pnl_pct = (pnl / investment) * 100 if investment != 0 else 0
        
        stocks.append({
            "Symbol": symbol,
            "Quantity": quantity,
            "Avg Cost": avg_cost,
            "Current Price": current_price,
            "Investment": investment,
            "Current Value": current_val,
            "P&L": pnl,
            "P&L %": pnl_pct,
            "Beta": beta
        })
        
        portfolio_value += current_val
        portfolio_beta += (current_val * beta)
        total_investment += investment
    
    if portfolio_value > 0:
        portfolio_beta = portfolio_beta / portfolio_value
    else:
        portfolio_beta = 1.0
    
    portfolio_pnl = portfolio_value - total_investment
    portfolio_pnl_pct = (portfolio_pnl / total_investment) * 100 if total_investment != 0 else 0
    
    return {
        "stocks": pd.DataFrame(stocks),
        "total_value": portfolio_value,
        "total_investment": total_investment,
        "total_pnl": portfolio_pnl,
        "total_pnl_pct": portfolio_pnl_pct,
        "portfolio_beta": portfolio_beta
    }

# Calculate hedging requirements
def calculate_hedge_requirements(portfolio_metrics, nifty_price, options_data, hedge_percentage=100):
    if portfolio_metrics is None or options_data is None or options_data.empty:
        return None
    
    portfolio_value = portfolio_metrics['total_value']
    portfolio_beta = portfolio_metrics['portfolio_beta']
    
    # Calculate portfolio's Nifty equivalent (beta-adjusted)
    nifty_equivalent = (portfolio_value * portfolio_beta) / nifty_price
    
    # We'll hedge using ATM puts (or slightly OTM)
    # Find ATM put
    options_data['distance_to_atm'] = abs(options_data['strike'] - nifty_price)
    atm_put = options_data[options_data['put_moneyness'] == 'ATM']
    
    if atm_put.empty:
        # If no exact ATM, get the closest
        atm_put = options_data.nsmallest(1, 'distance_to_atm')
    
    if atm_put.empty:
        return None
    
    put_data = atm_put.iloc[0]
    put_strike = put_data['strike']
    put_price = put_data['put_ltp']
    put_delta = abs(put_data['put_delta'])  # Put deltas are negative
    
    # Calculate number of lots needed (Nifty lot size is 50)
    hedge_quantity = (nifty_equivalent * (hedge_percentage / 100)) / (put_delta * 50)
    
    # Calculate cost of hedge
    hedge_cost = hedge_quantity * 50 * put_price
    
    return {
        "put_strike": put_strike,
        "put_price": put_price,
        "put_delta": put_delta,
        "nifty_equivalent": nifty_equivalent,
        "hedge_quantity": round(hedge_quantity),
        "hedge_cost": hedge_cost,
        "hedge_percentage": hedge_percentage,
        "portfolio_value": portfolio_value,
        "portfolio_beta": portfolio_beta
    }

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

# Get top ITM/OTM strikes
def get_top_strikes(df, spot_price, n=5):
    # For calls: ITM = strike < spot, OTM = strike > spot
    call_itm = df[df['strike'] < spot_price].sort_values('strike', ascending=False).head(n)
    call_otm = df[df['strike'] > spot_price].sort_values('strike', ascending=True).head(n)
    
    # For puts: ITM = strike > spot, OTM = strike < spot
    put_itm = df[df['strike'] > spot_price].sort_values('strike', ascending=True).head(n)
    put_otm = df[df['strike'] < spot_price].sort_values('strike', ascending=False).head(n)
    
    return {
        'call_itm': call_itm,
        'call_otm': call_otm,
        'put_itm': put_itm,
        'put_otm': put_otm
    }

# Calculate probability of profit
def calculate_pop(strike, premium, spot_price, iv, days_to_expiry, is_call=True):
    if days_to_expiry <= 0 or iv <= 0:
        return 0.5  # Neutral probability if data is invalid
    
    # Convert IV from percentage to decimal
    iv_decimal = iv / 100
    
    # Calculate break-even price
    if is_call:
        breakeven = strike + premium
    else:
        breakeven = strike - premium
    
    # Calculate expected move
    annualized_vol = iv_decimal * np.sqrt(365)
    daily_vol = annualized_vol / np.sqrt(365)
    expected_move = spot_price * daily_vol * np.sqrt(days_to_expiry)
    
    # Calculate z-score
    if is_call:
        z = (breakeven - spot_price) / (spot_price * iv_decimal * np.sqrt(days_to_expiry/365))
    else:
        z = (spot_price - breakeven) / (spot_price * iv_decimal * np.sqrt(days_to_expiry/365))
    
    # Calculate probability using normal CDF
    pop = norm.cdf(z)
    
    return pop

# Calculate expected move
def calculate_expected_move(spot_price, iv, days_to_expiry):
    if days_to_expiry <= 0 or iv <= 0:
        return (spot_price, spot_price)  # Neutral if data is invalid
    
    # Convert IV from percentage to decimal
    iv_decimal = iv / 100
    
    # Calculate expected move (1 standard deviation)
    move = spot_price * iv_decimal * np.sqrt(days_to_expiry/365)
    upper = spot_price + move
    lower = spot_price - move
    
    return (lower, upper)

# Detect market regimes using HMM
def detect_market_regime(df, n_regimes=3):
    try:
        # Prepare features (using IV and volume as indicators)
        features = df[['call_iv', 'put_iv', 'call_volume', 'put_volume']].dropna()
        features = np.log1p(features)  # Log transform for better normality
        
        # Fit HMM
        model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000)
        model.fit(features)
        
        # Predict regimes
        regimes = model.predict(features)
        
        # Classify regimes based on IV levels
        regime_stats = []
        for i in range(n_regimes):
            mask = (regimes == i)
            avg_iv = np.mean(df.loc[mask, ['call_iv', 'put_iv']].values)
            regime_stats.append((i, avg_iv))
        
        # Sort regimes by IV (low to high)
        regime_stats.sort(key=lambda x: x[1])
        
        # Create mapping from regime number to label
        regime_labels = {}
        for i, (regime_num, _) in enumerate(regime_stats):
            if i == 0:
                regime_labels[regime_num] = "Low Volatility"
            elif i == 1:
                regime_labels[regime_num] = "Normal Volatility"
            else:
                regime_labels[regime_num] = "High Volatility"
        
        # Get current regime (most recent)
        current_regime = regimes[-1]
        
        return regime_labels[current_regime], model.means_, model.covars_
    except Exception as e:
        st.error(f"Regime detection failed: {str(e)}")
        return "Unknown", None, None

# Check for put-call parity violations
def detect_arbitrage_opportunities(df, spot_price, risk_free_rate=0.05, days_to_expiry=7):
    try:
        arbitrage_ops = []
        
        # Convert days to years
        t = days_to_expiry / 365
        
        for _, row in df.iterrows():
            strike = row['strike']
            call_price = row['call_ltp']
            put_price = row['put_ltp']
            
            # Theoretical put price based on call-put parity
            theoretical_put = call_price - spot_price + strike * np.exp(-risk_free_rate * t)
            
            # Theoretical call price based on put-call parity
            theoretical_call = put_price + spot_price - strike * np.exp(-risk_free_rate * t)
            
            # Check for violations with 1% threshold
            put_discrepancy = (put_price - theoretical_put) / theoretical_put
            call_discrepancy = (call_price - theoretical_call) / theoretical_call
            
            if put_discrepancy > 0.01:
                arbitrage_ops.append({
                    'type': 'Put Overpriced',
                    'strike': strike,
                    'actual_put': put_price,
                    'theoretical_put': theoretical_put,
                    'discrepancy_pct': put_discrepancy * 100,
                    'call_price': call_price,
                    'action': 'Sell Put + Buy Call + Buy Stock'
                })
            
            if call_discrepancy > 0.01:
                arbitrage_ops.append({
                    'type': 'Call Overpriced',
                    'strike': strike,
                    'actual_call': call_price,
                    'theoretical_call': theoretical_call,
                    'discrepancy_pct': call_discrepancy * 100,
                    'put_price': put_price,
                    'action': 'Sell Call + Buy Put + Sell Stock'
                })
        
        # Sort by largest discrepancy
        arbitrage_ops.sort(key=lambda x: x['discrepancy_pct'], reverse=True)
        
        return arbitrage_ops[:5]  # Return top 5 opportunities
    except Exception as e:
        st.error(f"Arbitrage detection failed: {str(e)}")
        return []

# Generate trade recommendations with risk metrics
def generate_trade_recommendations(df, spot_price, days_to_expiry=7):
    recommendations = []
    
    # Calculate metrics for all strikes
    df['call_premium_ratio'] = (df['call_ask'] - df['call_bid']) / df['call_ltp']
    df['put_premium_ratio'] = (df['put_ask'] - df['put_bid']) / df['put_ltp']
    df['call_risk_reward'] = (spot_price - df['strike'] + df['call_ltp']) / df['call_ltp']
    df['put_risk_reward'] = (df['strike'] - spot_price + df['put_ltp']) / df['put_ltp']
    
    # Find best calls to buy (low premium ratio, high OI change, good risk/reward)
    best_calls = df[(df['call_moneyness'] == 'OTM') & 
                   (df['call_premium_ratio'] < 0.1) &
                   (df['call_oi_change'] > 0)].sort_values(
        by=['call_premium_ratio', 'call_oi_change'], 
        ascending=[True, False]
    ).head(3)
    
    for _, row in best_calls.iterrows():
        pop = calculate_pop(row['strike'], row['call_ltp'], spot_price, row['call_iv'], days_to_expiry, is_call=True)
        lower_move, upper_move = calculate_expected_move(spot_price, row['call_iv'], days_to_expiry)
        
        recommendations.append({
            'type': 'BUY CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'risk_reward': f"{row['call_risk_reward']:.1f}:1",
            'reason': "Low spread, OI buildup, good risk/reward",
            'pop': pop,
            'expected_lower': lower_move,
            'expected_upper': upper_move
        })
    
    # Find best puts to buy (low premium ratio, high OI change, good risk/reward)
    best_puts = df[(df['put_moneyness'] == 'OTM') & 
                  (df['put_premium_ratio'] < 0.1) &
                  (df['put_oi_change'] > 0)].sort_values(
        by=['put_premium_ratio', 'put_oi_change'], 
        ascending=[True, False]
    ).head(3)
    
    for _, row in best_puts.iterrows():
        pop = calculate_pop(row['strike'], row['put_ltp'], spot_price, row['put_iv'], days_to_expiry, is_call=False)
        lower_move, upper_move = calculate_expected_move(spot_price, row['put_iv'], days_to_expiry)
        
        recommendations.append({
            'type': 'BUY PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'risk_reward': f"{row['put_risk_reward']:.1f}:1",
            'reason': "Low spread, OI buildup, good risk/reward",
            'pop': pop,
            'expected_lower': lower_move,
            'expected_upper': upper_move
        })
    
    # Find best calls to sell (high premium ratio, decreasing OI)
    best_sell_calls = df[(df['call_moneyness'] == 'ITM') & 
                        (df['call_premium_ratio'] > 0.15) &
                        (df['call_oi_change'] < 0)].sort_values(
        by=['call_premium_ratio', 'call_oi_change'], 
        ascending=[False, True]
    ).head(2)
    
    for _, row in best_sell_calls.iterrows():
        pop = calculate_pop(row['strike'], row['call_ltp'], spot_price, row['call_iv'], days_to_expiry, is_call=True)
        lower_move, upper_move = calculate_expected_move(spot_price, row['call_iv'], days_to_expiry)
        
        recommendations.append({
            'type': 'SELL CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'risk_reward': f"{1/row['call_risk_reward']:.1f}:1",
            'reason': "High spread, OI unwinding, favorable risk",
            'pop': pop,
            'expected_lower': lower_move,
            'expected_upper': upper_move
        })
    
    # Find best puts to sell (high premium ratio, decreasing OI)
    best_sell_puts = df[(df['put_moneyness'] == 'ITM') & 
                       (df['put_premium_ratio'] > 0.15) &
                       (df['put_oi_change'] < 0)].sort_values(
        by=['put_premium_ratio', 'put_oi_change'], 
        ascending=[False, True]
    ).head(2)
    
    for _, row in best_sell_puts.iterrows():
        pop = calculate_pop(row['strike'], row['put_ltp'], spot_price, row['put_iv'], days_to_expiry, is_call=False)
        lower_move, upper_move = calculate_expected_move(spot_price, row['put_iv'], days_to_expiry)
        
        recommendations.append({
            'type': 'SELL PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'risk_reward': f"{1/row['put_risk_reward']:.1f}:1",
            'reason': "High spread, OI unwinding, favorable risk",
            'pop': pop,
            'expected_lower': lower_move,
            'expected_upper': upper_move
        })
    
    return recommendations

# Train predictive model (simplified example)
def train_price_direction_model(df):
    # This is a simplified example - in a real app you'd use more sophisticated features
    try:
        # Create target variable (1 if call OI increase > put OI increase, else 0)
        df['target'] = (df['call_oi_change'] > df['put_oi_change']).astype(int)
        
        # Select features
        features = ['pcr', 'call_oi_change', 'put_oi_change', 'call_iv', 'put_iv']
        X = df[features].fillna(0)
        y = df['target']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, features
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None

# Generate price direction prediction
def predict_price_direction(model, features, df):
    try:
        X = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        proba = model.predict_proba(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]
        
        return {
            'direction': 'Up' if prediction == 1 else 'Down',
            'confidence': max(proba[0], proba[1])
        }
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Calculate Greeks exposure
def calculate_greeks_exposure(df):
    try:
        # Calculate net Greeks across all strikes
        greeks = {
            'Gamma': df['call_gamma'].sum() - df['put_gamma'].sum(),
            'Theta': df['call_theta'].sum() - df['put_theta'].sum(),
            'Vega': df['call_vega'].sum() - df['put_vega'].sum()
        }
        return greeks
    except Exception as e:
        st.error(f"Greeks calculation failed: {str(e)}")
        return None

# Main App
def main():
    st.markdown("<div class='header'><h1>📊 PyStatIQ Options Chain Dashboard</h1></div>", unsafe_allow_html=True)
    
    # Fetch spot price
    spot_price = fetch_nifty_price()
    if spot_price is None:
        st.error("Failed to fetch Nifty spot price. Using default value.")
        spot_price = 22000  # Default fallback
    
    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        asset_key = st.selectbox(
            "Underlying Asset",
            ["NSE_INDEX|Nifty 50", "NSE_INDEX|Bank Nifty"],
            index=0
        )
        
        expiry_date = st.date_input(
            "Expiry Date",
            datetime.strptime("03-04-2025", "%d-%m-%Y")
        ).strftime("%d-%m-%Y")
        
        days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=30, value=7)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0) / 100
        
        st.markdown("---")
        st.markdown(f"**Current Nifty Spot Price: {spot_price:,.2f}**")
        
        st.markdown("---")
        st.markdown("**Analysis Settings**")
        volume_threshold = st.number_input("High Volume Threshold", value=5000000)
        oi_change_threshold = st.number_input("Significant OI Change", value=1000000)
        
        st.markdown("---")
        st.markdown("**Portfolio Hedging**")
        hedge_percentage = st.slider("Hedge Percentage", 0, 100, 100)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This dashboard provides real-time options chain analysis using data.")
    
    # Fetch and process data
    with st.spinner("Fetching live options data..."):
        raw_data = fetch_options_data(asset_key, expiry_date)
    
    if raw_data is None:
        st.error("Failed to load data. Please try again later.")
        return
    
    df = process_options_data(raw_data, spot_price)
    if df is None or df.empty:
        st.error("No data available for the selected parameters.")
        return
    
    # Fetch portfolio data
    with st.spinner("Fetching portfolio data..."):
        current_prices = fetch_portfolio_prices(PORTFOLIO)
        portfolio_metrics = calculate_portfolio_metrics(PORTFOLIO, current_prices)
        hedge_info = calculate_hedge_requirements(portfolio_metrics, spot_price, df, hedge_percentage)
    
    # Market Regime Detection
    market_regime, regime_means, regime_covars = detect_market_regime(df)
    
    # Arbitrage Detection
    arbitrage_opportunities = detect_arbitrage_opportunities(df, spot_price, risk_free_rate, days_to_expiry)
    
    # Greeks Exposure
    greeks_exposure = calculate_greeks_exposure(df)
    
    # Train predictive model
    model, features = train_price_direction_model(df)
    
    # Get top strikes
    top_strikes = get_top_strikes(df, spot_price)
    
    # Default strike selection (ATM)
    atm_strike = df.iloc[(df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]
    
    # Portfolio Section
    st.markdown("### Portfolio Analysis & Hedging")
    
    if portfolio_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='portfolio-card'>", unsafe_allow_html=True)
            st.markdown("**Portfolio Summary**")
            st.markdown(f"Total Investment: ₹{portfolio_metrics['total_investment']:,.2f}")
            st.markdown(f"Current Value: ₹{portfolio_metrics['total_value']:,.2f}")
            st.markdown(f"P&L: ₹{portfolio_metrics['total_pnl']:,.2f} ({portfolio_metrics['total_pnl_pct']:.2f}%)")
            st.markdown(f"Portfolio Beta: {portfolio_metrics['portfolio_beta']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='hedge-card'>", unsafe_allow_html=True)
            st.markdown("**Hedging Recommendation**")
            
            if hedge_info:
                st.markdown(f"Hedge {hedge_percentage}% of portfolio risk")
                st.markdown(f"Buy {hedge_info['hedge_quantity']} lots of Nifty {hedge_info['put_strike']:.0f} PUT")
                st.markdown(f"Put Premium: ₹{hedge_info['put_price']:.2f}")
                st.markdown(f"Delta: {hedge_info['put_delta']:.2f}")
                st.markdown(f"Estimated Cost: ₹{hedge_info['hedge_cost']:,.2f}")
                st.markdown(f"(Nifty Equivalent: {hedge_info['nifty_equivalent']:,.2f})")
            else:
                st.warning("Could not calculate hedge requirements")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display portfolio holdings
        st.dataframe(
            portfolio_metrics['stocks'].style.format({
                "Avg Cost": "{:,.2f}",
                "Current Price": "{:,.2f}",
                "Investment": "{:,.2f}",
                "Current Value": "{:,.2f}",
                "P&L": "{:,.2f}",
                "P&L %": "{:.2f}%"
            }),
            use_container_width=True,
            height=200
        )
    else:
        st.warning("Could not load portfolio data")
    
    # Main columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Call OI**")
        total_call_oi = df['call_oi'].sum()
        st.markdown(f"<h2>{total_call_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Put OI**")
        total_put_oi = df['put_oi'].sum()
        st.markdown(f"<h2>{total_put_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Strike price selector
        selected_strike = st.selectbox(
            "Select Strike Price",
            df['strike'].unique(),
            index=int(np.where(df['strike'].unique() == atm_strike)[0][0])
        )
        
        # Market regime indicator
        regime_class = ""
        if "Low" in market_regime:
            regime_class = "regime-low"
        elif "Normal" in market_regime:
            regime_class = "regime-normal"
        elif "High" in market_regime:
            regime_class = "regime-high"
            
        st.markdown(f"""
            <div class='regime-indicator {regime_class}'>
                <h3>Market Regime: {market_regime}</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Call OI Change**")
        call_oi_change = df[df['strike'] == selected_strike]['call_oi_change'].values[0]
        change_color = "positive" if call_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{call_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Put OI Change**")
        put_oi_change = df[df['strike'] == selected_strike]['put_oi_change'].values[0]
        change_color = "positive" if put_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{put_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Top Strikes Section
    st.markdown("### Top ITM/OTM Strike Prices")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Top ITM Call Strikes**")
        for _, row in top_strikes['call_itm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['call_ltp']:.2f})<br>
                    OI: {row['call_oi']:,} (Δ: {row['call_oi_change']:,})<br>
                    IV: {row['call_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Top OTM Call Strikes**")
        for _, row in top_strikes['call_otm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['call_ltp']:.2f})<br>
                    OI: {row['call_oi']:,} (Δ: {row['call_oi_change']:,})<br>
                    IV: {row['call_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Top ITM Put Strikes**")
        for _, row in top_strikes['put_itm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['put_ltp']:.2f})<br>
                    OI: {row['put_oi']:,} (Δ: {row['put_oi_change']:,})<br>
                    IV: {row['put_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Top OTM Put Strikes**")
        for _, row in top_strikes['put_otm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['put_ltp']:.2f})<br>
                    OI: {row['put_oi']:,} (Δ: {row['put_oi_change']:,})<br>
                    IV: {row['put_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    # Trade Recommendations with Risk Metrics
    st.markdown("### Trade Recommendations with Risk Metrics")
    recommendations = generate_trade_recommendations(df, spot_price, days_to_expiry)
    
    if recommendations:
        for rec in recommendations:
            is_sell = 'SELL' in rec['type']
            pop_percent = rec['pop'] * 100
            
            st.markdown(f"""
                <div class='trade-recommendation{' sell' if is_sell else ''}'>
                    <h4>{rec['type']} @ {rec['strike']:.0f}</h4>
                    <p>
                        Premium: {rec['premium']:.2f} | IV: {rec['iv']:.1f}%<br>
                        OI Change: {rec['oi_change']:,} | Risk/Reward: {rec['risk_reward']}<br>
                        <b>Probability of Profit:</b> {pop_percent:.1f}%
                        <div class='probability-meter'>
                            <div class='probability-indicator' style='left: {pop_percent}%;'></div>
                        </div>
                        <b>Expected Move:</b> {rec['expected_lower']:.1f} to {rec['expected_upper']:.1f}<br>
                        <b>Reason:</b> {rec['reason']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No strong trade recommendations based on current market conditions")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Strike Analysis", "OI/Volume Trends", "Advanced Analytics", "Predictive Models", "Risk Monitoring"])
    
    with tab1:
        st.markdown(f"### Detailed Analysis for Strike: {selected_strike}")
        
        # Get selected strike data
        strike_data = df[df['strike'] == selected_strike].iloc[0]
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['LTP', 'Bid', 'Ask', 'Volume', 'OI', 'OI Change', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega'],
            'Call': [
                strike_data['call_ltp'],
                strike_data['call_bid'],
                strike_data['call_ask'],
                strike_data['call_volume'],
                strike_data['call_oi'],
                strike_data['call_oi_change'],
                strike_data['call_iv'],
                strike_data['call_delta'],
                strike_data['call_gamma'],
                strike_data['call_theta'],
                strike_data['call_vega']
            ],
            'Put': [
                strike_data['put_ltp'],
                strike_data['put_bid'],
                strike_data['put_ask'],
                strike_data['put_volume'],
                strike_data['put_oi'],
                strike_data['put_oi_change'],
                strike_data['put_iv'],
                strike_data['put_delta'],
                strike_data['put_gamma'],
                strike_data['put_theta'],
                strike_data['put_vega']
            ]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Call': '{:,.2f}',
                'Put': '{:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.markdown("### Open Interest & Volume Trends")
        
        # Nearby strikes
        all_strikes = sorted(df['strike'].unique())
        current_idx = all_strikes.index(selected_strike)
        nearby_strikes = all_strikes[max(0, current_idx-5):min(len(all_strikes), current_idx+6)]
        nearby_df = df[df['strike'].isin(nearby_strikes)]
        
        # OI Change plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_oi_change', 'put_oi_change'],
            barmode='group',
            title=f'OI Changes Around {selected_strike}',
            labels={'value': 'OI Change', 'strike': 'Strike Price'},
            color_discrete_map={'call_oi_change': '#3498db', 'put_oi_change': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_volume', 'put_volume'],
            barmode='group',
            title=f'Volume Around {selected_strike}',
            labels={'value': 'Volume', 'strike': 'Strike Price'},
            color_discrete_map={'call_volume': '#3498db', 'put_volume': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Advanced Analytics")
        
        # IV Skew Analysis
        st.markdown("#### IV Skew Analysis")
        fig = px.line(
            df,
            x='strike',
            y=['call_iv', 'put_iv'],
            title='Implied Volatility Skew',
            labels={'value': 'IV (%)', 'strike': 'Strike Price'},
            color_discrete_map={'call_iv': '#3498db', 'put_iv': '#e74c3c'}
        )
        fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Arbitrage Opportunities
        st.markdown("#### Arbitrage Opportunities")
        if arbitrage_opportunities:
            for opp in arbitrage_opportunities:
                st.markdown(f"""
                    <div class='arbitrage-opportunity'>
                        <h4>{opp['type']} @ {opp['strike']:.0f}</h4>
                        <p>
                            <b>Discrepancy:</b> {opp['discrepancy_pct']:.2f}%<br>
                            <b>Actual Price:</b> {opp['actual_put' if 'Put' in opp['type'] else 'actual_call']:.2f}<br>
                            <b>Theoretical Price:</b> {opp['theoretical_put' if 'Put' in opp['type'] else 'theoretical_call']:.2f}<br>
                            <b>Action:</b> {opp['action']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant arbitrage opportunities detected")
        
        # Risk Analysis
        st.markdown("#### Risk Analysis")
        
        # Max pain calculation
        pain_points = []
        for strike in df['strike'].unique():
            strike_row = df[df['strike'] == strike].iloc[0]
            pain_points.append((strike, strike_row['call_oi'] + strike_row['put_oi']))
        
        max_pain_strike = min(pain_points, key=lambda x: x[1])[0] if pain_points else selected_strike
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Maximum Pain**")
            st.markdown(f"Current Strike: {selected_strike}")
            st.markdown(f"Max Pain Strike: {max_pain_strike}")
            
            if abs(max_pain_strike - selected_strike) <= (all_strikes[1] - all_strikes[0]) * 2:
                st.warning("Close to max pain - increased pin risk")
            else:
                st.success("Not near max pain level")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Gamma Exposure**")
            
            net_gamma = strike_data['call_gamma'] - strike_data['put_gamma']
            if net_gamma > 0:
                st.info("Positive Gamma: Market makers likely to buy on dips, sell on rallies")
            else:
                st.warning("Negative Gamma: Market makers likely to sell on dips, buy on rallies")
            
            st.markdown(f"Net Gamma: {net_gamma:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Probability calculator
        st.markdown("#### Probability Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            target_price = st.number_input("Target Price", value=spot_price * 1.02, min_value=0.0)
        
        with col2:
            days = st.number_input("Time Horizon (days)", value=days_to_expiry, min_value=1, max_value=30)
        
        # Calculate probability using selected strike's IV (simplified)
        if strike_data['call_iv'] > 0 and strike_data['put_iv'] > 0:
            avg_iv = (strike_data['call_iv'] + strike_data['put_iv']) / 2
            z_score = (target_price - spot_price) / (spot_price * (avg_iv/100) * np.sqrt(days/365))
            prob = norm.cdf(z_score) if target_price > spot_price else 1 - norm.cdf(z_score)
            
            st.markdown(f"""
                <div class='risk-metric'>
                    <h4>Probability Analysis</h4>
                    <p>Probability of reaching {target_price:,.2f} in {days} days: <b>{prob*100:.1f}%</b></p>
                    <p>Using average IV: {avg_iv:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Predictive Models")
        
        if model is not None:
            # Get prediction for current market conditions
            prediction = predict_price_direction(model, features, df)
            
            if prediction:
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h3>Price Direction Prediction</h3>
                        <p>Next 1-3 days: <b>{prediction['direction']}</b></p>
                        <p>Confidence: <b>{prediction['confidence']*100:.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Feature importance
            st.markdown("#### Feature Importance")
            feature_importances = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importances,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Model Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model explanation
            st.markdown("""
                <div class='risk-metric'>
                    <h4>Model Explanation</h4>
                    <p>This predictive model analyzes options chain data to forecast short-term price direction.</p>
                    <p>Key factors considered:</p>
                    <ul>
                        <li>Put-Call Ratio (PCR)</li>
                        <li>Open Interest changes</li>
                        <li>Implied Volatility differences</li>
                    </ul>
                    <p>The model is a Random Forest classifier trained on historical options data.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Could not load predictive model. Please check data availability.")
    
    with tab5:
        st.markdown("### Greeks-Based Risk Monitoring")
        
        if greeks_exposure:
            # Greeks Exposure Metrics
            st.markdown("#### Net Greeks Exposure")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("**Gamma Exposure**")
                st.markdown(f"<h2>{greeks_exposure['Gamma']:.4f}</h2>", unsafe_allow_html=True)
                if greeks_exposure['Gamma'] > 0:
                    st.info("Positive Gamma: Market makers hedge by buying dips and selling rallies")
                else:
                    st.warning("Negative Gamma: Market makers hedge by selling dips and buying rallies")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("**Theta Exposure**")
                st.markdown(f"<h2>{greeks_exposure['Theta']:.4f}</h2>", unsafe_allow_html=True)
                if greeks_exposure['Theta'] > 0:
                    st.success("Positive Theta: Benefiting from time decay")
                else:
                    st.warning("Negative Theta: Losing from time decay")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("**Vega Exposure**")
                st.markdown(f"<h2>{greeks_exposure['Vega']:.4f}</h2>", unsafe_allow_html=True)
                if greeks_exposure['Vega'] > 0:
                    st.info("Positive Vega: Benefiting from volatility increases")
                else:
                    st.warning("Negative Vega: Benefiting from volatility decreases")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Greeks Exposure Charts
            st.markdown("#### Greeks Across Strikes")
            
            # Gamma Exposure
            fig = px.line(
                df,
                x='strike',
                y=['call_gamma', 'put_gamma'],
                title='Gamma Exposure by Strike',
                labels={'value': 'Gamma', 'strike': 'Strike Price'},
                color_discrete_map={'call_gamma': '#3498db', 'put_gamma': '#e74c3c'}
            )
            fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            # Theta Exposure
            fig = px.line(
                df,
                x='strike',
                y=['call_theta', 'put_theta'],
                title='Theta Exposure by Strike',
                labels={'value': 'Theta', 'strike': 'Strike Price'},
                color_discrete_map={'call_theta': '#3498db', 'put_theta': '#e74c3c'}
            )
            fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            # Vega Exposure
            fig = px.line(
                df,
                x='strike',
                y=['call_vega', 'put_vega'],
                title='Vega Exposure by Strike',
                labels={'value': 'Vega', 'strike': 'Strike Price'},
                color_discrete_map={'call_vega': '#3498db', 'put_vega': '#e74c3c'}
            )
            fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not calculate Greeks exposure. Please check data availability.")

if __name__ == "__main__":
    main()
