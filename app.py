import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import yfinance as yf
from io import BytesIO

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
    .portfolio-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        background-color: #f0f7ff;
        border-left: 5px solid #1e88e5;
    }
    .hedge-recommendation {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #fff8e1;
        border-left: 5px solid #ffb300;
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

# Generate trade recommendations
def generate_trade_recommendations(df, spot_price):
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
        recommendations.append({
            'type': 'BUY CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'risk_reward': f"{row['call_risk_reward']:.1f}:1",
            'reason': "Low spread, OI buildup, good risk/reward"
        })
    
    # Find best puts to buy (low premium ratio, high OI change, good risk/reward)
    best_puts = df[(df['put_moneyness'] == 'OTM') & 
                  (df['put_premium_ratio'] < 0.1) &
                  (df['put_oi_change'] > 0)].sort_values(
        by=['put_premium_ratio', 'put_oi_change'], 
        ascending=[True, False]
    ).head(3)
    
    for _, row in best_puts.iterrows():
        recommendations.append({
            'type': 'BUY PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'risk_reward': f"{row['put_risk_reward']:.1f}:1",
            'reason': "Low spread, OI buildup, good risk/reward"
        })
    
    # Find best calls to sell (high premium ratio, decreasing OI)
    best_sell_calls = df[(df['call_moneyness'] == 'ITM') & 
                        (df['call_premium_ratio'] > 0.15) &
                        (df['call_oi_change'] < 0)].sort_values(
        by=['call_premium_ratio', 'call_oi_change'], 
        ascending=[False, True]
    ).head(2)
    
    for _, row in best_sell_calls.iterrows():
        recommendations.append({
            'type': 'SELL CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'risk_reward': f"{1/row['call_risk_reward']:.1f}:1",
            'reason': "High spread, OI unwinding, favorable risk"
        })
    
    # Find best puts to sell (high premium ratio, decreasing OI)
    best_sell_puts = df[(df['put_moneyness'] == 'ITM') & 
                       (df['put_premium_ratio'] > 0.15) &
                       (df['put_oi_change'] < 0)].sort_values(
        by=['put_premium_ratio', 'put_oi_change'], 
        ascending=[False, True]
    ).head(2)
    
    for _, row in best_sell_puts.iterrows():
        recommendations.append({
            'type': 'SELL PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'risk_reward': f"{1/row['put_risk_reward']:.1f}:1",
            'reason': "High spread, OI unwinding, favorable risk"
        })
    
    return recommendations

# Analyze portfolio and generate hedging recommendations
def analyze_portfolio(portfolio_df, nifty_price, options_data):
    if portfolio_df.empty:
        return None
    
    # Add .NS suffix for NSE stocks in yfinance
    portfolio_df['yf_symbol'] = portfolio_df['Symbol'].apply(lambda x: x+'.NS' if '.' not in x else x)
    
    # Get current prices
    current_prices = yf.download(list(portfolio_df['yf_symbol']), period='1d')['Close'].iloc[-1]
    
    # Calculate portfolio metrics
    portfolio_df['Current Price'] = portfolio_df['yf_symbol'].map(current_prices)
    portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Current Price']
    portfolio_df['Cost Value'] = portfolio_df['Quantity'] * portfolio_df['Avg. Cost Price']
    portfolio_df['P&L'] = portfolio_df['Current Value'] - portfolio_df['Cost Value']
    portfolio_df['P&L %'] = (portfolio_df['Current Value'] / portfolio_df['Cost Value'] - 1) * 100
    
    total_investment = portfolio_df['Cost Value'].sum()
    total_value = portfolio_df['Current Value'].sum()
    total_pnl = total_value - total_investment
    total_pnl_pct = (total_value / total_investment - 1) * 100
    
    # Calculate beta-weighted delta (simplified)
    # In a real scenario, we'd need to calculate each stock's beta to Nifty
    # Here we'll assume an average beta of 1 for simplicity
    portfolio_delta = total_value / nifty_price  # Approximate Nifty points equivalent
    
    # Generate hedging recommendations
    hedging_recommendations = []
    
    if total_pnl > 0:
        # Portfolio is in profit - consider protective puts
        put_strikes = options_data[options_data['put_moneyness'] == 'OTM'].sort_values('put_iv').head(3)
        
        for _, row in put_strikes.iterrows():
            hedge_ratio = portfolio_delta * abs(row['put_delta'])
            contracts_needed = int(hedge_ratio / (nifty_price * 0.01))  # Nifty lot size assumed as 50
            
            if contracts_needed > 0:
                hedging_recommendations.append({
                    'type': 'Protective Put',
                    'strike': row['strike'],
                    'premium': row['put_ltp'],
                    'iv': row['put_iv'],
                    'contracts': contracts_needed,
                    'cost': contracts_needed * row['put_ltp'] * 50,  # Assuming 50 is lot size
                    'reason': f"Protect {total_pnl_pct:.1f}% portfolio gains with OTM puts"
                })
    else:
        # Portfolio is in loss - consider call spreads to finance recovery
        call_strikes = options_data[options_data['call_moneyness'] == 'OTM'].sort_values('call_iv').head(3)
        
        if len(call_strikes) >= 2:
            sell_strike = call_strikes.iloc[0]['strike']
            buy_strike = call_strikes.iloc[1]['strike']
            premium_received = call_strikes.iloc[0]['call_ltp']
            premium_paid = call_strikes.iloc[1]['call_ltp']
            net_credit = premium_received - premium_paid
            
            contracts_needed = int(abs(portfolio_delta) / (nifty_price * 0.01))
            
            if contracts_needed > 0 and net_credit > 0:
                hedging_recommendations.append({
                    'type': 'Bear Call Spread',
                    'sell_strike': sell_strike,
                    'buy_strike': buy_strike,
                    'net_credit': net_credit,
                    'contracts': contracts_needed,
                    'max_gain': net_credit * 50 * contracts_needed,
                    'max_loss': (buy_strike - sell_strike - net_credit) * 50 * contracts_needed,
                    'reason': f"Finance portfolio recovery with credit spread (net credit: {net_credit:.1f} points)"
                })
    
    # Add Nifty delta hedge recommendation
    nifty_contracts = int(abs(portfolio_delta) / (nifty_price * 0.01))
    if nifty_contracts > 0:
        direction = "SELL" if portfolio_delta > 0 else "BUY"
        hedging_recommendations.append({
            'type': f'Nifty Futures {direction}',
            'contracts': nifty_contracts,
            'notional': nifty_contracts * nifty_price * 50,
            'reason': f"Direct delta hedge ({portfolio_delta:.1f} Nifty points equivalent)"
        })
    
    return {
        'portfolio_metrics': {
            'total_investment': total_investment,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'portfolio_delta': portfolio_delta
        },
        'portfolio_details': portfolio_df,
        'hedging_recommendations': hedging_recommendations
    }

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
        
        st.markdown("---")
        st.markdown(f"**Current Nifty Spot Price: {spot_price:,.2f}**")
        
        st.markdown("---")
        st.markdown("**Analysis Settings**")
        volume_threshold = st.number_input("High Volume Threshold", value=5000000)
        oi_change_threshold = st.number_input("Significant OI Change", value=1000000)
        
        st.markdown("---")
        st.markdown("**Portfolio Upload**")
        uploaded_file = st.file_uploader("Upload Portfolio (CSV/Excel)", type=['csv', 'xlsx'])
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This dashboard provides real-time options chain analysis and portfolio hedging recommendations.")
    
    # Portfolio analysis section
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                portfolio_df = pd.read_csv(uploaded_file)
            else:
                portfolio_df = pd.read_excel(uploaded_file)
            
            # Check required columns
            required_cols = ['Symbol', 'Quantity', 'Avg. Cost Price']
            if not all(col in portfolio_df.columns for col in required_cols):
                st.error("Uploaded file must contain columns: Symbol, Quantity, Avg. Cost Price")
            else:
                st.markdown("### Portfolio Analysis")
                
                with st.spinner("Analyzing portfolio and generating hedging recommendations..."):
                    # Fetch options data for hedging
                    raw_data = fetch_options_data(asset_key, expiry_date)
                    if raw_data is None:
                        st.error("Failed to fetch options data for hedging analysis")
                        return
                    
                    df = process_options_data(raw_data, spot_price)
                    if df is None or df.empty:
                        st.error("No options data available for hedging analysis")
                        return
                    
                    # Analyze portfolio
                    analysis_result = analyze_portfolio(portfolio_df, spot_price, df)
                    
                    if analysis_result:
                        # Display portfolio metrics
                        pm = analysis_result['portfolio_metrics']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown("**Total Investment**")
                            st.markdown(f"<h3>₹{pm['total_investment']:,.2f}</h3>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown("**Current Value**")
                            st.markdown(f"<h3>₹{pm['total_value']:,.2f}</h3>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown("**Total P&L**")
                            pnl_color = "positive" if pm['total_pnl'] >= 0 else "negative"
                            st.markdown(f"<h3 class='{pnl_color}'>₹{pm['total_pnl']:,.2f}</h3>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown("**P&L %**")
                            pnl_color = "positive" if pm['total_pnl_pct'] >= 0 else "negative"
                            st.markdown(f"<h3 class='{pnl_color}'>{pm['total_pnl_pct']:.2f}%</h3>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display portfolio details
                        st.markdown("#### Portfolio Holdings")
                        st.dataframe(
                            analysis_result['portfolio_details'].style.format({
                                'Avg. Cost Price': '{:.2f}',
                                'Current Price': '{:.2f}',
                                'Current Value': '{:,.2f}',
                                'Cost Value': '{:,.2f}',
                                'P&L': '{:,.2f}',
                                'P&L %': '{:.2f}%'
                            }),
                            use_container_width=True
                        )
                        
                        # Display hedging recommendations
                        st.markdown("#### Hedging Recommendations")
                        if analysis_result['hedging_recommendations']:
                            for rec in analysis_result['hedging_recommendations']:
                                if rec['type'] in ['Protective Put', 'Bear Call Spread']:
                                    st.markdown(f"""
                                        <div class='hedge-recommendation'>
                                            <h4>{rec['type']}</h4>
                                            <p>
                                                {rec['reason']}<br>
                                                {f"Strike: {rec['strike']}" if 'strike' in rec else ""}
                                                {f"Sell Strike: {rec['sell_strike']}, Buy Strike: {rec['buy_strike']}" if 'sell_strike' in rec else ""}<br>
                                                Contracts: {rec['contracts']} | 
                                                {f"Cost: ₹{rec['cost']:,.2f}" if 'cost' in rec else f"Net Credit: {rec['net_credit']:.2f} points"}<br>
                                                {f"Max Gain: ₹{rec['max_gain']:,.2f}" if 'max_gain' in rec else ""}
                                                {f"Max Loss: ₹{rec['max_loss']:,.2f}" if 'max_loss' in rec else ""}
                                            </p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                        <div class='hedge-recommendation'>
                                            <h4>{rec['type']}</h4>
                                            <p>
                                                {rec['reason']}<br>
                                                Contracts: {rec['contracts']} | 
                                                Notional: ₹{rec['notional']:,.2f}
                                            </p>
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No hedging recommendations based on current portfolio and market conditions")
        except Exception as e:
            st.error(f"Error processing portfolio file: {str(e)}")
    
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
    
    # Get top strikes
    top_strikes = get_top_strikes(df, spot_price)
    
    # Default strike selection (ATM)
    atm_strike = df.iloc[(df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]
    
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
        
        # PCR gauge
        pcr = df[df['strike'] == selected_strike]['pcr'].values[0]
        fig = px.bar(x=[pcr], range_x=[0, 2], title=f"Put-Call Ratio: {pcr:.2f}")
        fig.update_layout(
            xaxis_title="PCR",
            yaxis_visible=False,
            height=150,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.add_vline(x=0.7, line_dash="dot", line_color="green")
        fig.add_vline(x=1.3, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Trade Recommendations
    st.markdown("### Trade Recommendations")
    recommendations = generate_trade_recommendations(df, spot_price)
    
    if recommendations:
        for rec in recommendations:
            is_sell = 'SELL' in rec['type']
            st.markdown(f"""
                <div class='trade-recommendation{' sell' if is_sell else ''}'>
                    <h4>{rec['type']} @ {rec['strike']:.0f}</h4>
                    <p>
                        Premium: {rec['premium']:.2f} | IV: {rec['iv']:.1f}%<br>
                        OI Change: {rec['oi_change']:,} | Risk/Reward: {rec['risk_reward']}<br>
                        <b>Reason:</b> {rec['reason']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No strong trade recommendations based on current market conditions")
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Strike Analysis", "OI/Volume Trends", "Advanced Analytics"])
    
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

if __name__ == "__main__":
    main()
