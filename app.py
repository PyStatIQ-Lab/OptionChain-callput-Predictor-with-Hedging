import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
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
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .recommendation-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    .strike-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f1f8fe;
        border-left: 5px solid #3498db;
    }
    .warning-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #fff3e0;
        border-left: 5px solid #ffa000;
    }
</style>
""", unsafe_allow_html=True)

# Load portfolio data
@st.cache_data
def load_portfolio():
    portfolio = pd.read_csv("sample_portfolio (2).csv")
    return portfolio

# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio):
    portfolio['Investment'] = portfolio['Quantity'] * portfolio['Avg. Cost Price']
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    portfolio['P&L'] = portfolio['Current Value'] - portfolio['Investment']
    portfolio['P&L%'] = (portfolio['P&L'] / portfolio['Investment']) * 100
    return portfolio

# Fetch NIFTY spot price from Upstox API
@st.cache_data(ttl=60)
def get_nifty_spot():
    try:
        url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            return data['data']['lastPrice']
        else:
            st.error(f"Failed to fetch NIFTY spot price: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching NIFTY spot: {str(e)}")
        return None

# Fetch NIFTY options data from Upstox API
@st.cache_data(ttl=300)
def get_nifty_options(expiry_date="03-04-2025"):
    try:
        url = f"{BASE_URL}/strategy-chains?assetKey=NSE_INDEX|Nifty%2050&strategyChainType=PC_CHAIN&expiry={expiry_date}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            raw_data = response.json()
            return process_options_data(raw_data)
        else:
            st.error(f"Failed to fetch options data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return None

# Process raw options data
def process_options_data(raw_data):
    if not raw_data or 'data' not in raw_data:
        return None
    
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed_data = []
    
    for strike, data in strike_map.items():
        put_data = data.get('putOptionData', {})
        put_market = put_data.get('marketData', {})
        put_analytics = put_data.get('analytics', {})
        
        processed_data.append({
            'Strike': float(strike),
            'Type': 'PUT',
            'Premium': put_market.get('ltp', 0),
            'Bid': put_market.get('bidPrice', 0),
            'Ask': put_market.get('askPrice', 0),
            'IV': put_analytics.get('iv', 0),
            'Delta': put_analytics.get('delta', 0),
            'Gamma': put_analytics.get('gamma', 0),
            'Theta': put_analytics.get('theta', 0),
            'Vega': put_analytics.get('vega', 0),
            'OI': put_market.get('oi', 0),
            'OI Change': put_market.get('oi', 0) - put_market.get('prevOi', 0),
            'Volume': put_market.get('volume', 0)
        })
    
    return pd.DataFrame(processed_data)

# Calculate portfolio beta (simplified)
def calculate_portfolio_beta(portfolio):
    return 1.2  # Default beta, replace with actual calculation

# Calculate hedge requirements
def calculate_hedge(portfolio_value, portfolio_beta, nifty_spot, lot_size=75):
    if nifty_spot is None or nifty_spot == 0:
        return 0
    hedge_ratio = portfolio_beta * (portfolio_value / (nifty_spot * lot_size))
    return hedge_ratio

# Main app
def main():
    st.markdown("<div class='header'><h1>📊 Live Portfolio Hedging Calculator</h1></div>", unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Hedging Parameters")
        expiry_date = st.date_input(
            "Options Expiry Date",
            datetime.strptime("03-04-2025", "%d-%m-%Y")
        ).strftime("%d-%m-%Y")
        
        st.markdown("---")
        st.markdown("**Portfolio Beta Settings**")
        use_custom_beta = st.checkbox("Use custom beta instead of calculated")
        custom_beta = st.number_input("Custom Portfolio Beta", value=1.2, min_value=0.1, max_value=3.0, step=0.1)
        
        st.markdown("---")
        st.markdown("**Hedge Intensity**")
        hedge_percentage = st.slider("Percentage of portfolio to hedge", 0, 100, 100)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This tool calculates protective put options for portfolio hedging using live market data.")
    
    # Load and process portfolio
    portfolio = load_portfolio()
    portfolio = calculate_portfolio_metrics(portfolio)
    total_investment = portfolio['Investment'].sum()
    total_current = portfolio['Current Value'].sum()
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment) * 100
    
    # Fetch live market data
    with st.spinner("Fetching live market data..."):
        nifty_spot = get_nifty_spot()
        options_data = get_nifty_options(expiry_date)
    
    # Display portfolio summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Investment**")
        st.markdown(f"<h3>₹{total_investment:,.2f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Current Value**")
        st.markdown(f"<h3>₹{total_current:,.2f}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total P&L**")
        pnl_color = "positive" if total_pnl >= 0 else "negative"
        st.markdown(f"<h3 class='{pnl_color}'>₹{total_pnl:,.2f} ({total_pnl_pct:.2f}%)</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Portfolio details
    st.markdown("### Portfolio Holdings")
    st.dataframe(portfolio.style.format({
        'Avg. Cost Price': '{:,.2f}',
        'Current Price': '{:,.2f}',
        'Investment': '{:,.2f}',
        'Current Value': '{:,.2f}',
        'P&L': '{:,.2f}',
        'P&L%': '{:.2f}%'
    }), use_container_width=True)
    
    # Hedging calculation
    st.markdown("### Hedging Analysis")
    
    if nifty_spot is None or options_data is None:
        st.error("Failed to load market data. Please try again later.")
        return
    
    st.markdown(f"**NIFTY Spot Price:** {nifty_spot:,.2f}")
    
    # Calculate portfolio beta
    portfolio_beta = custom_beta if use_custom_beta else calculate_portfolio_beta(portfolio)
    st.markdown(f"**Portfolio Beta:** {portfolio_beta:.2f}")
    
    # Calculate hedge ratio
    hedge_ratio = calculate_hedge(total_current, portfolio_beta, nifty_spot)
    adjusted_hedge_ratio = hedge_ratio * (hedge_percentage / 100)
    
    st.markdown(f"**Hedge Ratio:** {hedge_ratio:.2f} lots ({(hedge_ratio * 75):.0f} units)")
    st.markdown(f"**Adjusted Hedge Ratio ({hedge_percentage}% of portfolio):** {adjusted_hedge_ratio:.2f} lots ({(adjusted_hedge_ratio * 75):.0f} units)")
    
    # Round to nearest whole lot
    recommended_lots = round(adjusted_hedge_ratio)
    
    if recommended_lots == 0:
        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        st.markdown("**Portfolio size is too small for effective hedging with NIFTY options (minimum 1 lot required)**")
        st.markdown("Consider alternative hedging strategies or increasing portfolio size")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Filter suitable put options
    suitable_puts = options_data[(options_data['Premium'] > 0) & 
                               (options_data['IV'] > 0)].sort_values('Strike')
    
    if suitable_puts.empty:
        st.error("No valid put options found for hedging")
        return
    
    # Determine strikes to recommend
    atm_strike = min(suitable_puts['Strike'], key=lambda x: abs(x - nifty_spot))
    otm2_strike = min(suitable_puts['Strike'], key=lambda x: abs(x - (nifty_spot * 0.98)))
    otm5_strike = min(suitable_puts['Strike'], key=lambda x: abs(x - (nifty_spot * 0.95)))
    
    # Get the options data for these strikes
    recommended_options = []
    for strike in [atm_strike, otm2_strike, otm5_strike]:
        option = suitable_puts[suitable_puts['Strike'] == strike]
        if not option.empty:
            recommended_options.append(option.iloc[0])
    
    # Display hedging recommendations
    st.markdown("### Recommended Put Options for Hedging")
    
    if not recommended_options:
        st.warning("Could not find suitable options for hedging at calculated strikes")
    else:
        for opt in recommended_options:
            protection_level = (nifty_spot - opt['Strike']) / nifty_spot * 100
            cost = opt['Premium'] * 75 * recommended_lots
            cost_pct = (cost / total_current) * 100
            
            st.markdown(f"""
            <div class='strike-card'>
                <h4>{opt['Strike']:.0f} PUT ({'ATM' if opt['Strike'] == atm_strike else 'OTM'})</h4>
                <p>
                    <b>Premium:</b> ₹{opt['Premium']:.2f} | <b>IV:</b> {opt['IV']:.1f}% | <b>Delta:</b> {opt['Delta']:.2f}<br>
                    <b>Protection Level:</b> {protection_level:.1f}% below current spot<br>
                    <b>Total Cost:</b> ₹{cost:,.2f} ({cost_pct:.2f}% of portfolio)<br>
                    <b>Lots Recommended:</b> {recommended_lots} ({(recommended_lots * 75):.0f} units)<br>
                    <b>Open Interest:</b> {opt['OI']:,} (Δ: {opt['OI Change']:,})
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional analysis
    st.markdown("### Additional Analysis")
    
    # Moneyness calculation
    suitable_puts['Moneyness'] = suitable_puts['Strike'].apply(
        lambda x: 'ITM' if x > nifty_spot else ('ATM' if x == nifty_spot else 'OTM'))
    
    # IV vs Strike plot
    fig = px.line(suitable_puts, x='Strike', y='IV', color='Moneyness',
                  title='Put Option IV by Strike Price',
                  labels={'Strike': 'Strike Price', 'IV': 'Implied Volatility (%)'},
                  color_discrete_map={'ITM': '#e74c3c', 'ATM': '#3498db', 'OTM': '#2ecc71'})
    fig.add_vline(x=nifty_spot, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost vs Protection analysis
    suitable_puts['Protection%'] = (nifty_spot - suitable_puts['Strike']) / nifty_spot * 100
    suitable_puts['TotalCost'] = suitable_puts['Premium'] * 75 * recommended_lots
    suitable_puts['Cost%'] = (suitable_puts['TotalCost'] / total_current) * 100
    
    fig = px.scatter(suitable_puts, x='Protection%', y='Cost%', color='Moneyness',
                     hover_data=['Strike', 'IV', 'Delta'], 
                     title='Protection Level vs Cost',
                     labels={'Protection%': 'Protection Level (%)', 'Cost%': 'Cost (% of portfolio)'},
                     color_discrete_map={'ITM': '#e74c3c', 'ATM': '#3498db', 'OTM': '#2ecc71'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Final recommendation
    st.markdown("### Final Recommendation")
    if recommended_lots > 0 and len(recommended_options) > 0:
        final_rec = recommended_options[len(recommended_options)//2]
        protection_level = (nifty_spot - final_rec['Strike']) / nifty_spot * 100
        cost = final_rec['Premium'] * 75 * recommended_lots
        
        st.markdown(f"""
        <div class='recommendation-card'>
            <h3>Recommended Hedge</h3>
            <p>
                <b>Action:</b> BUY {recommended_lots} lots of {final_rec['Strike']:.0f} PUT<br>
                <b>Total Quantity:</b> {(recommended_lots * 75):.0f} units<br>
                <b>Premium:</b> ₹{final_rec['Premium']:.2f} per unit<br>
                <b>Total Cost:</b> ₹{cost:,.2f} ({(cost/total_current*100):.2f}% of portfolio)<br>
                <b>Protection Level:</b> {protection_level:.1f}% below current spot<br>
                <b>Implied Volatility:</b> {final_rec['IV']:.1f}%<br>
                <b>Delta:</b> {final_rec['Delta']:.2f}<br>
                <b>Expiration:</b> {expiry_date}<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Implementation Notes:**
        - Place the order as a LIMIT order between ₹{final_rec['Bid']:.2f}-₹{final_rec['Ask']:.2f}
        - This hedge will protect against market declines below ₹{final_rec['Strike']:.0f}
        - The cost represents the maximum potential loss on the hedge
        - Monitor the position and adjust as market conditions change
        - Consider rolling the hedge as expiration approaches if still needed
        """)
    else:
        st.info("No specific recommendation generated due to portfolio size or data limitations")

if __name__ == "__main__":
    main()
