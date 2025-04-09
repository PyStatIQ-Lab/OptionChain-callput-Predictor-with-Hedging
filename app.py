import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress

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

# Get NIFTY spot price (mock function - in reality you'd fetch this from API)
def get_nifty_spot():
    return 22000  # Example value - replace with actual API call

# Get NIFTY options data (mock function - replace with actual API)
def get_nifty_options():
    # Mock data - in reality you'd fetch this from options API
    strikes = np.arange(21500, 22500, 100)
    data = []
    for strike in strikes:
        moneyness = "ITM" if strike > 22000 else ("OTM" if strike < 22000 else "ATM")
        iv = 15 + abs(strike - 22000)/100  # Mock IV calculation
        premium = max(50, (abs(strike - 22000)/10)  # Mock premium calculation
        data.append({
            'Strike': strike,
            'Type': 'PUT',
            'Premium': premium,
            'IV': iv,
            'Moneyness': moneyness,
            'Delta': -0.05 if strike < 22000 else -0.5 if strike == 22000 else -0.95
        })
    return pd.DataFrame(data)

# Calculate portfolio beta (simplified - in reality you'd use historical data)
def calculate_portfolio_beta(portfolio):
    # Mock beta calculation - in reality you'd use historical returns
    # For this example, we'll assume an average beta of 1.2 for the portfolio
    return 1.2

# Calculate hedge requirements
def calculate_hedge(portfolio_value, portfolio_beta, nifty_spot, lot_size=75):
    # Calculate hedge ratio
    hedge_ratio = portfolio_beta * (portfolio_value / (nifty_spot * lot_size))
    return hedge_ratio

# Main app
def main():
    st.markdown("<div class='header'><h1>📊 Portfolio Hedging Calculator</h1></div>", unsafe_allow_html=True)
    
    # Load and process portfolio
    portfolio = load_portfolio()
    portfolio = calculate_portfolio_metrics(portfolio)
    total_investment = portfolio['Investment'].sum()
    total_current = portfolio['Current Value'].sum()
    total_pnl = total_current - total_investment
    total_pnl_pct = (total_pnl / total_investment) * 100
    
    # Get market data
    nifty_spot = get_nifty_spot()
    options_data = get_nifty_options()
    
    # Calculate portfolio beta (simplified)
    portfolio_beta = calculate_portfolio_beta(portfolio)
    
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
    st.markdown(f"**Portfolio Beta (Estimated):** {portfolio_beta:.2f}")
    st.markdown(f"**NIFTY Spot Price:** {nifty_spot:,.2f}")
    
    hedge_ratio = calculate_hedge(total_current, portfolio_beta, nifty_spot)
    st.markdown(f"**Hedge Ratio:** {hedge_ratio:.2f} lots ({(hedge_ratio * 75):.0f} units)")
    
    # Round to nearest whole lot
    recommended_lots = round(hedge_ratio)
    st.markdown(f"**Recommended Lots for Full Hedge:** {recommended_lots} ({(recommended_lots * 75):.0f} units)")
    
    # Filter suitable put options
    suitable_puts = options_data[options_data['Type'] == 'PUT'].sort_values('Strike')
    
    # Display hedging recommendations
    st.markdown("### Recommended Put Options for Hedging")
    
    if recommended_lots == 0:
        st.warning("Portfolio size is too small for effective hedging with NIFTY options (minimum 1 lot required)")
    else:
        # We'll recommend 3 strikes: ATM, 2% OTM, and 5% OTM
        atm_strike = nifty_spot - (nifty_spot % 100)  # Round to nearest 100
        otm2_strike = atm_strike - (0.02 * nifty_spot // 100) * 100
        otm5_strike = atm_strike - (0.05 * nifty_spot // 100) * 100
        
        # Get the options data for these strikes
        recommended_options = []
        for strike in [atm_strike, otm2_strike, otm5_strike]:
            option = suitable_puts[suitable_puts['Strike'] == strike]
            if not option.empty:
                recommended_options.append(option.iloc[0])
        
        if recommended_options:
            for opt in recommended_options:
                protection_level = (nifty_spot - opt['Strike']) / nifty_spot * 100
                cost = opt['Premium'] * 75 * recommended_lots
                cost_pct = (cost / total_current) * 100
                
                st.markdown(f"""
                <div class='strike-card'>
                    <h4>{opt['Strike']:.0f} {opt['Type']} ({(opt['Moneyness'])})</h4>
                    <p>
                        <b>Premium:</b> ₹{opt['Premium']:.2f} | <b>IV:</b> {opt['IV']:.1f}% | <b>Delta:</b> {opt['Delta']:.2f}<br>
                        <b>Protection Level:</b> {protection_level:.1f}% below current spot<br>
                        <b>Total Cost:</b> ₹{cost:,.2f} ({cost_pct:.2f}% of portfolio)<br>
                        <b>Lots Recommended:</b> {recommended_lots} ({(recommended_lots * 75):.0f} units)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Could not find suitable options for hedging at calculated strikes")
    
    # Additional analysis
    st.markdown("### Additional Analysis")
    
    # IV vs Strike plot
    fig = px.line(suitable_puts, x='Strike', y='IV', 
                  title='Put Option IV by Strike Price',
                  labels={'Strike': 'Strike Price', 'IV': 'Implied Volatility (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost vs Protection analysis
    suitable_puts['Protection%'] = (nifty_spot - suitable_puts['Strike']) / nifty_spot * 100
    suitable_puts['TotalCost'] = suitable_puts['Premium'] * 75 * recommended_lots
    suitable_puts['Cost%'] = (suitable_puts['TotalCost'] / total_current) * 100
    
    fig = px.scatter(suitable_puts, x='Protection%', y='Cost%', 
                     hover_data=['Strike', 'IV'], 
                     title='Protection Level vs Cost',
                     labels={'Protection%': 'Protection Level (%)', 'Cost%': 'Cost (% of portfolio)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Final recommendation
    st.markdown("### Final Recommendation")
    if recommended_lots > 0 and len(recommended_options) > 0:
        # Select the middle option (balanced between cost and protection)
        final_rec = recommended_options[len(recommended_options)//2]
        protection_level = (nifty_spot - final_rec['Strike']) / nifty_spot * 100
        cost = final_rec['Premium'] * 75 * recommended_lots
        
        st.markdown(f"""
        <div class='recommendation-card'>
            <h3>Recommended Hedge</h3>
            <p>
                <b>Action:</b> BUY {recommended_lots} lots of {final_rec['Strike']:.0f} PUT<br>
                <b>Total Quantity:</b> {(recommended_lots * 75):.0f} units<br>
                <b>Protection Level:</b> {protection_level:.1f}% below current spot<br>
                <b>Estimated Cost:</b> ₹{cost:,.2f} ({(cost/total_current*100):.2f}% of portfolio)<br>
                <b>Expiration:</b> Next monthly expiry (adjust as needed)<br>
                <b>Rationale:</b> Provides balanced protection at reasonable cost
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Notes:**
        - This hedge will protect against market declines below the strike price
        - The cost represents the maximum potential loss on the hedge
        - Adjust quantity based on your risk tolerance (you may hedge partially)
        - Monitor and adjust as market conditions change
        """)
    else:
        st.info("No specific recommendation generated due to portfolio size or data limitations")

if __name__ == "__main__":
    main()
