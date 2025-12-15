# psx_advisor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PSX Stock Advisory Bot",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 PSX Stock Advisory Bot")
st.markdown("""
This tool analyzes PSX-listed companies based on:
- **Price range** you specify
- **Historical closing prices** (last available)
- **Simulated broker research analysis**
- **Company fundamentals** (simulated EPS, P/E, debt ratios)
- **Technical trend analysis** (SMA-based logic)
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Price range input
    st.subheader("Price Range Filter")
    min_price = st.number_input("Minimum Price (PKR)", min_value=1.0, max_value=5000.0, value=50.0, step=10.0)
    max_price = st.number_input("Maximum Price (PKR)", min_value=1.0, max_value=5000.0, value=500.0, step=10.0)
    
    # Validation
    if min_price >= max_price:
        st.error("Maximum price must be greater than minimum price!")
        st.stop()
    
    # Additional filters
    st.subheader("Additional Filters")
    min_market_cap = st.selectbox(
        "Minimum Market Cap",
        ["Any", "Small (<10B)", "Medium (10B-50B)", "Large (>50B)"]
    )
    
    st.markdown("---")
    st.caption("Note: This app uses historical data and simulated analysis for demonstration purposes.")

# Sample PSX stocks data
@st.cache_data
def get_psx_stocks():
    """Get sample PSX stocks with simulated data"""
    stocks = [
        # Banking Sector
        {"symbol": "HBL", "name": "Habib Bank Limited", "sector": "Banking", 
         "market_cap": 200.5, "volume": 1500000, "eps": 25.4, "pe_ratio": 4.2, 
         "debt_ratio": 0.65, "profitability": "High", "broker_rating": "Buy"},
        
        {"symbol": "UBL", "name": "United Bank Limited", "sector": "Banking", 
         "market_cap": 150.2, "volume": 1200000, "eps": 18.7, "pe_ratio": 3.8, 
         "debt_ratio": 0.60, "profitability": "High", "broker_rating": "Buy"},
        
        {"symbol": "MCB", "name": "MCB Bank Limited", "sector": "Banking", 
         "market_cap": 180.3, "volume": 900000, "eps": 32.1, "pe_ratio": 5.1, 
         "debt_ratio": 0.55, "profitability": "High", "broker_rating": "Strong Buy"},
        
        # Oil & Gas
        {"symbol": "PPL", "name": "Pakistan Petroleum Ltd", "sector": "Oil & Gas", 
         "market_cap": 120.4, "volume": 800000, "eps": 15.6, "pe_ratio": 6.2, 
         "debt_ratio": 0.40, "profitability": "Medium", "broker_rating": "Buy"},
        
        {"symbol": "OGDC", "name": "Oil & Gas Development Co.", "sector": "Oil & Gas", 
         "market_cap": 250.7, "volume": 2500000, "eps": 22.3, "pe_ratio": 4.8, 
         "debt_ratio": 0.35, "profitability": "High", "broker_rating": "Buy"},
        
        {"symbol": "POL", "name": "Pakistan Oilfields Ltd", "sector": "Oil & Gas", 
         "market_cap": 95.8, "volume": 600000, "eps": 12.8, "pe_ratio": 7.1, 
         "debt_ratio": 0.30, "profitability": "Medium", "broker_rating": "Hold"},
        
        # Cement
        {"symbol": "LUCK", "name": "Lucky Cement Limited", "sector": "Cement", 
         "market_cap": 140.6, "volume": 700000, "eps": 28.9, "pe_ratio": 8.4, 
         "debt_ratio": 0.50, "profitability": "High", "broker_rating": "Strong Buy"},
        
        {"symbol": "DGKC", "name": "D.G. Khan Cement Co.", "sector": "Cement", 
         "market_cap": 45.3, "volume": 400000, "eps": 8.7, "pe_ratio": 9.2, 
         "debt_ratio": 0.75, "profitability": "Low", "broker_rating": "Hold"},
        
        {"symbol": "FCCL", "name": "Fauji Cement Company Ltd", "sector": "Cement", 
         "market_cap": 38.9, "volume": 350000, "eps": 6.5, "pe_ratio": 11.3, 
         "debt_ratio": 0.80, "profitability": "Low", "broker_rating": "Sell"},
        
        # Fertilizer
        {"symbol": "EFERT", "name": "Engro Fertilizers Ltd", "sector": "Fertilizer", 
         "market_cap": 85.4, "volume": 850000, "eps": 14.2, "pe_ratio": 6.9, 
         "debt_ratio": 0.45, "profitability": "Medium", "broker_rating": "Buy"},
        
        {"symbol": "FATIMA", "name": "Fatima Fertilizer Co.", "sector": "Fertilizer", 
         "market_cap": 62.7, "volume": 500000, "eps": 10.8, "pe_ratio": 8.1, 
         "debt_ratio": 0.55, "profitability": "Medium", "broker_rating": "Hold"},
        
        # Power
        {"symbol": "HUBC", "name": "Hub Power Company Ltd", "sector": "Power", 
         "market_cap": 110.3, "volume": 950000, "eps": 9.4, "pe_ratio": 12.5, 
         "debt_ratio": 0.70, "profitability": "Medium", "broker_rating": "Buy"},
        
        {"symbol": "KAPCO", "name": "Kot Addu Power Co.", "sector": "Power", 
         "market_cap": 42.8, "volume": 300000, "eps": 5.6, "pe_ratio": 15.2, 
         "debt_ratio": 0.85, "profitability": "Low", "broker_rating": "Hold"},
        
        # Pharmaceuticals
        {"symbol": "SEARL", "name": "Searle Pakistan Ltd", "sector": "Pharmaceuticals", 
         "market_cap": 55.6, "volume": 450000, "eps": 7.9, "pe_ratio": 10.8, 
         "debt_ratio": 0.40, "profitability": "Medium", "broker_rating": "Buy"},
        
        {"symbol": "AGP", "name": "AGP Limited", "sector": "Pharmaceuticals", 
         "market_cap": 33.4, "volume": 280000, "eps": 4.3, "pe_ratio": 14.7, 
         "debt_ratio": 0.60, "profitability": "Low", "broker_rating": "Hold"},
        
        # Technology
        {"symbol": "NETSOL", "name": "NetSol Technologies Ltd", "sector": "Technology", 
         "market_cap": 18.9, "volume": 150000, "eps": 3.2, "pe_ratio": 22.4, 
         "debt_ratio": 0.25, "profitability": "Low", "broker_rating": "Buy"},
        
        {"symbol": "AVN", "name": "Avanceon Limited", "sector": "Technology", 
         "market_cap": 25.7, "volume": 120000, "eps": 5.1, "pe_ratio": 18.9, 
         "debt_ratio": 0.35, "profitability": "Medium", "broker_rating": "Strong Buy"},
        
        # Textile
        {"symbol": "NML", "name": "Nishat Mills Limited", "sector": "Textile", 
         "market_cap": 40.2, "volume": 320000, "eps": 6.8, "pe_ratio": 9.8, 
         "debt_ratio": 0.65, "profitability": "Low", "broker_rating": "Hold"},
        
        {"symbol": "GATM", "name": "Gul Ahmed Textile Mills", "sector": "Textile", 
         "market_cap": 28.5, "volume": 210000, "eps": 4.2, "pe_ratio": 13.5, 
         "debt_ratio": 0.75, "profitability": "Low", "broker_rating": "Sell"},
        
        # Food & Personal Care
        {"symbol": "NESTLE", "name": "Nestle Pakistan Ltd", "sector": "Food & Personal Care", 
         "market_cap": 320.8, "volume": 1800000, "eps": 45.6, "pe_ratio": 25.3, 
         "debt_ratio": 0.20, "profitability": "High", "broker_rating": "Strong Buy"},
        
        {"symbol": "ENGRO", "name": "Engro Corporation Ltd", "sector": "Conglomerate", 
         "market_cap": 180.5, "volume": 1400000, "eps": 21.4, "pe_ratio": 7.8, 
         "debt_ratio": 0.60, "profitability": "High", "broker_rating": "Buy"},
    ]
    
    return pd.DataFrame(stocks)

@st.cache_data
def generate_price_data(stock_df, min_price, max_price):
    """Generate simulated price data for stocks"""
    np.random.seed(42)  # For reproducibility
    
    results = []
    for _, stock in stock_df.iterrows():
        # Generate random price within user's range
        last_price = np.random.uniform(min_price, max_price)
        
        # Adjust based on fundamentals
        if stock['broker_rating'] == 'Strong Buy':
            last_price *= np.random.uniform(1.0, 1.3)
        elif stock['broker_rating'] == 'Buy':
            last_price *= np.random.uniform(0.9, 1.2)
        elif stock['broker_rating'] == 'Hold':
            last_price *= np.random.uniform(0.8, 1.1)
        else:  # Sell
            last_price *= np.random.uniform(0.7, 1.0)
        
        # Cap price within range
        last_price = max(min_price, min(max_price, last_price))
        
        # Generate historical prices for SMA calculation
        historical_prices = []
        for i in range(30):
            daily_change = np.random.uniform(-0.03, 0.03)
            last_price *= (1 + daily_change)
            historical_prices.append(last_price)
        
        # Calculate SMA trends
        sma_10 = np.mean(historical_prices[-10:])
        sma_20 = np.mean(historical_prices[-20:])
        
        # Determine trend
        if sma_10 > sma_20 and last_price > sma_10:
            trend = "Bullish"
        elif sma_10 < sma_20 and last_price < sma_10:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # Generate broker analysis
        brokers = ["AKD", "Arif Habib", "JS Global", "Foundation Securities"]
        broker_analysis = np.random.choice(brokers, 2, replace=False)
        
        # Recent news simulation
        news_items = [
            "Strong quarterly results announced",
            "New contract secured",
            "Dividend declaration expected",
            "Expansion plans underway",
            "Sector outlook positive",
            "Cost optimization measures showing results"
        ]
        
        results.append({
            'symbol': stock['symbol'],
            'name': stock['name'],
            'sector': stock['sector'],
            'last_price': round(last_price, 2),
            'sma_trend': trend,
            'broker_analysis': ", ".join(broker_analysis),
            'recent_news': np.random.choice(news_items),
            'market_cap': stock['market_cap'],
            'eps': stock['eps'],
            'pe_ratio': stock['pe_ratio'],
            'debt_ratio': stock['debt_ratio'],
            'profitability': stock['profitability'],
            'broker_rating': stock['broker_rating']
        })
    
    return pd.DataFrame(results)

def calculate_recommendations(filtered_stocks):
    """Calculate recommendations for different time horizons"""
    
    recommendations = {
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }
    
    # Target and Stop Loss parameters
    targets = {
        'short_term': {'target_pct': 0.06, 'sl_pct': -0.03},
        'medium_term': {'target_pct': 0.18, 'sl_pct': -0.07},
        'long_term': {'target_pct': 0.40, 'sl_pct': -0.12}
    }
    
    # Sort stocks by various criteria for different time horizons
    for _, stock in filtered_stocks.iterrows():
        # Calculate scores for different time horizons
        short_term_score = 0
        medium_term_score = 0
        long_term_score = 0
        
        # Factor weights
        if stock['sma_trend'] == 'Bullish':
            short_term_score += 30
            medium_term_score += 20
            long_term_score += 10
            
        if stock['broker_rating'] == 'Strong Buy':
            short_term_score += 25
            medium_term_score += 25
            long_term_score += 25
        elif stock['broker_rating'] == 'Buy':
            short_term_score += 20
            medium_term_score += 20
            long_term_score += 20
            
        if stock['profitability'] == 'High':
            short_term_score += 15
            medium_term_score += 20
            long_term_score += 25
        elif stock['profitability'] == 'Medium':
            short_term_score += 10
            medium_term_score += 15
            long_term_score += 20
            
        if stock['pe_ratio'] < 8:
            short_term_score += 15
            medium_term_score += 20
            long_term_score += 25
        elif stock['pe_ratio'] < 12:
            short_term_score += 10
            medium_term_score += 15
            long_term_score += 15
            
        if stock['debt_ratio'] < 0.5:
            short_term_score += 15
            medium_term_score += 20
            long_term_score += 25
        elif stock['debt_ratio'] < 0.7:
            short_term_score += 10
            medium_term_score += 10
            long_term_score += 10
        
        # Create stock recommendation object
        stock_rec = {
            'symbol': stock['symbol'],
            'name': stock['name'],
            'last_price': stock['last_price'],
            'sector': stock['sector'],
            'broker_rating': stock['broker_rating'],
            'sma_trend': stock['sma_trend'],
            'pe_ratio': stock['pe_ratio'],
            'profitability': stock['profitability']
        }
        
        # Add to appropriate lists based on scores
        if short_term_score >= 60:
            recommendations['short_term'].append((short_term_score, stock_rec))
        if medium_term_score >= 70:
            recommendations['medium_term'].append((medium_term_score, stock_rec))
        if long_term_score >= 80:
            recommendations['long_term'].append((long_term_score, stock_rec))
    
    # Sort and select top 5 for each category
    final_recommendations = {}
    for horizon in ['short_term', 'medium_term', 'long_term']:
        sorted_stocks = sorted(recommendations[horizon], key=lambda x: x[0], reverse=True)[:5]
        
        horizon_recommendations = []
        for score, stock in sorted_stocks:
            # Calculate target prices
            target_params = targets[horizon]
            buy_price = stock['last_price']
            target_price = round(buy_price * (1 + target_params['target_pct']), 2)
            stop_loss = round(buy_price * (1 + target_params['sl_pct']), 2)
            upside_pct = target_params['target_pct'] * 100
            
            # Generate explanation
            explanation = generate_explanation(stock, horizon)
            
            horizon_recommendations.append({
                'Company': f"{stock['name']} ({stock['symbol']})",
                'Sector': stock['sector'],
                'Last Price (PKR)': buy_price,
                'Buy Range (PKR)': f"{buy_price * 0.97:.2f} - {buy_price * 1.03:.2f}",
                'Target Price (PKR)': target_price,
                'Stop Loss (PKR)': stop_loss,
                'Upside (%)': f"{upside_pct:.1f}%",
                'Explanation': explanation,
                'Broker Rating': stock['broker_rating'],
                'Trend': stock['sma_trend']
            })
        
        final_recommendations[horizon] = pd.DataFrame(horizon_recommendations)
    
    return final_recommendations

def generate_explanation(stock, horizon):
    """Generate explanation for recommendation"""
    
    explanations = []
    
    # Base on broker rating
    if stock['broker_rating'] == 'Strong Buy':
        explanations.append("Strong buy ratings from multiple brokers")
    elif stock['broker_rating'] == 'Buy':
        explanations.append("Positive broker consensus")
    
    # Base on trend
    if stock['sma_trend'] == 'Bullish':
        explanations.append("Bullish technical trend")
    
    # Base on fundamentals
    if stock['pe_ratio'] < 10:
        explanations.append("Attractive valuation with low P/E ratio")
    elif stock['pe_ratio'] < 15:
        explanations.append("Reasonable valuation")
    
    if stock['profitability'] == 'High':
        explanations.append("Strong profitability metrics")
    
    # Horizon-specific factors
    if horizon == 'short_term':
        explanations.append("Favorable short-term momentum")
    elif horizon == 'medium_term':
        explanations.append("Solid medium-term growth prospects")
    else:  # long_term
        explanations.append("Excellent long-term fundamentals")
    
    return " | ".join(explanations)

def apply_market_cap_filter(stocks_df, filter_option):
    """Apply market cap filter"""
    if filter_option == "Any":
        return stocks_df
    
    filtered = stocks_df.copy()
    if filter_option == "Small (<10B)":
        filtered = filtered[filtered['market_cap'] < 10]
    elif filter_option == "Medium (10B-50B)":
        filtered = filtered[(filtered['market_cap'] >= 10) & (filtered['market_cap'] <= 50)]
    elif filter_option == "Large (>50B)":
        filtered = filtered[filtered['market_cap'] > 50]
    
    return filtered

# Main app logic
def main():
    # Get base stock data
    all_stocks = get_psx_stocks()
    
    # Apply market cap filter
    filtered_stocks = apply_market_cap_filter(all_stocks, min_market_cap)
    
    if len(filtered_stocks) == 0:
        st.warning("No stocks match your market cap filter. Please adjust your selection.")
        return
    
    # Generate price data
    with st.spinner("Analyzing stocks..."):
        price_data = generate_price_data(filtered_stocks, min_price, max_price)
    
    # Display filtered stocks count
    st.success(f"Found {len(price_data)} stocks within price range PKR {min_price:,.2f} - PKR {max_price:,.2f}")
    
    # Calculate recommendations
    recommendations = calculate_recommendations(price_data)
    
    # Display recommendations in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Short Term (1-3 months)", "📈 Medium Term (3-12 months)", "🚀 Long Term (1+ years)"])
    
    with tab1:
        st.subheader("Short Term Recommendations")
        st.caption("Target: +6% | Stop Loss: -3%")
        if len(recommendations['short_term']) > 0:
            st.dataframe(
                recommendations['short_term'],
                column_config={
                    "Last Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Target Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Stop Loss (PKR)": st.column_config.NumberColumn(format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Display as cards
            st.subheader("Top Short Term Pick")
            if len(recommendations['short_term']) > 0:
                top_st = recommendations['short_term'].iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Company", top_st['Company'].split('(')[0])
                    st.metric("Symbol", top_st['Company'].split('(')[1].rstrip(')'))
                with col2:
                    st.metric("Current Price", f"PKR {top_st['Last Price (PKR)']:.2f}")
                    st.metric("Target Price", f"PKR {top_st['Target Price (PKR)']:.2f}", 
                             delta=f"+{top_st['Upside (%)']}")
                with col3:
                    st.metric("Stop Loss", f"PKR {top_st['Stop Loss (PKR)']:.2f}")
                    st.metric("Broker Rating", top_st['Broker Rating'])
        else:
            st.info("No strong short-term recommendations found with current filters.")
    
    with tab2:
        st.subheader("Medium Term Recommendations")
        st.caption("Target: +18% | Stop Loss: -7%")
        if len(recommendations['medium_term']) > 0:
            st.dataframe(
                recommendations['medium_term'],
                column_config={
                    "Last Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Target Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Stop Loss (PKR)": st.column_config.NumberColumn(format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No strong medium-term recommendations found with current filters.")
    
    with tab3:
        st.subheader("Long Term Recommendations")
        st.caption("Target: +40% | Stop Loss: -12%")
        if len(recommendations['long_term']) > 0:
            st.dataframe(
                recommendations['long_term'],
                column_config={
                    "Last Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Target Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Stop Loss (PKR)": st.column_config.NumberColumn(format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No strong long-term recommendations found with current filters.")
    
    # Display analysis methodology
    with st.expander("📖 Analysis Methodology"):
        st.markdown("""
        ### How stocks are analyzed:
        
        1. **Price Range Filter**: Only stocks within your specified price range are considered
        2. **Broker Research Simulation**: Analysis based on simulated broker ratings (AKD, Arif Habib, JS Global, Foundation Securities)
        3. **Fundamental Analysis**: 
           - EPS (Earnings Per Share)
           - P/E Ratio (Price to Earnings)
           - Debt Ratios
           - Profitability Assessment
        4. **Technical Analysis**:
           - Simple Moving Average (SMA) trends
           - Price momentum
        5. **Sector & Company News**: Simulated recent developments
        
        ### Scoring System:
        - **Short Term**: Emphasizes technical trends and immediate catalysts
        - **Medium Term**: Balances technical and fundamental factors
        - **Long Term**: Focuses on strong fundamentals and growth prospects
        
        ### Important Notes:
        - This is a **demonstration tool** using simulated data
        - Always conduct your own research before investing
        - Past performance is not indicative of future results
        - Consider consulting with a financial advisor
        """)
    
    # Display raw data if needed
    with st.expander("🔍 View Raw Analysis Data"):
        st.dataframe(price_data[['symbol', 'name', 'sector', 'last_price', 'broker_rating', 
                                'sma_trend', 'pe_ratio', 'profitability']], 
                    use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
