# web_app.py - Streamlit Web Interface for Portfolio Beta
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import io

# Configuration
END_DATE = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=365)
YAHOO_INDEX_TICKER = "^NSEI"

def download_yahoo_adjclose(ticker, start, end):
    try:
        data = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            threads=False
        )
        if data is None or data.empty:
            return None
        return data.get("Adj Close") or data.get("Close")
    except Exception as e:
        st.warning(f"Failed to fetch {ticker}: {e}")
        return None

def compute_beta(stock_series, index_series):
    df = pd.concat([stock_series, index_series], axis=1, join="inner").dropna()
    df = df.sort_index()
    if df.shape[0] < 30:
        return np.nan
    returns = df.pct_change().dropna()
    if returns.shape[0] < 20:
        return np.nan
    cov = returns.cov().iloc[0,1]
    var_index = returns.iloc[:,1].var()
    return cov/var_index if var_index != 0 else np.nan

def get_stock_beta(symbol, index_series):
    yf_ticker = f"{symbol}.NS"
    series = download_yahoo_adjclose(yf_ticker, START_DATE, END_DATE)
    if series is None or series.empty:
        return symbol, np.nan
    beta = compute_beta(series, index_series)
    return symbol, beta

# Streamlit Web Interface
st.set_page_config(page_title="Portfolio Beta Calculator", layout="wide")
st.title("📊 Portfolio Beta Calculator")
st.write("Calculate your portfolio's weighted beta and hedging costs")

# Input Section
st.header("1. Portfolio Input")

input_method = st.radio("Choose input method:", ["Manual Entry", "CSV Upload"])

portfolio_data = None

if input_method == "Manual Entry":
    st.subheader("Enter Stocks Manually")
    
    num_stocks = st.number_input("Number of stocks:", min_value=1, max_value=20, value=3)
    
    stocks = []
    for i in range(num_stocks):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input(f"Stock Symbol {i+1}", value="RELIANCE", key=f"sym_{i}")
        with col2:
            amount = st.number_input(f"Investment Amount (₹) {i+1}", min_value=0, value=10000, key=f"amt_{i}")
        stocks.append({"SYMBOL": symbol.upper().replace('.NS', ''), "AMOUNT": amount})
    
    if stocks:
        portfolio_data = pd.DataFrame(stocks)
        st.write("Your Portfolio:")
        st.dataframe(portfolio_data)

else:  # CSV Upload
    st.subheader("Upload Portfolio CSV")
    st.info("Your CSV should have columns: SYMBOL, AMOUNT")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        portfolio_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio:")
        st.dataframe(portfolio_data)

# Calculation Section
if portfolio_data is not None:
    st.header("2. Calculate Beta")
    
    if st.button("🚀 Calculate Portfolio Beta", type="primary"):
        if "AMOUNT" not in portfolio_data.columns:
            st.error("❌ Portfolio must have 'AMOUNT' column")
        else:
            with st.spinner("📊 Calculating betas... This may take 30-60 seconds"):
                # Calculate weights
                total_amount = portfolio_data["AMOUNT"].sum()
                portfolio_data["WEIGHT"] = portfolio_data["AMOUNT"] / total_amount

                # Download index data
                st.info("📡 Downloading Nifty 50 index data...")
                index_series = download_yahoo_adjclose(YAHOO_INDEX_TICKER, START_DATE, END_DATE)
                
                if index_series is None or index_series.empty:
                    st.error("❌ Failed to download index data. Please check your internet connection and try again.")
                else:
                    index_series = index_series.dropna().sort_index()

                    # Calculate betas with progress
                    betas = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, sym in enumerate(portfolio_data["SYMBOL"]):
                        status_text.text(f"Calculating beta for {sym}... ({i+1}/{len(portfolio_data['SYMBOL'])})")
                        symbol, beta = get_stock_beta(sym, index_series)
                        betas.append((symbol, beta))
                        progress_bar.progress((i + 1) / len(portfolio_data["SYMBOL"]))
                    
                    status_text.text("✅ All calculations complete!")
                    
                    # Merge results
                    beta_df = pd.DataFrame(betas, columns=["SYMBOL", "BETA"])
                    merged = pd.merge(portfolio_data, beta_df, on="SYMBOL", how="left")
                    merged["WEIGHTED_BETA"] = merged["WEIGHT"] * merged["BETA"]
                    portfolio_beta = merged["WEIGHTED_BETA"].sum()

                    # Display Results
                    st.header("3. Results")
                    st.success("✅ Calculation Complete!")
                    
                    # Key Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Portfolio Value", f"₹{total_amount:,.2f}")
                    with col2:
                        st.metric("Weighted Average Beta", f"{portfolio_beta:.4f}")
                    with col3:
                        # Simple hedging cost estimation
                        hedging_cost = total_amount * portfolio_beta * 0.005
                        st.metric("Estimated Hedging Cost", f"₹{hedging_cost:,.2f}")
                    
                    # Detailed Breakdown
                    st.subheader("Portfolio Breakdown")
                    
                    display_df = merged.copy()
                    display_df['AMOUNT'] = display_df['AMOUNT'].apply(lambda x: f"₹{x:,.2f}")
                    display_df['WEIGHT'] = display_df['WEIGHT'].apply(lambda x: f"{x:.2%}")
                    display_df['BETA'] = display_df['BETA'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                    display_df['WEIGHTED_BETA'] = display_df['WEIGHTED_BETA'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                    
                    st.dataframe(display_df)
                    
                    # Download Results
                    st.subheader("Download Results")
                    csv = merged.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results as CSV",
                        data=csv,
                        file_name="portfolio_beta_results.csv",
                        mime="text/csv"
                    )

# Footer
st.markdown("---")
st.info("💡 **Note**: This calculates beta against Nifty 50 (^NSEI) using 1 year of historical data. Beta values are calculated using Yahoo Finance data.")