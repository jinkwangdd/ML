import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance # For fetching shares outstanding in example

# Import data_loader for testing purposes if needed
# from stock_predictor import data_loader # Assuming it's in the parent directory or installed

def add_technical_indicators(price_df: pd.DataFrame, rsi_period: int = 14, 
                             macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                             sma_short: int = 50, sma_long: int = 200, 
                             bb_period: int = 20, bb_std: int = 2) -> pd.DataFrame:
    """
    Calculates and adds technical indicators to the price DataFrame.

    Args:
        price_df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', 'Volume' columns and DatetimeIndex.
        rsi_period (int): Period for RSI calculation.
        macd_fast (int): Fast period for MACD.
        macd_slow (int): Slow period for MACD.
        macd_signal (int): Signal period for MACD.
        sma_short (int): Period for short-term Simple Moving Average.
        sma_long (int): Period for long-term Simple Moving Average.
        bb_period (int): Period for Bollinger Bands Moving Average.
        bb_std (int): Standard deviation for Bollinger Bands.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    if not isinstance(price_df, pd.DataFrame):
        print("Error: price_df must be a pandas DataFrame.")
        return pd.DataFrame() # Return empty DataFrame on error
    
    if price_df.empty or 'Close' not in price_df.columns:
        print("Error: price_df is empty or missing 'Close' column.")
        return price_df # Return original if not usable

    df = price_df.copy()

    try:
        # RSI
        if len(df) >= rsi_period:
            df.ta.rsi(length=rsi_period, append=True) # Appends as RSI_14
        else:
            print(f"Warning: DataFrame too short for RSI calculation (length {len(df)}, period {rsi_period}). Skipping RSI.")

        # MACD
        # MACD requires length > slow period + signal period for full calculation
        if len(df) >= macd_slow + macd_signal: 
            df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True) # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        else:
            print(f"Warning: DataFrame too short for MACD calculation (length {len(df)}, need > {macd_slow + macd_signal}). Skipping MACD.")

        # Moving Averages
        if len(df) >= sma_short:
            df.ta.sma(length=sma_short, append=True) # Appends as SMA_50
        else:
            print(f"Warning: DataFrame too short for SMA {sma_short} (length {len(df)}). Skipping SMA_{sma_short}.")
        
        if len(df) >= sma_long:
            df.ta.sma(length=sma_long, append=True) # Appends as SMA_200
        else:
            print(f"Warning: DataFrame too short for SMA {sma_long} (length {len(df)}). Skipping SMA_{sma_long}.")

        # Bollinger Bands
        if len(df) >= bb_period:
            df.ta.bbands(length=bb_period, std=bb_std, append=True) # Appends BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        else:
            print(f"Warning: DataFrame too short for Bollinger Bands (length {len(df)}, period {bb_period}). Skipping Bollinger Bands.")
            
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        # Return df with any indicators that were successfully added before the error
    
    return df

def calculate_financial_ratios(financial_data: dict[str, pd.DataFrame], 
                               price_df: pd.DataFrame, 
                               ticker_symbol: str = None) -> pd.DataFrame | None:
    """
    Calculates financial ratios by combining financial statements and price data.
    NOTE: This is a simplified version. Historical shares outstanding and precise alignment
          of financial reporting dates with daily prices are complex.

    Args:
        financial_data (dict[str, pd.DataFrame]): Dict with 'financials' (Income Stmt), 
                                                  'balance_sheet', 'cashflow' DataFrames.
                                                  Index of these DFs should be DatetimeIndex (annual/quarterly).
        price_df (pd.DataFrame): Daily stock price data, used for market cap and merging. Must have 'Close' column.
        ticker_symbol (str, optional): Ticker symbol, used to fetch current shares outstanding if needed.

    Returns:
        pd.DataFrame | None: DataFrame with financial ratios indexed by date, or None if critical data is missing.
    """
    if not financial_data or not all(key in financial_data for key in ['financials', 'balance_sheet', 'cashflow']):
        print("Error: financial_data dictionary is incomplete.")
        return None
    if price_df.empty or 'Close' not in price_df.columns:
        print("Error: price_df is empty or missing 'Close' column.")
        return None

    income_stmt = financial_data['financials']
    balance_sheet = financial_data['balance_sheet']
    # cash_flow = financial_data['cashflow'] # Not used in current simplified ratios but available

    # --- Identify common financial statement items ---
    # Exact names can vary. These are common patterns from yfinance.
    # Income Statement items
    NET_INCOME = 'Net Income' if 'Net Income' in income_stmt else 'NetIncome'
    TOTAL_REVENUE = 'Total Revenue' if 'Total Revenue' in income_stmt else 'TotalRevenue'
    
    # Balance Sheet items
    TOTAL_ASSETS = 'Total Assets' if 'Total Assets' in balance_sheet else 'TotalAssets'
    TOTAL_LIABILITIES = 'Total Liab' if 'Total Liab' in balance_sheet else 'TotalLiabilitiesNetMinorityInterest' # yfinance often uses 'Total Liab'
    TOTAL_STOCKHOLDER_EQUITY = 'Total Stockholder Equity' if 'Total Stockholder Equity' in balance_sheet else 'StockholdersEquity'
    
    # Attempt to find shares outstanding.
    # 1. From income statement (diluted average shares often preferred)
    # yfinance often has 'Diluted Average Shares' or 'Basic Average Shares'
    SHARES_OUTSTANDING_KEY_OPTIONS = ['Diluted Average Shares', 'Basic Average Shares', 'Shares Outstanding']
    shares_outstanding_col = next((col for col in SHARES_OUTSTANDING_KEY_OPTIONS if col in income_stmt), None)
    
    historical_shares_df = pd.DataFrame(index=price_df.index)

    if shares_outstanding_col:
        # Financial statements are usually annual/quarterly. Reindex and ffill.
        shares_series = income_stmt[shares_outstanding_col].reindex(price_df.index, method='ffill')
        historical_shares_df['SharesOutstanding'] = shares_series
    elif ticker_symbol:
        print(f"Warning: Historical shares outstanding not found in financials. Using current shares outstanding for {ticker_symbol} for PER. This is less accurate for historical PER.")
        try:
            ticker_obj = yfinance.Ticker(ticker_symbol)
            current_shares = ticker_obj.info.get('sharesOutstanding')
            if current_shares:
                historical_shares_df['SharesOutstanding'] = current_shares # Apply same value across all days
            else:
                print(f"Warning: Could not fetch current shares outstanding for {ticker_symbol}. PER will be NaN.")
                historical_shares_df['SharesOutstanding'] = np.nan
        except Exception as e:
            print(f"Error fetching current shares outstanding for {ticker_symbol}: {e}. PER will be NaN.")
            historical_shares_df['SharesOutstanding'] = np.nan
    else:
        print("Warning: Cannot calculate PER. No historical shares in financials and no ticker_symbol provided for current shares.")
        historical_shares_df['SharesOutstanding'] = np.nan

    # --- Prepare financial data for merging (reindex to daily, ffill) ---
    # Ratios will be calculated on a daily basis using the latest available financials.
    daily_financials = pd.DataFrame(index=price_df.index)

    # Helper to reindex and fill
    def reindex_and_ffill(df, column_name_map):
        temp_df = pd.DataFrame(index=price_df.index)
        for new_col, old_col in column_name_map.items():
            if old_col in df:
                # Ensure the index of df is datetime-like for reindexing
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        print(f"Could not convert index of financial statement for {old_col} to datetime: {e}")
                        temp_df[new_col] = np.nan
                        continue
                temp_df[new_col] = df[old_col].reindex(price_df.index, method='ffill')
            else:
                print(f"Warning: Column '{old_col}' not found in financial data. Ratio relying on it will be NaN.")
                temp_df[new_col] = np.nan
        return temp_df

    income_items_map = {'NetIncome': NET_INCOME, 'TotalRevenue': TOTAL_REVENUE}
    daily_income = reindex_and_ffill(income_stmt, income_items_map)
    
    balance_sheet_items_map = {'TotalAssets': TOTAL_ASSETS, 'TotalLiabilities': TOTAL_LIABILITIES, 'TotalStockholderEquity': TOTAL_STOCKHOLDER_EQUITY}
    daily_balance_sheet = reindex_and_ffill(balance_sheet, balance_sheet_items_map)

    daily_financials = pd.concat([daily_income, daily_balance_sheet, historical_shares_df], axis=1)
    daily_financials['MarketCap'] = price_df['Close'] * daily_financials['SharesOutstanding']

    # --- Calculate Ratios ---
    ratios_df = pd.DataFrame(index=price_df.index)

    # PER
    ratios_df['PER'] = daily_financials['MarketCap'] / daily_financials['NetIncome']
    
    # PBR
    book_value = daily_financials['TotalAssets'] - daily_financials['TotalLiabilities'] # Could also use TotalStockholderEquity
    # Using TotalStockholderEquity directly can be more robust if available and correctly reported
    book_value_per_share = daily_financials['TotalStockholderEquity'] / daily_financials['SharesOutstanding']
    ratios_df['PBR_calc_bvps'] = price_df['Close'] / book_value_per_share # PBR using calculated BVPS
    ratios_df['PBR_direct_equity'] = daily_financials['MarketCap'] / daily_financials['TotalStockholderEquity']


    # ROE
    ratios_df['ROE'] = daily_financials['NetIncome'] / daily_financials['TotalStockholderEquity']
    
    # Debt-to-Equity
    # Assuming 'TotalLiabilities' represents total debt for simplicity.
    # More precise would be ShortTermDebt + LongTermDebt from balance sheet if available.
    ratios_df['DebtToEquity'] = daily_financials['TotalLiabilities'] / daily_financials['TotalStockholderEquity']

    # Growth Rates (YoY)
    # Financial data is already reindexed and ffilled. To get YoY, we need original annual/quarterly data.
    # This simplified version calculates pct_change on the ffilled data, which is not true YoY if data is sparse.
    # A more accurate YoY would involve:
    # 1. Taking the original annual/quarterly data.
    # 2. Calculating pct_change(periods=N) where N is 1 for annual, 4 for quarterly.
    # 3. Then reindexing and ffilling that result.

    # For simplicity now, we'll show an example with pct_change on the original (annual/quarterly)
    # and then reindex. This is more accurate.
    
    def calculate_yoy_growth(original_series, name):
        # Ensure index is sorted for pct_change
        original_series_sorted = original_series.sort_index()
        # Determine period: 1 if index freq is annual/undetected, 4 if quarterly
        # This is a heuristic. yfinance data is usually annual, sometimes quarterly.
        # If index has no freq, assume annual for safety.
        period = 1 
        if pd.infer_freq(original_series_sorted.index) and pd.infer_freq(original_series_sorted.index).startswith('Q'):
            period = 4
        
        yoy_growth = original_series_sorted.pct_change(periods=period) * 100 # as percentage
        yoy_growth_reindexed = yoy_growth.reindex(price_df.index, method='ffill')
        return yoy_growth_reindexed

    if TOTAL_REVENUE in income_stmt:
        ratios_df['SalesGrowthYoY'] = calculate_yoy_growth(income_stmt[TOTAL_REVENUE], 'SalesGrowthYoY')
    else:
        ratios_df['SalesGrowthYoY'] = np.nan

    if NET_INCOME in income_stmt:
        ratios_df['NetIncomeGrowthYoY'] = calculate_yoy_growth(income_stmt[NET_INCOME], 'NetIncomeGrowthYoY')
    else:
        ratios_df['NetIncomeGrowthYoY'] = np.nan
        
    # Replace inf values that can occur from division by zero (e.g., NetIncome is 0 for PER)
    ratios_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return ratios_df


def combine_features(price_df: pd.DataFrame, 
                     technical_indicators_df: pd.DataFrame = None, 
                     financial_ratios_df: pd.DataFrame = None,
                     dropna_how: str = 'any') -> pd.DataFrame:
    """
    Merges price data, technical indicators, and financial ratios.

    Args:
        price_df (pd.DataFrame): The base DataFrame with price data.
        technical_indicators_df (pd.DataFrame, optional): DataFrame with technical indicators.
        financial_ratios_df (pd.DataFrame, optional): DataFrame with financial ratios.
        dropna_how (str): How to handle NaN values after merging ('any', 'all', or None to keep NaNs).

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    combined_df = price_df.copy()

    if technical_indicators_df is not None and not technical_indicators_df.empty:
        # Technical indicators are often calculated directly on price_df, so columns might already exist
        # or they might be separate. If separate, ensure index alignment for merge.
        # The add_technical_indicators function returns a df with original + new columns.
        # So, if that output is passed here, it's already combined.
        # This function assumes technical_indicators_df might be a separate df to be merged.
        if not all(col in combined_df.columns for col in technical_indicators_df.columns if col not in price_df.columns):
             # Only merge if technical_indicators_df is not the same object or doesn't just contain price_df columns
            if combined_df.index.equals(technical_indicators_df.index):
                combined_df = combined_df.merge(technical_indicators_df, left_index=True, right_index=True, how='left', suffixes=('', '_tech'))
            else: #This case should not happen if add_technical_indicators is used as intended
                 print("Warning: Index mismatch between price_df and technical_indicators_df. Trying to merge but check data.")
                 combined_df = combined_df.merge(technical_indicators_df, left_index=True, right_index=True, how='left', suffixes=('', '_tech'))

    if financial_ratios_df is not None and not financial_ratios_df.empty:
        combined_df = combined_df.merge(financial_ratios_df, left_index=True, right_index=True, how='left')
    
    # Clean up potential duplicate columns from merges if suffixes weren't perfect
    combined_df = combined_df.loc[:,~combined_df.columns.duplicated()]

    if dropna_how:
        original_len = len(combined_df)
        combined_df.dropna(how=dropna_how, inplace=True)
        print(f"Dropped {original_len - len(combined_df)} rows due to NaN values (method: {dropna_how}).")
        
    return combined_df

if __name__ == '__main__':
    # --- Test Data Setup ---
    # 1. Sample Price Data (mimicking data_loader.get_stock_price_data output)
    dates = pd.to_datetime(['2022-12-01', '2022-12-02', '2022-12-03', '2022-12-04', '2022-12-05',
                            '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-10',
                            '2022-12-11', '2022-12-12', '2022-12-13', '2022-12-14', '2022-12-15',
                            '2022-12-16', '2022-12-17', '2022-12-18', '2022-12-19', '2022-12-20',
                            '2022-12-21', '2022-12-22', '2022-12-23', '2022-12-24', '2022-12-25',
                            '2022-12-26', '2022-12-27', '2022-12-28', '2022-12-29', '2022-12-30',
                            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']) # Longer for indicators
    
    price_data = {
        'Open': np.random.uniform(100, 105, len(dates)),
        'High': np.random.uniform(105, 110, len(dates)),
        'Low': np.random.uniform(95, 100, len(dates)),
        'Close': np.random.uniform(100, 105, len(dates)),
        'Volume': np.random.randint(100000, 500000, len(dates))
    }
    sample_price_df = pd.DataFrame(price_data, index=dates)
    sample_price_df.index.name = 'Date'
    
    print("--- Original Sample Price Data ---")
    print(sample_price_df.head())

    # --- Test add_technical_indicators ---
    print("\n--- Testing add_technical_indicators ---")
    tech_df = add_technical_indicators(sample_price_df.copy()) # Use copy to preserve original
    print(tech_df.head())
    print("Columns added:", [col for col in tech_df.columns if col not in sample_price_df.columns])
    
    # Test with short DataFrame
    print("\n--- Testing add_technical_indicators with short DataFrame ---")
    short_price_df = sample_price_df.head(10).copy()
    short_tech_df = add_technical_indicators(short_price_df)
    print(short_tech_df.head())


    # --- Test calculate_financial_ratios ---
    print("\n--- Testing calculate_financial_ratios ---")
    # 2. Sample Financial Data (mimicking data_loader.get_us_financial_data output)
    # Annual data for simplicity
    financial_dates_annual = pd.to_datetime(['2021-12-31', '2022-12-31'])
    income_data_annual = {
        'Total Revenue': [1000000, 1200000], # Example: Sales Growth
        'Net Income': [100000, 120000],      # Example: ROE, PER, Net Income Growth
        'Diluted Average Shares': [50000, 52000] # For PER
    }
    balance_sheet_data_annual = {
        'Total Assets': [500000, 550000],
        'Total Liab': [200000, 220000],      # Example: Debt-to-Equity, PBR
        'Total Stockholder Equity': [300000, 330000] # Example: ROE, PBR, Debt-to-Equity
    }
    cashflow_data_annual = { # Not used in current simplified ratios but structure shown
        'Operating Cash Flow': [150000, 180000]
    }

    sample_financials_annual_df = pd.DataFrame(income_data_annual, index=financial_dates_annual)
    sample_balance_sheet_annual_df = pd.DataFrame(balance_sheet_data_annual, index=financial_dates_annual)
    sample_cashflow_annual_df = pd.DataFrame(cashflow_data_annual, index=financial_dates_annual)

    sample_financial_data = {
        'financials': sample_financials_annual_df,
        'balance_sheet': sample_balance_sheet_annual_df,
        'cashflow': sample_cashflow_annual_df
    }

    # Use a slice of price_df that aligns with the financial data for more meaningful test
    test_price_df_for_ratios = sample_price_df[sample_price_df.index >= '2022-01-01'].copy()
    
    # If using a real ticker for shares outstanding, ensure yfinance is installed and network available
    # For isolated testing, you can mock yfinance.Ticker().info or rely on 'Diluted Average Shares'
    ratios_df = calculate_financial_ratios(sample_financial_data, test_price_df_for_ratios, ticker_symbol="MSFT") # MSFT as example if live fetch
    
    if ratios_df is not None:
        print(ratios_df.head())
        print(ratios_df.tail())
        print("Ratio columns:", ratios_df.columns.tolist())
    else:
        print("Financial ratios calculation failed or returned None.")

    # Test with missing 'Diluted Average Shares' and no ticker
    income_data_no_shares = {'Total Revenue': [1000000], 'Net Income': [100000]}
    sample_financials_no_shares_df = pd.DataFrame(income_data_no_shares, index=pd.to_datetime(['2022-12-31']))
    sample_financial_data_no_shares = {
        'financials': sample_financials_no_shares_df,
        'balance_sheet': sample_balance_sheet_annual_df, # Re-use for structure
        'cashflow': sample_cashflow_annual_df
    }
    print("\n--- Testing calculate_financial_ratios (no shares info) ---")
    ratios_no_shares_df = calculate_financial_ratios(sample_financial_data_no_shares, test_price_df_for_ratios.copy())
    if ratios_no_shares_df is not None:
        print("PER with no shares info (should be NaN or not present):")
        if 'PER' in ratios_no_shares_df: print(ratios_no_shares_df[['PER']].head())
        else: print("PER column not generated as expected.")


    # --- Test combine_features ---
    print("\n--- Testing combine_features ---")
    # Assuming tech_df already contains price data columns + new tech indicators
    # And ratios_df is separate
    
    # Re-run tech indicators on the same price_df used for ratios for proper alignment
    aligned_tech_df = add_technical_indicators(test_price_df_for_ratios.copy()) 

    # Case 1: All dataframes present
    combined_df_all = combine_features(test_price_df_for_ratios.copy(), # Start with base prices
                                       aligned_tech_df,                 # Add technicals
                                       ratios_df,                       # Add ratios
                                       dropna_how='any')
    print("Combined (all, dropna='any'):")
    print(combined_df_all.head())
    print(f"Shape: {combined_df_all.shape}")
    # print(combined_df_all.info())


    # Case 2: Missing financial ratios
    combined_df_no_ratios = combine_features(test_price_df_for_ratios.copy(), 
                                             aligned_tech_df, 
                                             None, 
                                             dropna_how='any')
    print("\nCombined (no ratios, dropna='any'):")
    print(combined_df_no_ratios.head())
    print(f"Shape: {combined_df_no_ratios.shape}")

    # Case 3: Keeping NaNs
    combined_df_keep_nans = combine_features(test_price_df_for_ratios.copy(), 
                                             aligned_tech_df, 
                                             ratios_df, 
                                             dropna_how=None) # or dropna_how=False
    print("\nCombined (all, keep NaNs):")
    print(combined_df_keep_nans.head())
    print(combined_df_keep_nans.tail()) # See NaNs from early indicator calculations or ratio ffill limits
    print(f"Shape: {combined_df_keep_nans.shape}")
    # print(combined_df_keep_nans.info())

    print("\n--- Feature Engineering Tests Complete ---")


def create_labels(df: pd.DataFrame, n_days: int = 5, price_col: str = 'Close') -> pd.DataFrame:
    """
    Creates a binary label based on future price movement.

    Args:
        df (pd.DataFrame): Input DataFrame, must contain 'price_col'.
        n_days (int): Number of future days to compare price against.
        price_col (str): Name of the column containing the price (e.g., 'Close').

    Returns:
        pd.DataFrame: DataFrame with added 'label' column.
                      'label' is 1 if price_col at D+n_days > price_col at D, 0 otherwise.
                      NaN if future price is not available.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df must be a pandas DataFrame.")
        return pd.DataFrame()
    
    if price_col not in df.columns:
        print(f"Error: price_col '{price_col}' not found in DataFrame columns.")
        return df # Return original df if price_col is missing

    df_labeled = df.copy()
    
    # Calculate the future price
    future_price = df_labeled[price_col].shift(-n_days)
    
    # Create the label: 1 if future price is higher, 0 otherwise
    # np.nan will be assigned where future_price is NaN (last n_days rows)
    df_labeled['label'] = np.where(future_price > df_labeled[price_col], 1, 
                                   np.where(future_price.notna(), 0, np.nan))
    
    return df_labeled

if __name__ == '__main__':
    # --- Test Data Setup ---
    # 1. Sample Price Data (mimicking data_loader.get_stock_price_data output)
    dates = pd.to_datetime(['2022-12-01', '2022-12-02', '2022-12-03', '2022-12-04', '2022-12-05',
                            '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-10',
                            '2022-12-11', '2022-12-12', '2022-12-13', '2022-12-14', '2022-12-15',
                            '2022-12-16', '2022-12-17', '2022-12-18', '2022-12-19', '2022-12-20',
                            '2022-12-21', '2022-12-22', '2022-12-23', '2022-12-24', '2022-12-25',
                            '2022-12-26', '2022-12-27', '2022-12-28', '2022-12-29', '2022-12-30',
                            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']) # Longer for indicators
    
    price_data = {
        'Open': np.random.uniform(100, 105, len(dates)),
        'High': np.random.uniform(105, 110, len(dates)),
        'Low': np.random.uniform(95, 100, len(dates)),
        'Close': np.random.uniform(100, 105, len(dates)),
        'Volume': np.random.randint(100000, 500000, len(dates))
    }
    sample_price_df = pd.DataFrame(price_data, index=dates)
    sample_price_df.index.name = 'Date'
    
    print("--- Original Sample Price Data ---")
    print(sample_price_df.head())

    # --- Test add_technical_indicators ---
    print("\n--- Testing add_technical_indicators ---")
    tech_df = add_technical_indicators(sample_price_df.copy()) # Use copy to preserve original
    print(tech_df.head())
    print("Columns added:", [col for col in tech_df.columns if col not in sample_price_df.columns])
    
    # Test with short DataFrame
    print("\n--- Testing add_technical_indicators with short DataFrame ---")
    short_price_df = sample_price_df.head(10).copy()
    short_tech_df = add_technical_indicators(short_price_df)
    print(short_tech_df.head())


    # --- Test calculate_financial_ratios ---
    print("\n--- Testing calculate_financial_ratios ---")
    # 2. Sample Financial Data (mimicking data_loader.get_us_financial_data output)
    # Annual data for simplicity
    financial_dates_annual = pd.to_datetime(['2021-12-31', '2022-12-31'])
    income_data_annual = {
        'Total Revenue': [1000000, 1200000], # Example: Sales Growth
        'Net Income': [100000, 120000],      # Example: ROE, PER, Net Income Growth
        'Diluted Average Shares': [50000, 52000] # For PER
    }
    balance_sheet_data_annual = {
        'Total Assets': [500000, 550000],
        'Total Liab': [200000, 220000],      # Example: Debt-to-Equity, PBR
        'Total Stockholder Equity': [300000, 330000] # Example: ROE, PBR, Debt-to-Equity
    }
    cashflow_data_annual = { # Not used in current simplified ratios but structure shown
        'Operating Cash Flow': [150000, 180000]
    }

    sample_financials_annual_df = pd.DataFrame(income_data_annual, index=financial_dates_annual)
    sample_balance_sheet_annual_df = pd.DataFrame(balance_sheet_data_annual, index=financial_dates_annual)
    sample_cashflow_annual_df = pd.DataFrame(cashflow_data_annual, index=financial_dates_annual)

    sample_financial_data = {
        'financials': sample_financials_annual_df,
        'balance_sheet': sample_balance_sheet_annual_df,
        'cashflow': sample_cashflow_annual_df
    }

    # Use a slice of price_df that aligns with the financial data for more meaningful test
    test_price_df_for_ratios = sample_price_df[sample_price_df.index >= '2022-01-01'].copy()
    
    # If using a real ticker for shares outstanding, ensure yfinance is installed and network available
    # For isolated testing, you can mock yfinance.Ticker().info or rely on 'Diluted Average Shares'
    ratios_df = calculate_financial_ratios(sample_financial_data, test_price_df_for_ratios, ticker_symbol="MSFT") # MSFT as example if live fetch
    
    if ratios_df is not None:
        print(ratios_df.head())
        print(ratios_df.tail())
        print("Ratio columns:", ratios_df.columns.tolist())
    else:
        print("Financial ratios calculation failed or returned None.")

    # Test with missing 'Diluted Average Shares' and no ticker
    income_data_no_shares = {'Total Revenue': [1000000], 'Net Income': [100000]}
    sample_financials_no_shares_df = pd.DataFrame(income_data_no_shares, index=pd.to_datetime(['2022-12-31']))
    sample_financial_data_no_shares = {
        'financials': sample_financials_no_shares_df,
        'balance_sheet': sample_balance_sheet_annual_df, # Re-use for structure
        'cashflow': sample_cashflow_annual_df
    }
    print("\n--- Testing calculate_financial_ratios (no shares info) ---")
    ratios_no_shares_df = calculate_financial_ratios(sample_financial_data_no_shares, test_price_df_for_ratios.copy())
    if ratios_no_shares_df is not None:
        print("PER with no shares info (should be NaN or not present):")
        if 'PER' in ratios_no_shares_df: print(ratios_no_shares_df[['PER']].head())
        else: print("PER column not generated as expected.")


    # --- Test combine_features ---
    print("\n--- Testing combine_features ---")
    # Assuming tech_df already contains price data columns + new tech indicators
    # And ratios_df is separate
    
    # Re-run tech indicators on the same price_df used for ratios for proper alignment
    aligned_tech_df = add_technical_indicators(test_price_df_for_ratios.copy()) 

    # Case 1: All dataframes present
    combined_df_all = combine_features(test_price_df_for_ratios.copy(), # Start with base prices
                                       aligned_tech_df,                 # Add technicals
                                       ratios_df,                       # Add ratios
                                       dropna_how='any')
    print("Combined (all, dropna='any'):")
    print(combined_df_all.head())
    print(f"Shape: {combined_df_all.shape}")
    # print(combined_df_all.info())


    # Case 2: Missing financial ratios
    combined_df_no_ratios = combine_features(test_price_df_for_ratios.copy(), 
                                             aligned_tech_df, 
                                             None, 
                                             dropna_how='any')
    print("\nCombined (no ratios, dropna='any'):")
    print(combined_df_no_ratios.head())
    print(f"Shape: {combined_df_no_ratios.shape}")

    # Case 3: Keeping NaNs
    combined_df_keep_nans = combine_features(test_price_df_for_ratios.copy(), 
                                             aligned_tech_df, 
                                             ratios_df, 
                                             dropna_how=None) # or dropna_how=False
    print("\nCombined (all, keep NaNs):")
    print(combined_df_keep_nans.head())
    print(combined_df_keep_nans.tail()) # See NaNs from early indicator calculations or ratio ffill limits
    print(f"Shape: {combined_df_keep_nans.shape}")
    # print(combined_df_keep_nans.info())

    # --- Test create_labels ---
    print("\n--- Testing create_labels ---")
    # Use combined_df_keep_nans as it has the most complete set of rows before any dropping
    labeled_df = create_labels(combined_df_keep_nans.copy(), n_days=5, price_col='Close')
    print("DataFrame with labels (Head):")
    print(labeled_df[['Close', 'label']].head(10))
    print("\nDataFrame with labels (Tail):")
    print(labeled_df[['Close', 'label']].tail(10)) # Shows NaNs for last n_days

    # Demonstrate dropping rows where 'label' is NaN
    original_len_labeled = len(labeled_df)
    final_df_for_training = labeled_df.dropna(subset=['label'])
    print(f"\nDropped {original_len_labeled - len(final_df_for_training)} rows with NaN labels.")
    print("DataFrame after dropping NaN labels (Tail):")
    print(final_df_for_training[['Close', 'label']].tail())
    print(f"Shape after dropping NaN labels: {final_df_for_training.shape}")
    
    print("\n--- All Feature Engineering Tests Complete ---")
```
