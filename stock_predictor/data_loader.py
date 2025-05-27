import pandas as pd
import yfinance
import FinanceDataReader

def get_stock_price_data(ticker_symbol: str, start_date: str, end_date: str, country: str = 'US') -> pd.DataFrame | None:
    """
    Fetches historical stock price data.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        start_date (str): The start date for the data (YYYY-MM-DD).
        end_date (str): The end date for the data (YYYY-MM-DD).
        country (str): The country of the stock exchange ('US' or 'KR'). Defaults to 'US'.

    Returns:
        pd.DataFrame | None: A DataFrame with date as index and columns like 'Open', 'High', 'Low', 'Close', 'Volume',
                             or None if data fetching fails.
    """
    try:
        if country == 'US':
            data = yfinance.download(ticker_symbol, start=start_date, end=end_date)
        elif country == 'KR':
            # For simplicity, initially assume ticker_symbol for KR stocks is the numeric code.
            # Name-to-code mapping can be added later if needed.
            # Example: fdr.StockListing('KRX') can be used to find symbols.
            data = FinanceDataReader.DataReader(ticker_symbol, start_date, end_date)
        else:
            print(f"Country {country} not supported.")
            return None

        if data.empty:
            print(f"No data found for {ticker_symbol} from {start_date} to {end_date}.")
            return None
        
        # Standardize column names if necessary (yfinance uses capitalized, FinanceDataReader might use different)
        data.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume',
            # Add other potential variations from FinanceDataReader if needed
        }, inplace=True)
        
        return data
    except Exception as e:
        print(f"Error fetching stock price data for {ticker_symbol}: {e}")
        return None

def get_us_financial_data(ticker_symbol: str) -> dict[str, pd.DataFrame] | None:
    """
    Fetches US stock financial statements (income statement, balance sheet, cash flow).

    Args:
        ticker_symbol (str): The US stock ticker symbol.

    Returns:
        dict[str, pd.DataFrame] | None: A dictionary with keys 'financials', 'balance_sheet', 'cashflow'
                                         and their corresponding DataFrames as values.
                                         Returns None if data fetching fails or data is incomplete.
    """
    try:
        ticker = yfinance.Ticker(ticker_symbol)
        
        financials_df = ticker.financials
        balance_sheet_df = ticker.balance_sheet
        cashflow_df = ticker.cashflow

        if financials_df is None or financials_df.empty:
            print(f"Financials data not available for {ticker_symbol}")
            return None
        if balance_sheet_df is None or balance_sheet_df.empty:
            print(f"Balance sheet data not available for {ticker_symbol}")
            return None
        if cashflow_df is None or cashflow_df.empty:
            print(f"Cash flow data not available for {ticker_symbol}")
            return None

        # Data is typically columns as fiscal years, transpose to have dates as index
        return {
            'financials': financials_df.T,
            'balance_sheet': balance_sheet_df.T,
            'cashflow': cashflow_df.T
        }
    except Exception as e:
        print(f"Error fetching US financial data for {ticker_symbol}: {e}")
        return None

def get_kr_financial_data_csv(csv_path: str) -> pd.DataFrame | None:
    """
    Reads Korean financial data from a pre-formatted CSV file.
    This is a placeholder for potential OpenDART integration.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame | None: A DataFrame containing the financial data, or None if reading fails.
    """
    try:
        data = pd.read_csv(csv_path)
        # Assume the CSV is pre-formatted with dates and relevant financial data.
        # Further processing (e.g., setting index) might be needed depending on CSV structure.
        if 'date' in data.columns or 'Date' in data.columns: # Basic check for a date column
            date_col = 'date' if 'date' in data.columns else 'Date'
            try:
                data[date_col] = pd.to_datetime(data[date_col])
                data.set_index(date_col, inplace=True)
            except Exception as e:
                print(f"Could not parse date column or set index: {e}")
        else:
            print("Warning: No 'date' or 'Date' column found in CSV. Data returned without date index.")
            
        return data
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading Korean financial data from CSV {csv_path}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    
    # US Stock Price Data
    print("\n--- US Stock Price Data (AAPL) ---")
    aapl_price = get_stock_price_data('AAPL', '2023-01-01', '2023-03-31')
    if aapl_price is not None:
        print(aapl_price.head())
        print(aapl_price.tail())

    # KR Stock Price Data (Samsung Electronics)
    print("\n--- KR Stock Price Data (005930) ---")
    samsung_price = get_stock_price_data('005930', '2023-01-01', '2023-03-31', country='KR')
    if samsung_price is not None:
        print(samsung_price.head())
        print(samsung_price.tail())
    
    # US Financial Data
    print("\n--- US Financial Data (MSFT) ---")
    msft_financials = get_us_financial_data('MSFT')
    if msft_financials:
        print("Income Statement (Head):")
        print(msft_financials['financials'].head())
        print("\nBalance Sheet (Head):")
        print(msft_financials['balance_sheet'].head())
        print("\nCash Flow (Head):")
        print(msft_financials['cashflow'].head())

    # Test with a ticker that might have missing data
    print("\n--- US Financial Data (NonExistentTicker) ---")
    non_existent_financials = get_us_financial_data('NONEXISTENTTICKERXYZ')
    if non_existent_financials is None:
        print("Correctly handled non-existent ticker for financials.")

    # Test with a ticker that might be valid but lack some financial statements
    # This would depend on yfinance's behavior for specific thinly traded or new stocks
    # For now, the existing error handling should catch attribute errors if statements are None

    # Korean Financial Data (Placeholder CSV)
    print("\n--- KR Financial Data (CSV) ---")
    # Create a dummy CSV for testing
    dummy_data = {
        'Date': ['2022-12-31', '2021-12-31', '2020-12-31'],
        'Revenue': [1000, 900, 800],
        'Net Income': [100, 90, 80]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'dummy_kr_financials.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)
    
    kr_fin_csv = get_kr_financial_data_csv(dummy_csv_path)
    if kr_fin_csv is not None:
        print(kr_fin_csv.head())

    # Test non-existent CSV
    print("\n--- KR Financial Data (Non-existent CSV) ---")
    kr_fin_non_existent_csv = get_kr_financial_data_csv('non_existent.csv')
    if kr_fin_non_existent_csv is None:
        print("Correctly handled non-existent CSV.")

    # Clean up dummy CSV
    import os
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
        print(f"\nCleaned up {dummy_csv_path}")

    print("\n--- Testing Error Cases for get_stock_price_data ---")
    # Non-existent US ticker
    price_non_existent_us = get_stock_price_data('NONEXISTENTUSXYZ', '2023-01-01', '2023-01-10')
    if price_non_existent_us is None:
        print("Correctly handled non-existent US ticker for price data.")

    # Non-existent KR ticker
    price_non_existent_kr = get_stock_price_data('999999', '2023-01-01', '2023-01-10', country='KR')
    if price_non_existent_kr is None:
        print("Correctly handled non-existent KR ticker for price data.")
    
    # Invalid country
    price_invalid_country = get_stock_price_data('AAPL', '2023-01-01', '2023-01-10', country='GB')
    if price_invalid_country is None:
        print("Correctly handled invalid country for price data.")

```
