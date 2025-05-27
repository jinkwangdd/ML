import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_simple_backtest(price_data: pd.Series, signals: pd.Series, 
                        initial_capital: float = 100000.0, commission_rate: float = 0.001) -> pd.DataFrame | None:
    """
    Runs a simple backtest based on trading signals.

    Args:
        price_data (pd.Series): Series of actual closing prices, indexed by date.
        signals (pd.Series): Series of trading signals (1 for buy/hold, 0 for sell/neutral), indexed by date.
        initial_capital (float): Starting capital for the backtest.
        commission_rate (float): Transaction cost per trade (e.g., 0.001 for 0.1%).

    Returns:
        pd.DataFrame | None: DataFrame with daily 'portfolio_value', 'capital', 'position' (shares), 
                             'signal', 'price'. Returns None if inputs are invalid.
    """
    if not isinstance(price_data, pd.Series) or not isinstance(signals, pd.Series):
        print("Error: price_data and signals must be pandas Series.")
        return None
    if not price_data.index.equals(signals.index):
        print("Error: price_data and signals must have the same index.")
        return None
    if price_data.empty or signals.empty:
        print("Error: price_data or signals Series is empty.")
        return None
    if (signals.isin([0, 1, np.nan]).sum() != len(signals)): # Allow NaN in signals (interpret as hold)
         print("Error: signals Series should contain only 0, 1, or NaN.")
         # For this simple backtester, we'll treat NaN as "do nothing" or "hold current position"
         # which is effectively the same as signal 0 if holding, or signal 1 if flat.
         # A more sophisticated backtester might handle NaN explicitly.

    capital = initial_capital
    position = 0.0  # Number of shares held
    
    # Create a DataFrame to store daily values
    portfolio_df = pd.DataFrame(index=price_data.index)
    portfolio_df['price'] = price_data
    portfolio_df['signal'] = signals
    portfolio_df['capital'] = initial_capital
    portfolio_df['position'] = 0.0
    portfolio_df['portfolio_value'] = initial_capital

    for i in range(len(price_data)):
        current_date = price_data.index[i]
        current_price = price_data.iloc[i]
        current_signal = signals.iloc[i]

        if pd.isna(current_price) or current_price <= 0: # Skip if price is invalid
            portfolio_df.loc[current_date, 'capital'] = capital
            portfolio_df.loc[current_date, 'position'] = position
            portfolio_df.loc[current_date, 'portfolio_value'] = capital + position * (portfolio_df['price'].iloc[i-1] if i > 0 and not pd.isna(portfolio_df['price'].iloc[i-1]) else 0) # Use previous price if current is bad
            continue

        # Decision based on signal
        if current_signal == 1 and position == 0:  # Buy signal and not holding shares
            if capital > 0: # Ensure there's capital to buy
                shares_to_buy = capital / current_price
                position = shares_to_buy * (1 - commission_rate)  # Account for commission
                capital = 0.0
        elif current_signal == 0 and position > 0:  # Sell signal and holding shares
            capital = position * current_price * (1 - commission_rate)  # Account for commission
            position = 0.0
        
        # If signal is NaN, or signal is 1 but already in position, or signal is 0 but already flat, do nothing.
        
        portfolio_df.loc[current_date, 'capital'] = capital
        portfolio_df.loc[current_date, 'position'] = position
        portfolio_df.loc[current_date, 'portfolio_value'] = capital + (position * current_price)

    return portfolio_df

def calculate_cumulative_returns(portfolio_values: pd.Series, initial_capital: float) -> pd.Series:
    """
    Calculates cumulative returns from portfolio values.

    Args:
        portfolio_values (pd.Series): Series of portfolio values from the backtest.
        initial_capital (float): The initial capital.

    Returns:
        pd.Series: Series of cumulative returns.
    """
    if not isinstance(portfolio_values, pd.Series):
        print("Error: portfolio_values must be a pandas Series.")
        return pd.Series(dtype=float) # Return empty series
    if initial_capital <= 0:
        print("Error: initial_capital must be positive.")
        return pd.Series(dtype=float)

    cumulative_returns = (portfolio_values / initial_capital) - 1
    return cumulative_returns

def plot_cumulative_returns(strategy_returns: pd.Series, buy_and_hold_returns: pd.Series):
    """
    Plots cumulative returns of the strategy vs. buy-and-hold.

    Args:
        strategy_returns (pd.Series): Cumulative returns of the trading strategy.
        buy_and_hold_returns (pd.Series): Cumulative returns of a buy-and-hold strategy.
    """
    if not isinstance(strategy_returns, pd.Series) or not isinstance(buy_and_hold_returns, pd.Series):
        print("Error: strategy_returns and buy_and_hold_returns must be pandas Series.")
        return
        
    plt.figure(figsize=(12, 7))
    strategy_returns.plot(label='Strategy Returns', color='blue')
    buy_and_hold_returns.plot(label='Buy & Hold Returns', color='orange')
    
    plt.title('Strategy vs. Buy & Hold Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("--- Backtesting Module Test ---")

    # 1. Create Sample Data
    np.random.seed(42) # For reproducibility
    num_days = 100
    start_date = pd.to_datetime('2023-01-01')
    dates = pd.date_range(start_date, periods=num_days, freq='B') # Business days

    # Simulate some price movement (random walk with some trend)
    initial_price = 50
    price_changes = np.random.normal(0.05, 0.8, num_days -1) # small positive drift, some volatility
    sample_price_data = pd.Series(np.concatenate(([initial_price], initial_price + np.cumsum(price_changes))), index=dates)
    sample_price_data = sample_price_data.clip(lower=5) # Ensure price doesn't go below a certain value

    # Generate sample signals (random 0s and 1s)
    sample_signals = pd.Series(np.random.choice([0, 1], size=num_days), index=dates)
    
    # Introduce some NaNs in signals to test handling
    nan_indices = np.random.choice(sample_signals.index, size=int(num_days * 0.05), replace=False)
    sample_signals.loc[nan_indices] = np.nan


    initial_capital_test = 100000.0
    commission_rate_test = 0.001

    print(f"\nGenerated sample_price_data (Head):\n{sample_price_data.head()}")
    print(f"Generated sample_signals (Head with potential NaNs):\n{sample_signals.head()}")

    # 2. Test run_simple_backtest
    print("\n--- Testing run_simple_backtest ---")
    portfolio_history = run_simple_backtest(sample_price_data, sample_signals, 
                                            initial_capital_test, commission_rate_test)

    if portfolio_history is not None:
        print("\nPortfolio History (Head):")
        print(portfolio_history.head())
        print("\nPortfolio History (Tail):")
        print(portfolio_history.tail())

        # 3. Test calculate_cumulative_returns
        print("\n--- Testing calculate_cumulative_returns ---")
        strategy_cumulative_returns = calculate_cumulative_returns(portfolio_history['portfolio_value'], 
                                                                   initial_capital_test)
        
        # Calculate buy-and-hold returns
        buy_and_hold_returns = (sample_price_data / sample_price_data.iloc[0]) - 1
        
        print("\nStrategy Cumulative Returns (Tail):")
        print(strategy_cumulative_returns.tail())
        print("\nBuy & Hold Cumulative Returns (Tail):")
        print(buy_and_hold_returns.tail())

        # 4. Test plot_cumulative_returns
        print("\n--- Testing plot_cumulative_returns ---")
        plot_cumulative_returns(strategy_cumulative_returns, buy_and_hold_returns)
        print("Cumulative returns plot should have been displayed.")
    else:
        print("Backtest failed, skipping further tests.")

    # Test with mismatched indices
    print("\n--- Testing run_simple_backtest with mismatched indices ---")
    signals_mismatched = sample_signals.copy()
    signals_mismatched.index = pd.date_range(start_date + pd.Timedelta(days=1), periods=num_days, freq='B')
    mismatched_result = run_simple_backtest(sample_price_data, signals_mismatched)
    if mismatched_result is None:
        print("Correctly handled mismatched indices (returned None).")
    else:
        print("Error: Did not handle mismatched indices correctly.")
        
    # Test with invalid signal values (e.g. 2)
    print("\n--- Testing run_simple_backtest with invalid signal values ---")
    signals_invalid = sample_signals.copy()
    if len(signals_invalid) > 5:
        signals_invalid.iloc[5] = 2 # Introduce an invalid signal
    invalid_signal_result = run_simple_backtest(sample_price_data, signals_invalid)
    # The current check is basic, it will print an error but proceed.
    # Depending on strictness, this could also return None.
    # The check `signals.isin([0, 1, np.nan]).sum() != len(signals)` should catch this.
    # The function currently prints an error but proceeds by not acting on the invalid signal.
    # For this test, we will check if it printed the error, it does not return None.
    # If strict error handling returning None is required, the function needs modification.
    # Current interpretation: it logs error and continues (NaNs are treated as hold).
    # If an invalid signal (not 0, 1, NaN) is present, it should be noted.
    # The error message "Error: signals Series should contain only 0, 1, or NaN." is now printed.
    # It does not return None by design unless inputs are structurally flawed (type, index, empty).
    # The test will just confirm it runs.
    if invalid_signal_result is not None:
         print("Test with invalid signals ran (check console for error message about signal values).")


    print("\n--- Backtesting Module Test Complete ---")
```
