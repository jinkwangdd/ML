"""
Streamlit web application for the Stock Predictor.

This application provides an interactive dashboard to:
- Load stock price and financial data.
- Perform feature engineering (technical indicators, financial ratios).
- Train an XGBoost model to predict stock price movements.
- Evaluate the model's performance.
- Visualize feature importances.
- Run a simple backtest of the generated trading signals.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these modules are in stock_predictor directory and accessible
from stock_predictor.data_loader import get_stock_price_data, get_us_financial_data
from stock_predictor.feature_engineering import add_technical_indicators, calculate_financial_ratios, combine_features, create_labels
from stock_predictor.model_train import train_xgboost_model, evaluate_model # Removed plot_feature_importance from direct import
from stock_predictor.backtest import run_simple_backtest, calculate_cumulative_returns, plot_cumulative_returns

# --- App Title ---
st.set_page_config(layout="wide")
st.title("📈 Stock Predictor App")

# --- Helper function to capture matplotlib plots for Streamlit ---
def _capture_plot_and_show(plot_func, *args, **kwargs):
    """Captures a matplotlib plot and displays it in Streamlit."""
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6))) # Allow figsize override
    plot_func(*args, **kwargs, ax=ax) # Pass ax to the plotting function
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

# --- User Inputs (Sidebar) ---
st.sidebar.header("⚙️ User Inputs")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
country = st.sidebar.selectbox("Country", ['US', 'KR'], index=0)

# Sensible date defaults
default_start_date = datetime.date.today() - datetime.timedelta(days=3*365) # 3 years ago
default_end_date = datetime.date.today()

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date, min_value=start_date + datetime.timedelta(days=30))

n_days_label = st.sidebar.number_input("N-Days for Label Creation (future price > current)", value=5, min_value=1, max_value=30)
train_test_split_ratio = st.sidebar.slider("Train/Test Split Ratio (for Training Data)", 0.5, 0.9, 0.8, 0.05)

run_button = st.sidebar.button("🚀 Run Analysis")

# --- Main App Logic ---
if run_button:
    if not ticker_symbol:
        st.error("Ticker symbol cannot be empty.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            st.header(f"Analysis for {ticker_symbol} ({country})")

            # --- 1. Data Loading ---
            with st.spinner(f"Loading price data for {ticker_symbol}..."):
                @st.cache_data # Cache the price data loading
                def cached_get_stock_price_data(ticker, start, end, ctry):
                    return get_stock_price_data(ticker, str(start), str(end), country=ctry)
                
                price_df_full = cached_get_stock_price_data(ticker_symbol, start_date, end_date, country)

            if price_df_full is None or price_df_full.empty:
                st.error(f"Could not load price data for {ticker_symbol}. Please check ticker and dates.")
            else:
                st.success(f"Price data loaded for {ticker_symbol} ({len(price_df_full)} rows).")
                st.subheader("Raw Price Data Preview")
                st.dataframe(price_df_full.head())
                
                financial_data = None
                if country == 'US':
                    with st.spinner(f"Loading financial statement data for {ticker_symbol}..."):
                        @st.cache_data # Cache financial data loading
                        def cached_get_us_financial_data(ticker):
                            return get_us_financial_data(ticker)
                        financial_data = cached_get_us_financial_data(ticker_symbol)
                    
                    if financial_data:
                        st.success("US financial data loaded.")
                        # st.write("Financials (Income Statement) Head:")
                        # st.dataframe(financial_data['financials'].head())
                    else:
                        st.warning("Could not load or financial data is incomplete for US ticker. Proceeding without it.")
                elif country == 'KR':
                    st.info("Financial data loading for KR stocks via CSV is not yet implemented in this app. Proceeding with price data only.")
                    # Placeholder for future KR financial data CSV upload
                    # uploaded_file = st.file_uploader("Upload KR Financial Data CSV (Optional)")
                    # if uploaded_file:
                    #     financial_data_kr = get_kr_financial_data_csv(uploaded_file) # This needs modification for Streamlit upload
                    #     # Process financial_data_kr as needed

                # --- 2. Feature Engineering ---
                with st.spinner("Engineering features..."):
                    price_df_for_features = price_df_full.copy()

                    # Technical Indicators
                    tech_indicators_df = add_technical_indicators(price_df_for_features)
                    st.write("Technical indicators added.")

                    # Financial Ratios
                    ratios_df = None
                    if financial_data:
                        # Pass price_df_full for market cap calc, but ratios_df will be aligned with its index
                        ratios_df = calculate_financial_ratios(financial_data, price_df_full, ticker_symbol=ticker_symbol)
                        if ratios_df is not None and not ratios_df.empty:
                             st.write("Financial ratios calculated.")
                        else:
                            st.warning("Could not calculate financial ratios or data was insufficient.")
                    
                    # Combine features
                    combined_df = combine_features(price_df_for_features, tech_indicators_df, ratios_df, dropna_how=None)
                    
                    # Create Labels
                    combined_df = create_labels(combined_df, n_days=n_days_label, price_col='Close')
                    st.write(f"Labels created using N={n_days_label} days shift.")

                    st.subheader("Feature Set & NaN Handling")
                    st.write(f"Combined data length before NaN handling: {len(combined_df)}")
                    
                    # NaN Handling
                    combined_df.dropna(subset=['label'], inplace=True) # Crucial: drop rows where label couldn't be created
                    st.write(f"Data length after dropping rows with NaN labels: {len(combined_df)}")
                    
                    # Fill features - use ffill then bfill for robustness
                    # Define feature columns (exclude price columns that are not lagged, and label)
                    # This is a common source of error: ensuring no future-leaking data is in X
                    potential_feature_cols = [col for col in combined_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'label']]
                    
                    # For any features that are all NaN (e.g. a ratio that couldn't be computed)
                    # combined_df.dropna(axis=1, how='all', inplace=True)
                    # potential_feature_cols = [col for col in potential_feature_cols if col in combined_df.columns]


                    if not potential_feature_cols:
                        st.error("No feature columns found after initial processing. Check data and feature engineering steps.")
                        st.stop()

                    combined_df[potential_feature_cols] = combined_df[potential_feature_cols].fillna(method='ffill')
                    combined_df[potential_feature_cols] = combined_df[potential_feature_cols].fillna(method='bfill') # fill any leading NaNs
                    
                    # Final drop of any rows that might still have NaNs in features (e.g., if all values in a window were NaN)
                    combined_df.dropna(subset=potential_feature_cols, inplace=True)
                    st.write(f"Data length after ffill/bfill and final drop of feature NaNs: {len(combined_df)}")

                    if combined_df.empty:
                        st.error("No data remaining after NaN handling. Check data quality, date ranges, or feature engineering parameters.")
                        st.stop()

                    features_df = combined_df.copy()
                    st.subheader("Processed Features Preview (for Model)")
                    st.dataframe(features_df.head())

                # --- 3. Train/Test Split ---
                if len(features_df) < 20: # Arbitrary small number
                    st.error(f"Not enough data ({len(features_df)} rows) for training after processing. Adjust parameters or date range.")
                    st.stop()

                # Define X and y
                # Exclude original price columns that are not features (e.g. current day's OHLVC)
                # Also exclude any columns that might be all NaN if not handled before
                X = features_df[potential_feature_cols].copy() 
                y = features_df['label'].copy()

                # Chronological split
                split_idx = int(len(X) * train_test_split_ratio)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                st.subheader("Data Split")
                st.write(f"Training set: {len(X_train)} samples")
                st.write(f"Test set: {len(X_test)} samples")

                if X_train.empty or X_test.empty:
                    st.error("Training or test set is empty after split. Adjust split ratio or data range.")
                    st.stop()
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    st.warning(f"Warning: Training target unique values: {y_train.nunique()}, Test target unique values: {y_test.nunique()}. Model might not train/evaluate well if only one class is present.")
                    if y_train.nunique() < 2: st.stop("Training target has less than 2 unique classes. Cannot train model.")


                # --- 4. Model Training ---
                with st.spinner("Training XGBoost model... This may take a moment."):
                    # Using fewer estimators for quicker Streamlit app demo
                    model, cv_scores = train_xgboost_model(X_train, y_train, k_folds=3, n_estimators=100) 
                
                st.subheader("🚀 Model Training - Cross-Validation")
                if model:
                    st.success("Model training complete!")
                    st.write("Cross-Validation Scores:")
                    st.json(cv_scores) # Nicely formats dict
                else:
                    st.error("Model training failed.")
                    st.stop()

                # --- 5. Model Evaluation ---
                st.subheader("📊 Model Evaluation on Test Set")
                if y_test.nunique() > 1:
                    with st.spinner("Evaluating model..."):
                        # The evaluate_model from model_train.py already creates a plot.
                        # We need to modify it or create a new function here to pass `ax`.
                        
                        # Re-implementing evaluation plotting part here for Streamlit
                        y_pred_test = model.predict(X_test)
                        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
                        
                        # Corrected import for sklearn.metrics
                        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
                        
                        test_accuracy = accuracy_score(y_test, y_pred_test) # Corrected: was sns.accuracy_score
                        test_roc_auc = roc_auc_score(y_test, y_pred_proba_test) # Corrected: was sns.roc_auc_score
                        cm_test = confusion_matrix(y_test, y_pred_test) # Corrected: was sns.confusion_matrix
                        
                        st.write(f"Test Set Accuracy: {test_accuracy:.4f}")
                        st.write(f"Test Set ROC AUC: {test_roc_auc:.4f}")
                        st.text("Classification Report (Test Set):")
                        st.text(classification_report(y_test, y_pred_test)) # Corrected: was sns.classification_report

                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                                    xticklabels=model.classes_, yticklabels=model.classes_)
                    st.stop()

                # --- 5. Model Evaluation ---
                st.subheader("📊 Model Evaluation on Test Set")
                if y_test.nunique() > 1:
                    with st.spinner("Evaluating model..."):
                        # The evaluate_model from model_train.py already creates a plot.
                        # We need to modify it or create a new function here to pass `ax`.
                        
                        # Re-implementing evaluation plotting part here for Streamlit
                        y_pred_test = model.predict(X_test)
                        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
                        
                        test_accuracy = sns.accuracy_score(y_test, y_pred_test)
                        test_roc_auc = sns.roc_auc_score(y_test, y_pred_proba_test)
                        cm_test = sns.confusion_matrix(y_test, y_pred_test)
                        
                        st.write(f"Test Set Accuracy: {test_accuracy:.4f}")
                        st.write(f"Test Set ROC AUC: {test_roc_auc:.4f}")
                        st.text("Classification Report (Test Set):")
                        st.text(sns.classification_report(y_test, y_pred_test))

                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                                    xticklabels=model.classes_, yticklabels=model.classes_)
                        ax_cm.set_title('Confusion Matrix (Test Set)')
                        ax_cm.set_xlabel('Predicted Label')
                        ax_cm.set_ylabel('True Label')
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                else:
                    st.warning("Test set has less than 2 unique classes. Skipping evaluation metrics that require multiple classes.")


                # --- 6. Feature Importance ---
                st.subheader("🌟 Feature Importances")
                with st.spinner("Generating feature importance plot..."):
                    fig_fi, ax_fi = plt.subplots(figsize=(10, 8)) # Adjust size as needed
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
                    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20) # Top 20
                    
                    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax_fi)
                    ax_fi.set_title('Top 20 Feature Importances')
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)

                # --- 7. Backtesting ---
                st.subheader("📈 Backtesting Results on Test Period")
                with st.spinner("Running backtest..."):
                    # Use model predictions on the test set as signals
                    # Ensure X_test used for prediction is the same as used for training (same columns, same order)
                    test_predictions = model.predict(X_test)
                    test_signals = pd.Series(test_predictions, index=X_test.index)

                    # Align signals with the original price data for the test period
                    # Ensure 'Close' is present in price_df_full
                    if 'Close' not in price_df_full.columns:
                        st.error("Critical error: 'Close' column not found in price_df_full for backtesting.")
                        st.stop()
                    
                    # Make sure indices align for price_df_full and X_test
                    # X_test.index should be a subset of price_df_full.index
                    if not X_test.index.isin(price_df_full.index).all():
                        st.error("Index mismatch between X_test and price_df_full. Cannot run backtest.")
                        st.stop()
                    
                    # Ensure test_price_data is not empty and signals align
                    if X_test.index.empty:
                        st.warning("X_test index is empty, cannot create test_price_data for backtesting.")
                        st.stop()

                    test_price_data = price_df_full.loc[X_test.index, 'Close']
                    
                    if test_price_data.empty or test_signals.empty:
                        st.warning("Test price data or signals are empty. Skipping backtest.")
                    else:
                        backtest_results_df = run_simple_backtest(test_price_data, test_signals)

                        if backtest_results_df is not None and not backtest_results_df.empty:
                            st.write("Backtest Portfolio Performance (Test Period):")
                            st.dataframe(backtest_results_df.tail())

                            strategy_returns = calculate_cumulative_returns(backtest_results_df['portfolio_value'], 
                                                                            backtest_results_df['portfolio_value'].iloc[0]) # Initial capital for this period
                            
                            # Buy and Hold for the test period
                            buy_and_hold_test_period = (test_price_data / test_price_data.iloc[0]) - 1

                            fig_bt, ax_bt = plt.subplots(figsize=(12,6))
                            ax_bt.plot(strategy_returns.index, strategy_returns, label='Strategy Returns', color='blue')
                            ax_bt.plot(buy_and_hold_test_period.index, buy_and_hold_test_period, label='Buy & Hold Returns', color='orange')
                            ax_bt.set_title('Backtest: Strategy vs. Buy & Hold (Test Period)')
                            ax_bt.set_xlabel('Date')
                            ax_bt.set_ylabel('Cumulative Returns')
                            ax_bt.legend()
                            ax_bt.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig_bt)
                            plt.close(fig_bt)
                        else:
                            st.warning("Backtest did not generate results.")
        except Exception as e:
            st.error(f"An error occurred during the analysis: {e}")
            st.exception(e) # Shows stack trace for debugging in console, and a shorter message in UI

else:
    st.info("Adjust parameters in the sidebar and click 'Run Analysis' to start.")

```
