import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost_model(X: pd.DataFrame, y: pd.Series, k_folds: int = 5, random_state: int = 42, 
                        n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
    """
    Trains an XGBoost model using Stratified K-Fold cross-validation and then on the full dataset.

    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable.
        k_folds (int): Number of folds for Stratified K-Fold cross-validation.
        random_state (int): Random state for reproducibility.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Learning rate.
        max_depth (int): Maximum depth of a tree.

    Returns:
        tuple: 
            - model: Trained XGBoost classifier on the full dataset.
            - cv_scores (dict): Dictionary containing average cross-validation accuracy and ROC AUC.
    """
    if not isinstance(X, pd.DataFrame):
        print("Error: X must be a pandas DataFrame.")
        return None, {}
    if not isinstance(y, pd.Series):
        print("Error: y must be a pandas Series.")
        return None, {}
    if X.shape[0] != y.shape[0]:
        print("Error: X and y must have the same number of samples.")
        return None, {}

    print(f"Starting XGBoost training with {k_folds}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    cv_accuracy_scores = []
    cv_roc_auc_scores = []

    # Define the model
    model_proto = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False, # Recommended for newer XGBoost versions
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{k_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_fold = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        model_fold.fit(X_train, y_train,
                       early_stopping_rounds=10, # Optional: for early stopping
                       eval_set=[(X_val, y_val)],
                       verbose=False) # Suppress XGBoost training output during CV

        preds_val = model_fold.predict(X_val)
        proba_val = model_fold.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds_val)
        roc_auc = roc_auc_score(y_val, proba_val)
        
        cv_accuracy_scores.append(acc)
        cv_roc_auc_scores.append(roc_auc)
        print(f"    Fold {fold + 1} - Validation Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")

    avg_cv_accuracy = np.mean(cv_accuracy_scores)
    avg_cv_roc_auc = np.mean(cv_roc_auc_scores)
    
    print(f"\nAverage Cross-Validation Accuracy: {avg_cv_accuracy:.4f}")
    print(f"Average Cross-Validation ROC AUC: {avg_cv_roc_auc:.4f}")

    # Train final model on the entire dataset
    print("\nTraining final model on the entire dataset...")
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    final_model.fit(X, y, verbose=False) # Suppress verbose output for final training
    print("Final model training complete.")
    
    cv_scores = {
        'avg_cv_accuracy': avg_cv_accuracy,
        'avg_cv_roc_auc': avg_cv_roc_auc
    }
    
    return final_model, cv_scores

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the test set.

    Args:
        model: Trained model (e.g., XGBoost classifier).
        X_test (pd.DataFrame): Features of the test set.
        y_test (pd.Series): Target variable of the test set.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if model is None:
        print("Error: Model is None. Cannot evaluate.")
        return {}
    if not isinstance(X_test, pd.DataFrame) or not isinstance(y_test, pd.Series):
        print("Error: X_test must be a DataFrame and y_test a Series.")
        return {}

    print("\nEvaluating model on the test set...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True) # Get as dict for easier return

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Test Set ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred)) # Print formatted string report

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(), # Convert numpy array to list for easier serialization if needed
        'classification_report': cr
    }
    
    return evaluation_metrics

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20):
    """
    Plots the top N feature importances from the trained model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list[str]): List of feature names corresponding to X.
        top_n (int): Number of top features to display.
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        print("Error: Model is None or does not have feature_importances_ attribute.")
        return
    if not feature_names:
        print("Error: feature_names list is empty.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Get top N features
    top_features_df = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, top_n * 0.3 if top_n * 0.3 > 6 else 6)) # Adjust height based on N
    sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("--- Model Training Module Test ---")

    # 1. Create Sample Data
    n_samples = 100
    n_features = 15 # Increased for better feature importance demo
    X_sample = pd.DataFrame(np.random.rand(n_samples, n_features), 
                            columns=[f'feature_{i+1}' for i in range(n_features)])
    
    # Create somewhat imbalanced labels for more realistic StratifiedKFold scenario
    # And ensure both classes are present in smaller test splits
    y_sample_np = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    y_sample = pd.Series(y_sample_np, name='target')

    print(f"\nGenerated X_sample with shape: {X_sample.shape}")
    print(f"Generated y_sample with shape: {y_sample.shape}, value counts:\n{y_sample.value_counts(normalize=True)}")


    # 2. Simulate Train-Test Split
    # Ensure that both train and test sets have representatives of both classes.
    # This simple split might not guarantee it for very small test sets and high imbalance.
    # For robust testing, one might use sklearn.model_selection.train_test_split with stratify=y_sample
    split_idx = int(n_samples * 0.8)
    X_train_sample, y_train_sample = X_sample.iloc[:split_idx], y_sample.iloc[:split_idx]
    X_test_sample, y_test_sample = X_sample.iloc[split_idx:], y_sample.iloc[split_idx:]

    # Verify class distribution in splits
    print(f"\nTraining set y_train_sample value counts:\n{y_train_sample.value_counts(normalize=True)}")
    if y_test_sample.nunique() < 2:
        print(f"Warning: Test set y_test_sample has only {y_test_sample.nunique()} unique classes. This might cause issues in evaluation.")
        # Try to re-balance y_test_sample for testing if it's problematic
        if len(y_test_sample) > 1 and y_test_sample.nunique() < 2:
            # Flip one label if all are same to ensure roc_auc works
            y_test_sample_list = y_test_sample.tolist()
            if all(lbl == y_test_sample_list[0] for lbl in y_test_sample_list):
                y_test_sample_list[0] = 1 - y_test_sample_list[0]
                y_test_sample = pd.Series(y_test_sample_list, index=X_test_sample.index, name='target')
                print(f"Adjusted y_test_sample for testing. New counts:\n{y_test_sample.value_counts(normalize=True)}")

    print(f"X_train_sample shape: {X_train_sample.shape}, y_train_sample shape: {y_train_sample.shape}")
    print(f"X_test_sample shape: {X_test_sample.shape}, y_test_sample shape: {y_test_sample.shape}")


    # 3. Test train_xgboost_model
    print("\n--- Testing train_xgboost_model ---")
    # Using fewer estimators for faster testing
    trained_model, cv_results = train_xgboost_model(X_train_sample, y_train_sample, k_folds=3, n_estimators=50) 
                                                    
    if trained_model:
        print("\nCV Results from training:")
        for key, value in cv_results.items():
            print(f"  {key}: {value:.4f}")

        # 4. Test evaluate_model
        print("\n--- Testing evaluate_model ---")
        if y_test_sample.nunique() > 1: # Ensure evaluate_model can calculate ROC AUC etc.
            eval_metrics = evaluate_model(trained_model, X_test_sample, y_test_sample)
            if eval_metrics:
                print("\nEvaluation Metrics from test set:")
                # Print selected metrics, confusion matrix and report are printed by the function
                print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"  ROC AUC: {eval_metrics['roc_auc']:.4f}")
        else:
            print("Skipping evaluate_model as y_test_sample does not have multiple classes.")

        # 5. Test plot_feature_importance
        print("\n--- Testing plot_feature_importance ---")
        plot_feature_importance(trained_model, list(X_sample.columns), top_n=10)
        print("Feature importance plot should have been displayed.")
    else:
        print("Model training failed. Skipping evaluation and feature importance plotting.")

    print("\n--- Model Training Module Test Complete ---")
```
