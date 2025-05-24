import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from utils import get_config

# Load configuration
config = get_config()
viz_config = config['visualization']

def predict_new_data(model, preprocessor, new_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Make predictions on new data using the trained model and preprocessor.
    
    Args:
        model: Trained model with predict and predict_proba methods
        preprocessor: Fitted preprocessor for feature transformation
        new_data: DataFrame containing new data to predict on
        
    Returns:
        Tuple of (DataFrame with predictions, metrics dictionary)
    """
    # Preprocess the new data
    X_new = preprocessor.transform(new_data)
    
    # Make predictions
    y_pred = model.predict(X_new)
    y_prob = model.predict_proba(X_new)[:, 1]
    
    # Add predictions to the original dataframe
    results = new_data.copy()
    results['predicted_fraud'] = y_pred
    results['fraud_probability'] = y_prob
    
    # Calculate prediction statistics
    metrics = {
        'total_transactions': len(results),
        'predicted_frauds': sum(y_pred),
        'fraud_rate': sum(y_pred) / len(y_pred)
    }
    
    return results, metrics

def plot_probability_distribution(predictions: pd.DataFrame) -> None:
    """
    Plot the distribution of fraud probabilities.
    
    Args:
        predictions: DataFrame containing fraud_probability column
    """
    plt.figure(figsize=tuple(viz_config['probability_dist']['figsize']))
    plt.hist(predictions['fraud_probability'], 
            bins=viz_config['probability_dist']['n_bins'],
            edgecolor='black')
    plt.title('Distribution of Fraud Probabilities')
    plt.xlabel('Probability of Fraud')
    plt.ylabel('Number of Transactions')
    plt.axvline(x=0.5, 
                color=viz_config['probability_dist']['threshold_color'],
                linestyle='--',
                label='Decision Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_high_risk_transactions(predictions: pd.DataFrame, 
                             threshold: float = None) -> pd.DataFrame:
    """
    Get transactions with high fraud probability for review.
    
    Args:
        predictions: DataFrame containing predictions
        threshold: Probability threshold for high-risk transactions
        
    Returns:
        DataFrame containing high-risk transactions
    """
    if threshold is None:
        threshold = config['prediction']['high_risk_threshold']
        
    high_risk = predictions[predictions['fraud_probability'] >= threshold].copy()
    high_risk = high_risk.sort_values('fraud_probability', ascending=False)
    return high_risk 