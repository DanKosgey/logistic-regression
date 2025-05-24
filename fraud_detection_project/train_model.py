from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from typing import Tuple, Dict, Any

def split_data(X, y, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Target variable
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of remaining data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                class_weight: str = 'balanced', 
                max_iter: int = 1000,
                cv: int = 5) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a logistic regression model with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weight: Weight balancing strategy for classes
        max_iter: Maximum iterations for convergence
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (fitted model, training metrics dictionary)
    """
    # Initialize model
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Fit the model on full training data
    model.fit(X_train, y_train)
    
    # Collect training metrics
    metrics = {
        'cv_scores': cv_scores,
        'cv_mean_auc': cv_scores.mean(),
        'cv_std_auc': cv_scores.std(),
    }
    
    return model, metrics 