import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Optional

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(csv_path)

def preprocess_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target (str): Name of the target column
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Processed features
        - pd.Series: Target variable
        - ColumnTransformer: Fitted preprocessor
    """
    # Separate target
    y = df[target]
    X = df.drop(columns=[target])
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False  # Simplify feature names
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Convert to DataFrame with feature names
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_processed, y, preprocessor 