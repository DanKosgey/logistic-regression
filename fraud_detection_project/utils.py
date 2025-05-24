import json
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file {config_path}")

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Raises:
        ValueError: If required configuration keys are missing or invalid
    """
    required_sections = [
        'data',
        'data_preprocessing',
        'model',
        'training',
        'prediction',
        'paths',
        'visualization'
    ]
    
    # Check for required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data configuration
    if 'training_data' not in config['data']:
        raise ValueError("Missing training_data configuration in data section")
    if 'path' not in config['data']['training_data']:
        raise ValueError("Missing path in training_data configuration")
    if 'target_column' not in config['data']['training_data']:
        raise ValueError("Missing target_column in training_data configuration")
    
    # Validate paths
    required_paths = ['data_dir', 'models_dir', 'results_dir']
    for path in required_paths:
        if path not in config['paths']:
            raise ValueError(f"Missing required path: {path}")
    
    # Validate numeric values
    if config['training']['test_size'] <= 0 or config['training']['test_size'] >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if config['training']['validation_size'] <= 0 or config['training']['validation_size'] >= 1:
        raise ValueError("validation_size must be between 0 and 1")
    
    if config['prediction']['high_risk_threshold'] <= 0 or config['prediction']['high_risk_threshold'] > 1:
        raise ValueError("high_risk_threshold must be between 0 and 1")

def get_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load and validate configuration.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Validated configuration dictionary
    """
    config = load_config(config_path)
    validate_config(config)
    return config

def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure that all required directories exist.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    for dir_name in ['data_dir', 'models_dir', 'results_dir']:
        Path(config['paths'][dir_name]).mkdir(exist_ok=True)

def get_data_path(config: Dict[str, Any], data_type: str = 'training') -> Path:
    """
    Get the full path for a data file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        data_type (str): Type of data path to get ('training' or 'prediction')
        
    Returns:
        Path: Full path to the data file
        
    Raises:
        ValueError: If data_type is invalid
    """
    if data_type not in ['training', 'prediction']:
        raise ValueError("data_type must be either 'training' or 'prediction'")
        
    data_key = f"{data_type}_data"
    if data_key not in config['data']:
        raise ValueError(f"Missing {data_key} configuration")
        
    return Path(config['data'][data_key]['path']) 