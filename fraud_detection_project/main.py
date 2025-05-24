import argparse
import pandas as pd
import joblib
from pathlib import Path
from data_preprocessing import load_data, preprocess_data
from train_model import split_data, train_model
from evaluate_model import evaluate_model
from predict import predict_new_data, plot_probability_distribution, get_high_risk_transactions
from visualize_metrics import create_metrics_summary
from utils import get_config, ensure_directories

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the training data CSV file')
    parser.add_argument('--target', type=str, default='is_fraud',
                      help='Name of the target column')
    parser.add_argument('--new_data', type=str,
                      help='Path to new data CSV file for prediction')
    parser.add_argument('--model_output', type=str,
                      help='Path to save the trained model')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main function to run the fraud detection pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Ensure output directories exist
    ensure_directories(config)
    
    print("Loading data...")
    df = load_data(args.data)
    
    print("Preprocessing data...")
    X_processed, y, preprocessor = preprocess_data(df, args.target)
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_processed, y,
        test_size=config['training']['test_size'],
        val_size=config['training']['validation_size'],
        random_state=config['training']['random_state']
    )
    
    print("Training model...")
    model, train_metrics = train_model(
        X_train, y_train,
        class_weight=config['model']['class_weight'],
        max_iter=config['model']['max_iter'],
        cv=config['training']['cv_folds']
    )
    
    print("\nCross-validation results:")
    print(f"Mean ROC-AUC: {train_metrics['cv_mean_auc']:.3f} (+/- {train_metrics['cv_std_auc']*2:.3f})")
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, X_val, y_val, dataset_name='validation')
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name='test')
    
    print("\nGenerating performance visualizations...")
    create_metrics_summary(
        str(Path(config['paths']['results_dir']) / 'validation' / 'metrics.json'),
        Path(config['paths']['results_dir']) / 'validation'
    )
    create_metrics_summary(
        str(Path(config['paths']['results_dir']) / 'test' / 'metrics.json'),
        Path(config['paths']['results_dir']) / 'test'
    )
    
    # Save the model and preprocessor
    model_path = Path(config['paths']['models_dir']) / (
        args.model_output or config['paths']['default_model_name']
    )
    print(f"\nSaving model to {model_path}...")
    joblib.dump({'model': model, 'preprocessor': preprocessor}, model_path)
    
    # If new data is provided, make predictions
    if args.new_data:
        print("\nMaking predictions on new data...")
        new_df = load_data(args.new_data)
        predictions, pred_metrics = predict_new_data(model, preprocessor, new_df)
        
        print("\nPrediction Summary:")
        print(f"Total transactions: {pred_metrics['total_transactions']}")
        print(f"Predicted frauds: {pred_metrics['predicted_frauds']}")
        print(f"Fraud rate: {pred_metrics['fraud_rate']:.2%}")
        
        # Plot probability distribution
        plot_probability_distribution(predictions)
        
        # Get high-risk transactions
        high_risk = get_high_risk_transactions(
            predictions,
            threshold=config['prediction']['high_risk_threshold']
        )
        if len(high_risk) > 0:
            print("\nHigh-risk transactions (probability >= {:.1f}):".format(
                config['prediction']['high_risk_threshold']
            ))
            print(high_risk)
            
            # Save predictions
            results_dir = Path(config['paths']['results_dir'])
            predictions.to_csv(results_dir / 'predictions.csv', index=False)
            high_risk.to_csv(results_dir / 'high_risk_transactions.csv', index=False)

if __name__ == '__main__':
    main() 