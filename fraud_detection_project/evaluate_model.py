import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, Any
from utils import get_config

# Load configuration
config = get_config()
viz_config = config['visualization']

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, labels: list = None) -> None:
    """Plot and save confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=tuple(viz_config['confusion_matrix']['figsize']))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=viz_config['confusion_matrix']['cmap'],
        xticklabels=labels or ['Non-Fraud', 'Fraud'],
        yticklabels=labels or ['Non-Fraud', 'Fraud']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> float:
    """Plot and save ROC curve, return AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=tuple(viz_config['roc_curve']['figsize']))
    plt.plot(fpr, tpr, 
             color=viz_config['roc_curve']['line_color'],
             lw=viz_config['roc_curve']['line_width'],
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curve.png')
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path) -> float:
    """Plot and save Precision-Recall curve, return average precision score."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=tuple(viz_config['pr_curve']['figsize']))
    plt.plot(recall, precision,
             color=viz_config['pr_curve']['line_color'],
             lw=viz_config['pr_curve']['line_width'],
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path / 'precision_recall_curve.png')
    plt.close()
    
    return avg_precision

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate all relevant classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_curve(y_true, y_prob)[0].mean(),
        'avg_precision': average_precision_score(y_true, y_prob)
    }

def evaluate_model(model, X: np.ndarray, y: np.ndarray, dataset_name: str = 'validation') -> Dict[str, Any]:
    """
    Evaluate model performance with multiple metrics and save results.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X: Features
        y: True labels
        dataset_name: Name of the dataset being evaluated (e.g., 'validation' or 'test')
        
    Returns:
        Dict containing evaluation metrics
    """
    # Create results directory if it doesn't exist
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Create dataset-specific directory
    dataset_dir = results_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate and save metrics
    metrics = calculate_metrics(y, y_pred, y_prob)
    
    # Add classification report
    report = classification_report(y, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    # Save metrics to JSON file
    with open(dataset_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save plots
    plot_confusion_matrix(y, y_pred, dataset_dir)
    metrics['roc_auc'] = plot_roc_curve(y, y_prob, dataset_dir)
    metrics['avg_precision'] = plot_precision_recall_curve(y, y_prob, dataset_dir)
    
    # Create a summary text file
    with open(dataset_dir / 'summary.txt', 'w') as f:
        f.write(f"Model Evaluation Results - {dataset_name.capitalize()} Set\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall: {metrics['recall']:.3f}\n")
        f.write(f"F1 Score: {metrics['f1']:.3f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"Average Precision: {metrics['avg_precision']:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y, y_pred))
    
    return metrics 