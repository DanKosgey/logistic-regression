import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_metrics_summary(metrics_file: str, output_dir: Path):
    """Create visually appealing summaries of model metrics."""
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Set style
    plt.style.use('bmh')  # Using a built-in style instead of seaborn
    
    # 1. Key Metrics Bar Chart
    create_key_metrics_plot(metrics, output_dir)
    
    # 2. Radar Plot
    create_radar_plot(metrics, output_dir)
    
    # 3. Class-wise Metrics Plot
    create_class_metrics_plot(metrics, output_dir)
    
    # 4. Summary Dashboard
    create_summary_dashboard(metrics, output_dir)

def create_key_metrics_plot(metrics: dict, output_dir: Path):
    """Create a bar chart of key metrics."""
    key_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'ROC AUC': metrics['roc_auc']
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(key_metrics.keys(), key_metrics.values())
    plt.title('Model Performance Metrics', pad=20)
    plt.ylabel('Score')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # Set y-axis limit to 0-1 with some padding
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'key_metrics.png')
    plt.close()

def create_radar_plot(metrics: dict, output_dir: Path):
    """Create a radar/spider plot of model metrics."""
    # Prepare data
    metrics_for_radar = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'ROC AUC': metrics['roc_auc']
    }
    
    # Number of variables
    categories = list(metrics_for_radar.keys())
    values = list(metrics_for_radar.values())
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    values += values[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add title
    plt.title('Model Performance Radar Plot', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_plot.png')
    plt.close()

def create_class_metrics_plot(metrics: dict, output_dir: Path):
    """Create a grouped bar chart for class-wise metrics."""
    class_metrics = metrics['classification_report']
    
    # Prepare data
    classes = ['Non-Fraud (0)', 'Fraud (1)']
    metrics_by_class = {
        'Precision': [class_metrics['0']['precision'], class_metrics['1']['precision']],
        'Recall': [class_metrics['0']['recall'], class_metrics['1']['recall']],
        'F1-Score': [class_metrics['0']['f1-score'], class_metrics['1']['f1-score']]
    }
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    multiplier = 0
    
    # Plot bars for each metric
    for metric, values in metrics_by_class.items():
        offset = width * multiplier
        ax.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    # Customize the plot
    ax.set_ylabel('Score')
    ax.set_title('Class-wise Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_metrics.png')
    plt.close()

def create_summary_dashboard(metrics: dict, output_dir: Path):
    """Create a summary dashboard combining multiple visualizations."""
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Key metrics in top left
    ax1 = fig.add_subplot(gs[0, 0])
    key_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1']
    }
    colors = sns.color_palette("husl", len(key_metrics))
    bars = ax1.bar(key_metrics.keys(), key_metrics.values(), color=colors)
    ax1.set_title('Key Metrics')
    ax1.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 2. Class distribution in top right
    ax2 = fig.add_subplot(gs[0, 1])
    class_dist = [metrics['classification_report']['0']['support'],
                 metrics['classification_report']['1']['support']]
    ax2.pie(class_dist, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%',
            colors=['lightblue', 'salmon'])
    ax2.set_title('Class Distribution')
    
    # 3. Class-wise metrics in bottom
    ax3 = fig.add_subplot(gs[1, :])
    class_metrics = {
        'Non-Fraud': [metrics['classification_report']['0']['precision'],
                     metrics['classification_report']['0']['recall'],
                     metrics['classification_report']['0']['f1-score']],
        'Fraud': [metrics['classification_report']['1']['precision'],
                 metrics['classification_report']['1']['recall'],
                 metrics['classification_report']['1']['f1-score']]
    }
    
    x = np.arange(3)
    width = 0.35
    ax3.bar(x - width/2, class_metrics['Non-Fraud'], width, label='Non-Fraud',
            color='lightblue')
    ax3.bar(x + width/2, class_metrics['Fraud'], width, label='Fraud',
            color='salmon')
    
    ax3.set_ylabel('Score')
    ax3.set_title('Class-wise Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax3.legend()
    
    plt.suptitle('Fraud Detection Model Performance Dashboard', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_dashboard.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example usage
    metrics_file = "results/test/metrics.json"
    output_dir = Path("results/test")
    create_metrics_summary(metrics_file, output_dir) 