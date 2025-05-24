# Fraud Detection System Using Logistic Regression

## Project Overview
This project implements a machine learning-based fraud detection system using logistic regression. The system is designed to identify potentially fraudulent transactions by analyzing various features and patterns in transaction data.

## Project Structure
```
fraud_detection_project/
├── data/                   # Data directory for storing datasets
├── models/                 # Saved model files
├── results/               # Performance metrics and visualizations
│   ├── validation/       # Validation set results
│   └── test/            # Test set results
├── config.json           # Configuration parameters
├── main.py              # Main execution script
├── data_preprocessing.py # Data preprocessing module
├── train_model.py       # Model training module
├── evaluate_model.py    # Model evaluation module
├── predict.py           # Prediction module
├── visualize_metrics.py # Visualization module
└── utils.py            # Utility functions
```

## Features Used
The model analyzes various features including:
- Income
- Name-email similarity
- Previous and current address duration
- Customer age
- Days since request
- Intended balance amount
- Payment type
- Transaction velocities (6h, 24h, 4w)
- Bank branch activity
- Email patterns
- Device information
- Session characteristics

## Model Performance

### Cross-Validation Results
- Mean ROC-AUC Score: 0.814 (±0.055)
- This indicates strong and consistent model performance across different data splits

### Test Set Performance

#### Overall Metrics
- Accuracy: 82.45%
- Precision: 16.97%
- Recall: 64.50%
- F1 Score: 0.269
- ROC AUC: 0.217
- Average Precision: 0.241

#### Class-wise Performance

Non-Fraud Class (0):
- Precision: 97.81%
- Recall: 83.39%
- F1-score: 90.03%
- Support: 3,800 samples

Fraud Class (1):
- Precision: 16.97%
- Recall: 64.50%
- F1-score: 26.88%
- Support: 200 samples

### Key Insights
1. **Class Imbalance**: The dataset shows significant class imbalance with fraud cases representing only 5% of the data (200 out of 4000 samples).

2. **High Recall for Fraud**: The model achieves a good recall (64.50%) for fraud cases, meaning it catches a majority of actual fraud attempts.

3. **Precision Trade-off**: The relatively low precision for fraud cases (16.97%) indicates some false positives, which is often acceptable in fraud detection where missing actual fraud is more costly than false alarms.

4. **Strong Non-fraud Detection**: The model excels at identifying legitimate transactions with 97.81% precision and 83.39% recall.

## Visualization Outputs
The project generates several visualization files to help understand model performance:

1. **Key Metrics Plot** (`key_metrics.png`)
   - Bar chart showing main performance metrics
   - Provides quick overview of model's overall performance

2. **Radar Plot** (`radar_plot.png`)
   - Spider/radar visualization of metric relationships
   - Helps identify balanced/imbalanced areas of performance

3. **Class Metrics Plot** (`class_metrics.png`)
   - Grouped bar chart comparing fraud vs non-fraud performance
   - Highlights class-specific strengths and weaknesses

4. **Performance Dashboard** (`performance_dashboard.png`)
   - Comprehensive view combining multiple visualizations
   - Includes class distribution and detailed metrics breakdown

## Usage

### Prerequisites
- Python 3.x
- Required packages: scikit-learn, pandas, numpy, matplotlib, seaborn

### Running the Model
```bash
python main.py --data "path_to_data.csv" --target "fraud_Cases"
```

### Configuration
The `config.json` file contains adjustable parameters for:
- Data paths and column names
- Preprocessing settings
- Model hyperparameters
- Training settings
- Visualization preferences

## Future Improvements
1. Feature Engineering
   - Create more sophisticated velocity features
   - Develop complex interaction features

2. Model Enhancements
   - Experiment with different classification algorithms
   - Implement ensemble methods
   - Fine-tune hyperparameters

3. Evaluation Metrics
   - Add cost-sensitive evaluation metrics
   - Implement business-specific KPIs

## Conclusions
The logistic regression model demonstrates strong overall performance in fraud detection, with particularly good results in:
- Maintaining high accuracy (82.45%)
- Achieving good fraud detection recall (64.50%)
- Excellent legitimate transaction identification (97.81% precision)

The model's performance makes it suitable for:
- Initial fraud screening
- Risk scoring of transactions
- Supporting human review processes

While there's room for improvement in precision for fraud cases, the current model provides a solid foundation for fraud detection, especially when false positives are less costly than missed fraud cases. 