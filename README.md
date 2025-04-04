# Titanic Survival Prediction - Ensemble Model

## Overview

This repository contains an ensemble machine learning model for the classic "Titanic: Machine Learning from Disaster" Kaggle competition. The goal of this project is to predict which passengers survived the Titanic shipwreck based on features like passenger class, sex, age, and more.

The model combines three powerful classifiers (Random Forest, Gradient Boosting, and Logistic Regression) using a voting strategy to achieve high prediction accuracy.

## Features

- **Comprehensive Feature Engineering**
  - Title extraction from passenger names
  - Family size calculation
  - Binning of continuous variables
  - Creation of derived features (e.g., IsAlone)

- **Robust Preprocessing Pipeline**
  - Separate handling for numeric and categorical features
  - Intelligent missing value imputation
  - Feature scaling
  - One-hot encoding for categorical variables

- **Ensemble Classification**
  - Three complementary base models
  - Soft voting strategy using predicted probabilities
  - Hyperparameter tuning via GridSearchCV

- **Performance Evaluation**
  - Cross-validation scoring
  - Individual model performance analysis
  - Final prediction file generation

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install required packages using:

```bash
pip install pandas numpy scikit-learn
```

## Usage

### 1. Download Data

First, download the Titanic dataset from Kaggle:
- Go to [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
- Download `train.csv` and `test.csv`
- Place both files in the project directory

### 2. Run the Model

Execute the main script:

```bash
python titanic_ensemble.py
```

This will:
- Load and preprocess the data
- Train individual models with optimized hyperparameters
- Create the ensemble model
- Generate predictions
- Create a submission file (`ensemble_submission.csv`)

### 3. Submit Results

Upload the generated `ensemble_submission.csv` file to Kaggle to see your score.

## Code Structure

- **Data Loading**: `load_and_prepare_data()` function handles initial data import
- **Feature Engineering**: `engineer_features()` creates and transforms features
- **Preprocessing**: `create_preprocessor()` builds the preprocessing pipeline
- **Model Building**: `build_ensemble_model()` creates the base voting classifier
- **Model Training**: `train_and_tune_model()` optimizes and combines models
- **Main Execution**: `titanic_prediction()` orchestrates the entire workflow

## Model Performance

The ensemble approach typically achieves 80-83% accuracy on Kaggle's test set. Individual model contributions are analyzed and displayed during execution.

## Customization

You can easily modify the hyperparameter search space in the `train_and_tune_model()` function to explore different model configurations. Additional feature engineering ideas can be implemented in the `engineer_features()` function.

## License

[MIT License](LICENSE)

## Acknowledgments

- Kaggle for hosting the competition
- The scikit-learn team for their excellent machine learning library
